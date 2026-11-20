"""Cold start: a paused Aurora cluster must produce a fast 503 'warming_up', never a gateway timeout."""

import boto3
import httpx
import pytest
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from knowledgemcp import api
from knowledgemcp.auth import ApiKeyAuth, hash_key
from knowledgemcp.db import DataApiExecutor, DatabaseWaking
from knowledgemcp.mcp_server import HttpBackend, KnowledgeBaseWarmingUp

CLUSTER = "arn:aws:rds:us-west-2:123456789012:cluster:kmcp"
SECRET = "arn:aws:secretsmanager:us-west-2:123456789012:secret:kmcp-abc"
KEY = "kmcp_test"


def resuming_client(times):
    client = boto3.client("rds-data", region_name="us-west-2")
    stub = Stubber(client)
    for _ in range(times):
        stub.add_client_error("execute_statement", service_error_code="DatabaseResumingException")
    return client, stub


def test_wait_never_exceeds_budget_and_raises_database_waking():
    client, stub = resuming_client(10)
    sleeps = []
    with stub, pytest.raises(DatabaseWaking):
        DataApiExecutor(client, CLUSTER, SECRET, "kmcp", max_wait_s=15, sleep=sleeps.append).execute("SELECT 1")
    assert sum(sleeps) == 15  # 1 + 2 + 4 + 8 = 15: capped exactly at the budget, never over
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


def test_ping_starts_resume_without_waiting():
    client, stub = resuming_client(1)
    with stub:
        assert DataApiExecutor(client, CLUSTER, SECRET, "kmcp").ping() is False


class WakingStore:
    def ping(self):
        return False

    def list_documents(self, teams):
        raise DatabaseWaking("resuming")


@pytest.fixture
def client():
    api.app.dependency_overrides = {
        api.get_store: lambda: WakingStore(),
        api.get_auth: lambda: ApiKeyAuth({hash_key(KEY): {"user_id": "u", "teams": ["hr"]}})}
    yield TestClient(api.app)
    api.app.dependency_overrides = {}


def test_api_answers_503_with_retry_after(client):
    r = client.get("/documents", headers={"x-api-key": KEY})
    assert r.status_code == 503 and r.headers["retry-after"] == "10"
    assert r.json()["error"] == "warming_up"


def test_warmup_endpoint(client):
    assert client.post("/warmup").status_code == 401
    assert client.post("/warmup", headers={"x-api-key": KEY}).json() == {"database": "resuming"}


def mock_api(warming_responses):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= warming_responses:
            return httpx.Response(503, json={"error": "warming_up", "retry_after_s": 10})
        return httpx.Response(200, json=[{"doc_id": "d", "title": "T", "team": "hr", "chunks": 1}])

    return httpx.Client(base_url="http://kmcp", transport=httpx.MockTransport(handler)), calls


def test_mcp_backend_waits_through_warmup_within_budget():
    http, calls = mock_api(warming_responses=2)
    sleeps = []
    backend = HttpBackend("http://kmcp", KEY, client=http, wait_budget_s=45, sleep=sleeps.append)
    assert backend.list_documents()[0]["title"] == "T"
    assert calls["n"] == 3 and sleeps == [10.0, 10.0]


def test_mcp_backend_gives_up_at_its_budget():
    http, _ = mock_api(warming_responses=99)
    sleeps = []
    backend = HttpBackend("http://kmcp", KEY, client=http, wait_budget_s=8, sleep=sleeps.append)
    with pytest.raises(KnowledgeBaseWarmingUp, match="warming_up"):
        backend.list_documents()
    assert sum(sleeps) == 8  # a caller with its own deadline is never held past its budget
