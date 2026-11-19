import io
import json

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from knowledgemcp.db import DataApiExecutor, data_api_param, to_pyformat
from knowledgemcp.embeddings import BedrockTitanEmbedder, HashEmbedder, to_pgvector

CLUSTER = "arn:aws:rds:us-west-2:123456789012:cluster:kmcp"
SECRET = "arn:aws:secretsmanager:us-west-2:123456789012:secret:kmcp-abc"


def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder(64)
    a, b = e.embed(["refund policy", "refund policy"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_titan_embedder_calls_bedrock_with_dimensions():
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    stub = Stubber(client)
    body = io.BytesIO(json.dumps({"embedding": [0.1] * 256, "inputTextTokenCount": 7}).encode())
    stub.add_response(
        "invoke_model",
        {"body": body, "contentType": "application/json"},
        {"modelId": "amazon.titan-embed-text-v2:0", "body": json.dumps(
            {"inputText": "hello", "dimensions": 256, "normalize": True}),
         "contentType": "application/json", "accept": "application/json"},
    )
    with stub:
        e = BedrockTitanEmbedder(client, dim=256)
        assert len(e.embed(["hello"])[0]) == 256
        assert e.token_count(["hello"]) == 7


def test_titan_rejects_bad_dimension():
    with pytest.raises(ValueError):
        BedrockTitanEmbedder(object(), dim=300)


def test_to_pyformat_keeps_casts():
    sql = "SELECT CAST(:q AS vector), x::text FROM t WHERE a = :a AND b LIKE '50%'"
    assert to_pyformat(sql) == "SELECT CAST(%(q)s AS vector), x::text FROM t WHERE a = %(a)s AND b LIKE '50%%'"


def test_data_api_param_types():
    assert data_api_param("n", None) == {"name": "n", "value": {"isNull": True}}
    assert data_api_param("n", 3)["value"] == {"longValue": 3}
    assert data_api_param("n", 0.5)["value"] == {"doubleValue": 0.5}
    assert data_api_param("n", True)["value"] == {"booleanValue": True}
    assert data_api_param("n", "x")["value"] == {"stringValue": "x"}
    assert to_pgvector([0.5, -1.0]) == "[0.500000,-1.000000]"


def test_data_api_executor_retries_while_aurora_resumes():
    client = boto3.client("rds-data", region_name="us-west-2")
    stub = Stubber(client)
    stub.add_client_error("execute_statement", service_error_code="DatabaseResumingException")
    stub.add_response("execute_statement", {"formattedRecords": json.dumps([{"x": 1}])})
    sleeps = []
    with stub:
        ex = DataApiExecutor(client, CLUSTER, SECRET, "kmcp", sleep=sleeps.append)
        assert ex.execute("SELECT 1 AS x") == [{"x": 1}]
    assert sleeps == [1.0]


def test_data_api_executor_does_not_retry_real_errors():
    client = boto3.client("rds-data", region_name="us-west-2")
    stub = Stubber(client)
    stub.add_client_error("execute_statement", service_error_code="BadRequestException")
    with stub, pytest.raises(ClientError):
        DataApiExecutor(client, CLUSTER, SECRET, "kmcp", sleep=lambda _: None).execute("SELEC 1")
