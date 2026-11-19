import os
from pathlib import Path

import pytest

# Never touch real AWS from tests.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("AWS_REGION", "us-west-2")
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ.setdefault("POWERTOOLS_TRACE_DISABLED", "1")
os.environ.setdefault("POWERTOOLS_METRICS_NAMESPACE", "KnowledgeMCP")

from knowledgemcp.embeddings import HashEmbedder  # noqa: E402
from knowledgemcp.ingest import Ingestor  # noqa: E402
from knowledgemcp.store import InMemoryStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "sample_docs"


@pytest.fixture
def embedder():
    return HashEmbedder(512)


@pytest.fixture
def memory_store(embedder):
    store = InMemoryStore()
    Ingestor(store, embedder).ingest_folder(SAMPLE_DOCS)
    return store


def _pg_dsn() -> str | None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    try:
        import psycopg

        psycopg.connect(dsn, connect_timeout=2).close()
        return dsn
    except Exception:
        return None


@pytest.fixture
def pg_store():
    dsn = _pg_dsn()
    if not dsn:
        pytest.skip("set DATABASE_URL to a Postgres with pgvector to run integration tests")
    from knowledgemcp.db import PsycopgExecutor
    from knowledgemcp.store import PgVectorStore

    ex = PsycopgExecutor(dsn)
    ex.execute("DROP TABLE IF EXISTS chunks")
    ex.execute("DROP TABLE IF EXISTS documents")
    store = PgVectorStore(ex, dim=512)
    store.ensure_schema()
    return store
