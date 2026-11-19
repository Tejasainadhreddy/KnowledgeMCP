import json

import boto3
import pytest
from conftest import SAMPLE_DOCS
from moto import mock_aws

from knowledgemcp.ingest import Ingestor, doc_id_for, handle_s3_event, team_for
from knowledgemcp.store import InMemoryStore


def test_team_comes_from_first_folder():
    assert team_for("HR/pto.md") == "hr"
    with pytest.raises(ValueError):
        team_for("pto.md")


def test_ingest_is_idempotent(embedder):
    store = InMemoryStore()
    ing = Ingestor(store, embedder)
    first = ing.ingest_bytes("hr/a.md", b"# A\nhello world", "file://a")
    again = ing.ingest_bytes("hr/a.md", b"# A\nhello world", "file://a")
    changed = ing.ingest_bytes("hr/a.md", b"# A\nhello there", "file://a")
    assert (first.status, again.status, changed.status) == ("indexed", "unchanged", "indexed")
    assert ing.ingest_bytes("hr/a.docx", b"x", "file://x").status == "skipped"


@mock_aws
def test_s3_event_ingests_and_deletes(embedder):
    s3 = boto3.client("s3", region_name="us-west-2")
    s3.create_bucket(Bucket="docs", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    s3.put_object(Bucket="docs", Key="hr/pto policy.md", Body=(SAMPLE_DOCS / "hr/pto-policy.md").read_bytes())
    store = InMemoryStore()
    ing = Ingestor(store, embedder)
    event = {"Records": [{"eventName": "ObjectCreated:Put",
                          "s3": {"bucket": {"name": "docs"}, "object": {"key": "hr/pto+policy.md"}}}]}
    [result] = handle_s3_event(event, ing, s3)
    assert result.status == "indexed" and result.chunks > 0
    assert store.docs[doc_id_for("hr/pto policy.md")].source_uri == "s3://docs/hr/pto policy.md"

    event["Records"][0]["eventName"] = "ObjectRemoved:Delete"
    assert handle_s3_event(event, ing, s3)[0].status == "deleted"
    assert not store.docs and not store.chunks
    json.dumps([r.__dict__ for r in [result]])  # results are JSON-serializable for logs


def test_memory_store_enforces_teams(memory_store, embedder):
    hits = memory_store.vector_search(embedder.embed(["parental leave weeks"])[0], ["engineering", "public"], 50)
    assert hits and all(memory_store.docs[h.doc_id].team in {"engineering", "public"} for h in hits)
    assert memory_store.get_document(doc_id_for("hr/pto-policy.md"), ["engineering"]) is None


@pytest.mark.integration
def test_pgvector_store_end_to_end(pg_store, embedder):
    ing = Ingestor(pg_store, embedder)
    results = ing.ingest_folder(SAMPLE_DOCS)
    assert all(r.status == "indexed" for r in results) and len(results) == 8

    q = "how many weeks of parental leave"
    hr_hits = pg_store.keyword_search(q, ["hr", "public"], 5)
    assert hr_hits[0].title == "Paid Time Off Policy"
    assert all(h.title != "Paid Time Off Policy" for h in pg_store.keyword_search(q, ["engineering", "public"], 5))

    vec_hits = pg_store.vector_search(embedder.embed([q])[0], ["hr"], 3)
    assert vec_hits and all(h.doc_id in {doc_id_for("hr/pto-policy.md"), doc_id_for("hr/remote-work-policy.md")}
                            for h in vec_hits)

    pdf = pg_store.keyword_search("department head approval above 1,000 dollars", ["finance"], 3)
    assert pdf[0].page == 2

    docs = pg_store.list_documents(["public"])
    assert {d.team for d in docs} == {"public"}
    doc, chunks = pg_store.get_document(doc_id_for("hr/pto-policy.md"), ["hr"])
    assert doc.num_chunks == len(chunks) > 0
    assert pg_store.get_document(doc_id_for("hr/pto-policy.md"), ["finance"]) is None

    assert ing.ingest_bytes("hr/pto-policy.md", (SAMPLE_DOCS / "hr/pto-policy.md").read_bytes(), "x").status \
        == "unchanged"
