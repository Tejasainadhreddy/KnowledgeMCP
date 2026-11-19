import boto3
import pytest
from botocore.stub import Stubber

from knowledgemcp.models import Hit, User
from knowledgemcp.rerank import BedrockReranker, LexicalReranker, terms
from knowledgemcp.search import HybridRetriever, reciprocal_rank_fusion

RERANK_ARN = "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"


def hit(cid, text="x"):
    return Hit(cid, "d", "Doc", text, None, None, 1.0)


def test_rrf_rewards_chunks_found_by_both_retrievers():
    fused = reciprocal_rank_fusion([hit("a"), hit("b"), hit("c")], [hit("c"), hit("d")])
    assert fused[0].chunk_id == "c"
    assert (fused[0].vector_rank, fused[0].keyword_rank) == (3, 1)
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)
    assert {h.chunk_id for h in fused} == {"a", "b", "c", "d"}


def test_terms_drop_stopwords_and_stem():
    assert terms("How many refunds are allowed?") == {"many", "refund", "allow"}


def test_lexical_reranker_prefers_coverage():
    hits = [hit("a", "the weather is nice"), hit("b", "parental leave is 16 weeks")]
    ranked = LexicalReranker().rerank("parental leave weeks", hits, 2)
    assert ranked[0].chunk_id == "b" and ranked[0].score > ranked[1].score
    assert LexicalReranker().rerank("the a of", hits, 2) == []  # only stopwords -> nothing relevant


def test_bedrock_reranker_maps_results():
    client = boto3.client("bedrock-agent-runtime", region_name="us-west-2")
    stub = Stubber(client)
    results = [{"index": 1, "relevanceScore": 0.91}, {"index": 0, "relevanceScore": 0.2}]
    stub.add_response("rerank", {"results": results})
    with stub:
        out = BedrockReranker(client, RERANK_ARN).rerank("q", [hit("a"), hit("b")], 2)
    assert [(h.chunk_id, h.score) for h in out] == [("b", 0.91), ("a", 0.2)]


def test_bedrock_reranker_falls_back_on_error():
    client = boto3.client("bedrock-agent-runtime", region_name="us-west-2")
    stub = Stubber(client)
    stub.add_client_error("rerank", service_error_code="AccessDeniedException")
    with stub:
        rr = BedrockReranker(client, RERANK_ARN)
        out = rr.rerank("leave", [hit("a", "leave policy")], 1)
    assert out[0].chunk_id == "a" and rr.last_used == "lexical"


def test_hybrid_retriever_end_to_end_respects_teams(memory_store, embedder):
    r = HybridRetriever(memory_store, embedder, LexicalReranker(), 20, 3)
    hr = r.retrieve("How many weeks of parental leave do parents get?", User("u1", ["hr"]))
    assert hr.hits[0].title == "Paid Time Off Policy" and "16 weeks" in hr.hits[0].text
    assert set(hr.timer.steps) == {"embed_query", "vector_search", "keyword_search", "rerank"}

    eng = r.retrieve("How many weeks of parental leave do parents get?", User("u2", ["engineering"]))
    assert all(h.title != "Paid Time Off Policy" for h in eng.hits)

    exact = r.retrieve("db-failover.sh", User("u3", ["engineering"]))
    assert "db-failover.sh" in exact.hits[0].text
