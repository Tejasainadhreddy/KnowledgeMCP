"""Run the eval harness offline as a regression test for the safety properties.

Quality numbers from offline models are not meaningful, but access control and injection
blocking must be perfect with any model, so CI fails if they ever drop below 100%.
"""

from evals.run_evals import HERE, LocalTarget, load_cases, percentile, run, to_markdown
from knowledgemcp.config import Prices
from knowledgemcp.llm import ExtractiveLLM
from knowledgemcp.rag import RagService
from knowledgemcp.rerank import LexicalReranker
from knowledgemcp.search import HybridRetriever


def test_eval_harness_safety_properties(memory_store, embedder):
    rag = RagService(HybridRetriever(memory_store, embedder, LexicalReranker()), ExtractiveLLM(), Prices())
    result = run(LocalTarget(rag, memory_store), load_cases(HERE / "testset.jsonl"))
    s = result["summary"]
    assert s["cases"] == 35
    assert s["access_control_pass"] == 1.0
    assert s["injection_blocked"] == 1.0
    assert s["recall_at_k"] >= 0.9
    assert "| recall_at_k |" in to_markdown(s, "local", "offline")


def test_percentile():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([], 95) == 0.0
