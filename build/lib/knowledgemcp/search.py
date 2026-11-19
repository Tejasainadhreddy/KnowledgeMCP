"""Hybrid retrieval: vector search + keyword search, merged with Reciprocal Rank Fusion, then re-ranked.

Why both? Vectors find answers phrased differently ("time off" vs "vacation"); keywords find exact
terms (error codes, product names, "db-failover.sh") that vectors can miss.

Reciprocal Rank Fusion (RRF): each list votes 1 / (60 + rank) for every chunk it returned; we add the
votes up. It only needs ranks, so the two very different score scales never have to be compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .embeddings import Embedder
from .models import Hit, User
from .observability import StepTimer
from .rerank import Reranker
from .store import Store

RRF_K = 60


def reciprocal_rank_fusion(vector_hits: list[Hit], keyword_hits: list[Hit], k: int = RRF_K) -> list[Hit]:
    fused: dict[str, Hit] = {}
    for kind, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            current = fused.get(hit.chunk_id) or replace(hit, score=0.0)
            current.score += 1.0 / (k + rank)
            if kind == "vector":
                current.vector_rank = rank
            else:
                current.keyword_rank = rank
            fused[hit.chunk_id] = current
    return sorted(fused.values(), key=lambda h: -h.score)


@dataclass
class Retrieval:
    hits: list[Hit]
    candidates: int
    embed_tokens: int
    reranker: str
    timer: StepTimer = field(default_factory=StepTimer)


class HybridRetriever:
    def __init__(self, store: Store, embedder: Embedder, reranker: Reranker,
                 candidates_per_retriever: int = 20, top_k: int = 5):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates_per_retriever
        self.top_k = top_k

    def retrieve(self, query: str, user: User, top_k: int | None = None, timer: StepTimer | None = None) -> Retrieval:
        timer = timer or StepTimer()
        teams = user.allowed_teams()
        with timer.step("embed_query"):
            qvec = self.embedder.embed([query])[0]
        with timer.step("vector_search"):
            vec = self.store.vector_search(qvec, teams, self.candidates)
        with timer.step("keyword_search"):
            kw = self.store.keyword_search(query, teams, self.candidates)
        fused = reciprocal_rank_fusion(vec, kw)
        with timer.step("rerank"):
            hits = self.reranker.rerank(query, fused, top_k or self.top_k)
        used = getattr(self.reranker, "last_used", self.reranker.name)
        return Retrieval(hits, len(fused), self.embedder.token_count([query]), used, timer)
