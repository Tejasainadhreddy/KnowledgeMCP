"""Re-ranking: look at the best candidates again, more carefully, and give each a 0..1 relevance score.

BedrockReranker uses Amazon Rerank 1.0 (available in us-west-2). If the call fails, we fall back to
LexicalReranker so a single model error never takes the API down.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol

from .models import Hit
from .observability import logger

STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it its me my of on or our should so "
    "that the their them there this to us was we what when where which who why will with you your".split()
)


def terms(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOPWORDS or len(w) < 2:
            continue
        for suffix in ("ing", "ed", "es", "s"):  # tiny stemmer: "refunds" ~ "refund"
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


class LexicalReranker:
    """Offline reranker: how many of the question's key words appear in the chunk, plus fusion rank."""

    name = "lexical"

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        q = terms(query)
        if not q or not hits:
            return []
        best_fused = max(h.score for h in hits) or 1.0
        scored = []
        for h in hits:
            coverage = len(q & terms(h.text)) / len(q)
            scored.append(replace(h, score=round(0.8 * coverage + 0.2 * (h.score / best_fused), 4)))
        scored.sort(key=lambda h: -h.score)
        return scored[:top_n]


class BedrockReranker:
    name = "bedrock"

    def __init__(self, client, model_arn: str, fallback: Reranker | None = None):
        self.client = client
        self.model_arn = model_arn
        self.fallback = fallback or LexicalReranker()
        self.last_used = self.name

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        try:
            resp = self.client.rerank(
                queries=[{"type": "TEXT", "textQuery": {"text": query}}],
                sources=[{"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": h.text}}}
                         for h in hits],
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "numberOfResults": min(top_n, len(hits)),
                        "modelConfiguration": {"modelArn": self.model_arn},
                    },
                },
            )
            self.last_used = self.name
            return [replace(hits[r["index"]], score=round(float(r["relevanceScore"]), 4)) for r in resp["results"]]
        except Exception as exc:
            logger.warning("bedrock rerank failed, using lexical fallback", extra={"error": str(exc)})
            self.last_used = self.fallback.name
            return self.fallback.rerank(query, hits, top_n)
