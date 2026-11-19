"""Embeddings turn text into vectors; similar meaning -> nearby vectors.

BedrockTitanEmbedder is the real one. HashEmbedder is a tiny offline stand-in (word hashing) so tests
and local demos run without AWS. It only matches shared words, not meaning, so never quote eval
numbers produced with it as the system's quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def token_count(self, texts: list[str]) -> int: ...


def rough_tokens(texts: list[str]) -> int:
    return sum(max(1, len(t) // 4) for t in texts)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.last_tokens = 0

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return _normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.last_tokens = rough_tokens(texts)
        return [self._one(t) for t in texts]

    def token_count(self, texts: list[str]) -> int:
        return rough_tokens(texts)


class BedrockTitanEmbedder:
    """Amazon Titan Text Embeddings V2 (supports 256, 512 or 1024 dimensions)."""

    def __init__(self, client, model_id: str = "amazon.titan-embed-text-v2:0", dim: int = 512):
        if dim not in (256, 512, 1024):
            raise ValueError("Titan V2 supports 256, 512 or 1024 dimensions")
        self.client = client
        self.model_id = model_id
        self.dim = dim
        self.last_tokens = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors, tokens = [], 0
        for text in texts:  # Titan V2 embeds one text per call
            body = json.dumps({"inputText": text[:40000], "dimensions": self.dim, "normalize": True})
            resp = self.client.invoke_model(
                modelId=self.model_id, body=body, contentType="application/json", accept="application/json"
            )
            payload = json.loads(resp["body"].read())
            vectors.append(payload["embedding"])
            tokens += int(payload.get("inputTextTokenCount", 0))
        self.last_tokens = tokens
        return vectors

    def token_count(self, texts: list[str]) -> int:
        return self.last_tokens or rough_tokens(texts)


def to_pgvector(vec: list[float]) -> str:
    """pgvector accepts the text form '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
