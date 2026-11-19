"""Plain data objects passed between the pipeline steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PUBLIC_TEAM = "public"  # documents under s3://bucket/public/ are visible to every user


@dataclass
class Section:
    """A piece of a source document before chunking (a PDF page or a Markdown section)."""

    text: str
    page: int | None = None
    section: str | None = None


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    team: str
    page: int | None = None
    section: str | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}:{self.chunk_index}"


@dataclass
class Document:
    doc_id: str
    title: str
    source_uri: str
    team: str
    content_hash: str
    num_chunks: int = 0


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    page: int | None
    section: str | None
    score: float = 0.0
    vector_rank: int | None = None
    keyword_rank: int | None = None

    def location(self) -> str:
        if self.page is not None:
            return f"page {self.page}"
        if self.section:
            return f"section '{self.section}'"
        return "whole document"


@dataclass
class User:
    user_id: str
    teams: list[str]

    def allowed_teams(self) -> list[str]:
        return sorted(set(self.teams) | {PUBLIC_TEAM})


@dataclass
class Citation:
    ref: str  # "S1"
    doc_id: str
    title: str
    location: str
    chunk_id: str


@dataclass
class Usage:
    embed_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    rerank_queries: int = 0
    cost_usd: float = 0.0


@dataclass
class Answer:
    answer: str
    citations: list[Citation]
    refused: bool
    reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    steps_ms: dict[str, float] = field(default_factory=dict)
    guardrail_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
