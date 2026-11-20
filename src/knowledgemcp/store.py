"""Where chunks live. PgVectorStore is the real one; InMemoryStore is for fast unit tests.

Access control happens here, in SQL: every query filters on the caller's allowed teams, so a
chunk from another team can never reach the LLM prompt.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from .db import Executor
from .embeddings import to_pgvector
from .models import Chunk, Document, Hit


class Store(Protocol):
    def ensure_schema(self) -> None: ...
    def ping(self) -> bool: ...
    def get_content_hash(self, doc_id: str) -> str | None: ...
    def upsert_document(self, doc: Document, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def delete_document(self, doc_id: str) -> None: ...
    def vector_search(self, vector: list[float], teams: list[str], limit: int) -> list[Hit]: ...
    def keyword_search(self, query: str, teams: list[str], limit: int) -> list[Hit]: ...
    def list_documents(self, teams: list[str]) -> list[Document]: ...
    def get_document(self, doc_id: str, teams: list[str]) -> tuple[Document, list[Hit]] | None: ...


def schema_statements(dim: int) -> list[str]:
    # One statement per call: the Data API does not run multi-statement strings.
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        """CREATE TABLE IF NOT EXISTS documents (
            doc_id text PRIMARY KEY,
            title text NOT NULL,
            source_uri text NOT NULL,
            team text NOT NULL,
            content_hash text NOT NULL,
            num_chunks int NOT NULL DEFAULT 0,
            updated_at timestamptz NOT NULL DEFAULT now())""",
        f"""CREATE TABLE IF NOT EXISTS chunks (
            chunk_id text PRIMARY KEY,
            doc_id text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            chunk_index int NOT NULL,
            team text NOT NULL,
            page int,
            section text,
            content text NOT NULL,
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            embedding vector({dim}) NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)",
        "CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)",
        "CREATE INDEX IF NOT EXISTS chunks_team ON chunks (team)",
        "CREATE INDEX IF NOT EXISTS documents_team ON documents (team)",
    ]


_TEAM_FILTER = "c.team = ANY(string_to_array(:teams, ','))"
_HIT_COLUMNS = "c.chunk_id, c.doc_id, d.title, c.content, c.page, c.section"


def _hit(row: dict, score: float) -> Hit:
    return Hit(
        chunk_id=row["chunk_id"], doc_id=row["doc_id"], title=row["title"], text=row["content"],
        page=row.get("page"), section=row.get("section"), score=float(score),
    )


def _doc(row: dict) -> Document:
    return Document(
        doc_id=row["doc_id"], title=row["title"], source_uri=row["source_uri"], team=row["team"],
        content_hash=row["content_hash"], num_chunks=int(row["num_chunks"]),
    )


class PgVectorStore:
    def __init__(self, executor: Executor, dim: int = 512):
        self.db = executor
        self.dim = dim

    def ensure_schema(self) -> None:
        for stmt in schema_statements(self.dim):
            self.db.execute(stmt)

    def ping(self) -> bool:
        return self.db.ping()

    def get_content_hash(self, doc_id: str) -> str | None:
        rows = self.db.execute("SELECT content_hash FROM documents WHERE doc_id = :doc_id", {"doc_id": doc_id})
        return rows[0]["content_hash"] if rows else None

    def upsert_document(self, doc: Document, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.delete_document(doc.doc_id)  # simple re-index: replace all chunks of this document
        self.db.execute(
            """INSERT INTO documents (doc_id, title, source_uri, team, content_hash, num_chunks)
               VALUES (:doc_id, :title, :source_uri, :team, :content_hash, :num_chunks)""",
            {"doc_id": doc.doc_id, "title": doc.title, "source_uri": doc.source_uri, "team": doc.team,
             "content_hash": doc.content_hash, "num_chunks": len(chunks)},
        )
        self.db.execute_many(
            """INSERT INTO chunks (chunk_id, doc_id, chunk_index, team, page, section, content, embedding)
               VALUES (:chunk_id, :doc_id, :chunk_index, :team, :page, :section, :content,
                       CAST(:embedding AS vector))""",
            [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "chunk_index": c.chunk_index, "team": c.team,
                 "page": c.page, "section": c.section, "content": c.text, "embedding": to_pgvector(v)}
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )

    def delete_document(self, doc_id: str) -> None:
        self.db.execute("DELETE FROM documents WHERE doc_id = :doc_id", {"doc_id": doc_id})

    def vector_search(self, vector: list[float], teams: list[str], limit: int) -> list[Hit]:
        rows = self.db.execute(
            f"""SELECT {_HIT_COLUMNS}, 1 - (c.embedding <=> CAST(:q AS vector)) AS score
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE {_TEAM_FILTER}
                ORDER BY c.embedding <=> CAST(:q AS vector)
                LIMIT :limit""",
            {"q": to_pgvector(vector), "teams": ",".join(teams), "limit": limit},
        )
        return [_hit(r, r["score"]) for r in rows]

    def keyword_search(self, query: str, teams: list[str], limit: int) -> list[Hit]:
        # plainto_tsquery ANDs every word; questions rarely match all words, so we OR them instead.
        # Ranking uses Postgres full-text ts_rank_cd (similar idea to BM25, but not BM25).
        rows = self.db.execute(
            f"""WITH q AS (
                  SELECT NULLIF(replace(plainto_tsquery('english', :query)::text, '&', '|'), '') AS t)
                SELECT {_HIT_COLUMNS}, ts_rank_cd(c.tsv, to_tsquery('english', q.t)) AS score
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id, q
                WHERE q.t IS NOT NULL AND c.tsv @@ to_tsquery('english', q.t) AND {_TEAM_FILTER}
                ORDER BY score DESC
                LIMIT :limit""",
            {"query": query, "teams": ",".join(teams), "limit": limit},
        )
        return [_hit(r, r["score"]) for r in rows]

    def list_documents(self, teams: list[str]) -> list[Document]:
        rows = self.db.execute(
            """SELECT doc_id, title, source_uri, team, content_hash, num_chunks FROM documents
               WHERE team = ANY(string_to_array(:teams, ',')) ORDER BY team, title""",
            {"teams": ",".join(teams)},
        )
        return [_doc(r) for r in rows]

    def get_document(self, doc_id: str, teams: list[str]) -> tuple[Document, list[Hit]] | None:
        docs = self.db.execute(
            """SELECT doc_id, title, source_uri, team, content_hash, num_chunks FROM documents
               WHERE doc_id = :doc_id AND team = ANY(string_to_array(:teams, ','))""",
            {"doc_id": doc_id, "teams": ",".join(teams)},
        )
        if not docs:
            return None
        rows = self.db.execute(
            f"""SELECT {_HIT_COLUMNS} FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.doc_id = :doc_id ORDER BY c.chunk_index""",
            {"doc_id": doc_id},
        )
        return _doc(docs[0]), [_hit(r, 0.0) for r in rows]


class InMemoryStore:
    """Same behaviour as PgVectorStore for unit tests (no database needed)."""

    def __init__(self):
        self.docs: dict[str, Document] = {}
        self.chunks: dict[str, tuple[Chunk, list[float]]] = {}

    def ensure_schema(self) -> None:
        pass

    def ping(self) -> bool:
        return True

    def get_content_hash(self, doc_id: str) -> str | None:
        doc = self.docs.get(doc_id)
        return doc.content_hash if doc else None

    def upsert_document(self, doc: Document, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.delete_document(doc.doc_id)
        doc.num_chunks = len(chunks)
        self.docs[doc.doc_id] = doc
        for c, v in zip(chunks, vectors, strict=True):
            self.chunks[c.chunk_id] = (c, v)

    def delete_document(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)
        for cid in [k for k, (c, _) in self.chunks.items() if c.doc_id == doc_id]:
            del self.chunks[cid]

    def _visible(self, teams: list[str]):
        return [(c, v) for c, v in self.chunks.values() if c.team in teams]

    def _to_hit(self, c: Chunk, score: float) -> Hit:
        return Hit(c.chunk_id, c.doc_id, self.docs[c.doc_id].title, c.text, c.page, c.section, score)

    def vector_search(self, vector: list[float], teams: list[str], limit: int) -> list[Hit]:
        scored = [(sum(a * b for a, b in zip(vector, v, strict=True)), c) for c, v in self._visible(teams)]
        scored.sort(key=lambda x: -x[0])
        return [self._to_hit(c, s) for s, c in scored[:limit]]

    def keyword_search(self, query: str, teams: list[str], limit: int) -> list[Hit]:
        terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
        scored = []
        for c, _ in self._visible(teams):
            words = re.findall(r"[a-z0-9]+", c.text.lower())
            score = sum(math.log1p(words.count(t)) for t in terms)
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return [self._to_hit(c, s) for s, c in scored[:limit]]

    def list_documents(self, teams: list[str]) -> list[Document]:
        return sorted((d for d in self.docs.values() if d.team in teams), key=lambda d: (d.team, d.title))

    def get_document(self, doc_id: str, teams: list[str]) -> tuple[Document, list[Hit]] | None:
        doc = self.docs.get(doc_id)
        if not doc or doc.team not in teams:
            return None
        chunks = sorted((c for c, _ in self.chunks.values() if c.doc_id == doc_id), key=lambda c: c.chunk_index)
        return doc, [self._to_hit(c, 0.0) for c in chunks]
