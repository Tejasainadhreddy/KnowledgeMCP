"""Ingestion: file -> sections -> chunks -> embeddings -> pgvector.

On AWS this runs as a Lambda triggered by S3. The first folder of the S3 key is the team:
  s3://<bucket>/hr/leave-policy.pdf   -> team "hr"
  s3://<bucket>/public/overview.md    -> visible to everyone
"""

from __future__ import annotations

import hashlib
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_sections
from .embeddings import Embedder
from .loaders import SUPPORTED_EXTENSIONS, extension, load_document, title_from
from .models import Document
from .store import Store


@dataclass
class IngestResult:
    doc_id: str
    status: str  # "indexed" | "unchanged" | "skipped" | "deleted"
    chunks: int = 0
    embed_tokens: int = 0
    seconds: float = 0.0


def doc_id_for(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def team_for(key: str) -> str:
    parts = key.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Key must start with a team folder, e.g. 'hr/file.pdf' (got '{key}')")
    return parts[0].lower()


class Ingestor:
    def __init__(self, store: Store, embedder: Embedder, chunk_chars: int = 1200, overlap: int = 150):
        self.store = store
        self.embedder = embedder
        self.chunk_chars = chunk_chars
        self.overlap = overlap

    def ingest_bytes(self, key: str, data: bytes, source_uri: str) -> IngestResult:
        start = time.perf_counter()
        doc_id = doc_id_for(key)
        if extension(key) not in SUPPORTED_EXTENSIONS:
            return IngestResult(doc_id, "skipped")
        content_hash = hashlib.sha256(data).hexdigest()
        if self.store.get_content_hash(doc_id) == content_hash:
            return IngestResult(doc_id, "unchanged")
        team = team_for(key)
        sections = load_document(key, data)
        chunks = chunk_sections(doc_id, team, sections, self.chunk_chars, self.overlap)
        if not chunks:
            return IngestResult(doc_id, "skipped")
        vectors = self.embedder.embed([c.text for c in chunks])
        doc = Document(doc_id, title_from(key, sections), source_uri, team, content_hash)
        self.store.upsert_document(doc, chunks, vectors)
        return IngestResult(doc_id, "indexed", len(chunks), self.embedder.token_count([c.text for c in chunks]),
                            round(time.perf_counter() - start, 3))

    def delete(self, key: str) -> IngestResult:
        doc_id = doc_id_for(key)
        self.store.delete_document(doc_id)
        return IngestResult(doc_id, "deleted")

    def ingest_folder(self, root: Path) -> list[IngestResult]:
        """Local helper: index a folder laid out like the S3 bucket (team/file)."""
        results = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                key = path.relative_to(root).as_posix()
                results.append(self.ingest_bytes(key, path.read_bytes(), f"file://{path.resolve()}"))
        return results


def handle_s3_event(event: dict, ingestor: Ingestor, s3_client) -> list[IngestResult]:
    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        if record.get("eventName", "").startswith("ObjectRemoved"):
            results.append(ingestor.delete(key))
            continue
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        results.append(ingestor.ingest_bytes(key, body, f"s3://{bucket}/{key}"))
    return results
