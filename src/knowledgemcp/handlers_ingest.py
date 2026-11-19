"""Lambda entry points for ingestion and the one-time schema migration."""

from __future__ import annotations

from dataclasses import asdict

import boto3

from .ingest import Ingestor, handle_s3_event
from .observability import MetricUnit, add_metric, logger, metrics, tracer
from .services import build_embedder, build_store, settings

_ingestor: Ingestor | None = None


def _get_ingestor() -> Ingestor:
    global _ingestor
    if _ingestor is None:
        s = settings()
        _ingestor = Ingestor(build_store(s), build_embedder(s), s.chunk_chars, s.chunk_overlap)
    return _ingestor


@logger.inject_lambda_context
@metrics.log_metrics
@tracer.capture_lambda_handler
def ingest_handler(event, context):
    results = handle_s3_event(event, _get_ingestor(), boto3.client("s3"))
    for r in results:
        logger.info("ingest", extra=asdict(r))
        if r.status == "indexed":
            add_metric("DocumentsIndexed", 1)
            add_metric("ChunksIndexed", r.chunks)
            add_metric("EmbedTokens", r.embed_tokens)
            add_metric("IngestSeconds", r.seconds, MetricUnit.Seconds)
    return {"results": [asdict(r) for r in results]}


@logger.inject_lambda_context
def migrate_handler(event, context):
    """Run by a CDK Trigger after each deploy. Safe to run many times (IF NOT EXISTS)."""
    build_store(settings()).ensure_schema()
    logger.info("schema ready")
    return {"status": "ok"}
