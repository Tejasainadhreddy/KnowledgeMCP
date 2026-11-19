"""Build the real or offline components from Settings (one place to wire everything)."""

from __future__ import annotations

from functools import lru_cache

import boto3

from .config import Settings, get_settings
from .db import DataApiExecutor, PsycopgExecutor
from .embeddings import BedrockTitanEmbedder, Embedder, HashEmbedder
from .store import PgVectorStore, Store


def build_store(s: Settings) -> Store:
    if s.db_mode == "dataapi":
        executor = DataApiExecutor(boto3.client("rds-data", region_name=s.region),
                                   s.db_cluster_arn, s.db_secret_arn, s.db_name)
    else:
        executor = PsycopgExecutor(s.database_url)
    return PgVectorStore(executor, dim=s.embed_dim)


def build_embedder(s: Settings) -> Embedder:
    if s.embedder == "bedrock":
        return BedrockTitanEmbedder(boto3.client("bedrock-runtime", region_name=s.region),
                                    s.embed_model_id, s.embed_dim)
    return HashEmbedder(s.embed_dim)


@lru_cache(maxsize=1)
def settings() -> Settings:
    return get_settings()
