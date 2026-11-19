"""All settings come from environment variables, so the same code runs locally and on Lambda."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Prices:
    """USD per 1,000 tokens (or per rerank query). Check the Bedrock pricing page for your region."""

    embed_per_1k: float = float(_env("PRICE_EMBED_PER_1K", "0.00002"))
    llm_input_per_1k: float = float(_env("PRICE_LLM_INPUT_PER_1K", "0.001"))
    llm_output_per_1k: float = float(_env("PRICE_LLM_OUTPUT_PER_1K", "0.005"))
    rerank_per_query: float = float(_env("PRICE_RERANK_PER_QUERY", "0.001"))


@dataclass(frozen=True)
class Settings:
    region: str = field(default_factory=lambda: _env("AWS_REGION", "us-west-2"))

    # Database: "dataapi" (Aurora via RDS Data API, used on AWS) or "psycopg" (local Postgres)
    db_mode: str = field(default_factory=lambda: _env("DB_MODE", "psycopg"))
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "postgresql://postgres@localhost:5433/kmcp")
    )
    db_cluster_arn: str = field(default_factory=lambda: _env("DB_CLUSTER_ARN"))
    db_secret_arn: str = field(default_factory=lambda: _env("DB_SECRET_ARN"))
    db_name: str = field(default_factory=lambda: _env("DB_NAME", "kmcp"))

    # Models: "bedrock" on AWS; "hash"/"fake"/"lexical" are offline stand-ins for tests and local dev
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "hash"))
    embed_model_id: str = field(default_factory=lambda: _env("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"))
    embed_dim: int = field(default_factory=lambda: int(_env("EMBED_DIM", "512")))
    llm: str = field(default_factory=lambda: _env("LLM", "fake"))
    llm_model_id: str = field(
        default_factory=lambda: _env("LLM_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    )
    reranker: str = field(default_factory=lambda: _env("RERANKER", "lexical"))
    rerank_model_id: str = field(default_factory=lambda: _env("RERANK_MODEL_ID", "amazon.rerank-v1:0"))

    # Retrieval
    candidates_per_retriever: int = field(default_factory=lambda: int(_env("CANDIDATES", "20")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.25")))
    chunk_chars: int = field(default_factory=lambda: int(_env("CHUNK_CHARS", "1200")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))

    # Auth: API keys are stored hashed, either inline JSON (local) or in Secrets Manager (AWS)
    api_keys_json: str = field(default_factory=lambda: _env("API_KEYS_JSON"))
    api_keys_secret_arn: str = field(default_factory=lambda: _env("API_KEYS_SECRET_ARN"))

    prices: Prices = field(default_factory=Prices)


def get_settings() -> Settings:
    return Settings()
