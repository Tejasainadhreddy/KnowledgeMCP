"""Build the real or offline components from Settings (one place to wire everything)."""

from __future__ import annotations

from functools import lru_cache

import boto3

from .config import Settings, get_settings
from .db import DataApiExecutor, PsycopgExecutor
from .embeddings import BedrockTitanEmbedder, Embedder, HashEmbedder
from .llm import LLM, BedrockLLM, ExtractiveLLM
from .rag import RagService
from .rerank import BedrockReranker, LexicalReranker, Reranker
from .search import HybridRetriever
from .store import PgVectorStore, Store


def build_store(s: Settings) -> Store:
    if s.db_mode == "dataapi":
        executor = DataApiExecutor(boto3.client("rds-data", region_name=s.region),
                                   s.db_cluster_arn, s.db_secret_arn, s.db_name,
                                   max_wait_s=s.db_resume_wait_s)
    else:
        executor = PsycopgExecutor(s.database_url)
    return PgVectorStore(executor, dim=s.embed_dim)


def build_embedder(s: Settings) -> Embedder:
    if s.embedder == "bedrock":
        return BedrockTitanEmbedder(boto3.client("bedrock-runtime", region_name=s.region),
                                    s.embed_model_id, s.embed_dim)
    return HashEmbedder(s.embed_dim)


def build_reranker(s: Settings) -> Reranker:
    if s.reranker == "bedrock":
        arn = f"arn:aws:bedrock:{s.region}::foundation-model/{s.rerank_model_id}"
        return BedrockReranker(boto3.client("bedrock-agent-runtime", region_name=s.region), arn)
    return LexicalReranker()


def build_llm(s: Settings) -> LLM:
    if s.llm == "bedrock":
        return BedrockLLM(boto3.client("bedrock-runtime", region_name=s.region), s.llm_model_id)
    return ExtractiveLLM()


def build_rag(s: Settings, store: Store | None = None) -> RagService:
    retriever = HybridRetriever(store or build_store(s), build_embedder(s), build_reranker(s),
                                s.candidates_per_retriever, s.top_k)
    return RagService(retriever, build_llm(s), s.prices, s.min_relevance)


@lru_cache(maxsize=1)
def settings() -> Settings:
    return get_settings()
