"""FastAPI HTTP API. Runs on Lambda behind API Gateway (via Mangum) or locally with uvicorn.

Every route except /health needs an `x-api-key` header; the key decides which teams' documents
the caller can see.
"""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel, Field

from . import __version__
from .auth import ApiKeyAuth, load_auth
from .db import DatabaseWaking
from .guardrails import GuardrailViolation
from .models import User
from .observability import add_metric, logger, metrics, tracer
from .rag import RagService
from .services import build_rag, build_store, settings
from .store import Store

app = FastAPI(
    title="KnowledgeMCP API",
    version=__version__,
    description="Team-scoped RAG over your documents, with citations. Send your key in the `x-api-key` header.",
)


@lru_cache(maxsize=1)
def get_store() -> Store:
    return build_store(settings())


@lru_cache(maxsize=1)
def get_rag() -> RagService:
    return build_rag(settings(), get_store())


_AUTH_TTL_S = 60.0
_auth_cache: tuple[float, ApiKeyAuth] | None = None


def get_auth() -> ApiKeyAuth:
    """Reload keys every 60 s so keys added with scripts/create_api_key.py work without a redeploy."""
    global _auth_cache
    now = time.monotonic()
    if _auth_cache is None or now - _auth_cache[0] > _AUTH_TTL_S:
        _auth_cache = (now, load_auth(settings()))
    return _auth_cache[1]


def current_user(x_api_key: str | None = Header(default=None), auth: ApiKeyAuth = Depends(get_auth)) -> User:
    user = auth.authenticate(x_api_key)
    if user is None:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return user


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["How many vacation days can I carry over?"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000, examples=["database failover"])
    top_k: int = Field(default=5, ge=1, le=10)


@app.exception_handler(GuardrailViolation)
async def guardrail_error(_: Request, exc: GuardrailViolation):
    add_metric("GuardrailBlocks", 1)
    logger.warning("guardrail blocked request", extra={"code": exc.code})
    return JSONResponse(status_code=400, content={"error": exc.code, "message": exc.message})


WARMING_RETRY_S = 10


@app.exception_handler(DatabaseWaking)
async def database_waking(_: Request, exc: DatabaseWaking):
    add_metric("WarmingUp503", 1)
    return JSONResponse(
        status_code=503, headers={"Retry-After": str(WARMING_RETRY_S)},
        content={"error": "warming_up", "retry_after_s": WARMING_RETRY_S,
                 "message": "The knowledge base is waking up after being idle. Retry in about 10 seconds."})


@app.post("/warmup")
def warmup(user: User = Depends(current_user), store: Store = Depends(get_store)):
    """Start resuming the database without waiting for it. Clients call this ahead of real questions."""
    return {"database": "ready" if store.ping() else "resuming"}


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/me")
def me(user: User = Depends(current_user)):
    return {"user_id": user.user_id, "teams": user.allowed_teams()}


@app.get("/documents")
def list_documents(user: User = Depends(current_user), store: Store = Depends(get_store)):
    return [{"doc_id": d.doc_id, "title": d.title, "team": d.team, "chunks": d.num_chunks}
            for d in store.list_documents(user.allowed_teams())]


@app.get("/documents/{doc_id}")
def get_document(doc_id: str, user: User = Depends(current_user), store: Store = Depends(get_store)):
    found = store.get_document(doc_id, user.allowed_teams())
    if not found:  # same answer for "missing" and "not allowed", so IDs can't be probed
        raise HTTPException(status_code=404, detail="Document not found")
    doc, chunks = found
    return {"doc_id": doc.doc_id, "title": doc.title, "team": doc.team, "source_uri": doc.source_uri,
            "chunks": [{"chunk_id": c.chunk_id, "location": c.location(), "text": c.text} for c in chunks]}


@app.post("/search")
def search(req: SearchRequest, user: User = Depends(current_user), rag: RagService = Depends(get_rag)):
    return [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title, "location": h.location(),
             "score": h.score, "text": h.text} for h in rag.search(req.query, user, req.top_k)]


@app.post("/ask")
def ask(req: AskRequest, user: User = Depends(current_user), rag: RagService = Depends(get_rag)):
    return rag.ask(req.question, user).to_dict()


_mangum = Mangum(app, lifespan="off")


@logger.inject_lambda_context(correlation_id_path="requestContext.requestId")
@metrics.log_metrics
@tracer.capture_lambda_handler
def handler(event, context):
    try:  # Mangum needs a current event loop; recreate it if something closed it
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return _mangum(event, context)
