"""MCP server: lets Claude Desktop, Claude Code or any MCP client use the knowledge base.

Tools:      search_documents, ask_question, list_documents
Resources:  doc://{doc_id}   (full text of one document, as Markdown)

By default it talks to the deployed HTTP API, so access control is enforced on the server:
    KMCP_API_URL=https://xxxx.execute-api.us-west-2.amazonaws.com  KMCP_API_KEY=kmcp_...  knowledgemcp-mcp
For local development without AWS you can run it in-process instead:
    KMCP_LOCAL=1 KMCP_TEAMS=hr,engineering knowledgemcp-mcp
"""

from __future__ import annotations

import os
import time
from typing import Any, Protocol

import httpx
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel


class Passage(BaseModel):
    doc_id: str
    title: str
    location: str
    score: float
    text: str


class CitationOut(BaseModel):
    ref: str
    title: str
    location: str
    doc_id: str


class AskResult(BaseModel):
    answer: str
    refused: bool
    citations: list[CitationOut]


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    team: str
    chunks: int


class Backend(Protocol):
    def search(self, query: str, top_k: int) -> list[dict]: ...
    def ask(self, question: str) -> dict: ...
    def list_documents(self) -> list[dict]: ...
    def get_document(self, doc_id: str) -> dict | None: ...


class KnowledgeBaseWarmingUp(RuntimeError):
    """The API kept answering 503 warming_up for longer than this client's wait budget."""


class HttpBackend:
    """Calls the KnowledgeMCP API. While the database wakes up (503 warming_up), it waits and retries,
    but only for `wait_budget_s` (env KMCP_WAIT_BUDGET_S). Desktop clients can wait ~45 s; a caller with
    its own deadline (like a Lambda) should pass a small budget and handle KnowledgeBaseWarmingUp."""

    def __init__(self, base_url: str, api_key: str, client: httpx.Client | None = None,
                 wait_budget_s: float | None = None, sleep=time.sleep):
        self.client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=60.0)
        self.headers = {"x-api-key": api_key}
        self.wait_budget_s = float(os.environ.get("KMCP_WAIT_BUDGET_S", "45")) if wait_budget_s is None \
            else wait_budget_s
        self.sleep = sleep

    def _call(self, method: str, path: str, **kw) -> Any:
        waited = 0.0
        while True:
            resp = self.client.request(method, path, headers=self.headers, **kw)
            if resp.status_code == 503 and resp.json().get("error") == "warming_up":
                remaining = self.wait_budget_s - waited
                if remaining <= 0:
                    raise KnowledgeBaseWarmingUp(
                        "warming_up: the knowledge base is waking up after being idle; try again shortly")
                pause = min(float(resp.json().get("retry_after_s", 10)), remaining)
                self.sleep(pause)
                waited += pause
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code == 400:
                raise ValueError(resp.json().get("message", "Request blocked by guardrail"))
            resp.raise_for_status()
            return resp.json()

    def warmup(self) -> str:
        return self.client.post("/warmup", headers=self.headers).json().get("database", "unknown")

    def search(self, query, top_k):
        return self._call("POST", "/search", json={"query": query, "top_k": top_k})

    def ask(self, question):
        return self._call("POST", "/ask", json={"question": question})

    def list_documents(self):
        return self._call("GET", "/documents")

    def get_document(self, doc_id):
        return self._call("GET", f"/documents/{doc_id}")


class LocalBackend:
    """Runs the pipeline in-process (local Postgres, offline or Bedrock models per env settings)."""

    def __init__(self, teams: list[str], rag=None, store=None):
        from .models import User
        from .services import build_rag, build_store, settings

        self.user = User("local-mcp", teams)
        self.store = store or build_store(settings())
        self.rag = rag or build_rag(settings(), self.store)

    def search(self, query, top_k):
        return [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title, "location": h.location(),
                 "score": h.score, "text": h.text} for h in self.rag.search(query, self.user, top_k)]

    def ask(self, question):
        return self.rag.ask(question, self.user).to_dict()

    def list_documents(self):
        return [{"doc_id": d.doc_id, "title": d.title, "team": d.team, "chunks": d.num_chunks}
                for d in self.store.list_documents(self.user.allowed_teams())]

    def get_document(self, doc_id):
        found = self.store.get_document(doc_id, self.user.allowed_teams())
        if not found:
            return None
        doc, chunks = found
        return {"doc_id": doc.doc_id, "title": doc.title, "team": doc.team,
                "chunks": [{"location": c.location(), "text": c.text} for c in chunks]}


def create_server(backend: Backend) -> MCPServer:
    server = MCPServer(
        name="KnowledgeMCP",
        instructions=(
            "Search and answer questions over the company's internal documents. Prefer ask_question for "
            "direct questions (it returns a cited answer or says it could not find one). Use "
            "search_documents to explore, and read doc://{doc_id} for a full document."
        ),
    )

    @server.tool(structured_output=True,
                 description="Hybrid (semantic + keyword) search over documents you can access. "
                             "Returns the best matching passages with their document and page/section.")
    def search_documents(query: str, top_k: int = 5) -> list[Passage]:
        top_k = max(1, min(int(top_k), 10))
        return [Passage(**{k: h[k] for k in Passage.model_fields}) for h in backend.search(query, top_k)]

    @server.tool(structured_output=True,
                 description="Answer a question from the documents with citations. "
                             "If the documents do not contain the answer it says so instead of guessing.")
    def ask_question(question: str) -> AskResult:
        a = backend.ask(question)
        return AskResult(answer=a["answer"], refused=a["refused"],
                         citations=[CitationOut(**{k: c[k] for k in CitationOut.model_fields})
                                    for c in a["citations"]])

    @server.tool(structured_output=True, description="List the documents you have access to (id, title, team).")
    def list_documents() -> list[DocumentInfo]:
        return [DocumentInfo(**d) for d in backend.list_documents()]

    @server.resource("doc://{doc_id}", name="document", mime_type="text/markdown",
                     description="Full text of one document, split by page or section.")
    def document(doc_id: str) -> str:
        doc = backend.get_document(doc_id)
        if doc is None:
            raise ValueError(f"Document {doc_id} not found")
        parts = [f"# {doc['title']}\n\n_team: {doc['team']}_"]
        parts += [f"## {c['location']}\n\n{c['text']}" for c in doc["chunks"]]
        return "\n\n".join(parts)

    return server


def backend_from_env() -> Backend:
    if os.environ.get("KMCP_LOCAL") == "1":
        teams = [t for t in os.environ.get("KMCP_TEAMS", "").split(",") if t]
        return LocalBackend(teams)
    url, key = os.environ.get("KMCP_API_URL"), os.environ.get("KMCP_API_KEY")
    if not url or not key:
        raise SystemExit("Set KMCP_API_URL and KMCP_API_KEY (or KMCP_LOCAL=1 for local mode).")
    return HttpBackend(url, key)


def main() -> None:
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)  # stdout/stderr stay clean for MCP clients
    create_server(backend_from_env()).run("stdio")


if __name__ == "__main__":
    main()
