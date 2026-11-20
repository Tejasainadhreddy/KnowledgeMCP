# KnowledgeMCP

[![CI](https://github.com/Tejasainadhreddy/KnowledgeMCP/actions/workflows/ci.yml/badge.svg)](https://github.com/Tejasainadhreddy/KnowledgeMCP/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK%20(Python)-orange)
![MCP](https://img.shields.io/badge/MCP-server-7aa2ff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A team-scoped RAG knowledge base on AWS, served as an MCP server and a REST API.**
Upload PDFs, Markdown or text files to S3. Ask questions from Claude Desktop, Claude Code or any app,
and get answers with citations (document + page/section). Each user only sees answers from the
documents their team may read. If the answer isn't in those documents, it says so instead of guessing.

## Status (November 2026): code complete, not yet deployed

**Works and is tested**
- Ingestion, hybrid search, team access control, guardrails, cited answers, REST API and MCP server: 50 tests, including an integration test against real Postgres + pgvector.
- CDK synth tests (7) and a Docker image build run in CI on every push.
- The preview below is a real local run (Postgres + pgvector, offline stand-in models).

**Not proven yet**
- `cdk deploy` has never been run, so the AWS stack itself is untested.
- Bedrock calls (Titan embeddings, Amazon Rerank, Claude) are only tested against recorded responses, never the real models.
- pgvector queries through the RDS Data API have not been run against Aurora (only over a direct Postgres connection).
- No quality numbers yet: recall, faithfulness, latency and cost will be measured on the deployed stack.

**Known issues**
- After Aurora auto-pauses, the first request waits about 15 s for it to resume; with model time added it may hit API Gateway's 30 s limit.
- Prompt-injection detection is pattern-based, so reworded attacks can get through (team filtering in SQL still limits what the model can see).
- Filtered vector search can return fewer than `k` results when a team owns few chunks.
- The offline stand-in models used in tests and local runs give weak answers; they only check the plumbing.

![KnowledgeMCP preview](docs/preview.png)

## What it does

| | |
|---|---|
| **Ingestion** | S3 upload → Lambda parses PDF/Markdown/text → section-aware chunks (page or heading kept) → Amazon Titan Text Embeddings V2 → Aurora PostgreSQL + pgvector. Re-uploads of unchanged files are skipped; deletes remove the document. |
| **Hybrid search** | Vector search (pgvector HNSW, cosine) + keyword search (Postgres full-text), merged with Reciprocal Rank Fusion, then re-ranked by Amazon Rerank 1.0 (falls back to a local re-ranker if the call fails). |
| **Answers with citations** | Claude on Bedrock answers only from the retrieved passages, citing `[S1]`, `[S2]`… Citations are verified; uncited answers are refused. |
| **Access control** | The first S3 folder is the team (`hr/`, `finance/`, `public/`). API keys map to users and teams; team filtering happens **in SQL**, so other teams' text never reaches the model. |
| **Guardrails** | Input validation and prompt-injection checks on questions; retrieved text is sanitized and wrapped as untrusted data; relevance gate and `NOT_FOUND` path for refusals. |
| **MCP server** | Tools `ask_question`, `search_documents`, `list_documents` (typed results) and `doc://{doc_id}` resources. Talks to the deployed API, so access control stays server-side. |
| **Evaluation** | 35-case test set: recall@k, MRR, answer correctness, citation accuracy, refusal accuracy, access control, injection blocking, optional LLM-judge faithfulness, p50/p95 latency, cost per query. |
| **Observability** | Structured JSON logs, CloudWatch metrics via Embedded Metric Format, X-Ray traces, per-step timings in every answer, and a CloudWatch dashboard. |
| **Infrastructure** | AWS CDK (Python): Aurora Serverless v2 (scales to 0), S3, three container Lambdas, HTTP API with throttling, Secrets Manager, dashboard, optional $10/month budget alert. |

## Architecture

![Architecture](docs/architecture.png)

**Asking a question:** API key → user and teams → input guardrail → embed question → vector + keyword search
(both filtered to the user's teams) → RRF fusion → re-rank → relevance gate → sanitize sources → Claude →
citation check → answer, or a refusal.

### Design decisions

- **Aurora Serverless v2 with min 0 ACU instead of an always-on RDS instance.** The database pauses when idle,
  so a portfolio project doesn't pay for compute 24/7. Trade-off: the first request after a pause waits while
  it resumes (about 15 s); the Data API client retries automatically.
- **RDS Data API instead of Lambda-in-VPC.** Lambdas reach the database over HTTPS with IAM, so there is no
  NAT gateway or VPC endpoint to pay for, and no connection pooling to manage.
- **One container image, three handlers** (API, ingest, migrate), selected with the Lambda `cmd`.
- **Schema migrations run after every deploy** through a CDK Trigger (idempotent `IF NOT EXISTS`).
- **Same SQL locally and on AWS:** `PsycopgExecutor` (local Postgres) and `DataApiExecutor` (Aurora) share
  one store, so integration tests run against a real pgvector database.
- **Honest naming:** the keyword side uses Postgres full-text ranking (`ts_rank_cd`), which is similar in
  spirit to BM25 but is not BM25.

## Results

Measured against the deployed stack (Bedrock models) with `python -m evals.run_evals --target http --judge bedrock`.

| Metric | Value |
|---|---|
| Retrieval recall@5 | _to be measured after deployment_ |
| Answer correctness | _to be measured_ |
| Citation accuracy | _to be measured_ |
| Faithfulness (LLM judge) | _to be measured_ |
| Refusal accuracy (unanswerable + other-team) | _to be measured_ |
| Access control / injection blocking | _to be measured_ |
| Latency p50 / p95 (warm) | _to be measured_ |
| Model cost per question | _to be measured_ |

CI also runs the evals with offline stand-in models on every push. Those numbers only test the plumbing,
but CI fails if access control or injection blocking drops below 100%.

## Run it locally (no AWS needed)

```bash
docker compose up --build        # Postgres + pgvector and the API; sample docs are indexed on start
open http://localhost:8000/docs  # interactive API docs
```

Or without Docker, against any Postgres that has pgvector:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local]"
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/kmcp
python -m knowledgemcp.cli ingest sample_docs
python -m knowledgemcp.cli ask "How many vacation days can I carry over?" --teams hr
pytest -q                        # 50 tests (includes the pgvector integration test)
```

Local mode uses offline stand-ins (word-hash embeddings, an extractive "LLM", lexical re-ranker) so everything
runs without credentials. Set `EMBEDDER=bedrock LLM=bedrock RERANKER=bedrock` to use the real models.

## Deploy to AWS

Prerequisites: AWS CLI configured (`aws sts get-caller-identity` works), Docker running, Node.js 18+, Python 3.12.
In the Bedrock console (us-west-2), make sure your account can use **Titan Text Embeddings V2**,
**Amazon Rerank 1.0** and **Claude Haiku 4.5**.

```bash
pip install -e ".[dev,local]" -r infra/requirements.txt
npm install -g aws-cdk
cd infra
cdk bootstrap                                    # once per account/region
cdk deploy -c alert_email=you@example.com        # prints ApiUrl, BucketName, ApiKeysSecretArn, ...
```

Then load documents, create a key and ask:

```bash
aws s3 sync ../sample_docs s3://<BucketName>/                      # team = first folder
python ../scripts/create_api_key.py --user teja --teams hr,engineering,finance --secret-id <ApiKeysSecretArn>
curl -s <ApiUrl>/ask -H "x-api-key: kmcp_..." -H "content-type: application/json" \
  -d '{"question": "Who approves expenses above 1,000 US dollars?"}'
```

Tear everything down with `cdk destroy`.

## Use it from Claude Desktop (MCP)

```bash
pip install -e ".[mcp]"
```

Add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "knowledgemcp": {
      "command": "/absolute/path/to/.venv/bin/knowledgemcp-mcp",
      "env": { "KMCP_API_URL": "https://<id>.execute-api.us-west-2.amazonaws.com", "KMCP_API_KEY": "kmcp_..." }
    }
  }
}
```

For development without AWS, use `"env": {"KMCP_LOCAL": "1", "KMCP_TEAMS": "hr,engineering", "DATABASE_URL": "..."}`.

## Evaluate

```bash
# one key per persona: hr, engineering, finance, none (public only)
python scripts/create_api_key.py --user eval-hr --teams hr --secret-id <ApiKeysSecretArn>   # repeat per persona
# evals/keys.json  ->  {"hr": "kmcp_...", "engineering": "kmcp_...", "finance": "kmcp_...", "none": "kmcp_..."}
python -m evals.run_evals --target http --url <ApiUrl> --keys evals/keys.json --judge bedrock
```

Results are written to `evals/results/` as JSON and Markdown.

## Project structure

```
src/knowledgemcp/
  loaders.py, chunking.py     PDF/Markdown/text → sections → chunks (page/section kept)
  embeddings.py               Titan V2 embedder (+ offline hash embedder)
  db.py, store.py             Data API / psycopg executors; pgvector store with team filtering
  search.py, rerank.py        hybrid retrieval, RRF, Bedrock re-ranker (+ lexical fallback)
  guardrails.py, llm.py       input/source/output guardrails; Claude via Bedrock Converse
  rag.py                      the question-answering pipeline, metrics and cost
  api.py, auth.py             FastAPI app + Lambda handler; hashed API keys
  mcp_server.py               MCP tools and resources
  ingest.py, handlers_ingest.py, cli.py
infra/                        AWS CDK app (Python) + synth tests
evals/                        test set and evaluation harness
tests/                        pytest suite (moto, botocore Stubber, real pgvector)
sample_docs/                  fictional company documents (hr, engineering, finance, public)
```

## Limitations

- Prompt-injection detection is pattern-based. It is one layer of several (system prompt, data wrapping,
  SQL team filtering, citation checks), not a complete defense.
- HNSW search with a team filter can return fewer than `k` results when a team owns a small share of chunks;
  at larger scale, use pgvector iterative scans or per-team partitions.
- The sample documents describe a fictional company.

## License

MIT
