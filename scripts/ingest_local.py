"""Index a local folder (laid out as team/file) into the database from your laptop.

Local Postgres:  python scripts/ingest_local.py sample_docs
Aurora on AWS:   DB_MODE=dataapi DB_CLUSTER_ARN=... DB_SECRET_ARN=... EMBEDDER=bedrock \
                 python scripts/ingest_local.py sample_docs
(On AWS the normal path is uploading to S3; this is a shortcut for development.)
"""

import sys
from pathlib import Path

from knowledgemcp.ingest import Ingestor
from knowledgemcp.services import build_embedder, build_store, settings


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_docs")
    s = settings()
    store = build_store(s)
    store.ensure_schema()
    for r in Ingestor(store, build_embedder(s), s.chunk_chars, s.chunk_overlap).ingest_folder(root):
        print(f"{r.status:9} {r.doc_id} chunks={r.chunks}")


if __name__ == "__main__":
    main()
