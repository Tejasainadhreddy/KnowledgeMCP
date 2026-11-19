"""Small command line tool for local development.

    python -m knowledgemcp.cli migrate              # create tables and indexes
    python -m knowledgemcp.cli ingest sample_docs   # index a folder laid out as team/file
    python -m knowledgemcp.cli ask "How many PTO days carry over?" --teams hr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingest import Ingestor
from .models import User
from .services import build_embedder, build_rag, build_store, settings


def main() -> None:
    p = argparse.ArgumentParser(prog="knowledgemcp")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    ing = sub.add_parser("ingest")
    ing.add_argument("folder", type=Path)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--teams", default="")
    args = p.parse_args()

    s = settings()
    store = build_store(s)
    if args.cmd == "migrate":
        store.ensure_schema()
        print("schema ready")
    elif args.cmd == "ingest":
        store.ensure_schema()
        for r in Ingestor(store, build_embedder(s), s.chunk_chars, s.chunk_overlap).ingest_folder(args.folder):
            print(f"{r.status:9} {r.doc_id} chunks={r.chunks}")
    else:
        user = User("cli", [t for t in args.teams.split(",") if t])
        print(json.dumps(build_rag(s, store).ask(args.question, user).to_dict(), indent=2))


if __name__ == "__main__":
    main()
