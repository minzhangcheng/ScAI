"""Command routing for source-tree and outcome proof operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .discovery import ContractError
from .evidence import validate_outcomes
from .identity import (
    _write_json, build_commit_manifest, build_index_manifest, build_worktree_manifest,
    compare_index, compare_post_summary, verify_final_commit,
)

def cli() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name, kind in (("worktree", "worktree"), ("index", "index")):
        command = sub.add_parser(name)
        command.add_argument("--base", required=True)
        command.add_argument("--output", type=Path, required=True)
    commit_parser = sub.add_parser("commit")
    commit_parser.add_argument("--base", required=True)
    commit_parser.add_argument("--commit", required=True)
    commit_parser.add_argument("--output", type=Path, required=True)
    summary_parser = sub.add_parser("compare-summary")
    summary_parser.add_argument("--tested", type=Path, required=True)
    summary_parser.add_argument("--post", type=Path, required=True)
    summary_parser.add_argument("--summary-path", type=Path, required=True)
    index_parser = sub.add_parser("compare-index")
    index_parser.add_argument("--post", type=Path, required=True)
    index_parser.add_argument("--index", type=Path, required=True)
    verify_parser = sub.add_parser("verify-commit")
    verify_parser.add_argument("--base", required=True)
    verify_parser.add_argument("--index", type=Path, required=True)
    verify_parser.add_argument("--subject", required=True)
    outcomes_parser = sub.add_parser("validate-outcomes")
    outcomes_parser.add_argument("--collection", type=Path, required=True)
    outcomes_parser.add_argument("--outcomes", type=Path, required=True)
    outcomes_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        repo = Path.cwd()
        if args.command in {"worktree", "index", "commit"}:
            if args.command == "worktree":
                result = build_worktree_manifest(repo, args.base)
            elif args.command == "index":
                result = build_index_manifest(repo, args.base)
            else:
                result = build_commit_manifest(repo, args.base, args.commit)
            _write_json(args.output, result)
        elif args.command == "compare-summary":
            compare_post_summary(args.tested, args.post, args.summary_path)
        elif args.command == "compare-index":
            compare_index(args.post, args.index)
        elif args.command == "verify-commit":
            print(verify_final_commit(repo, args.base, args.index, args.subject))
        elif args.command == "validate-outcomes":
            collection = json.loads(args.collection.read_text(encoding="utf-8"))
            nodeids = tuple(collection["nodeids"])
            _write_json(args.output, validate_outcomes(args.outcomes.read_bytes(), nodeids))
    except ContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2
    return 0
