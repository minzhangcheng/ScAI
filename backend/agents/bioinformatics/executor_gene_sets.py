"""Small invariant helpers for gene-set fetch dispatch."""
from __future__ import annotations


def resolve_fetch_organism(args, state) -> str:
    return str(args.get("organism") or getattr(state, "species", "") or "").strip()


def fetch_failure(error) -> str:
    return f"fetch_gene_set failed: {error}"


def partition_failure(error) -> str:
    return f"fetch_gene_set could not partition: {error}"
