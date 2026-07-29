# Network access for gene-set resolution. Runs in the FastAPI process, never in the sandbox.
"""Fetch MSigDB collections and project them across organisms.

Sandbox code cannot make network calls (utils/sandbox_parts/shared.py blocks
socket/requests/urllib/httpx/aiohttp), so this module runs backend-side and the
resolved genes are handed to the kernel as data -- the same division
``scripts/fetch_markers.py`` and ``_compute_database_priors`` already use.

Everything here fails soft. A gene set that cannot be fetched must degrade to a
clear message the agent can act on, never to an exception that aborts an
analysis mid-round.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import httpx

from .gene_set_resolver import (
    DEFAULT_HOMOLOGY_TYPES,
    ENSEMBL_SOURCE_SPECIES,
    GeneSetResult,
    MSIGDB_ORGANISMS,
    _DEFAULT_CACHE,
    build_provenance,
    cache_path,
    msigdb_gmt_url,
    normalize_organism,
    parse_gmt,
    summarize_projection,
    suggest_similar,
    validate_set_id,
)

logger = logging.getLogger(__name__)

ENSEMBL_REST = "https://rest.ensembl.org"

_HTTP_TIMEOUT = 60
_ENSEMBL_TIMEOUT = 30


def _get_text(url: str, timeout: int = _HTTP_TIMEOUT) -> str:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def load_collection(version: str, collection: str, organism: str,
                    cache_dir: Optional[Path] = None) -> dict[str, list[str]]:
    """Return {set_id: genes} for one MSigDB collection, caching the GMT on disk.

    The cache key includes the release, so a pinned version is served from disk
    forever and a version bump is a different file rather than a silent change
    under the same name.
    """
    organism = normalize_organism(organism)
    code = MSIGDB_ORGANISMS.get(organism)
    if not code:
        raise ValueError(
            f"MSigDB publishes native collections for "
            f"{sorted(MSIGDB_ORGANISMS)} only; {organism!r} must be projected."
        )
    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, version, collection, code)

    if path.exists():
        return parse_gmt(path.read_text(encoding="utf-8"))

    url = msigdb_gmt_url(version, collection, code)
    text = _get_text(url)
    sets = parse_gmt(text)
    if not sets:
        raise ValueError(f"{url} parsed to zero gene sets")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)  # atomic: never leave a half-written GMT for the next read
    return sets


def project_genes(genes: Iterable[str], source_organism: str,
                  target_species: str,
                  homology_types: Iterable[str] = DEFAULT_HOMOLOGY_TYPES,
                  ) -> dict[str, list[str]]:
    """Map genes to a target organism through Ensembl Compara orthology.

    Returns {source_gene: [target_symbols]} for genes that mapped. A gene with
    no orthologue of an accepted type is simply absent -- callers compare the
    size of this dict against the input to get the mapping rate.
    """
    source_species = ENSEMBL_SOURCE_SPECIES.get(normalize_organism(source_organism))
    if not source_species:
        raise ValueError(f"cannot project from {source_organism!r}")

    accepted = set(homology_types)
    mapped: dict[str, list[str]] = {}
    with httpx.Client(timeout=_ENSEMBL_TIMEOUT, follow_redirects=True) as client:
        for gene in genes:
            url = (
                f"{ENSEMBL_REST}/homology/symbol/{source_species}/{gene}"
                f"?target_species={target_species}&type=orthologues"
                f"&content-type=application/json"
            )
            try:
                response = client.get(url)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # one gene failing must not lose the set
                logger.debug("ortholog lookup failed for %s: %s", gene, exc)
                continue
            data = payload.get("data") or []
            if not data:
                continue
            targets: list[str] = []
            for homology in data[0].get("homologies") or []:
                if homology.get("type") not in accepted:
                    continue
                symbol = (homology.get("target") or {}).get("id")
                # Ensembl returns a stable ID here; the display symbol is not
                # always present, so keep whatever identifier we are given and
                # let the caller reconcile against its own var_names.
                if symbol:
                    targets.append(str(symbol))
            if targets:
                mapped[gene] = targets
    return mapped


def resolve_symbols(stable_ids: Iterable[str]) -> dict[str, str]:
    """Map Ensembl stable gene IDs to display symbols in one batched request.

    Ensembl's POST /lookup/id accepts up to 1000 ids per call. Any id without a
    display name is simply absent from the result; callers keep the stable id in
    that case rather than dropping the gene, since a stable id is still a valid
    identifier and dropping it would silently shrink the panel.
    """
    ids = [i for i in dict.fromkeys(stable_ids) if i]
    if not ids:
        return {}
    out: dict[str, str] = {}
    with httpx.Client(timeout=_ENSEMBL_TIMEOUT, follow_redirects=True) as client:
        for start in range(0, len(ids), 900):
            chunk = ids[start:start + 900]
            try:
                response = client.post(
                    f"{ENSEMBL_REST}/lookup/id",
                    headers={"Content-Type": "application/json",
                             "Accept": "application/json"},
                    json={"ids": chunk},
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                logger.debug("symbol lookup failed for %d ids: %s", len(chunk), exc)
                continue
            for stable_id, record in (payload or {}).items():
                name = (record or {}).get("display_name")
                if name:
                    out[stable_id] = str(name)
    return out


def resolve_gene_set(set_id: str, organism: str, *, version: str,
                     collection: str, resource: str = "msigdb",
                     cache_dir: Optional[Path] = None,
                     homology_types: Iterable[str] = DEFAULT_HOMOLOGY_TYPES,
                     ) -> tuple[Optional[GeneSetResult], Optional[str]]:
    """Resolve one named gene set for one organism.

    Returns ``(result, None)`` or ``(None, error)`` where ``error`` is written
    for the agent to act on.
    """
    problem = validate_set_id(set_id)
    if problem:
        return None, problem

    organism = normalize_organism(organism)
    native = organism in MSIGDB_ORGANISMS

    # For a non-native organism, pull the set from whichever published organism
    # we are projecting from. Mouse is the closer source for most non-human
    # work, but the caller may already have told us otherwise via `organism`.
    source_organism = organism if native else "mouse"

    try:
        sets = load_collection(version, collection, source_organism, cache_dir)
    except Exception as exc:
        return None, (
            f"could not load collection {collection!r} version {version!r} for "
            f"{source_organism}: {exc}"
        )

    genes = sets.get(set_id)
    if genes is None:
        hints = suggest_similar(set_id, sets.keys())
        hint_text = f" Did you mean: {', '.join(hints)}?" if hints else ""
        return None, (
            f"{set_id!r} is not in {collection} v{version} "
            f"({len(sets)} sets available).{hint_text}"
        )

    provenance = build_provenance(resource, f"{version}.{MSIGDB_ORGANISMS[source_organism]}", set_id)

    if native:
        return GeneSetResult(
            set_id=set_id, genes=list(genes), resource=resource,
            version=f"{version}.{MSIGDB_ORGANISMS[organism]}",
            collection=collection, organism=organism,
            source_organism=organism, projected=False,
            provenance=provenance,
        ), None

    target_species = organism.replace(" ", "_").lower()
    try:
        mapped = project_genes(genes, source_organism, target_species, homology_types)
    except Exception as exc:
        return None, f"orthology projection to {organism!r} failed: {exc}"

    if not mapped:
        return None, (
            f"no gene of {set_id!r} has a {'/'.join(homology_types)} orthologue in "
            f"{organism!r}. The set cannot be projected; use a natively published "
            f"organism or supply the panel explicitly."
        )

    projected_ids: list[str] = []
    for targets in mapped.values():
        projected_ids.extend(targets)

    # Stable IDs are correct but will not match a symbol-indexed matrix, so
    # resolve display symbols and fall back to the stable id where none exists.
    try:
        symbols = resolve_symbols(projected_ids)
    except Exception as exc:
        logger.warning("symbol resolution failed, keeping stable ids: %s", exc)
        symbols = {}
    n_unresolved = sum(1 for i in set(projected_ids) if i not in symbols)

    # Deduplicate while preserving order; one2many mappings legitimately produce
    # repeats and double-counting them would inflate any downstream score.
    seen: set[str] = set()
    unique = []
    for stable_id in projected_ids:
        gene = symbols.get(stable_id, stable_id)
        if gene not in seen:
            seen.add(gene)
            unique.append(gene)

    projection = summarize_projection(len(genes), mapped, homology_types)
    projection["n_unresolved_symbols"] = n_unresolved
    return GeneSetResult(
        set_id=set_id, genes=unique, resource=resource,
        version=f"{version}.{MSIGDB_ORGANISMS[source_organism]}",
        collection=collection, organism=organism,
        source_organism=source_organism, projected=True,
        projection=projection,
        provenance=provenance,
    ), None
