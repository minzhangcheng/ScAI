# Fetching named gene sets from versioned external resources.
"""Resolve a named gene set to genes, for any organism, with provenance.

Why this exists
---------------
The bioinformatics agent could not fetch a curated gene set at all: sandbox code
is blocked from making network calls (see utils/sandbox_parts/shared.py), so
every panel it scored was assembled from the model's own memory. Two runs of the
same analysis therefore built different panels for the same question, and
neither was reproducible.

The fetch has to happen in the backend process and be handed to the kernel as a
file -- the pattern ``_compute_database_priors`` already uses for marker priors.

Organism handling
-----------------
MSigDB publishes native collections for exactly two organisms, Hs and Mm, with
symbols already in the right convention for each. For anything else the set is
projected through Ensembl Compara orthology, and the projection is recorded:
which source organism, how many genes mapped, and under what homology type.

Projection is lossy and, across enough evolutionary distance, meaningless -- a
mammalian pathway set mapped onto a distant organism can lose most of its
members and retain the ones that are least specific. So the mapping rate travels
with the result and a low rate is reported, not hidden. Callers decide; this
module does not silently pretend a projected set is equivalent to a native one.

Domain neutrality
-----------------
Resource and collection identifiers are configuration. No pathway, tissue,
disease or organism-specific default appears here beyond the two organisms the
upstream resource itself publishes, and no curated biology ships with ScAI.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

#: Where cached GMTs live. Mirrors backend/data/markers; both are runtime caches
#: rather than tracked content, so nothing curated ships with the repository.
_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "gene_sets"

MSIGDB_BASE = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"

#: MSigDB's own two-organism split. Its collection prefixes differ per organism
#: (human uses h/c2/c5, mouse uses mh/m2/m5), which is why this is a table and
#: not string interpolation.
MSIGDB_ORGANISMS = {
    "human": "Hs",
    "mouse": "Mm",
}

_ORGANISM_ALIASES = {
    "human": "human", "homo sapiens": "human", "hsapiens": "human", "hs": "human",
    "mouse": "mouse", "mus musculus": "mouse", "mmusculus": "mouse", "mm": "mouse",
}

#: Ensembl species names for the two organisms we can project *from*.
ENSEMBL_SOURCE_SPECIES = {
    "human": "homo_sapiens",
    "mouse": "mus_musculus",
}

#: Below this fraction of the source set surviving projection, the result is
#: flagged. Not a hard refusal: for some organism pairs a low rate is expected
#: and the caller may still want the set, but it must not pass unremarked.
LOW_MAPPING_RATE = 0.5

#: One-to-one orthologues are the defensible default. one2many/many2many
#: introduce duplicate counting and ambiguity, so they are opt-in.
DEFAULT_HOMOLOGY_TYPES = ("ortholog_one2one",)

_SET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]*$")


def normalize_organism(organism: str) -> str:
    """Map an organism string to a canonical key, or return it lowercased."""
    text = str(organism or "").strip().lower()
    return _ORGANISM_ALIASES.get(text, text)


@dataclass
class GeneSetResult:
    """A resolved gene set plus everything needed to cite and re-fetch it."""

    set_id: str
    genes: list[str]
    resource: str
    version: str
    collection: str
    organism: str
    #: Organism the set was actually published for. Differs from ``organism``
    #: only when the set was projected.
    source_organism: str = ""
    projected: bool = False
    projection: dict = field(default_factory=dict)
    provenance: str = ""

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    def to_dict(self) -> dict:
        return {
            "set_id": self.set_id,
            "genes": self.genes,
            "n_genes": self.n_genes,
            "resource": self.resource,
            "version": self.version,
            "collection": self.collection,
            "organism": self.organism,
            "source_organism": self.source_organism,
            "projected": self.projected,
            "projection": self.projection,
            "provenance": self.provenance,
        }

    def warnings(self) -> list[str]:
        """Caveats a caller must surface rather than swallow."""
        out: list[str] = []
        if self.projected:
            rate = self.projection.get("mapping_rate")
            src = self.source_organism or "another organism"
            out.append(
                f"Projected from {src} via Ensembl orthology "
                f"({self.projection.get('n_mapped', '?')}/"
                f"{self.projection.get('n_source', '?')} genes mapped"
                + (f", {rate:.0%}" if isinstance(rate, float) else "")
                + "). A projected set is not equivalent to a natively curated one."
            )
            unresolved = self.projection.get("n_unresolved_symbols") or 0
            if unresolved:
                out.append(
                    f"{unresolved} projected gene(s) had no display symbol and are "
                    f"reported as stable identifiers; those will not match a "
                    f"symbol-indexed matrix."
                )
            if isinstance(rate, float) and rate < LOW_MAPPING_RATE:
                out.append(
                    f"Mapping rate is below {LOW_MAPPING_RATE:.0%}: most of the "
                    f"original set did not survive projection, and the genes that "
                    f"did are not a random sample of it. Treat this set as "
                    f"exploratory and say so in any finding that rests on it."
                )
        return out


def build_provenance(resource: str, version: str, set_id: str) -> str:
    """Provenance string in the vocabulary services/gene_set_provenance parses."""
    return f"database:{resource}:{version}:{set_id}"


def msigdb_gmt_url(version: str, collection: str, organism_code: str) -> str:
    """URL of one MSigDB collection GMT.

    The release identifier is part of the path, which is what makes an MSigDB
    citation re-fetchable: ``2025.1.Mm`` names an immutable file.
    """
    stamp = f"{version}.{organism_code}"
    return f"{MSIGDB_BASE}/{stamp}/{collection}.v{stamp}.symbols.gmt"


def parse_gmt(text: str) -> dict[str, list[str]]:
    """Parse GMT content into {set_id: genes}.

    GMT is tab-separated: name, description, then one gene per remaining field.
    Blank fields appear in real files and are dropped.
    """
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        name = parts[0].strip()
        genes = [g.strip() for g in parts[2:] if g.strip()]
        if name and genes:
            out[name] = genes
    return out


def cache_path(cache_dir: Path, version: str, collection: str,
               organism_code: str) -> Path:
    return Path(cache_dir) / f"{collection}.v{version}.{organism_code}.symbols.gmt"


def validate_set_id(set_id: str) -> Optional[str]:
    """Return an error message if the identifier is unusable, else None."""
    text = str(set_id or "").strip()
    if not text:
        return "set_id is required."
    if not _SET_ID_RE.match(text):
        return f"set_id {text!r} is not a valid identifier."
    return None


def suggest_similar(set_id: str, available: Iterable[str], limit: int = 8) -> list[str]:
    """Candidate identifiers for a miss, so a typo is recoverable in one step.

    Matches on shared tokens rather than edit distance: gene-set names are
    underscore-joined words, and a caller who gets one word right is usually
    closer than raw string distance suggests.
    """
    wanted = {t for t in re.split(r"[^A-Za-z0-9]+", str(set_id).upper()) if len(t) > 2}
    if not wanted:
        return []
    scored: list[tuple[int, str]] = []
    for name in available:
        tokens = {t for t in re.split(r"[^A-Za-z0-9]+", name.upper()) if len(t) > 2}
        overlap = len(wanted & tokens)
        if overlap:
            scored.append((overlap, name))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [name for _, name in scored[:limit]]


def summarize_projection(n_source: int, mapped: dict[str, list[str]],
                         homology_types: Iterable[str]) -> dict:
    """Projection statistics recorded alongside a projected set."""
    n_mapped = len(mapped)
    return {
        "n_source": int(n_source),
        "n_mapped": int(n_mapped),
        "mapping_rate": round(n_mapped / n_source, 4) if n_source else 0.0,
        "homology_types": list(homology_types),
        "one_to_many": sorted(g for g, targets in mapped.items() if len(targets) > 1),
    }
