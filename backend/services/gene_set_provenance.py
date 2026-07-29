# Validation of gene-set provenance strings for register_score.
"""Where a scored gene panel came from, in a machine-checkable form.

Motivation
----------
``register_score`` used to accept a free-text ``gene_list_source`` ("curated
gene set") and validate nothing. A five-gene panel the model assembled from
memory registered identically to a 200-gene versioned database collection, so
downstream nothing -- not the report, not the supervisor -- could tell the two
apart or reproduce either. Two runs of the same analysis produced panels that
disagreed, and there was no record of what either panel actually was.

This module defines a small controlled vocabulary so that provenance is a
parseable value rather than prose. It is deliberately *structural*: it checks
the shape of the claim, never its biology. No pathway, tissue, disease or
organism name appears here, and none may be added -- ScAI is a general-purpose
platform (see CLAUDE.md) and a domain-neutrality test guards the tool schemas.

Grammar
-------
    database:<resource>:<version>:<set_id>
        A named set pulled from a versioned external resource. ``version`` must
        be a concrete release identifier, not "latest" -- the whole point is
        that the panel can be re-fetched byte-identically later.

    derived:<method>[:<detail>]
        Computed from this dataset (differential expression, matrix
        factorisation, marker ranking, ...). Reproducible from the notebook.

    literature:PMID:<id> | literature:DOI:<id>
        Taken from a specific publication.

    bespoke:<justification>
        Assembled by hand. Permitted, but never silently: the justification is
        mandatory, ``is_bespoke`` is True, and callers are expected to surface
        that in the report and to weigh such panels differently.

Anything else is rejected with a corrective message aimed at the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

KIND_DATABASE = "database"
KIND_DERIVED = "derived"
KIND_LITERATURE = "literature"
KIND_BESPOKE = "bespoke"

VALID_KINDS = (KIND_DATABASE, KIND_DERIVED, KIND_LITERATURE, KIND_BESPOKE)

# A version that pins nothing defeats the purpose of recording one.
_UNPINNED_VERSIONS = {"latest", "current", "head", "master", "main", "unknown", "na", "n/a", ""}

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]*$")

_MIN_JUSTIFICATION_CHARS = 20

_EXAMPLES = (
    "database:<resource>:<version>:<set_id>  |  "
    "derived:<method>[:<detail>]  |  "
    "literature:PMID:<id>  |  "
    "bespoke:<why no versioned set fits, >=20 chars>"
)


@dataclass
class GeneSetProvenance:
    """A parsed ``gene_list_source``."""

    kind: str
    raw: str
    resource: str = ""
    version: str = ""
    set_id: str = ""
    detail: str = ""
    justification: str = ""

    @property
    def is_bespoke(self) -> bool:
        return self.kind == KIND_BESPOKE

    @property
    def is_reproducible(self) -> bool:
        """True when the panel can be reconstructed from the record alone."""
        return self.kind in (KIND_DATABASE, KIND_DERIVED)

    def citation(self) -> str:
        """Short human-readable form for the report methodology section."""
        if self.kind == KIND_DATABASE:
            return f"{self.resource} {self.version}: {self.set_id}"
        if self.kind == KIND_DERIVED:
            return f"derived from this dataset ({self.detail or self.resource})"
        if self.kind == KIND_LITERATURE:
            return f"{self.resource.upper()}:{self.set_id}"
        return f"hand-assembled: {self.justification}"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "raw": self.raw,
            "resource": self.resource,
            "version": self.version,
            "set_id": self.set_id,
            "detail": self.detail,
            "justification": self.justification,
        }


def _reject(msg: str) -> tuple[None, str]:
    return None, f"{msg} Expected one of: {_EXAMPLES}"


def parse_provenance(raw) -> tuple[Optional[GeneSetProvenance], Optional[str]]:
    """Parse a ``gene_list_source``.

    Returns ``(provenance, None)`` on success or ``(None, error)`` where
    ``error`` is a corrective message written for the model to act on.
    """
    if raw is None or not str(raw).strip():
        return _reject(
            "gene_list_source is required: record where this gene panel came from."
        )

    text = str(raw).strip()
    kind, _, rest = text.partition(":")
    kind = kind.strip().lower()

    if kind not in VALID_KINDS:
        return _reject(
            f"gene_list_source must start with one of {list(VALID_KINDS)}; got {text!r}."
        )

    if kind == KIND_BESPOKE:
        justification = rest.strip()
        if len(justification) < _MIN_JUSTIFICATION_CHARS:
            return _reject(
                "bespoke gene lists require a justification of at least "
                f"{_MIN_JUSTIFICATION_CHARS} characters saying why no versioned "
                "database set fits. Prefer a database set where one exists."
            )
        return GeneSetProvenance(kind=kind, raw=text, justification=justification), None

    if kind == KIND_LITERATURE:
        scheme, _, ident = rest.partition(":")
        scheme = scheme.strip().lower()
        ident = ident.strip()
        if scheme not in ("pmid", "doi") or not ident:
            return _reject(
                "literature provenance must be literature:PMID:<id> or literature:DOI:<id>."
            )
        return GeneSetProvenance(kind=kind, raw=text, resource=scheme, set_id=ident), None

    if kind == KIND_DERIVED:
        method, _, detail = rest.partition(":")
        method = method.strip()
        if not method:
            return _reject(
                "derived provenance must name the method, e.g. derived:deg:<contrast>."
            )
        return GeneSetProvenance(
            kind=kind, raw=text, resource=method, detail=detail.strip()
        ), None

    # kind == database
    parts = [p.strip() for p in rest.split(":")]
    if len(parts) < 3 or not all(parts[:3]):
        return _reject(
            "database provenance must be database:<resource>:<version>:<set_id>."
        )
    resource, version, set_id = parts[0], parts[1], ":".join(parts[2:])
    if version.lower() in _UNPINNED_VERSIONS:
        return _reject(
            f"database version {version!r} does not pin anything; record the concrete "
            "release identifier so the panel can be re-fetched later."
        )
    if not _ID_RE.match(resource):
        return _reject(f"database resource {resource!r} is not a valid identifier.")
    return GeneSetProvenance(
        kind=kind, raw=text, resource=resource.lower(), version=version, set_id=set_id
    ), None


# --------------------------------------------------------------------------- power

#: Panels smaller than this cannot support a *negative* conclusion. The value is
#: the conventional ``minGSSize`` floor used by mainstream enrichment tooling
#: (clusterProfiler and fgsea both default to 10); it is a reporting threshold,
#: not a claim that smaller panels are meaningless.
MIN_PANEL_SIZE_FOR_NEGATIVE_CLAIM = 10


def assess_panel_power(n_genes: int, n_matched: Optional[int] = None) -> dict:
    """Describe what a panel of this size can and cannot support.

    ``n_matched`` -- how many panel genes survived mapping into the analysed
    matrix -- is the number that actually governs power; ``n_genes`` is only the
    nominal size. When ``n_matched`` is unknown the nominal size is used and the
    result is marked as such.
    """
    effective = n_matched if n_matched is not None else n_genes
    # A panel with nothing left after mapping is the MOST underpowered case, not
    # an exempt one. The earlier `0 < effective` guard was meant to skip scores
    # that carry no panel at all, but it also excused a 200-gene panel of which
    # zero genes were measured -- precisely the case where a null result says
    # least. Distinguish "no panel was declared" (n_genes == 0) from "the panel
    # vanished on contact with the data" (n_genes > 0, effective == 0).
    declared_a_panel = n_genes > 0
    underpowered = declared_a_panel and effective < MIN_PANEL_SIZE_FOR_NEGATIVE_CLAIM
    return {
        "n_genes": int(n_genes),
        "n_matched": None if n_matched is None else int(n_matched),
        "effective_n": int(effective),
        "match_rate": (round(n_matched / n_genes, 4)
                       if n_matched is not None and n_genes else None),
        "underpowered": bool(underpowered),
        "supports_negative_claim": bool(declared_a_panel and not underpowered),
        "threshold": MIN_PANEL_SIZE_FOR_NEGATIVE_CLAIM,
        "basis": "matched" if n_matched is not None else "nominal",
    }


def power_caveat(power: dict) -> str:
    """One-line caveat for a panel, or '' when none is warranted.

    Carries the governing method rule so the model receives the reasoning, not
    just the verdict. The rule lookup is deferred to call time and degrades to
    the bare caveat if the corpus is unavailable -- guidance is an enhancement,
    never a dependency.
    """
    if not power.get("underpowered"):
        return ""
    if power.get("effective_n") == 0:
        head = (
            "NONE of this panel's genes are present in the analysed matrix, so it "
            "measures nothing at all. Check the organism and the gene-naming "
            "convention before using this score for anything."
        )
    else:
        head = ""
    caveat = head + (
        f" Panel has only {power['effective_n']} usable gene(s) "
        f"({power['basis']}), below the conventional minimum of "
        f"{power['threshold']}. Positive enrichment may still be reported as an "
        f"observation, but a non-significant result from this panel does NOT "
        f"support a negative conclusion -- it is underpowered, not evidence of absence."
    )
    try:
        from .method_rules import rule_text

        guidance = rule_text("minimum-gene-set-size", include_python=True)
    except Exception:
        guidance = ""
    return f"{caveat} {guidance}".strip()
