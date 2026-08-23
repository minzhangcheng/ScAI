"""Pure cross-file barcode canonicalization for the trusted kernel path."""

from __future__ import annotations

import re
from collections.abc import Iterable


_DNA_CORE = re.compile(r"[ACGTN]{8,}", re.IGNORECASE)


def canonicalize_barcode(value) -> str:
    """Return the canonical DNA-like core for one supported barcode form."""
    if not isinstance(value, str) or not value:
        raise ValueError("barcode must be a non-empty string")

    candidate = value
    if candidate[-1:] in {"x", "X"}:
        candidate = candidate[:-1]

    dash = candidate.rfind("-")
    if dash >= 0 and candidate[dash + 1 :].isdigit():
        candidate = candidate[:dash]

    if ":" in candidate:
        prefix, candidate = candidate.rsplit(":", 1)
        if not prefix or ":" in prefix:
            raise ValueError(f"unsupported barcode identifier: {value!r}")
    if "_" in candidate:
        prefix, candidate = candidate.rsplit("_", 1)
        if not prefix:
            raise ValueError(f"unsupported barcode identifier: {value!r}")

    if _DNA_CORE.fullmatch(candidate) is None:
        raise ValueError(f"unsupported barcode identifier: {value!r}")
    return candidate.upper()


def canonicalize_barcodes(values: Iterable[str], *, label="barcodes") -> list[str]:
    """Canonicalize values and reject duplicate identities after normalization."""
    canonical: list[str] = []
    seen: dict[str, str] = {}
    for raw in values:
        normalized = canonicalize_barcode(raw)
        prior = seen.get(normalized)
        if prior is not None and prior != raw:
            raise ValueError(
                f"{label} collision after canonicalization: "
                f"{prior!r} and {raw!r} -> {normalized!r}"
            )
        seen[normalized] = raw
        canonical.append(normalized)
    return canonical


def barcode_index(values: Iterable[str], *, label="barcodes") -> dict[str, int]:
    """Return canonical barcode positions while sharing collision validation."""
    canonical = canonicalize_barcodes(values, label=label)
    return {value: position for position, value in enumerate(canonical)}
