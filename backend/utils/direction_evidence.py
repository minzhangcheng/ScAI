"""Shared factory for degraded direction-literature-evidence rows.

Lives in utils (not workflow) so both the agent layer (PIAgent) and the
workflow layer (SynthesisMixin) can consume it without the agent layer
importing the orchestration layer.
"""
from __future__ import annotations


def build_fallback_direction_evidence(raw_candidates, *, status, reason, limitations=None):
    """Build a uniform degraded direction-evidence list (skipped/failed paths)."""
    lims = limitations if limitations is not None else [reason]
    return [
        {
            "direction_id": c.get("direction_id", f"dir_{idx + 1:03d}"),
            "direction": c.get("direction", ""),
            "status": status,
            "search_tier": "failed",
            "literature_novelty": "unknown",
            "confidence": "failed",
            "summary": reason,
            "limitations": lims,
        }
        for idx, c in enumerate(raw_candidates)
    ]
