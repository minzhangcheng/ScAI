"""Shared technical contract for the bounded supplementary executor.

The PI decides whether a scientific follow-up is useful.  This module only
describes the objective capabilities of the restricted execution path so the
planner and executor cannot drift onto different assumptions.
"""
from __future__ import annotations


SUPPLEMENTARY_CONTRACT_VERSION = 2


def render_supplementary_capability_contract() -> str:
    """Return the single prompt block shared by PI and executor."""
    return f"""SUPPLEMENTARY TECHNICAL CONTRACT v{SUPPLEMENTARY_CONTRACT_VERSION}
- The pass may read objects already present in the live kernel and exact paths
  registered in AnalysisState.data_files. A registered file need not already be
  loaded. It may not acquire an unregistered, external, or public input.
- The pass may add bounded quantitative evidence derived from the current
  canonical object, including differential expression, scores, enrichment,
  pathway activity, and composition summaries.
- The pass may mutate, remap, filter, normalize, scale, re-cluster, rebuild
  neighbourhoods or embeddings, or rebind the current canonical object when an
  approved supplementary task explicitly calls for that operation. It must not
  silently redirect a requested live-object change into a derived variable.
- The pass may not expand beyond the approved supplementary tasks or write
  analysis data files.
- The pass may not render, display, or save plots or images. Publication visuals
  belong to the existing main figure follow-up pipeline.
- Work that needs an external input, broader tools, or more execution than the
  bounded pass allows must remain a PI-visible follow-up for Chat or a later main
  plan; live-object mutation alone is not a reason to defer it."""


__all__ = [
    "SUPPLEMENTARY_CONTRACT_VERSION",
    "render_supplementary_capability_contract",
]
