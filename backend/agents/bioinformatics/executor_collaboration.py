"""Shared response helpers for collaboration and lookup tools."""
from __future__ import annotations


def species_validation_response(state, assessment, action=None) -> str:
    if assessment.get("match", True):
        return f"Gene names are consistent with {state.species}. Proceed with analysis."
    if action:
        if action.get("action") == "change_species":
            return f"Species updated to {action.get('new_species', '')}. Proceed."
        if action.get("action") == "custom":
            return "User instruction received. Proceed with analysis."
        return f"Keeping species as {state.species}. Proceed with caution."
    return (
        f"Warning: gene names may not match {state.species}. "
        f"{assessment.get('reasoning', '')}. Proceeding anyway."
    )


def method_guidance(method_question: str, *, format_rule, search_rules) -> str:
    hits = search_rules(method_question, limit=3)
    if hits:
        return "\n\n".join(format_rule(rule) for rule in hits)
    return (
        "No rule in the corpus covers that question. Proceed with standard "
        "practice and state your reasoning in the findings."
    )
