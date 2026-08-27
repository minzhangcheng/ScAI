"""Painter generation prompt construction."""
from __future__ import annotations


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

def _build_graphic_prompt(self, state: AnalysisState, summary: str) -> str:
    """
    Build a detailed prompt for graphic abstract generation.

    Args:
        state: Analysis state
        summary: PI's summary of findings

    Returns:
        Formatted prompt string
    """
    # Extract key findings from all rounds
    findings = []
    for round_result in state.round_results:
        findings.extend(round_result.bioinformatics_findings[:3])
        findings.extend(round_result.pathology_findings[:2])

    findings_text = "\n".join(f"- {f}" for f in findings[:8]) if findings else "No specific findings available"

    # Budget: ~30K input tokens total. Fixed template ≈ 500 tokens, leave rest for summary.
    # Cap summary at ~25K tokens (~75K chars) as safety net — PI should already keep it short.
    max_summary_chars = 75_000
    if len(summary) > max_summary_chars:
        logger.warning(
            f"[{self.session_id}][Painter] Summary too long ({len(summary)} chars), "
            f"truncating to {max_summary_chars} chars"
        )
        summary = summary[:max_summary_chars] + "\n... [truncated due to token limit]"

    prompt = f"""Create a professional scientific graphic abstract for a biomedical research paper.

Research Topic: {state.name}
Research Objective: {state.initial_instruction[:500] if state.initial_instruction else 'Biomedical research analysis'}

Key Findings Summary:
{summary}

Main Discoveries:
{findings_text}

Style Requirements:
- Professional scientific illustration style suitable for journal publication
- Clean, modern design with clear visual hierarchy
- Use appropriate scientific symbols and cell/tissue representations
- Include visual representations of:
  * Key cell types or tissues involved
  * Molecular pathways or mechanisms
  * Treatment effects or comparisons
  * Key biomarkers or genes
- Color scheme: Use professional, publication-appropriate colors
- Layout: Clear flow from left to right or top to bottom showing the research narrative
- Avoid excessive text - focus on visual communication
- Include simple labels for key elements
- Resolution and quality suitable for journal publication

The graphic abstract should tell the story of the research at a glance, highlighting the main findings and conclusions.
"""
    return prompt
