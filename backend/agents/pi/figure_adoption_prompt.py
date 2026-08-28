"""Prompt assembly for binding existing workspace figures to plan steps."""

from typing import Any, List


def build_figure_adoption_prompt(
    state: Any,
    requirements: List[Any],
    shown: List[Any],
) -> str:
    req_lines = "\n".join(
        f"- step {getattr(r, 'plan_step_id', None)} needs a "
        f"{getattr(r, 'figure_type', 'figure')}: {getattr(r, 'description', '')}"
        for r in requirements
    )
    cand_lines = "\n".join(
        f"[image {i + 1}] filename={c.name!r} — "
        f"{c.provenance or 'no recorded provenance'}"
        for i, c in enumerate(shown)
    )
    return f"""These images were produced during the analysis but were never registered
against the approved plan, so the platform cannot tell which of them satisfies
which planned deliverable. Decide that now.

Research objective: {state.initial_instruction}

Unmet plan requirements:
{req_lines}

Candidate images, in the order attached:
{cand_lines}

For each requirement you can satisfy, return one binding. Rules:
- Look at the image. Bind a requirement only when the image really is that kind
  of figure and really shows what the requirement describes.
- Several candidates may be successive attempts at the same figure. Choose the
  finished one by what the images show and by the recorded provenance. Do NOT
  infer version order from filename suffixes such as _final, _fixed or _v2 —
  those are unreliable and in past sessions the "_fixed" file was the older one.
- Use each filename at most once.
- If nothing among these images satisfies a requirement, leave it out. Do not
  stretch an unrelated figure to fill a gap; an unmet requirement is reported
  honestly, a wrong binding files a real figure under the wrong step.
- caption: one sentence describing what the figure shows, for the report.

Respond with ONLY a JSON array:
[{{"plan_step_id": <int>, "figure_type": "<type>", "filename": "<name>", "caption": "<one sentence>"}}]
Return [] if none of the images satisfy any requirement.
"""
