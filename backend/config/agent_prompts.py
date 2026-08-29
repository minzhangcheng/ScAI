"""Agent prompt templates for the ScAI multi-agent system.

The prompt TEXT is not defined here. It lives in
``frontend/src/i18n/agent-prompts.json``, which is the single source of truth
for both the browser (which seeds the prompts panel from it) and this module
(which supplies the defaults a headless / API session falls back to when it
sends no ``custom_prompt`` override).

The two used to be maintained as separate copies and drifted apart in both
directions: the frontend copy lost the organism-aware enrichment guidance and
the "never fabricate citations" rule, while the backend copy lost the
honest-reporting blocks that let pathology and biochemistry decline to answer.
Whichever copy a given session happened to load decided which guardrails it
had. Reading one file removes that class of bug entirely.
"""
from pathlib import Path
from typing import Dict
import json

from pydantic import BaseModel

#: Repo-relative location of the shared prompt text. ``COPY . .`` puts the
#: frontend sources in the runtime image and ``.dockerignore`` does not exclude
#: ``frontend/src``, so this resolves inside the container as well as in a
#: checkout.
PROMPTS_JSON_PATH = (
    Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n" / "agent-prompts.json"
)

AGENT_PROMPT_KEYS = (
    "pi",
    "pathology",
    "bioinformatics",
    "biochemistry",
    "literature",
    "painter",
    "supervisor",
    "technique",
)


def _load_default_prompts() -> Dict[str, str]:
    """Read the shared prompt text.

    Raises rather than degrading to empty strings. An agent running with no
    system prompt still produces fluent, plausible output, so a silent fallback
    would surface as bad science days later instead of as a startup error.
    """
    try:
        raw = json.loads(PROMPTS_JSON_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - deployment error
        raise RuntimeError(
            f"Default agent prompts not found at {PROMPTS_JSON_PATH}. This file is "
            "the single source of truth shared with the frontend and must ship with "
            "the backend."
        ) from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - deployment error
        raise RuntimeError(
            f"Default agent prompts at {PROMPTS_JSON_PATH} are not valid JSON: {exc}"
        ) from exc

    missing = [k for k in AGENT_PROMPT_KEYS if not str(raw.get(k, "")).strip()]
    if missing:
        raise RuntimeError(
            f"Default agent prompts at {PROMPTS_JSON_PATH} are missing or empty for: "
            f"{', '.join(missing)}"
        )
    return {k: str(raw[k]) for k in AGENT_PROMPT_KEYS}


DEFAULT_PROMPTS: Dict[str, str] = _load_default_prompts()


class AgentPrompts(BaseModel):
    """Customizable prompts for each agent - all in English for consistency.

    Defaults come from :data:`DEFAULT_PROMPTS`; edit the JSON, not this class.
    """

    pi_prompt: str = DEFAULT_PROMPTS["pi"]
    pathology_prompt: str = DEFAULT_PROMPTS["pathology"]
    bioinformatics_prompt: str = DEFAULT_PROMPTS["bioinformatics"]
    biochemistry_prompt: str = DEFAULT_PROMPTS["biochemistry"]
    literature_prompt: str = DEFAULT_PROMPTS["literature"]
    painter_prompt: str = DEFAULT_PROMPTS["painter"]
    technique_prompt: str = DEFAULT_PROMPTS["technique"]
    supervisor_prompt: str = DEFAULT_PROMPTS["supervisor"]
