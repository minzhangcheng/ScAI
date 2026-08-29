"""Default agent prompts must have exactly one source of truth.

They used to exist as two hand-maintained copies -- ``backend/config/agent_prompts.py``
and ``frontend/src/i18n/prompts.ts`` -- and drifted apart in both directions. Which
copy a session loaded decided which guardrails it had: a UI-created session ran the
literature agent without the "never fabricate citations" rule, and a headless session
ran pathology and biochemistry without the honest-reporting blocks that let them
decline to answer. Nothing detected the divergence because both copies were valid.

These tests fail if a copy is reintroduced.
"""
import json
import re
from pathlib import Path

import pytest

from backend.config.agent_prompts import (
    AGENT_PROMPT_KEYS,
    AgentPrompts,
    PROMPTS_JSON_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_PROMPTS_TS = REPO_ROOT / "frontend" / "src" / "i18n" / "prompts.ts"


def test_shared_prompt_file_exists_and_covers_every_agent():
    assert PROMPTS_JSON_PATH.is_file(), (
        f"{PROMPTS_JSON_PATH} is the single source of truth and must ship with the backend"
    )
    data = json.loads(PROMPTS_JSON_PATH.read_text(encoding="utf-8"))
    assert set(data) == set(AGENT_PROMPT_KEYS)
    for key, text in data.items():
        assert text.strip(), f"empty default prompt for {key}"


@pytest.mark.parametrize("agent", AGENT_PROMPT_KEYS)
def test_backend_defaults_come_from_the_shared_file(agent):
    shared = json.loads(PROMPTS_JSON_PATH.read_text(encoding="utf-8"))
    assert getattr(AgentPrompts(), f"{agent}_prompt") == shared[agent]


def test_frontend_reads_the_shared_file_instead_of_inlining_prompts():
    src = FRONTEND_PROMPTS_TS.read_text(encoding="utf-8")
    assert "agent-prompts.json" in src, (
        "frontend/src/i18n/prompts.ts must import the shared JSON, not restate the prompts"
    )
    # A reintroduced copy shows up as a long template literal. The file legitimately
    # contains only imports, a re-export, and comments.
    literals = re.findall(r"`([^`]*)`", src)
    assert not [lit for lit in literals if len(lit) > 200], (
        "prompts.ts contains an inlined prompt again; edit agent-prompts.json instead"
    )


def test_guardrails_lost_to_the_old_split_are_present_in_the_defaults():
    """Each of these existed in exactly one of the two former copies.

    Scope note: this asserts the DEFAULTS only. A session that supplies a
    custom_prompt REPLACES the whole system prompt
    (``AgentSettingsProvider.get_system_prompt`` returns the override verbatim),
    so none of these rules survive a user override. They are defaults, not a
    platform policy floor; do not read this test as "every session has them".
    """
    p = AgentPrompts()
    # Only the frontend copy had these.
    assert "DO NOT fabricate or guess conclusions" in p.pathology_prompt
    assert "DO NOT fabricate or guess conclusions" in p.biochemistry_prompt
    # Only the backend copy had these.
    assert "never fabricate citations, PMIDs, or DOIs" in p.literature_prompt
    assert "Do not invent findings" in p.painter_prompt
    assert "NEVER hardcode a `_Human` library" in p.bioinformatics_prompt
