"""Focused A2 tests for actionable, session-scoped failure guidance."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.supervisor_agent import SupervisorAgent
from backend.models.schemas import AnalysisStep


def _api_change_notes_module():
    try:
        return importlib.import_module("backend.services.api_change_notes")
    except ModuleNotFoundError as exc:  # Product RED, not test collection noise.
        pytest.fail(f"A2 API-note service is missing: {exc}")


def test_api_change_signature_reuses_one_domain_neutral_note():
    notes = _api_change_notes_module()
    first_error = "AttributeError: module 'acme.plot' has no attribute 'draw'"
    same_error_with_traceback = (
        "Cell In[91], line 4\n"
        "AttributeError: module 'acme.plot' has no attribute 'draw'\n"
    )

    signature = notes.api_change_signature(first_error)
    assert signature == "acme.plot.draw (missing attribute)"
    assert notes.api_change_signature(same_error_with_traceback) == signature

    state = SimpleNamespace()
    notes.record_api_note(state, signature, "Use the installed drawing entry point.")
    assert notes.recorded_api_note(state, signature) == (
        "Use the installed drawing entry point."
    )
    rendered = notes.format_api_note(signature, notes.recorded_api_note(state, signature))
    assert "acme.plot.draw" in rendered
    assert "Use the installed drawing entry point." in rendered


class _RecordingClient:
    def __init__(self):
        self.calls = []
        self.set_context = MagicMock()

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=(
                "The name may belong to Python or R; confirm the ecosystem before "
                "choosing an equivalent."
            )
        )


def _missing_package_supervisor(client):
    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    provider = MagicMock()
    provider.get_llm_client.return_value = client
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.1
    supervisor._provider = provider
    supervisor._last_config_revision = provider.get_config_revision.return_value
    supervisor.session_id = "missing-package-session"
    supervisor._vision_disabled_runtime = False
    supervisor._recent_errors = []
    supervisor._recent_rule_feedback_hashes = set()
    supervisor._pi_question_hashes = set()
    supervisor._diagnosed_packages = set()
    supervisor._step_count = 0
    supervisor.supervision_log = []
    supervisor.all_rounds_log = []
    supervisor.pi_callback = None
    supervisor.message_callback = None
    supervisor.technique_agent = None
    supervisor._sandbox = None
    supervisor._notify = AsyncMock()
    supervisor._llm_evaluate = AsyncMock(return_value=None)
    return supervisor


@pytest.mark.asyncio
async def test_missing_package_guidance_is_ecosystem_neutral_and_deduplicated():
    client = _RecordingClient()
    supervisor = _missing_package_supervisor(client)
    error = "ModuleNotFoundError: No module named 'sklearn.cluster'"

    first = await supervisor.evaluate_step(
        AnalysisStep(step_number=1, step_type="execute_code", code="import sklearn.cluster", code_error=error),
        [],
    )
    second = await supervisor.evaluate_step(
        AnalysisStep(step_number=2, step_type="execute_code", code="import sklearn.cluster", code_error=error),
        [],
    )

    assert len(client.calls) == 1
    prompt = client.calls[0]["messages"][0]["content"]
    assert "sklearn" in prompt
    assert "ecosystem" in prompt.lower()
    assert "[Package Support]" in (first.feedback or "")
    assert second.feedback is None or "[Package Support]" not in second.feedback


@pytest.mark.asyncio
async def test_figure_failure_wiring_records_and_reuses_api_note():
    from backend.agents.bioinformatics.figures import FiguresMixin

    class _Technique:
        def __init__(self):
            self.calls = 0

        async def lookup_from_error(self, sandbox, error_text, failed_code=""):
            self.calls += 1
            return "Use the installed drawing entry point."

    class _Host:
        sandbox = object()

        def __init__(self, technique):
            self.technique_agent = technique

    technique = _Technique()
    host = _Host(technique)
    state = SimpleNamespace()
    error = "AttributeError: module 'acme.plot' has no attribute 'draw'"

    first = await FiguresMixin._environment_api_hint(host, state, error, "plot()")
    second = await FiguresMixin._environment_api_hint(host, state, error, "plot()")

    assert "installed drawing entry point" in first
    assert "installed drawing entry point" in second
    assert technique.calls == 1
