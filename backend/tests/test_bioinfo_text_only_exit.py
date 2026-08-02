"""Tests for the bioinformatics text-only exit fixes.

Covers two helpers added to BioinformaticsAgent:
  - _compute_tool_choice: forces tool use on the last attempt before the
    MAX_TEXT_RESPONSES budget would be exhausted.
  - _raise_text_only_exhausted: raises UpstreamExhaustedError so the
    workflow @with_retry decorator can route to the LLM-failure HITL gate
    instead of silently fabricating a "[Fallback extraction]" success.
"""
import pytest

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AgentType
from backend.tests.split_facade import setattr_facade
from backend.utils.llm_client import UpstreamExhaustedError


class TestComputeToolChoice:
    def test_returns_auto_when_no_text_only_responses_yet(self):
        assert BioinformaticsAgent._compute_tool_choice(0, max_text_responses=2) == "auto"

    def test_returns_required_on_last_chance(self):
        # text_response_count == max - 1 means the next call is the LAST
        # chance before MAX_TEXT_RESPONSES would be reached. Force a tool.
        assert BioinformaticsAgent._compute_tool_choice(1, max_text_responses=2) == "required"

    def test_returns_required_when_already_at_or_past_limit(self):
        # Defensive: if the loop ever calls this at or past the limit,
        # still force a tool (never go back to "auto").
        assert BioinformaticsAgent._compute_tool_choice(2, max_text_responses=2) == "required"
        assert BioinformaticsAgent._compute_tool_choice(3, max_text_responses=2) == "required"

    def test_respects_custom_max(self):
        assert BioinformaticsAgent._compute_tool_choice(2, max_text_responses=4) == "auto"
        assert BioinformaticsAgent._compute_tool_choice(3, max_text_responses=4) == "required"


class TestRaiseTextOnlyExhausted:
    def _make_agent_shell(self):
        """Construct a bare object exposing the attributes the helper reads.

        The real BioinformaticsAgent.__init__ requires a settings_provider,
        sandbox, etc. — far more than the helper needs. We attach the
        unbound method to a SimpleNamespace and call it via __get__ to
        get exception-raising behavior without standing up the full stack.
        """
        from types import SimpleNamespace

        class _FakeClient:
            base_url = "https://fake.example.com/v1"

        shell = SimpleNamespace()
        shell.llm_client = _FakeClient()
        # `self.model` is a property on the real class; on the shell it's
        # just a plain attribute.
        shell.model = "google/gemini-3.1-pro-preview"
        return shell

    def test_raises_upstream_exhausted_with_context(self):
        shell = self._make_agent_shell()
        bound = BioinformaticsAgent._raise_text_only_exhausted.__get__(shell, type(shell))

        with pytest.raises(UpstreamExhaustedError) as excinfo:
            bound(text_response_count=2)

        err = excinfo.value
        assert err.agent_name == "bioinformatics"
        assert err.model == "google/gemini-3.1-pro-preview"
        assert err.llm_server == "https://fake.example.com/v1"
        assert "text-only" in str(err).lower()
        assert "2" in str(err)  # the count appears in the message

    def test_handles_llm_client_without_base_url(self):
        # Some test doubles may not set base_url; helper must not crash.
        from types import SimpleNamespace
        shell = SimpleNamespace(llm_client=SimpleNamespace(), model="m")
        bound = BioinformaticsAgent._raise_text_only_exhausted.__get__(shell, type(shell))
        with pytest.raises(UpstreamExhaustedError) as excinfo:
            bound(text_response_count=2)
        assert excinfo.value.llm_server == ""


class TestAutonomousLoopTextOnlyEscalation:
    """Integration: drive run_autonomous_analysis with a mocked LLM that
    only ever returns text — assert it raises UpstreamExhaustedError
    rather than silently producing a "[Fallback extraction]" success."""

    @pytest.mark.asyncio
    async def test_two_consecutive_text_only_responses_raise(self, monkeypatch, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from pathlib import Path
        from backend.utils.llm_client import LLMToolResponse
        from backend.models.schemas import AnalysisState

        # --- Bypass __init__ ---
        agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
        agent.session_id = "test-session"
        agent.code_history = []
        agent.analysis_context = {}
        agent.last_result = None
        agent._recent_sup_feedback = set()
        agent._pause_event = None
        agent.literature_callback = None
        agent.pi_callback = None
        agent.annotate_clusters_callback = None
        agent.message_callback = AsyncMock()
        agent.figure_review_callback = None
        agent.supervisor_callback = None
        agent.technique_agent = None
        agent._provider = MagicMock()
        agent._provider.get_config_revision = MagicMock(return_value=0)
        agent._last_config_revision = 0
        agent.agent_type = AgentType.BIOINFORMATICS

        # --- Stub read-only properties from BaseAgent ---
        monkeypatch.setattr(
            BioinformaticsAgent, "model",
            property(lambda self: "google/gemini-3.1-pro-preview"),
        )
        monkeypatch.setattr(
            BioinformaticsAgent, "system_prompt",
            property(lambda self: "You are a bioinformatics agent."),
        )
        monkeypatch.setattr(
            BioinformaticsAgent, "config",
            property(lambda self: MagicMock(temperature=0.7)),
        )
        monkeypatch.setattr(
            BioinformaticsAgent, "_static_system_prompt",
            property(lambda self: "Static bioinfo system prompt."),
        )
        monkeypatch.setattr(
            BioinformaticsAgent, "_build_dynamic_context",
            lambda self, state: "Dynamic context for test.",
        )
        monkeypatch.setattr(BioinformaticsAgent, "_set_llm_context", lambda self: None)
        monkeypatch.setattr(BioinformaticsAgent, "_notify", AsyncMock())

        # --- LLM client: returns text-only every call ---
        text_only = LLMToolResponse(
            content="I will now run marker gene analysis...",
            tool_calls=[],
            model="google/gemini-3.1-pro-preview",
            usage={},
            raw_response=None,
            raw_assistant_message=None,
        )
        fake_client = MagicMock()
        fake_client.base_url = "https://fake.example.com/v1"
        fake_client.chat_with_tools = AsyncMock(return_value=text_only)
        monkeypatch.setattr(
            BioinformaticsAgent, "llm_client",
            property(lambda self: fake_client),
        )

        # --- Sandbox stub: already started (km truthy), with state_dir ---
        state_dir = tmp_path / "states"
        state_dir.mkdir()
        agent.sandbox = MagicMock()
        agent.sandbox.state_dir = state_dir
        agent.sandbox.working_dir = tmp_path
        agent.sandbox.km = MagicMock()
        agent.sandbox.start = AsyncMock()
        agent.sandbox.save_session = AsyncMock()
        agent.sandbox.get_code_history = MagicMock(return_value=[])
        # The runtime persistence hook is synchronous.  An unstubbed
        # MagicMock returns another MagicMock, which is not a state mapping and
        # makes this text-only control-flow test fail before reaching the LLM
        # exhaustion path it owns.
        agent.sandbox.sync_runtime_persistence_state.return_value = {}

        # --- CheckpointManager: no-op load/save/clear ---
        class _NullCheckpoint:
            def __init__(self, *a, **kw): pass
            async def load(self): return None
            async def save(self, data): return None
            def clear(self): pass
            def should_checkpoint(self, step): return False

        setattr_facade(
            monkeypatch, "backend.agents.bioinformatics_agent", "CheckpointManager", _NullCheckpoint
        )

        state = AnalysisState(id="test-session", user_id="test-user")

        with pytest.raises(UpstreamExhaustedError) as excinfo:
            await agent.run_autonomous_analysis(
                state,
                initial_instruction="Run stage 1 analysis.",
                max_steps=10,
                resume_from_checkpoint=False,
            )

        assert excinfo.value.agent_name == "bioinformatics"
        assert excinfo.value.model == "google/gemini-3.1-pro-preview"
        assert excinfo.value.llm_server == "https://fake.example.com/v1"

        assert fake_client.chat_with_tools.call_count == 2

        first_call_kwargs = fake_client.chat_with_tools.call_args_list[0].kwargs
        second_call_kwargs = fake_client.chat_with_tools.call_args_list[1].kwargs
        assert first_call_kwargs.get("tool_choice") == "auto"
        assert second_call_kwargs.get("tool_choice") == "required"
