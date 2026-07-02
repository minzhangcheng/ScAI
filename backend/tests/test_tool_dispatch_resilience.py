"""Test that one bad tool handler result does not abort the dispatch loop.

The tool-call dispatch in run_autonomous_analysis iterates over all tool_calls
in a response. If one handler raises an unexpected exception (e.g. ValueError
from a callback), the whole loop should log the error and continue to the next
tool_call rather than propagating the exception and crashing.
"""
import pytest
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AgentType, AnalysisState
from backend.utils.llm_client import UpstreamExhaustedError


class TestToolDispatchResilience:

    @pytest.mark.asyncio
    async def test_one_bad_tool_result_does_not_abort_loop(self, monkeypatch, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from pathlib import Path
        from backend.utils.llm_client import LLMToolResponse, ToolCall
        from backend.agents import bioinformatics_agent as bio_mod

        # --- Minimal agent (bypass __init__) ---
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
        agent.message_callback = None
        agent.figure_review_callback = None
        agent.supervisor_callback = None
        agent.species_validation_callback = None
        agent.marker_db_intervention_callback = None
        agent._provider = MagicMock()
        agent._provider.get_config_revision = MagicMock(return_value=0)
        agent._last_config_revision = 0
        agent.agent_type = AgentType.BIOINFORMATICS

        # --- Stub read-only properties ---
        monkeypatch.setattr(
            BioinformaticsAgent, "model",
            property(lambda self: "test-model"),
        )
        monkeypatch.setattr(
            BioinformaticsAgent, "system_prompt",
            property(lambda self: "Test bioinfo agent."),
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

        # --- LLM responses ---
        # Response 0: Two tool calls — consult_pi (will raise) + register_score (should succeed)
        # Response 1: Text-only to let the loop exit via max_steps_reached
        tc1 = ToolCall(
            id="call_1",
            name="consult_pi",
            arguments={"question": "test question", "context": "", "options": []},
        )
        tc2 = ToolCall(
            id="call_2",
            name="register_score",
            arguments={
                "score_name": "test_score",
                "method": "test",
                "description": "test",
                "parameters": {},
                "gene_list": [],
                "gene_list_source": "",
            },
        )
        response_0 = LLMToolResponse(
            content="",
            tool_calls=[tc1, tc2],
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw_response=None,
            raw_assistant_message=None,
        )
        response_1 = LLMToolResponse(
            content="Analysis complete.",
            tool_calls=[],
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw_response=None,
            raw_assistant_message=None,
        )

        # --- LLM client mock ---
        fake_client = MagicMock()
        fake_client.base_url = "https://fake.example.com/v1"
        fake_client.chat_with_tools = AsyncMock(side_effect=[response_0, response_1])
        monkeypatch.setattr(
            BioinformaticsAgent, "llm_client",
            property(lambda self: fake_client),
        )

        # --- Sandbox stub ---
        agent.sandbox = MagicMock()
        agent.sandbox.state_dir = tmp_path / "states"
        agent.sandbox.working_dir = tmp_path
        agent.sandbox.km = MagicMock()
        agent.sandbox.start = AsyncMock()
        agent.sandbox.save_session = AsyncMock()
        agent.sandbox.get_code_history = MagicMock(return_value=[])

        # --- Null CheckpointManager (no-op load/save/clear) ---
        class _NullCheckpoint:
            def __init__(self, *a, **kw): pass
            async def load(self): return None
            async def save(self, data): return None
            def clear(self): pass
            def should_checkpoint(self, step): return False

        monkeypatch.setattr(bio_mod, "CheckpointManager", _NullCheckpoint)

        # --- Make consult_pi callback raise ---
        async def _raising_callback(question, context):
            raise ValueError("PI agent crashed unexpectedly")

        agent.pi_callback = _raising_callback

        # --- State ---
        state = AnalysisState(id="test-session", user_id="test-user")

        # --- This call should NOT raise. Before the fix, ValueError("PI agent crashed")
        #    propagates out of the tool-dispatch loop and aborts the entire method.
        #    After the fix, the exception is caught, logged, and the loop continues
        #    to register_score. ---
        result = await agent.run_autonomous_analysis(
            state,
            initial_instruction="Run test.",
            max_steps=2,
            resume_from_checkpoint=False,
        )

        # --- Assertions ---
        # The loop completed without crashing
        assert result["status"] in ("max_steps_reached", "completed", "completed_with_fallback"), (
            f"Expected max_steps_reached/completed/completed_with_fallback, got {result['status']}"
        )

        # register_score was processed (proves the loop continued past the failing tool)
        assert any(
            s.score_name == "test_score" for s in state.score_definitions
        ), "register_score was not processed — loop likely aborted after consult_pi failed"
