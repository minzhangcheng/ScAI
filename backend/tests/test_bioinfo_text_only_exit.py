"""Executor budget exhaustion must not promote prose to completed evidence."""
import pytest

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AgentType
from backend.tests.split_facade import setattr_facade


class TestAutonomousLoopBudgetExhaustion:
    """A substantive step without report_findings must remain incomplete."""

    @pytest.mark.asyncio
    async def test_budget_exhaustion_does_not_promote_assistant_prose(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from backend.models.schemas import AnalysisState
        from backend.utils.llm_client import LLMToolResponse, ToolCall

        agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
        agent.session_id = "budget-session"
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

        monkeypatch.setattr(
            BioinformaticsAgent,
            "model",
            property(lambda self: "test/model"),
        )
        monkeypatch.setattr(
            BioinformaticsAgent,
            "system_prompt",
            property(lambda self: "You are a bioinformatics agent."),
        )
        monkeypatch.setattr(
            BioinformaticsAgent,
            "config",
            property(lambda self: SimpleNamespace(temperature=0.2)),
        )
        monkeypatch.setattr(
            BioinformaticsAgent,
            "_static_system_prompt",
            property(lambda self: "Static bioinformatics prompt."),
        )
        monkeypatch.setattr(
            BioinformaticsAgent,
            "_build_dynamic_context",
            lambda self, state: "Dynamic context for test.",
        )
        monkeypatch.setattr(BioinformaticsAgent, "_set_llm_context", lambda self: None)
        monkeypatch.setattr(BioinformaticsAgent, "_notify", AsyncMock())
        monkeypatch.setattr(
            BioinformaticsAgent,
            "_detect_and_lookup_swallowed_error",
            AsyncMock(return_value=(None, "")),
        )

        response = LLMToolResponse(
            content="I observed a promising pattern; I will run a check now.",
            tool_calls=[ToolCall(
                id="call-execute",
                name="execute_code",
                arguments={"code": "x = 1", "description": "Run a check"},
            )],
            model="test/model",
            usage={},
            raw_response=None,
            raw_assistant_message=None,
        )
        client = MagicMock(base_url="https://fake.example.com/v1")
        client.chat_with_tools = AsyncMock(return_value=response)
        monkeypatch.setattr(
            BioinformaticsAgent,
            "llm_client",
            property(lambda self: client),
        )

        agent.sandbox = MagicMock()
        agent.sandbox.state_dir = tmp_path / "states"
        agent.sandbox.state_dir.mkdir()
        agent.sandbox.working_dir = tmp_path
        agent.sandbox.km = MagicMock()
        agent.sandbox.save_session = AsyncMock()
        agent.sandbox.get_code_history = MagicMock(return_value=[])
        agent.sandbox.runtime_dependency_tracker = None
        agent.sandbox.sync_runtime_persistence_state.return_value = {}
        agent.sandbox.execute = AsyncMock(
            return_value=SimpleNamespace(output="terminal output", error=None, figures=[])
        )

        checkpoints = []
        cleared = []

        class _Checkpoint:
            def __init__(self, *args, **kwargs):
                pass

            async def load(self):
                return None

            async def save(self, data):
                checkpoints.append(data)

            def clear(self):
                cleared.append(True)

            def should_checkpoint(self, step):
                return False

        setattr_facade(
            monkeypatch,
            "backend.agents.bioinformatics_agent",
            "CheckpointManager",
            _Checkpoint,
        )

        result = await agent.run_autonomous_analysis(
            AnalysisState(id="budget-session", user_id="test-user"),
            initial_instruction="Run the planned check.",
            max_steps=1,
            resume_from_checkpoint=False,
        )

        assert result["status"] == "max_steps_reached"
        assert result["findings"] == []
        assert not any("Fallback extraction" in finding for finding in result["findings"])
        # Dispatch identity is checkpointed before and after the kernel call;
        # the terminal budget record must remain the final position.
        assert checkpoints
        assert checkpoints[-1]["step_name"] == "step_1_budget_exhausted"
        assert checkpoints[-1]["budget_exhausted"] is True
        assert cleared == []
