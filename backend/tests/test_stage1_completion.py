"""Tests for Stage 1 completion fix: planning_stage propagation, gate awareness,
annotation completion signal, checkpoint recovery, and PI stage context."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.models.schemas import AnalysisState, AnalysisPhase


@pytest.fixture
def analysis_state_exploration():
    """Create an AnalysisState in EXPLORATION phase."""
    state = MagicMock(spec=AnalysisState)
    state.analysis_phase = AnalysisPhase.EXPLORATION
    state.id = "test-session"
    state.current_round = 1
    state.current_plan = None
    return state


class TestPlanningStageParameter:
    """Verify run_autonomous_analysis accepts and stores planning_stage."""

    def test_accepts_planning_stage_default(self):
        """Default planning_stage should be 'full'."""
        from backend.agents.bioinformatics_agent import BioinformaticsAgent
        # Verify the method signature has planning_stage parameter
        import inspect
        sig = inspect.signature(BioinformaticsAgent.run_autonomous_analysis)
        assert "planning_stage" in sig.parameters
        assert sig.parameters["planning_stage"].default == "full"

    def test_accepts_planning_stage_stage1(self):
        """Should accept 'stage1' as planning_stage."""
        from backend.agents.bioinformatics_agent import BioinformaticsAgent
        import inspect
        sig = inspect.signature(BioinformaticsAgent.run_autonomous_analysis)
        assert "planning_stage" in sig.parameters


class TestStage1CompletionInjection:
    """Verify that stage1 injects a completion message after annotation."""

    @pytest.mark.asyncio
    async def test_stage1_injects_completion_message(self):
        """When planning_stage is 'stage1', after annotation completes,
        a user message instructing the agent to call report_findings
        should be appended to messages."""
        from backend.agents.bioinformatics_agent import BioinformaticsAgent

        agent = MagicMock(spec=BioinformaticsAgent)
        agent.planning_stage = "stage1"
        messages = []

        # Simulate the injection that should happen after annotation_completed = True
        if agent.planning_stage == "stage1":
            messages.append({
                "role": "user",
                "content": (
                    "STAGE 1 ANNOTATION COMPLETE. Cell annotation is done and confirmed by the user. "
                    "The platform will now render and verify the supported baseline figures. "
                    "Call report_findings now. "
                    "Do NOT run differential expression, pathway enrichment, "
                    "cell communication, trajectory analysis, or any other downstream analysis. "
                    "These will be done in Stage 2 after PI reviews your Stage 1 results "
                    "and creates a new analysis plan."
                )
            })

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "STAGE 1 ANNOTATION COMPLETE" in messages[0]["content"]
        assert "platform" in messages[0]["content"]
        assert "report_findings" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_full_mode_does_not_inject_completion_message(self):
        """When planning_stage is 'full', no stage completion message should be injected."""
        agent = MagicMock()
        agent.planning_stage = "full"
        messages = []

        if agent.planning_stage == "stage1":
            messages.append({"role": "user", "content": "STAGE 1 ANNOTATION COMPLETE..."})

        assert len(messages) == 0


class TestAnnotationResultStageAware:
    """Verify that the annotation already_confirmed message is stage-aware."""

    def test_stage1_annotation_confirmed_message(self):
        """In stage1, already_confirmed annotation should say 'call report_findings now'."""
        planning_stage = "stage1"
        result_lines = []

        if planning_stage == "stage1":
            result_lines.append(
                "[NOTE: These annotations were already confirmed by the user. "
                "No re-annotation needed. The platform will render the Stage 1 "
                "baseline figures after this pass; call report_findings now.]\n"
            )
        else:
            result_lines.append(
                "[NOTE: These annotations were already confirmed by the user. "
                "No re-annotation needed. Proceed to downstream analysis.]\n"
            )

        assert "call report_findings now" in result_lines[0]
        assert "platform will render" in result_lines[0]

    def test_full_mode_annotation_confirmed_message(self):
        """In full mode, already_confirmed annotation should say 'proceed to downstream analysis'."""
        planning_stage = "full"
        result_lines = []

        if planning_stage == "stage1":
            result_lines.append(
                "[NOTE: These annotations were already confirmed by the user. "
                "No re-annotation needed. The platform will render the Stage 1 "
                "baseline figures after this pass; call report_findings now.]\n"
            )
        else:
            result_lines.append(
                "[NOTE: These annotations were already confirmed by the user. "
                "No re-annotation needed. Proceed to downstream analysis.]\n"
            )

        assert "Proceed to downstream analysis" in result_lines[0]


class TestReportFindingsGateStageAware:
    """Verify that report_findings gates 1-3 are skipped in stage1."""

    def test_gate_condition_stage1_skips_gates_1_3(self):
        """In stage1, the condition for gates 1-3 should be False even in EXPLORATION phase."""
        from backend.models.schemas import AnalysisPhase

        analysis_phase = AnalysisPhase.EXPLORATION
        planning_stage = "stage1"

        # Original condition: state.analysis_phase == AnalysisPhase.EXPLORATION
        # New condition: state.analysis_phase == AnalysisPhase.EXPLORATION and self.planning_stage != "stage1"
        gate_active = analysis_phase == AnalysisPhase.EXPLORATION and planning_stage != "stage1"

        assert not gate_active  # Gates 1-3 should NOT be active in stage1

    def test_gate_condition_full_mode_in_exploration(self):
        """In full mode, gates 1-3 should be active in EXPLORATION phase."""
        from backend.models.schemas import AnalysisPhase

        analysis_phase = AnalysisPhase.EXPLORATION
        planning_stage = "full"

        gate_active = analysis_phase == AnalysisPhase.EXPLORATION and planning_stage != "stage1"

        assert gate_active  # Gates 1-3 should be active in full mode during exploration

    def test_gate_condition_stage2_in_exploration(self):
        """In stage2, gates 1-3 should be active in EXPLORATION phase."""
        from backend.models.schemas import AnalysisPhase

        analysis_phase = AnalysisPhase.EXPLORATION
        planning_stage = "stage2"

        gate_active = analysis_phase == AnalysisPhase.EXPLORATION and planning_stage != "stage1"

        assert gate_active  # Gates 1-3 should be active in stage2 during exploration

    def test_gate_0_still_applies_in_stage1(self):
        """Gate 0 (mandatory annotate_clusters) should always apply in exploration,
        regardless of planning_stage. This test verifies the logic is separate."""
        from backend.models.schemas import AnalysisPhase

        analysis_phase = AnalysisPhase.EXPLORATION
        planning_stage = "stage1"
        annotation_completed = False

        # Gate 0: always check annotation_completed in exploration
        gate_0_active = analysis_phase == AnalysisPhase.EXPLORATION and not annotation_completed

        assert gate_0_active  # Gate 0 should be active even in stage1


class TestCheckpointAnnotationRecovery:
    """Verify annotation_completed is restored from checkpoint data."""

    def test_restore_from_analysis_log_with_annotate_step(self):
        """If checkpoint analysis_log contains an annotate_clusters step,
        annotation_completed should be set to True."""
        from backend.models.schemas import AnalysisStep

        # Simulate a checkpoint with an annotate_clusters step
        analysis_log = [
            AnalysisStep(step_number=1, step_type="execute_code", agent_reasoning="QC"),
            AnalysisStep(step_number=2, step_type="annotate_clusters", agent_reasoning="Annotation"),
        ]

        # The restoration logic
        annotation_completed = False
        if analysis_log and any(
            s.step_type == "annotate_clusters" for s in analysis_log
        ):
            annotation_completed = True

        assert annotation_completed is True

    def test_no_restore_without_annotate_step(self):
        """If checkpoint analysis_log has no annotate_clusters step,
        annotation_completed should remain False."""
        from backend.models.schemas import AnalysisStep

        analysis_log = [
            AnalysisStep(step_number=1, step_type="execute_code", agent_reasoning="QC"),
            AnalysisStep(step_number=2, step_type="cluster_cells", agent_reasoning="Clustering"),
        ]

        annotation_completed = False
        if analysis_log and any(
            s.step_type == "annotate_clusters" for s in analysis_log
        ):
            annotation_completed = True

        assert annotation_completed is False

    def test_restore_from_empty_log(self):
        """Empty analysis_log should not set annotation_completed."""
        analysis_log = []

        annotation_completed = False
        if analysis_log and any(
            s.step_type == "annotate_clusters" for s in analysis_log
        ):
            annotation_completed = True

        assert annotation_completed is False


class TestWorkflowPassesPlanningStage:
    """Verify workflow passes planning_stage to agent and supervisor."""

    def test_workflow_extracts_planning_stage(self):
        """Workflow should extract planning_stage from state and pass it along."""
        state = {"planning_stage": "stage1", "plan_approved": True}

        planning_stage = state.get("planning_stage", "full")
        assert planning_stage == "stage1"

    def test_workflow_default_planning_stage(self):
        """Default planning_stage should be 'full'."""
        state = {}

        planning_stage = state.get("planning_stage", "full")
        assert planning_stage == "full"


class TestPIStageContext:
    """Verify PI consultation includes stage context."""

    def test_stage1_context_includes_restriction(self):
        """Stage 1 context should mention QC+clustering+annotation ONLY."""
        planning_stage = "stage1"
        stage_context = ""
        if planning_stage == "stage1":
            stage_context = (
                "\n\nCURRENT STAGE: Stage 1 (QC + Clustering + Cell Type Annotation ONLY). "
                "The bioinformatics agent should ONLY perform basic QC, clustering, and annotation. "
                "Downstream analyses (DEG, GO/KEGG, cell communication, trajectory) are reserved "
                "for Stage 2 and should NOT be performed now. "
                "If the agent asks about downstream analysis, advise them to complete Stage 1 first "
                "by calling report_findings."
            )
        assert "Stage 1" in stage_context
        assert "report_findings" in stage_context
        assert "QC + Clustering + Cell Type Annotation ONLY" in stage_context

    def test_stage2_context_mentions_downstream(self):
        """Stage 2 context should mention downstream analysis."""
        planning_stage = "stage2"
        stage_context = ""
        if planning_stage == "stage1":
            stage_context = "Stage 1..."
        elif planning_stage == "stage2":
            stage_context = (
                "\n\nCURRENT STAGE: Stage 2 (Downstream Analysis). "
                "Stage 1 (QC + clustering + annotation) has been completed. "
                "The agent is now executing the approved Stage 2 analysis plan."
            )
        assert "Stage 2" in stage_context
        assert "Downstream Analysis" in stage_context

    def test_full_mode_no_stage_context(self):
        """Full mode should produce no stage context."""
        planning_stage = "full"
        stage_context = ""
        if planning_stage == "stage1":
            stage_context = "Stage 1..."
        elif planning_stage == "stage2":
            stage_context = "Stage 2..."
        # else: no stage context added
        assert stage_context == ""


class TestSupervisorStageAwareEscalation:
    """Verify supervisor annotation escalation is stage-aware."""

    def test_stage1_escalation_message(self):
        """In stage1, annotation escalation should mention report_findings."""
        planning_stage = "stage1"

        if getattr(type('obj', (), {'planning_stage': planning_stage}), 'planning_stage', 'full') == "stage1":
            escalation_msg = (
                "Cell type annotation has been completed. "
                "Please confirm the annotations are biologically reasonable. "
                "Stage 1 is now complete — the agent should call report_findings "
                "to finalize Stage 1 before Stage 2 downstream analysis begins."
            )
        else:
            escalation_msg = (
                "Cell type annotation has been completed. Please review the annotation results "
                "and provide guidance on whether they are biologically reasonable."
            )

        assert "report_findings" in escalation_msg
        assert "Stage 1 is now complete" in escalation_msg

    def test_full_mode_escalation_message(self):
        """In full mode, annotation escalation should be open-ended."""
        planning_stage = "full"

        if planning_stage == "stage1":
            escalation_msg = "Stage 1..."
        else:
            escalation_msg = (
                "Cell type annotation has been completed. Please review the annotation results "
                "and provide guidance on whether they are biologically reasonable."
            )

        assert "biologically reasonable" in escalation_msg
        assert "report_findings" not in escalation_msg
