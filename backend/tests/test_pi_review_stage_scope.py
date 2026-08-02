import inspect
from backend.agents.pi_agent import PIAgent


class TestPIReviewStageScope:
    """review_bioinfo_plan prompt must contain stage-scope review criteria."""

    def _prompt_template(self) -> str:
        return inspect.getsource(PIAgent.review_bioinfo_plan)

    def test_prompt_passes_plan_stage(self):
        src = self._prompt_template()
        assert "plan.stage" in src, "PI review prompt must surface plan.stage"

    def test_prompt_lists_stage_scope_criterion(self):
        src = self._prompt_template().lower()
        assert "stage scope" in src or "scope" in src
        assert "stage 1" in src
        assert "stage 2" in src

    def test_prompt_names_concrete_stage1_exclusions(self):
        src = self._prompt_template().lower()
        for token in ["spliced", "velocity", "trajectory"]:
            assert token in src, f"PI prompt missing scope token: {token}"

    def test_prompt_directs_revise_on_scope_leak(self):
        src = self._prompt_template().lower()
        assert "revise" in src
        assert "downstream" in src or "stage 2" in src

    def test_prompt_reviews_visual_coverage_and_explicit_exceptions(self):
        src = self._prompt_template().lower()
        assert "visual coverage" in src
        assert "figure_requirements" in src
        assert "allow_text_only=true" in src
        assert "text_only_reason" in src
        assert "doublet" in src and "before/after" in src
        assert "missing visual" in src and "revise" in src
