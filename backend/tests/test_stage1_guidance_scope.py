from backend.agents.bioinformatics_agent import (
    _STAGE1_GUIDANCE,
    _STAGE2_GUIDANCE,
    _stage_guidance_text,
)


class TestStage1GuidanceScope:
    """Stage 1 guidance must forbid loading/preprocessing data for Stage 2 analyses."""

    def test_stage1_forbids_downstream_analyses(self):
        text = _STAGE1_GUIDANCE.lower()
        assert "do not include downstream analysis" in text

    def test_stage1_forbids_loading_velocity_layers(self):
        """Stage 1 must explicitly disallow loading spliced/unspliced/velocyto layers."""
        text = _STAGE1_GUIDANCE.lower()
        assert "spliced" in text
        assert "unspliced" in text
        assert "velocity" in text

    def test_stage1_forbids_preparing_stage2_inputs(self):
        """Stage 1 must ban data prep whose only purpose is Stage 2."""
        text = _STAGE1_GUIDANCE.lower()
        assert "do not load" in text or "do not preprocess" in text
        assert "stage 2" in text

    def test_stage1_names_concrete_examples(self):
        """Concrete examples reduce model ambiguity — name the common offenders."""
        text = _STAGE1_GUIDANCE.lower()
        assert "loom" in text
        assert "trajectory" in text or "pseudotime" in text

    def test_stage1_keeps_in_scope_steps(self):
        text = _STAGE1_GUIDANCE.lower()
        for token in ["quality control", "doublet", "normalization", "clustering", "annotation"]:
            assert token in text, f"missing in-scope token: {token}"

    def test_stage_guidance_text_routes_correctly(self):
        assert _stage_guidance_text("stage1") == _STAGE1_GUIDANCE
        assert _stage_guidance_text("stage2") == _STAGE2_GUIDANCE
        assert _stage_guidance_text("full") == ""