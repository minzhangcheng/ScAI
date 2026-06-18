"""Confirm modality-branched workflow dead code has been removed (avoid dual source of truth with embedded scRNA pipeline)."""
import backend.config.agent_prompts as ap


def test_modality_helpers_removed():
    assert not hasattr(ap, "workflow_for_modality")
    assert not hasattr(ap, "_BULK_WORKFLOW")
    assert not hasattr(ap, "_MULTIMODAL_WORKFLOW")
