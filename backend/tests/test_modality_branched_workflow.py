"""P1-4: modality-branched workflow + adaptive bulk gates."""
import re

import backend.config.agent_prompts as ap_mod
import backend.workflow.agents as agents_mod


def test_workflow_for_modality_helper_exists():
    """P1-4: helper to pick workflow by modality must exist."""
    assert hasattr(ap_mod, "workflow_for_modality"), \
        "P1-4: must expose workflow_for_modality(modality) -> str helper"
    sc = ap_mod.workflow_for_modality("single-cell RNA-seq")
    bulk = ap_mod.workflow_for_modality("bulk RNA-seq")
    multi = ap_mod.workflow_for_modality("single-cell and bulk RNA-seq")
    assert sc != bulk, "scRNA and bulk workflows must differ"
    # bulk-only should not require single-cell clustering
    assert "clustering" not in bulk.lower() or "scrna" not in bulk.lower(), \
        "bulk-only workflow should not require single-cell clustering"
    # multimodal should include bulk integration content
    assert "bulk" in multi.lower() and (
        "deconvolution" in multi.lower() or "pseudobulk" in multi.lower()
    ), "multimodal workflow must include bulk integration"


def test_cross_validation_gate_requires_both_modalities():
    """P1-4: pseudobulk cross-validation / deconvolution MANDATORY only when both scRNA AND bulk present."""
    src = open(agents_mod.__file__).read()
    fn = re.search(r"(if has_bulk.*?)(?=else:|\n        else:|\Z)", src, re.DOTALL)
    assert fn
    assert ("has_bulk and has_scrna" in src) or ("has_scrna and has_bulk" in src), \
        "P1-4: cross-validation/deconvolution must require BOTH modalities"
