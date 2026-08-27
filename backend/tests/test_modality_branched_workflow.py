"""P1-4: modality-branched workflow + adaptive bulk gates."""
import re

import backend.workflow.agents as agents_mod
from backend.tests.split_facade import split_source


def test_cross_validation_gate_requires_both_modalities():
    """P1-4: pseudobulk cross-validation / deconvolution MANDATORY only when both scRNA AND bulk present."""
    src = split_source("backend.workflow.agents")
    fn = re.search(r"(if has_bulk.*?)(?=else:|\n        else:|\Z)", src, re.DOTALL)
    assert fn
    assert ("has_bulk and has_scrna" in src) or ("has_scrna and has_bulk" in src), \
        "P1-4: cross-validation/deconvolution must require BOTH modalities"
