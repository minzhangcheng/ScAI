"""P1-7: Workflow callbacks must not silently construct bare AnalysisState().

`AnalysisState` requires `user_id` (Field(..., min_length=1)), so bare
`AnalysisState()` raises ValidationError at runtime. The fallback should
instead raise an explicit RuntimeError signalling the misuse.
"""
import re

import backend.workflow.callbacks as cb_mod
import backend.workflow.annotation as anno_mod
from backend.tests.split_facade import source_for_path


def test_no_bare_analysis_state_construction_in_workflow():
    """Workflow callbacks must not contain bare `AnalysisState()` calls."""
    for mod in (cb_mod, anno_mod):
        src = source_for_path(mod.__name__.replace(".", "/") + ".py")
        matches = re.findall(r"\bAnalysisState\(\s*\)", src)
        assert not matches, (
            f"P1-7: {mod.__name__} contains bare AnalysisState() construction"
        )


def test_workflow_callbacks_raise_runtime_error_when_state_missing():
    """Modules previously using AnalysisState() fallback must now raise RuntimeError."""
    for mod in (cb_mod, anno_mod):
        src = source_for_path(mod.__name__.replace(".", "/") + ".py")
        assert "RuntimeError" in src, (
            f"P1-7: {mod.__name__} must raise RuntimeError when "
            "_current_analysis_state is missing"
        )
