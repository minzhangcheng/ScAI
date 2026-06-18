"""_build_planning_context must include literature_findings (current or previous round)."""
from backend.workflow.context import ContextMixin


class _Round:
    def __init__(self):
        self.biochemistry_findings = []
        self.pathology_findings = []
        self.bioinformatics_findings = []
        self.literature_findings = ["TGF-beta drives fibrosis (PMID 123)"]


class _State:
    def __init__(self):
        self.round_results = [_Round()]


def test_literature_findings_included_from_previous_round():
    mixin = object.__new__(ContextMixin)
    state = {"analysis_state": _State(), "planning_context": None}
    ctx = mixin._build_planning_context(state)
    assert ctx is not None
    assert ctx.get("literature_findings") == ["TGF-beta drives fibrosis (PMID 123)"]
