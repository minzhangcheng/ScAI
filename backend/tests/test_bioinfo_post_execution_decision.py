"""Unit tests for the post-execution stage decision used by
_bioinformatics_analysis. A mid-execution plan-revision pause must NOT be
treated as a stage completion."""
import pytest

from backend.workflow.helpers import decide_bioinfo_post_execution


@pytest.mark.parametrize("stage", ["stage1", "stage2", "full"])
def test_revision_pause_never_advances_stage(stage):
    """Revision requested → 'revision' regardless of stage."""
    assert decide_bioinfo_post_execution(stage, plan_revision_requested=True) == "revision"


def test_stage1_genuine_completion_advances():
    """stage1 finished without revision → advance to stage2."""
    assert decide_bioinfo_post_execution("stage1", plan_revision_requested=False) == "stage1_complete"


def test_stage2_genuine_completion():
    """stage2 finished without revision → normal completion."""
    assert decide_bioinfo_post_execution("stage2", plan_revision_requested=False) == "complete"


def test_full_genuine_completion():
    """full mode finished without revision → normal completion."""
    assert decide_bioinfo_post_execution("full", plan_revision_requested=False) == "complete"