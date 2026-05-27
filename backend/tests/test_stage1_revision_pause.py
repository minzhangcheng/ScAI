"""Structural guards for the _bioinformatics_analysis post-execution block.

These pin the two bugs that caused stage1 to advance to stage2 on a mere
revision pause:
  A) the stage transition must be gated by the decision helper (not an
     unconditional `planning_stage == "stage1"` branch);
  B) the revision flag must be read from the raw agent `result`, NOT from the
     rebuilt `final_result` (which omits the key)."""
import re

import backend.workflow.agents as agents_mod


def _bioinfo_body() -> str:
    source = open(agents_mod.__file__).read()
    fn = re.search(
        r"async def _bioinformatics_analysis\b.*?(?=\n    (?:async )?def |\Z)",
        source, re.DOTALL,
    )
    assert fn, "could not locate _bioinformatics_analysis"
    return fn.group(0)


def test_revision_flag_read_from_raw_result():
    """Bug B: must read plan_revision_needed from `result`, not `final_result`."""
    body = _bioinfo_body()
    assert 'result.get("plan_revision_needed")' in body
    assert 'final_result.get("plan_revision_needed")' not in body, (
        "final_result omits plan_revision_needed; reading it there is always falsy"
    )


def test_uses_decision_helper():
    """Bug A: stage transition must be driven by the pure decision helper."""
    body = _bioinfo_body()
    assert "decide_bioinfo_post_execution" in body


def test_helper_is_imported():
    source = open(agents_mod.__file__).read()
    assert "decide_bioinfo_post_execution" in source.split("async def")[0], (
        "decide_bioinfo_post_execution must be imported at module top"
    )


def test_cache_invalidate_still_near_revision_assignment():
    """Regression guard mirroring test_plan_revision_cache_invalidation.py."""
    body = _bioinfo_body()
    idx = body.find('"plan_revision_needed"] = True')
    assert idx != -1
    assert "cache_invalidate" in body[max(0, idx - 100):idx + 600]