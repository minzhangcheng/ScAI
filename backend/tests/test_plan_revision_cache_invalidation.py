"""P0-8: Verify bioinformatics cache is invalidated when plan_revision_needed.

When `_bioinformatics_analysis` triggers a mid-execution plan revision, the
result that was just stored in the cache is the *revision-trigger* result.
If we don't invalidate, the post-revision re-entry into the node will hit the
stale cache and skip executing the revised plan.
"""
import re
import inspect
import backend.workflow.agents as agents_mod
from backend.workflow.agent_nodes import bioinformatics_node


def test_plan_revision_invalidates_bioinformatics_cache():
    body = inspect.getsource(bioinformatics_node.finalize_bioinformatics_analysis)

    set_idx = body.find('"plan_revision_needed"] = True')
    if set_idx == -1:
        set_idx = body.find("plan_revision_needed'] = True")
    assert set_idx != -1, "could not find plan_revision_needed assignment"

    neighborhood = body[max(0, set_idx - 100):set_idx + 600]
    assert "cache_invalidate" in neighborhood, (
        "P0-8: missing cache_invalidate near plan_revision_needed block"
    )
