"""C3 — STOP_KEYWORDS is a single source of truth, both PI paths see same set."""
import pytest

from backend.tests.split_facade import split_source


def test_stop_keywords_exposed_in_constants():
    from backend.constants import STOP_KEYWORDS
    # Must be a frozenset (or set) and contain both English + Chinese forms.
    assert isinstance(STOP_KEYWORDS, (set, frozenset))
    expected = {"stop", "terminate", "end", "finish", "conclude",
                "summarize", "report",
                "停止", "终止", "结束", "总结", "报告"}
    assert expected <= set(STOP_KEYWORDS)


def test_pi_agent_uses_shared_stop_keywords():
    """Inspect pi_agent module: there must be no per-function literal stop-keyword list."""
    import re

    text = split_source("backend.agents.pi_agent")
    # No literal lists like ['stop', 'terminate', ...] should remain.
    # Match a list containing 'stop' immediately followed by 'terminate'.
    pattern = re.compile(r"\[\s*['\"]stop['\"]\s*,\s*['\"]terminate['\"]", re.UNICODE)
    assert not pattern.search(text), \
        "pi_agent still defines a local stop-keyword list — must use constants.STOP_KEYWORDS"


@pytest.mark.parametrize("keyword", ["summarize", "report", "总结", "报告"])
def test_both_paths_recognize_overlap_keywords(keyword):
    """Both keyword sets, after consolidation, must recognize these."""
    from backend.constants import STOP_KEYWORDS
    assert keyword in STOP_KEYWORDS