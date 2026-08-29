"""C3 — STOP_KEYWORDS is a single source of truth, both PI paths see same set."""
import pytest

from backend.agents.pi.interaction import _contains_stop_command
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


@pytest.mark.parametrize(
    "text",
    [
        "Continue the endothelial analysis.",
        "Compare reporter-positive and reporter-negative cells.",
        "Use the endpoint measurements.",
    ],
)
def test_stop_words_do_not_match_inside_domain_terms(text):
    assert _contains_stop_command(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "Do not stop the current analysis.",
        "Don't finish yet.",
        "No report now.",
        "不要停止当前分析。",
        "请勿结束。",
        "别报告结果。",
    ],
)
def test_clause_local_negation_suppresses_stop_commands(text):
    assert _contains_stop_command(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "Stop now.",
        "Please finish the analysis.",
        "停止并生成报告。",
        "Do not stop the current step, but finish the analysis now.",
        "不要停止当前步骤，但是现在结束分析。",
    ],
)
def test_explicit_or_later_affirmative_stop_command_wins(text):
    assert _contains_stop_command(text) is True
