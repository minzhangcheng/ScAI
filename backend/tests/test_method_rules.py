"""The methodological rule corpus must stay loadable, attributed and neutral.

The corpus is distilled from bioSkills (MIT) and translated for this stack; see
THIRD_PARTY_NOTICES.md. These tests pin the parts that would silently rot: the
attribution, the advisory/guidance split that other code keys off, and the
domain-neutrality that ScAI's general-purpose remit requires.
"""


from backend.services.method_rules import (
    ENFORCEMENT_ADVISORY,
    ENFORCEMENT_GUIDANCE,
    ENFORCEMENT_VALUES,
    all_rules,
)

_REQUIRED_FIELDS = ("id", "title", "enforcement", "applies_when", "rule",
                    "why", "python", "citations", "source_skill")


def test_corpus_has_only_advisories_and_guidance():
    """Scientific rules may inform the PI but may not refuse a call."""
    kinds = {r["enforcement"] for r in all_rules()}
    assert ENFORCEMENT_ADVISORY in kinds
    assert ENFORCEMENT_GUIDANCE in kinds
    assert kinds == set(ENFORCEMENT_VALUES)
