"""P0-9 regression: pathology synthesis prompt must use tissue-agnostic grading."""
import re

import backend.agents.pathology_agent as mod


def _synthesize_findings_body() -> str:
    src = open(mod.__file__).read()
    m = re.search(
        r"async def _synthesize_findings\b.*?(?=\n    (?:async )?def |\nclass )",
        src,
        re.DOTALL,
    )
    assert m, "Could not locate _synthesize_findings in pathology_agent.py"
    return m.group(0)


def test_synthesis_prompt_does_not_hardcode_tissue_or_disease_grading():
    """The synthesis prompt must derive grading dimensions from the supplied images."""
    body = _synthesize_findings_body()
    forbidden = [
        '"inflammation": 0-4',
        '"fibrosis": 0-4',
        '"steatosis": 0-4',
        '"injury": 0-4',
        "liver/NASH",
        "brain/AD",
        "tumor grade",
        "kidney/AKI",
        "lung/IPF",
    ]
    for token in forbidden:
        assert token not in body, f"P0-9: prompt still hardcodes liver axes: {token}"


def test_synthesis_prompt_uses_generic_grading_schema():
    body = _synthesize_findings_body()
    assert '"grading"' in body, "P0-9: prompt must use generic 'grading' schema"
    # Generic schema must instruct LLM to specify score / max / scale_description
    assert "scale_description" in body, (
        "P0-9: generic grading must include scale_description guidance"
    )
    assert '"score"' in body, "P0-9: generic grading must include score field"
    assert '"max"' in body, "P0-9: generic grading must include max field"
