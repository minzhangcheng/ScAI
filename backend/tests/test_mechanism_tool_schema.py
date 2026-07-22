from backend.agents.bioinformatics_agent import TOOLS
from backend.utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY


def test_register_mechanism_programs_tool_present():
    names = [t["function"]["name"] for t in TOOLS]
    assert "register_mechanism_programs" in names


def test_tool_schema_has_programs_and_reason():
    tool = next(t for t in TOOLS if t["function"]["name"] == "register_mechanism_programs")
    props = tool["function"]["parameters"]["properties"]
    assert "programs" in props
    assert "no_program_reason" in props
    item_props = props["programs"]["items"]["properties"]
    for f in ("label", "triggering_observations", "evidence_execution_ids", "evidence_strength",
              "specificity", "interpretation", "bulk_support"):
        assert f in item_props


def test_tool_schema_has_no_hardcoded_mechanisms():
    tool = next(t for t in TOOLS if t["function"]["name"] == "register_mechanism_programs")
    blob = str(tool).lower()
    for banned in ("ferroptosis", "apoptosis", "cuproptosis", "nrf2"):
        assert banned not in blob


def test_tool_schema_requires_evidence_based_naming_and_hypothesis_evaluation():
    tool = next(t for t in TOOLS if t["function"]["name"] == "register_mechanism_programs")
    description = tool["function"]["description"]
    assert MECHANISM_EVIDENCE_POLICY in description
    assert "name it explicitly" in description
    assert "prespecified hypotheses" in description
    assert "insufficiently supported" in description
