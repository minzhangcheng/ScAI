from backend.agents.bioinformatics_agent import validate_mechanism_program_payload


def _valid_program(**over):
    p = {
        "label": "iron-redox-lipid hepatocyte program",
        "triggering_observations": ["GPX4 down + ACSL4 up in hepatocytes"],
        "evidence_execution_ids": ["exec-1"],
        "evidence_strength": "moderate",
        "specificity": "medium",
        "interpretation": "Candidate redox/lipid imbalance; not confirmed.",
    }
    p.update(over)
    return p


def test_valid_payload_builds_programs():
    progs, err = validate_mechanism_program_payload({"programs": [_valid_program()]})
    assert err is None
    assert len(progs) == 1
    assert progs[0].label.startswith("iron-redox-lipid")


def test_does_not_invent_alternatives_and_adds_generic_missing_evidence():
    progs, err = validate_mechanism_program_payload({"programs": [_valid_program()]})
    assert err is None
    assert progs[0].alternative_explanations == []
    assert progs[0].missing_evidence == [
        "Direct causal validation is not available from the current data."
    ]


def test_overclaiming_interpretation_gets_warning():
    progs, err = validate_mechanism_program_payload(
        {"programs": [_valid_program(interpretation="This confirmed ferroptosis is causal.")]}
    )
    assert err is None
    joined = " ".join(progs[0].missing_evidence).lower()
    assert "overclaim" in joined or "not confirmed" in joined
