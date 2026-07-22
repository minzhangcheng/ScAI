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


def test_established_mechanism_name_is_accepted_when_evidence_is_supplied():
    progs, err = validate_mechanism_program_payload({
        "programs": [_valid_program(
            label="ferroptosis",
            supporting_genes=["GPX4", "ACSL4"],
            supporting_pathways=["lipid peroxidation"],
        )]
    })
    assert err is None
    assert progs[0].label == "ferroptosis"
    assert progs[0].supporting_genes == ["GPX4", "ACSL4"]


def test_does_not_invent_alternatives_and_adds_generic_missing_evidence():
    progs, err = validate_mechanism_program_payload({"programs": [_valid_program()]})
    assert err is None
    assert progs[0].alternative_explanations == []
    assert progs[0].missing_evidence == [
        "Direct causal validation is not available from the current data."
    ]


def test_missing_required_field_returns_error():
    bad = _valid_program()
    del bad["interpretation"]
    progs, err = validate_mechanism_program_payload({"programs": [bad]})
    assert progs == []
    assert err and "interpretation" in err


def test_empty_observations_returns_error():
    progs, err = validate_mechanism_program_payload(
        {"programs": [_valid_program(triggering_observations=[])]}
    )
    assert progs == []
    assert err


def test_missing_execution_evidence_returns_error():
    progs, err = validate_mechanism_program_payload(
        {"programs": [_valid_program(evidence_execution_ids=[])]}
    )
    assert progs == []
    assert err and "evidence_execution_id" in err


def test_empty_programs_requires_reason():
    progs, err = validate_mechanism_program_payload({"programs": []})
    assert progs == []
    assert err and "no_program_reason" in err

    progs2, err2 = validate_mechanism_program_payload(
        {"programs": [], "no_program_reason": "No cross-validated pattern reached moderate support."}
    )
    assert err2 is None
    assert progs2 == []


def test_truncates_program_count_to_8():
    progs, err = validate_mechanism_program_payload(
        {"programs": [_valid_program(label=f"p{i}") for i in range(12)]}
    )
    assert err is None
    assert len(progs) == 8


def test_truncates_genes_to_50():
    progs, err = validate_mechanism_program_payload(
        {"programs": [_valid_program(supporting_genes=[f"G{i}" for i in range(80)])]}
    )
    assert err is None
    assert len(progs[0].supporting_genes) == 50


def test_overclaiming_interpretation_gets_warning():
    progs, err = validate_mechanism_program_payload(
        {"programs": [_valid_program(interpretation="This confirmed ferroptosis is causal.")]}
    )
    assert err is None
    joined = " ".join(progs[0].missing_evidence).lower()
    assert "overclaim" in joined or "not confirmed" in joined
