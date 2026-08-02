from backend.models.schemas import RoundResult
from backend.routes.serialization import serialize_round_results


def test_serialize_preserves_integration_fields():
    rounds = [
        RoundResult(round_number=3, phase="innovation", pi_summary="Dir B",
                    bioinformatics_findings=["b1"]),
        RoundResult(
            round_number=3, phase="integration", pi_summary="Cross-direction",
            integration_report_summary="Shared: copper deficiency.",
            integration_candidates=[{"direction": "New axis X", "score": 0.85}],
        ),
    ]
    out = serialize_round_results(rounds)
    assert len(out) == 2

    innovation, integration = out
    assert innovation["phase"] == "innovation"
    assert integration["phase"] == "integration"
    assert integration["integration_report_summary"] == "Shared: copper deficiency."
    assert integration["integration_candidates"] == [{"direction": "New axis X", "score": 0.85}]
    assert innovation["bioinformatics_findings"] == ["b1"]


def test_serialize_preserves_forced_evidence_gate_disclosure():
    finding = {
        "gate": "register-score-plan",
        "rule_id": "prespecify-the-family-before-results",
        "forced_release": True,
        "rejection_count": 4,
        "rejection_limit": 4,
    }
    [payload] = serialize_round_results([RoundResult(
        round_number=5,
        phase="innovation",
        evidence_gate_degraded=True,
        bioinformatics_evidence_verified=False,
        evidence_gate_findings=[finding],
        evidence_notes=["Score registration was force-released."],
    )])

    assert payload["evidence_gate_degraded"] is True
    assert payload["bioinformatics_evidence_verified"] is False
    assert payload["evidence_gate_findings"] == [finding]
    assert payload["evidence_notes"] == [
        "Score registration was force-released."
    ]


def test_serialize_exposes_direction_for_legacy_identity_fallback():
    [payload] = serialize_round_results([RoundResult(
        round_number=2,
        phase="innovation",
        direction="Candidate Axis",
        direction_key="",
        result_identity_key="",
    )])

    assert payload["direction"] == "Candidate Axis"
    assert payload["direction_key"] == ""
    assert payload["result_identity_key"] == ""
