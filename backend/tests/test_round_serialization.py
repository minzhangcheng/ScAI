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
