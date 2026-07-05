"""Client-facing serialization shared by completed-session reconnect
(websockets) and report APIs, so integration-phase fields stay in sync."""
from typing import Any, Dict, List

from ..models.schemas import RoundResult


def serialize_round_results(round_results: List[RoundResult]) -> List[Dict[str, Any]]:
    """Serialize round results for the frontend. Includes integration-phase
    fields (phase, integration_candidates, integration_report_summary) so an
    integration round survives reload instead of degrading to a plain round."""
    return [
        {
            "round": r.round_number,
            "summary": r.pi_summary,
            "phase": r.phase,
            "pathology_findings": r.pathology_findings,
            "bioinformatics_findings": r.bioinformatics_findings,
            "biochemistry_findings": r.biochemistry_findings,
            "literature_findings": r.literature_findings,
            "integration_candidates": r.integration_candidates,
            "integration_report_summary": r.integration_report_summary,
        }
        for r in round_results
    ]
