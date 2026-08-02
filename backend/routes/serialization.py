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
            "round_status": r.round_status,
            "evidence_status": r.evidence_status,
            "direction_coverage_eligible": r.direction_coverage_eligible,
            "bioinformatics_evidence_verified": r.bioinformatics_evidence_verified,
            "evidence_gate_degraded": r.evidence_gate_degraded,
            "evidence_execution_ids": list(r.evidence_execution_ids or []),
            "evidence_notes": list(r.evidence_notes or []),
            "evidence_gate_findings": list(r.evidence_gate_findings or []),
            "lineage_id": r.lineage_id,
            "branch_id": r.branch_id,
            "branch_label": r.branch_label,
            "restored_from_snapshot_id": r.restored_from_snapshot_id,
            "direction": r.direction,
            "direction_key": r.direction_key,
            "result_identity_key": r.result_identity_key,
        }
        for r in round_results
    ]
