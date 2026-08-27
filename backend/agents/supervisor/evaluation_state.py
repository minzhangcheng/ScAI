"""Supervisor evaluation log and round-state mutation helpers."""
from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, List

from ...models.schemas import PlanStage
from .shared import SupervisorRecord, _serialize_record

logger = logging.getLogger(__name__)

def reset_for_round(self, planning_stage: PlanStage = "full"):
    """Archive current round log and reset for a new round."""
    if self.supervision_log:
        self.all_rounds_log.append(self.supervision_log.copy())
    self.supervision_log = []
    self._step_count = 0
    self._recent_errors = []
    self._pi_question_hashes = set()
    self.planning_stage = planning_stage

def get_summary_stats(self) -> Dict[str, Any]:
    """Return summary statistics for the current analysis."""
    all_records = []
    for round_log in self.all_rounds_log:
        all_records.extend(round_log)
    all_records.extend(self.supervision_log)

    total = len(all_records)
    with_issues = sum(1 for r in all_records if r.technical_issues)
    pi_escalations = sum(1 for r in all_records if r.pi_escalated)
    critical = sum(1 for r in all_records if r.llm_severity == "critical")

    # Build checkpoints with round identifiers
    checkpoints_reached = []
    for round_idx, round_log in enumerate(self.all_rounds_log):
        for r in round_log:
            if r.checkpoint_type:
                checkpoints_reached.append({"step": r.step_number, "type": r.checkpoint_type, "round": round_idx + 1})
    current_round = len(self.all_rounds_log) + 1
    for r in self.supervision_log:
        if r.checkpoint_type:
            checkpoints_reached.append({"step": r.step_number, "type": r.checkpoint_type, "round": current_round})

    return {
        "total_steps_monitored": total,
        "steps_with_technical_issues": with_issues,
        "pi_escalations": pi_escalations,
        "checkpoints_reached": checkpoints_reached,
        "critical_issues": critical,
        "rounds_completed": len(self.all_rounds_log),
    }

def get_supervision_log(self) -> List[Dict]:
    """Return serializable log of all rounds + current round, with round_number."""
    result = []
    for round_idx, round_log in enumerate(self.all_rounds_log):
        for r in round_log:
            result.append(_serialize_record(r, round_idx + 1))
    current_round = len(self.all_rounds_log) + 1
    for r in self.supervision_log:
        result.append(_serialize_record(r, current_round))
    return result

def restore_log_from_saved(self, records: List[Dict]):
    """Restore supervision log from saved analysis_log dicts, grouping by round_number.

    Records with round_number == max(round_number) go to supervision_log (current round).
    All other rounds are grouped into all_rounds_log.
    Old format records without round_number default to round 1.
    Empty records list short-circuits to empty logs.
    """
    if not records:
        self.all_rounds_log = []
        self.supervision_log = []
        return

    max_round = max(r.get("round_number", 1) for r in records)

    rounds: Dict[int, List[SupervisorRecord]] = {}
    for d in records:
        ts = d.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(ts) if ts else datetime.now()
        except (ValueError, TypeError):
            timestamp = datetime.now()
        record = SupervisorRecord(
            step_number=d.get("step_number", 0),
            step_type=d.get("step_type", ""),
            timestamp=timestamp,
            technical_issues=d.get("technical_issues", []),
            checkpoint_type=d.get("checkpoint_type"),
            llm_severity=d.get("llm_severity"),
            pi_escalated=d.get("pi_escalated", False),
            pi_response=d.get("pi_response"),
            feedback_injected=d.get("feedback_injected"),
            literature_topic=d.get("literature_topic"),
            literature_result=d.get("literature_result"),
            code=d.get("code"),
            code_output=d.get("code_output"),
            code_error=d.get("code_error"),
            saved_figure_filenames=d.get("saved_figure_filenames", []),
            agent_reasoning=d.get("agent_reasoning"),
        )
        rn = d.get("round_number", 1)
        rounds.setdefault(rn, []).append(record)

    self.all_rounds_log = []
    self.supervision_log = []
    for rn in sorted(rounds.keys()):
        if rn == max_round:
            self.supervision_log = rounds[rn]
        else:
            self.all_rounds_log.append(rounds[rn])

    logger.info(f"Restored {len(records)} records: {len(self.all_rounds_log)} completed rounds, "
                 f"{len(self.supervision_log)} current round records (round {max_round})")
