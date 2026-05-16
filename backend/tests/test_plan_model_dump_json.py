"""Tests that AnalysisPlan.model_dump() output is JSON-serializable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from datetime import datetime

from backend.models.schemas import AnalysisPlan, PlanStep


class TestPlanModelDumpJsonSerializable:
    """Verify that model_dump(mode='json') produces JSON-safe dicts."""

    def test_plan_model_dump_default_not_json_serializable(self):
        """Canary: model_dump() (no mode) preserves raw datetime — json.dumps fails.
        If this test breaks, Pydantic changed default serialization and
        model_dump(mode="json") callers may also need review."""
        plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="test plan",
            steps=[PlanStep(id=1, title="QC", method="scanpy")],
            status="draft",
        )
        raw = plan.model_dump()
        with pytest.raises(TypeError, match="Object of type datetime is not JSON serializable"):
            json.dumps(raw)

    def test_plan_model_dump_mode_json_is_serializable(self):
        """model_dump(mode='json') converts datetime to ISO strings — json.dumps succeeds."""
        plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="test plan",
            steps=[PlanStep(id=1, title="QC", method="scanpy")],
            status="approved",
            created_at=datetime(2026, 5, 15, 3, 5, 2),
            approved_at=datetime(2026, 5, 15, 3, 5, 2),
        )
        safe = plan.model_dump(mode="json")
        json.dumps(safe)
        assert safe["created_at"] == "2026-05-15T03:05:02"
        assert safe["approved_at"] == "2026-05-15T03:05:02"

    def test_plan_step_executed_at_is_serializable_with_mode_json(self):
        """PlanStep.executed_at is datetime — must be ISO string in mode='json'."""
        step = PlanStep(
            id=1, title="Step", method="test",
            executed_at=datetime(2026, 5, 15, 3, 5, 2),
        )
        safe = step.model_dump(mode="json")
        json.dumps(safe)
        assert safe["executed_at"] == "2026-05-15T03:05:02"

    def test_plan_model_dump_mode_json_preserves_all_fields(self):
        """model_dump(mode='json') has the same keys as model_dump()."""
        plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="test",
            steps=[PlanStep(id=1, title="S", method="m")],
        )
        default_keys = set(plan.model_dump().keys())
        json_keys = set(plan.model_dump(mode="json").keys())
        assert default_keys == json_keys
