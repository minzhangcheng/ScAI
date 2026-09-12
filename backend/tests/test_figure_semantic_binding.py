"""Supervisor semantic binding and durable figure-repair contracts."""

import logging

from backend.agents.supervisor.figure_binding import validate_figure_binding
from backend.models.schemas import AnalysisState, PublicationFigure
from backend.services.figure_contract import RequiredFigure, _matches
from backend.services.figure_revision_tasks import (
    resolve_figure_revision_tasks,
    upsert_figure_revision_task,
)


def _figure(**overrides) -> PublicationFigure:
    values = {
        "id": "figure-1",
        "name": "candidate",
        "description": "candidate",
        "figure_type": "volcano",
        "png_path": "candidate.png",
        "pdf_path": "candidate.pdf",
        "logical_figure_id": "logical-1",
        "plan_id": "plan-1",
        "plan_step_id": 2,
        "plan_requirement_key": "plan.2.1",
        "source_execution_id": "execution-1",
    }
    values.update(overrides)
    return PublicationFigure(**values)


def test_supervisor_binding_validator_is_closed_and_requires_exact_key(caplog):
    accepted, error = validate_figure_binding({
        "decision": "rebind",
        "plan_requirement_key": "plan.2.1",
        "reason": "The contrast plot matches the second visual requirement.",
        "confidence": 0.9,
    })
    assert error == ""
    assert accepted["plan_requirement_key"] == "plan.2.1"

    assert validate_figure_binding({
        "decision": "accept",
        "plan_requirement_key": None,
        "reason": "missing key",
    })[0] is None

    with caplog.at_level(
        logging.WARNING,
        logger="backend.agents.supervisor.figure_binding",
    ):
        supplementary, supplementary_error = validate_figure_binding({
            "decision": "supplementary",
            "plan_requirement_key": "plan.2.1",
            "reason": "the candidate is additional context",
        })
        rejected, rejected_error = validate_figure_binding({
            "decision": "reject",
            "plan_requirement_key": "k" * 150,
            "reason": "does not match the visual requirement",
        })

    assert supplementary_error == ""
    assert supplementary["decision"] == "supplementary"
    assert supplementary["plan_requirement_key"] is None

    assert rejected_error == ""
    assert rejected["decision"] == "reject"
    assert rejected["plan_requirement_key"] is None

    assert "plan.2.1" in caplog.text
    assert "k" * 120 in caplog.text
    assert "k" * 121 not in caplog.text


def test_exact_requirement_key_prevents_cross_step_figure_credit():
    step_one = RequiredFigure(
        key="plan.1.1",
        figure_type="volcano",
        description="first contrast",
        plan_step_id=1,
        plan_id="plan-1",
        require_plan_link=True,
    )
    step_two = RequiredFigure(
        key="plan.2.1",
        figure_type="volcano",
        description="second contrast",
        plan_step_id=2,
        plan_id="plan-1",
        require_plan_link=True,
    )
    figure = _figure()

    assert _matches(step_one, figure) is False
    assert _matches(step_two, figure) is True


def test_revision_task_is_updated_then_resolved_by_reviewed_replacement():
    state = AnalysisState(id="revision-session", user_id="user")
    figure = _figure(revision_count=0)

    first = upsert_figure_revision_task(
        state,
        figure,
        reason="visual_revision",
        must_fix=["Separate overlapping labels"],
    )
    figure.revision_count = 1
    second = upsert_figure_revision_task(
        state,
        figure,
        reason="revision_failed",
        must_fix=["Regenerate the plot"],
    )

    assert first.id == second.id
    assert len(state.figure_revision_tasks) == 1
    assert second.reason == "revision_failed"
    assert second.attempt_count == 1
    assert resolve_figure_revision_tasks(state, figure) == 1
    assert second.status == "resolved"


def test_second_failed_redraw_exhausts_and_excludes_the_task():
    state = AnalysisState(id="revision-exhausted", user_id="user")
    figure = _figure(revision_count=2)

    task = upsert_figure_revision_task(
        state,
        figure,
        reason="revision_failed",
        must_fix=["Regenerate the plot"],
    )

    assert task.attempt_count == 2
    assert task.reason == "attempts_exhausted"
    assert task.status == "exhausted_excluded"
    assert resolve_figure_revision_tasks(state, figure) == 0
    assert task.status == "exhausted_excluded"


def test_same_logical_figure_in_a_revised_plan_has_a_separate_task():
    state = AnalysisState(id="revision-plan-identity", user_id="user")
    old_figure = _figure(plan_id="plan-1")
    new_figure = _figure(id="figure-2", plan_id="plan-2")

    old_task = upsert_figure_revision_task(
        state,
        old_figure,
        reason="visual_revision",
        must_fix=["Fix the old-plan render"],
    )
    new_task = upsert_figure_revision_task(
        state,
        new_figure,
        reason="visual_revision",
        must_fix=["Fix the revised-plan render"],
    )

    assert old_task.id != new_task.id
    assert len(state.figure_revision_tasks) == 2
    assert resolve_figure_revision_tasks(state, new_figure) == 1
    assert new_task.status == "resolved"
    assert old_task.status == "pending"
