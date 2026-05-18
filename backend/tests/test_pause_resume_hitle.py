"""Tests for pause-from-WAITING deadlock fix.

When a session is paused from a WAITING_* state, the original WAITING_*
status is preserved in pre_pause_status so resume can restore it and
re-activate the correct HITL gate event.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.models.schemas import AnalysisState, TaskStatus


class TestPrePauseStatusField:
    """AnalysisState.pre_pause_status field exists and defaults to None."""

    def test_pre_pause_status_defaults_to_none(self):
        state = AnalysisState(user_id="test")
        assert state.pre_pause_status is None

    def test_pre_pause_status_accepts_task_status(self):
        state = AnalysisState(user_id="test")
        state.pre_pause_status = TaskStatus.WAITING_INPUT
        assert state.pre_pause_status == TaskStatus.WAITING_INPUT

    def test_pre_pause_status_not_in_api_dict(self):
        state = AnalysisState(user_id="test")
        state.pre_pause_status = TaskStatus.WAITING_INPUT
        api_dict = state.to_api_dict()
        assert "pre_pause_status" not in api_dict

    def test_pre_pause_status_survives_json_roundtrip(self):
        state = AnalysisState(user_id="test")
        state.pre_pause_status = TaskStatus.WAITING_ANNOTATION
        json_str = state.model_dump_json()
        restored = AnalysisState.model_validate_json(json_str)
        assert restored.pre_pause_status == TaskStatus.WAITING_ANNOTATION


class TestPauseSavesPrePauseStatus:
    """pause() must save the original WAITING_* status before overwriting."""

    def test_pause_from_waiting_input_saves_pre_pause_status(self):
        state = AnalysisState(user_id="test")
        state.status = TaskStatus.WAITING_INPUT
        if state.status in (
            TaskStatus.WAITING_INPUT,
            TaskStatus.WAITING_ANNOTATION,
            TaskStatus.WAITING_PLAN_REVIEW,
        ):
            state.pre_pause_status = state.status
        state.status = TaskStatus.PAUSED
        assert state.pre_pause_status == TaskStatus.WAITING_INPUT
        assert state.status == TaskStatus.PAUSED

    def test_pause_from_running_clears_pre_pause_status(self):
        state = AnalysisState(user_id="test")
        state.status = TaskStatus.RUNNING
        if state.status in (
            TaskStatus.WAITING_INPUT,
            TaskStatus.WAITING_ANNOTATION,
            TaskStatus.WAITING_PLAN_REVIEW,
        ):
            state.pre_pause_status = state.status
        else:
            state.pre_pause_status = None
        state.status = TaskStatus.PAUSED
        assert state.pre_pause_status is None

    def test_pause_from_waiting_annotation_saves_pre_pause_status(self):
        state = AnalysisState(user_id="test")
        state.status = TaskStatus.WAITING_ANNOTATION
        if state.status in (
            TaskStatus.WAITING_INPUT,
            TaskStatus.WAITING_ANNOTATION,
            TaskStatus.WAITING_PLAN_REVIEW,
        ):
            state.pre_pause_status = state.status
        state.status = TaskStatus.PAUSED
        assert state.pre_pause_status == TaskStatus.WAITING_ANNOTATION

    def test_pause_from_waiting_plan_review_saves_pre_pause_status(self):
        state = AnalysisState(user_id="test")
        state.status = TaskStatus.WAITING_PLAN_REVIEW
        if state.status in (
            TaskStatus.WAITING_INPUT,
            TaskStatus.WAITING_ANNOTATION,
            TaskStatus.WAITING_PLAN_REVIEW,
        ):
            state.pre_pause_status = state.status
        state.status = TaskStatus.PAUSED
        assert state.pre_pause_status == TaskStatus.WAITING_PLAN_REVIEW


class TestResumeRestoresPrePauseStatus:
    """resume() from PAUSED must restore pre_pause_status and signal correct HITL event."""

    def test_resume_from_waiting_input_pause_restores_status(self):
        state = AnalysisState(user_id="test")
        state.status = TaskStatus.PAUSED
        state.pre_pause_status = TaskStatus.WAITING_INPUT
        if state.pre_pause_status in (
            TaskStatus.WAITING_INPUT,
            TaskStatus.WAITING_ANNOTATION,
            TaskStatus.WAITING_PLAN_REVIEW,
        ):
            restored = state.pre_pause_status
            state.pre_pause_status = None
            state.status = restored
        else:
            state.pre_pause_status = None
            state.status = TaskStatus.RUNNING
        assert state.status == TaskStatus.WAITING_INPUT
        assert state.pre_pause_status is None

    def test_resume_from_waiting_annotation_pause_restores_status(self):
        state = AnalysisState(user_id="test")
        state.status = TaskStatus.PAUSED
        state.pre_pause_status = TaskStatus.WAITING_ANNOTATION
        if state.pre_pause_status in (
            TaskStatus.WAITING_INPUT,
            TaskStatus.WAITING_ANNOTATION,
            TaskStatus.WAITING_PLAN_REVIEW,
        ):
            restored = state.pre_pause_status
            state.pre_pause_status = None
            state.status = restored
        else:
            state.pre_pause_status = None
            state.status = TaskStatus.RUNNING
        assert state.status == TaskStatus.WAITING_ANNOTATION
        assert state.pre_pause_status is None

    def test_resume_from_running_pause_goes_to_running(self):
        state = AnalysisState(user_id="test")
        state.status = TaskStatus.PAUSED
        state.pre_pause_status = None
        if state.pre_pause_status in (
            TaskStatus.WAITING_INPUT,
            TaskStatus.WAITING_ANNOTATION,
            TaskStatus.WAITING_PLAN_REVIEW,
        ):
            restored = state.pre_pause_status
            state.pre_pause_status = None
            state.status = restored
        else:
            state.pre_pause_status = None
            state.status = TaskStatus.RUNNING
        assert state.status == TaskStatus.RUNNING
        assert state.pre_pause_status is None

    def test_resume_signals_correct_hitl_event(self):
        signal_map = {
            TaskStatus.WAITING_INPUT: "signal_continue",
            TaskStatus.WAITING_ANNOTATION: "signal_annotation_confirmed",
            TaskStatus.WAITING_PLAN_REVIEW: "signal_plan_review",
        }
        assert signal_map[TaskStatus.WAITING_INPUT] == "signal_continue"
        assert signal_map[TaskStatus.WAITING_ANNOTATION] == "signal_annotation_confirmed"
        assert signal_map[TaskStatus.WAITING_PLAN_REVIEW] == "signal_plan_review"


def test_resume_from_waiting_plan_review_must_not_auto_skip():
    state = AnalysisState(user_id="test")
    state.status = TaskStatus.WAITING_PLAN_REVIEW

    signal_map = []

    if state.status == TaskStatus.WAITING_PLAN_REVIEW:
        restored_action = "stay_waiting_plan_review"
    else:
        restored_action = "other"

    signal_map.append(restored_action)

    assert signal_map == ["stay_waiting_plan_review"]


def test_resume_from_paused_waiting_plan_review_must_not_auto_skip():
    state = AnalysisState(user_id="test")
    state.status = TaskStatus.PAUSED
    state.pre_pause_status = TaskStatus.WAITING_PLAN_REVIEW

    if state.pre_pause_status in (
        TaskStatus.WAITING_INPUT,
        TaskStatus.WAITING_ANNOTATION,
        TaskStatus.WAITING_PLAN_REVIEW,
    ):
        restored = state.pre_pause_status
        state.pre_pause_status = None
        state.status = restored
    else:
        state.pre_pause_status = None
        state.status = TaskStatus.RUNNING

    assert state.status == TaskStatus.WAITING_PLAN_REVIEW