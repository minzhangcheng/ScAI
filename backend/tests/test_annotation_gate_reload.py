"""The annotation gate must survive a client reload.

The confirmation dialog used to be reconstructible only by scanning the
transient realtime-message stream for an `annotation_confirmation` message.
That stream is server-authoritative but lossy by design: /load replays only the
last 500 messages, message_callback caps the stream at 1000, and snapshot
restore drops it entirely. Whenever the message fell out, the frontend had no
way back to the dialog — `annotation_state` held the full payload but was never
exposed by the API.

These tests pin the contract that /load carries the authoritative annotation
state, so the client can rebuild the gate without the message.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

from backend.models.schemas import (
    AnalysisState,
    AnnotationConfirmationState,
    ClusterAnnotation,
    TaskStatus,
)


def _pending_state() -> AnalysisState:
    """A session parked at the annotation gate, as after a browser restart."""
    state = AnalysisState(user_id="test")
    state.status = TaskStatus.WAITING_ANNOTATION
    state.annotation_state = AnnotationConfirmationState(
        cluster_annotations=[
            ClusterAnnotation(
                cluster_id="0",
                cell_type="Hepatocyte",
                key_markers=["Alb", "Cyp2e1"],
                confidence="high",
                group_proportions={"Con": 9.8, "DKO": 0.7},
            ),
        ],
        tissue="liver",
        organism="mouse",
        tissue_summary="Liver tissue summary",
        analysis_summary="Analysis summary",
        pending=True,
        proportion_group_labels=["Con", "DKO"],
        proportion_dimension="group",
        validation_status="completed",
        validation_summary="Validation summary",
    )
    return state


class TestAnnotationGateSurvivesReload:

    def test_load_exposes_annotation_state(self):
        """/load must carry annotation_state so a reloaded client can rebuild
        the dialog without the transient annotation_confirmation message."""
        api = _pending_state().to_api_dict()
        json.dumps(api)

        assert "annotation_state" in api, (
            "annotation_state missing from to_api_dict — a reloaded client "
            "cannot rebuild the annotation dialog"
        )

    def test_exposed_annotation_state_carries_dialog_payload(self):
        """The exposed state must contain every field the dialog renders."""
        ann = _pending_state().to_api_dict()["annotation_state"]

        assert ann["tissue"] == "liver"
        assert ann["organism"] == "mouse"
        assert ann["tissue_summary"] == "Liver tissue summary"
        assert ann["pending"] is True
        assert ann["proportion_group_labels"] == ["Con", "DKO"]
        assert ann["proportion_dimension"] == "group"
        assert ann["validation_status"] == "completed"
        assert ann["validation_summary"] == "Validation summary"

        assert len(ann["cluster_annotations"]) == 1
        cluster = ann["cluster_annotations"][0]
        assert cluster["cluster_id"] == "0"
        assert cluster["cell_type"] == "Hepatocyte"
        assert cluster["key_markers"] == ["Alb", "Cyp2e1"]
        assert cluster["group_proportions"] == {"Con": 9.8, "DKO": 0.7}

    def test_annotation_state_is_null_when_no_gate_pending(self):
        """A session that never reached the gate exposes the key as null,
        so the client can tell "no gate" from "gate data lost"."""
        api = AnalysisState(user_id="test").to_api_dict()

        assert "annotation_state" in api
        assert api["annotation_state"] is None
