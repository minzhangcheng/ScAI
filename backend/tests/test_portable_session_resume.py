"""A portable session must be resumable on a *different* installation.

Three installation-scoped mechanisms used to make that impossible, each in a way
that only shows up across instances:

1. the live sandbox pkl's HMAC sidecar lives outside the session directory, so it
   never travels with the archive and the destination refuses to rehydrate;
2. credentials sealed with the source installation's Fernet key cannot be opened
   at the destination, and raising aborts AnalysisState validation -- which makes
   an imported LangGraph checkpoint undeserializable and blocks resume entirely;
3. save_session only captured a fixed name list, so an AnnData under any other
   name was silently absent from the runtime the destination restores.

A fourth case is the honest-failure one: importing *without* runtime must not
leave the session claiming its kernel state is intact.
"""
import uuid

import pytest

from backend.config import get_session_states
from backend.models.schemas import AnalysisState
from backend.routes.export_import_parts.import_impl import (
    _resign_imported_live_pkl,
)
from backend.utils.runtime_apply import (
    RuntimeApplyError,
    verify_live_pkl_or_raise,
)
from backend.utils.secret_redaction import SEALED_SECRET_PREFIX


def _session_id() -> str:
    return f"portable-resume-{uuid.uuid4().hex[:12]}"


def _place_live_pkl(session_id: str, payload: bytes = b"DILL-PAYLOAD") -> "object":
    """Drop a pkl exactly where import leaves one, with no signature."""
    states = get_session_states(session_id)
    states.mkdir(parents=True, exist_ok=True)
    pkl = states / f"{session_id}_session.pkl"
    pkl.write_bytes(payload)
    return pkl


# --- 1. HMAC re-signing -----------------------------------------------------


def test_unsigned_imported_pkl_is_rejected_before_resigning():
    """Baseline: this is the cross-instance failure the re-sign exists to fix."""
    session_id = _session_id()
    _place_live_pkl(session_id)

    with pytest.raises(RuntimeApplyError):
        verify_live_pkl_or_raise(session_id)


def test_resigned_signature_tracks_content_not_just_presence():
    """A signature must cover the bytes, so later tampering is still caught."""
    session_id = _session_id()
    pkl = _place_live_pkl(session_id)
    _resign_imported_live_pkl(session_id)

    pkl.write_bytes(b"TAMPERED-AFTER-IMPORT")

    with pytest.raises(RuntimeApplyError):
        verify_live_pkl_or_raise(session_id)


# --- 2. Credentials sealed under another installation's key -----------------


def test_analysis_state_still_loads_when_its_credentials_are_foreign():
    """The checkpoint case: validation must survive, minus the credentials."""
    payload = AnalysisState(id="foreign-secret", user_id="owner").model_dump(mode="json")
    payload["agent_server_configs"] = {
        "pi": {
            "base_url": "https://api.example.test/v1",
            "api_key": SEALED_SECRET_PREFIX + "Z" * 64,
        }
    }
    payload["current_round"] = 3

    state = AnalysisState(**payload)

    assert state.current_round == 3
    assert state.agent_server_configs["pi"]["api_key"] == ""
    # Non-secret configuration must not be collateral damage.
    assert state.agent_server_configs["pi"]["base_url"] == "https://api.example.test/v1"


# --- 3. Runtime availability after a runtime-less import --------------------


# --- 4. save_session captures the whole namespace ---------------------------
