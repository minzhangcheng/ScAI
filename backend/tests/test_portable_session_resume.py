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
import asyncio
import json
import types
import uuid

import pytest

from backend.config import get_session_states
from backend.models.schemas import AnalysisState, TaskStatus
from backend.routes.export_import_parts.import_impl import (
    _mark_runtime_availability_after_import,
    _resign_imported_live_pkl,
)
from backend.utils.runtime_apply import (
    RuntimeApplyError,
    is_forward_resume_blocked,
    verify_live_pkl_or_raise,
)
from backend.utils.sandbox_parts.shared import _integrity_dir, _sig_path
from backend.utils.secret_redaction import SEALED_SECRET_PREFIX, open_secret


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


def test_resigning_an_imported_pkl_lets_the_kernel_rehydrate():
    session_id = _session_id()
    pkl = _place_live_pkl(session_id)

    assert _resign_imported_live_pkl(session_id) is True
    assert _sig_path(pkl, _integrity_dir(session_id)).exists()
    verify_live_pkl_or_raise(session_id)  # must not raise


def test_resigning_is_a_no_op_when_no_pkl_was_imported():
    assert _resign_imported_live_pkl(_session_id()) is False


def test_resigned_signature_tracks_content_not_just_presence():
    """A signature must cover the bytes, so later tampering is still caught."""
    session_id = _session_id()
    pkl = _place_live_pkl(session_id)
    _resign_imported_live_pkl(session_id)

    pkl.write_bytes(b"TAMPERED-AFTER-IMPORT")

    with pytest.raises(RuntimeApplyError):
        verify_live_pkl_or_raise(session_id)


# --- 2. Credentials sealed under another installation's key -----------------


def test_foreign_sealed_secret_degrades_to_empty_instead_of_raising():
    foreign = SEALED_SECRET_PREFIX + "Z" * 64  # undecryptable under this key

    assert open_secret(foreign) == ""


def test_plaintext_and_empty_values_are_untouched():
    assert open_secret("sk-plaintext-legacy") == "sk-plaintext-legacy"
    assert open_secret("") == ""


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


def test_locally_sealed_credentials_still_round_trip():
    """The degrade path must not weaken the normal same-instance round trip."""
    state = AnalysisState(id="local-secret", user_id="owner")
    state.agent_server_configs = {"pi": {"api_key": "sk-local-key"}}

    revived = AnalysisState(**json.loads(json.dumps(state.model_dump(mode="json"))))

    assert revived.agent_server_configs["pi"]["api_key"] == "sk-local-key"


# --- 3. Runtime availability after a runtime-less import --------------------


def test_import_without_runtime_marks_the_session_unresumable():
    state = AnalysisState(id=_session_id(), user_id="owner", current_round=2)
    state.status = TaskStatus.PAUSED

    _mark_runtime_availability_after_import(state)

    assert state.runtime_restore_status == "unavailable"
    assert is_forward_resume_blocked(state.runtime_restore_status)
    assert state.runtime_restore_warning


def test_import_with_runtime_stays_resumable():
    session_id = _session_id()
    _place_live_pkl(session_id)
    state = AnalysisState(id=session_id, user_id="owner", current_round=2)

    _mark_runtime_availability_after_import(state)

    assert state.runtime_restore_status == "complete"
    assert not is_forward_resume_blocked(state.runtime_restore_status)


def test_never_analysed_session_is_left_alone():
    """Round 0 has no kernel state to lose; a fresh kernel is already correct."""
    state = AnalysisState(id=_session_id(), user_id="owner", current_round=0)

    _mark_runtime_availability_after_import(state)

    assert state.runtime_restore_status == "complete"


# --- 4. save_session captures the whole namespace ---------------------------


def _extract_save_code(monkeypatch, live_names, tracker_required):
    """Run save_session far enough to capture the code it sends to the kernel."""
    from backend.utils import sandbox as sandbox_mod

    sandbox = sandbox_mod.LocalJupyterSandbox.__new__(sandbox_mod.LocalJupyterSandbox)
    sandbox.session_id = "save-code-probe"
    sandbox._exec_lock = asyncio.Lock()
    sandbox._last_save_skipped = []
    sandbox.notebook_state = types.SimpleNamespace(
        session_path=get_session_states("save-code-probe") / "probe_session.pkl"
    )
    sandbox.notebook_state.session_path.parent.mkdir(parents=True, exist_ok=True)

    captured = {}

    async def _snapshot_user_globals():
        return set(live_names)

    async def _execute_unlocked(code, **_kwargs):
        captured["code"] = code
        return types.SimpleNamespace(success=True, output="", error=None)

    sandbox._snapshot_user_globals = _snapshot_user_globals
    sandbox._prune_legacy_unobserved_required = lambda _live: set()
    sandbox.get_runtime_dependency_names = lambda: set(tracker_required)
    sandbox._execute_unlocked = _execute_unlocked
    monkeypatch.setattr(sandbox_mod, "_write_pkl_signature", lambda *a, **k: None)

    asyncio.run(sandbox.save_session())
    return captured["code"]


def test_save_code_enumerates_the_namespace_rather_than_a_name_list(monkeypatch):
    code = _extract_save_code(monkeypatch, {"adata", "scratch"}, set())

    # It must read globals() itself; a hardcoded target list cannot see a value
    # the agent invented a name for.
    assert "globals().keys()" in code
    assert "_scai_is_anndata" in code


def test_save_code_requires_canonical_artifacts_that_are_live(monkeypatch):
    code = _extract_save_code(monkeypatch, {"adata", "scratch"}, set())

    assert "_required_names = set(['adata'])" in code


def test_save_code_does_not_require_canonical_artifacts_that_are_absent(monkeypatch):
    code = _extract_save_code(monkeypatch, {"scratch"}, set())

    assert "_required_names = set([])" in code


def test_save_code_keeps_tracker_declarations_required(monkeypatch):
    code = _extract_save_code(monkeypatch, {"scratch"}, {"deg_table"})

    assert "_required_names = set(['deg_table'])" in code


def _run_save_code(code, namespace):
    """Execute the generated code against a synthetic kernel namespace."""
    import io
    import sys

    stdout, real = io.StringIO(), sys.stdout
    sys.stdout = stdout
    try:
        exec(code, namespace)
    finally:
        sys.stdout = real
    return stdout.getvalue()


def _read_frames(path):
    import dill
    import zstandard as zstd

    magic = b"SCAI_ZSTD_DILL_V1\n"
    out = {}
    with open(path, "rb") as raw:
        assert raw.read(len(magic)) == magic
        with zstd.ZstdDecompressor().stream_reader(raw) as stream:
            while True:
                try:
                    frame = dill.load(stream)
                except EOFError:
                    break
                if isinstance(frame, tuple) and len(frame) == 3 and frame[0] == "__SCAI_ALIAS__":
                    out[frame[1]] = out[frame[2]]
                else:
                    out[frame[0]] = frame[1]
    return out


def test_anndata_under_a_non_canonical_name_is_captured(monkeypatch, tmp_path):
    anndata = pytest.importorskip("anndata")
    import numpy as np

    code = _extract_save_code(monkeypatch, {"adata_hepatocyte"}, set())
    target = get_session_states("save-code-probe") / "probe_session.pkl"
    namespace = {
        "__builtins__": __builtins__,
        "adata_hepatocyte": anndata.AnnData(np.zeros((2, 2))),
        "counts": np.arange(6),
    }

    _run_save_code(code, namespace)
    restored = _read_frames(target)

    assert "adata_hepatocyte" in restored
    assert restored["adata_hepatocyte"].shape == (2, 2)
    assert "counts" in restored


def test_unpicklable_value_is_skipped_without_losing_the_save(monkeypatch):
    code = _extract_save_code(monkeypatch, set(), set())
    target = get_session_states("save-code-probe") / "probe_session.pkl"

    class Unpicklable:
        def __reduce__(self):
            raise TypeError("cannot pickle")

    namespace = {
        "__builtins__": __builtins__,
        "broken": Unpicklable(),
        "keepme": {"a": 1},
    }

    output = _run_save_code(code, namespace)
    restored = _read_frames(target)

    assert "__SCAI_SAVE_SKIPPED__" in output
    assert "broken" not in restored
    # The whole point: one bad value must not cost the user everything else.
    assert restored["keepme"] == {"a": 1}


def test_open_handles_are_excluded_by_type(monkeypatch, tmp_path):
    """dill will happily 'save' a file handle by path and then reopen it on a
    machine where that path does not exist, aborting the entire restore."""
    import sqlite3
    import threading

    code = _extract_save_code(monkeypatch, set(), set())
    target = get_session_states("save-code-probe") / "probe_session.pkl"
    handle = open(tmp_path / "scratch.txt", "w")
    connection = sqlite3.connect(":memory:")
    try:
        namespace = {
            "__builtins__": __builtins__,
            "log_handle": handle,
            "db": connection,
            "guard": threading.Lock(),
            "keepme": [1, 2, 3],
        }
        _run_save_code(code, namespace)
    finally:
        handle.close()
        connection.close()

    restored = _read_frames(target)

    assert "log_handle" not in restored
    assert "db" not in restored
    assert "guard" not in restored
    assert restored["keepme"] == [1, 2, 3]


def test_modules_and_callables_are_excluded(monkeypatch):
    import numpy as np

    code = _extract_save_code(monkeypatch, set(), set())
    target = get_session_states("save-code-probe") / "probe_session.pkl"

    def helper():
        return 1

    class Model:
        pass

    namespace = {
        "__builtins__": __builtins__,
        "np": np,
        "helper": helper,
        "Model": Model,
        "keepme": "value",
    }

    _run_save_code(code, namespace)
    restored = _read_frames(target)

    assert set(restored) == {"keepme"}


def test_missing_required_variable_still_fails_closed(monkeypatch):
    """Relaxing the optional tier must not relax the required one."""
    code = _extract_save_code(monkeypatch, set(), {"deg_table"})
    namespace = {"__builtins__": __builtins__, "something_else": 1}

    with pytest.raises(RuntimeError, match="required runtime variables are missing"):
        _run_save_code(code, namespace)


def test_aliased_objects_restore_to_one_object(monkeypatch):
    code = _extract_save_code(monkeypatch, set(), set())
    target = get_session_states("save-code-probe") / "probe_session.pkl"
    shared = {"x": 1}
    namespace = {"__builtins__": __builtins__, "first": shared, "second": shared}

    _run_save_code(code, namespace)
    restored = _read_frames(target)

    assert restored["first"] is restored["second"]
