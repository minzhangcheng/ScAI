"""Regression tests for the session export/import hardening pass.

Each test pins one defect that made a *snapshot-disabled* export/import round
trip lose data, fail outright, or damage something outside the session being
imported. They are grouped by the failure they prevent, not by the function they
call, because several fixes span the export and the import side together.
"""
import importlib
import io
import json
import os
import sqlite3
import tarfile
import uuid
from pathlib import Path

import pytest

from backend.tests.split_facade import reload_split


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point every path helper at tmp_path so a round trip uses a clean tree."""
    monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path))
    import backend.config._config as _cfg
    importlib.reload(_cfg)
    import backend.config as _config
    importlib.reload(_config)
    reload_split("backend.routes.export_import")
    import backend.services.session_store as _ss
    importlib.reload(_ss)
    yield


def _session_json(session_id: str, **overrides) -> dict:
    from backend.models.schemas import AnalysisState

    state = AnalysisState(id=session_id, name="hardening", user_id="owner")
    for key, value in overrides.items():
        setattr(state, key, value)
    return state.model_dump(mode="json")


def _build_archive(
    path: Path,
    session_id: str,
    members: dict[str, bytes],
    *,
    data_base_dir: str,
    state_payload: dict | None = None,
) -> Path:
    """Write a session archive with an arbitrary set of extra members."""
    with tarfile.open(path, mode="w:gz") as tf:
        def _add(arcname: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name=arcname)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

        _add(
            f"{session_id}/manifest.json",
            json.dumps({
                "session_id": session_id,
                "version": "2.0",
                "data_base_dir": data_base_dir,
            }).encode(),
        )
        payload = state_payload if state_payload is not None else _session_json(session_id)
        _add(
            f"{session_id}/sessions/{session_id}.json",
            json.dumps(payload).encode(),
        )
        for arcname, blob in members.items():
            _add(arcname, blob)
    return path


@pytest.mark.asyncio
async def test_import_phase_order_from_bind_through_publication(
    monkeypatch, tmp_path,
):
    from backend.config import DATA_BASE_DIR
    from backend.routes.export_import_parts import import_impl, import_tar
    from backend.state import import_jobs, sessions

    archive_session_id = str(uuid.uuid4())
    archive = _build_archive(
        tmp_path / "phase-order.tar.gz",
        archive_session_id,
        {f"{archive_session_id}/workspace/{archive_session_id}/result.txt": b"ok"},
        data_base_dir=str(DATA_BASE_DIR),
    )
    import_job_id = "phase-order-job"
    import_jobs[import_job_id] = {
        "created_at": 1.0,
        "cancel_requested": False,
    }
    trace = []
    extract_seen = False

    original_allocate = import_impl._allocate_import_session_id
    original_resolve = import_tar._resolve_import_dest
    original_rebind = import_tar._rebind_session_id_values
    original_model = import_tar.AnalysisState
    original_finalize = import_tar._finalize_imported_state
    original_save = import_impl.async_save_session

    def traced_allocate(*args, **kwargs):
        trace.append("bind")
        return original_allocate(*args, **kwargs)

    def traced_resolve(*args, **kwargs):
        nonlocal extract_seen
        if not extract_seen:
            trace.append("extract")
            extract_seen = True
        return original_resolve(*args, **kwargs)

    def traced_rebind(*args, **kwargs):
        trace.append("rebind")
        return original_rebind(*args, **kwargs)

    def traced_model(*args, **kwargs):
        trace.append("validate")
        return original_model(*args, **kwargs)

    def traced_finalize(*args, **kwargs):
        trace.append("finalize")
        return original_finalize(*args, **kwargs)

    async def traced_save(*args, **kwargs):
        trace.append("publish")
        return await original_save(*args, **kwargs)

    monkeypatch.setattr(import_impl, "_allocate_import_session_id", traced_allocate)
    monkeypatch.setattr(import_tar, "_resolve_import_dest", traced_resolve)
    monkeypatch.setattr(import_tar, "_rebind_session_id_values", traced_rebind)
    monkeypatch.setattr(import_tar, "AnalysisState", traced_model)
    monkeypatch.setattr(import_tar, "_finalize_imported_state", traced_finalize)
    monkeypatch.setattr(import_impl, "async_save_session", traced_save)

    await import_impl._process_import_job(
        str(archive), import_job_id, "owner",
    )

    assert import_jobs[import_job_id]["status"] == "completed"
    assert trace == ["bind", "extract", "rebind", "validate", "finalize", "publish"]
    imported_id = import_jobs[import_job_id]["session_id"]
    sessions.pop(imported_id, None)
    import_jobs.pop(import_job_id, None)


def test_import_finalization_revalidates_v2_figures_without_upgrading_legacy(
    monkeypatch,
):
    from backend.models.schemas import (
        AnalysisState,
        FigureReviewStatus,
        PublicationFigure,
    )
    from backend.routes.export_import_parts import import_tar
    from backend.services import session_lifecycle

    invalid = PublicationFigure(
        id="dangling-imported-figure",
        name="dangling",
        description="Imported figure with no source receipt",
        figure_type="heatmap",
        png_path="/missing/dangling.jpg",
        pdf_path="/missing/dangling.pdf",
        review_status=FigureReviewStatus.APPROVED,
        plan_id="plan-imported",
        round_number=2,
        provenance_schema_version=2,
        producer_run_kind="ordinary",
        source_execution_id="missing-imported-execution",
        source_execution_run_kind="ordinary",
        source_execution_validated=True,
    )
    legacy = PublicationFigure(
        id="legacy-imported-figure",
        name="legacy",
        description="Pre-provenance figure",
        figure_type="umap",
        png_path="/missing/legacy.jpg",
        pdf_path="/missing/legacy.pdf",
    )
    known_invalid_legacy = PublicationFigure(
        id="14fc685e-9ffa-4bd2-bc03-2d92da0a9c87",
        name="doublet_score_distribution_per_sample",
        description="Persisted legacy incident artifact",
        figure_type="histogram",
        png_path="/missing/known-invalid.jpg",
        pdf_path="/missing/known-invalid.pdf",
        plan_id="014b3726-7baa-44b2-a0e4-d78366063654",
        plan_step_id=3,
        plan_requirement_key="plan.3.1",
        round_number=2,
        source_execution_id="a4989f32-064d-4c1d-b2ef-d47e153dd6ee",
    )
    state = AnalysisState(
        id="import-provenance",
        user_id="owner",
        publication_figures=[invalid, legacy, known_invalid_legacy],
    )
    monkeypatch.setattr(
        session_lifecycle, "bind_import_generation", lambda _state: None
    )

    import_tar._finalize_imported_state(state, imported_runtime=True)

    assert "missing from durable code history" in (
        invalid.provenance_validation_reason or ""
    )
    assert invalid.source_execution_validated is True
    assert legacy.provenance_schema_version is None
    assert legacy.provenance_validation_reason is None
    assert "Known legacy figure was invalidated" in (
        known_invalid_legacy.provenance_validation_reason or ""
    )


# --- The catch-all destination: anything under the data root was writable ---


class TestImportStaysInsideTheSession:
    def test_unmapped_prefixes_resolve_to_nothing(self):
        from backend.routes.export_import import _resolve_import_dest

        for relative in (
            "app.db",
            ".jwt_secret",
            "sessions/some-other-session/session.json",
            "integrity/sid/runtime_apply.journal.json",
            "../escape.txt",
        ):
            assert _resolve_import_dest(relative, "sid") is None, relative

    def test_known_prefixes_land_inside_the_session_directory(self):
        from backend.config import get_session_dir
        from backend.routes.export_import import _resolve_import_dest

        session_dir = get_session_dir("sid").resolve()
        for relative in (
            "states/checkpoints.sqlite",
            "uploads/sid/matrix.h5ad",
            "workspace/sid/figures/umap.png",
            "reports/sid/report.html",
            "blobs/deadbeef.b64",
            "snapshots/snap1_blobs/cafe.b64",
            "logs/llm.jsonl",
            "reproducibility/manifest.json",
        ):
            dest = _resolve_import_dest(relative, "sid")
            assert dest is not None, relative
            assert dest.resolve().is_relative_to(session_dir), relative

    def test_archive_cannot_overwrite_the_account_database(self, tmp_path):
        """The concrete exploit: a member named <sid>/app.db.

        APP_DB_PATH is DATA_BASE_DIR/app.db, and the only guard used to be
        validate_path_within(dest, DATA_BASE_DIR) — which permits the entire
        data root, including every other user's session and the JWT signing key.
        """
        from backend.config import APP_DB_PATH, DATA_BASE_DIR
        from backend.routes.export_import import _import_tar_gz

        APP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        APP_DB_PATH.write_bytes(b"REAL-ACCOUNT-DB")
        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "evil.tar.gz",
            session_id,
            {
                f"{session_id}/app.db": b"PWNED",
                f"{session_id}/.jwt_secret": b"attacker-chosen-key",
            },
            data_base_dir=str(DATA_BASE_DIR),
        )

        _import_tar_gz(str(archive), user_id="attacker")

        assert APP_DB_PATH.read_bytes() == b"REAL-ACCOUNT-DB"
        assert not (DATA_BASE_DIR / ".jwt_secret").exists()

    def test_reproducibility_bundle_lands_in_the_workspace(self, tmp_path):
        """Benign proof the catch-all was live code, not a theoretical path."""
        from backend.config import DATA_BASE_DIR, get_session_workspace
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "repro.tar.gz",
            session_id,
            {f"{session_id}/reproducibility/manifest.json": b"{}"},
            data_base_dir=str(DATA_BASE_DIR),
        )

        state = _import_tar_gz(str(archive), user_id="owner")

        landed = get_session_workspace(state.id) / "reproducibility" / "manifest.json"
        assert landed.exists()
        assert not (DATA_BASE_DIR / "reproducibility").exists()


# --- Importing is creating a session, never overwriting one -----------------


class TestImportNeverOverwritesAnExistingSession:
    def test_colliding_id_becomes_a_new_session(self, tmp_path):
        from backend.config import DATA_BASE_DIR, get_session_dir, get_session_uploads
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        # A live session already owns that id, with a file the archive lacks.
        victim_uploads = get_session_uploads(session_id)
        victim_uploads.mkdir(parents=True, exist_ok=True)
        (victim_uploads / "victim.h5ad").write_bytes(b"VICTIM-DATA")
        (get_session_dir(session_id) / "session.json").write_text(
            json.dumps(_session_json(session_id, name="victim")), encoding="utf-8"
        )

        archive = _build_archive(
            tmp_path / "backup.tar.gz",
            session_id,
            {f"{session_id}/uploads/{session_id}/imported.h5ad": b"IMPORTED"},
            data_base_dir=str(DATA_BASE_DIR),
        )

        state = _import_tar_gz(str(archive), user_id="importer")

        assert state.id != session_id, "import must not reuse an occupied id"
        # The victim is untouched, still its own owner's session.
        assert (victim_uploads / "victim.h5ad").read_bytes() == b"VICTIM-DATA"
        victim_json = json.loads(
            (get_session_dir(session_id) / "session.json").read_text(encoding="utf-8")
        )
        assert victim_json["user_id"] == "owner"
        assert victim_json["name"] == "victim"
        # And the archive landed somewhere new, in full.
        assert (get_session_uploads(state.id) / "imported.h5ad").read_bytes() == b"IMPORTED"

    def test_free_id_is_kept(self, tmp_path):
        from backend.config import DATA_BASE_DIR
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "fresh.tar.gz", session_id, {}, data_base_dir=str(DATA_BASE_DIR),
        )

        state = _import_tar_gz(str(archive), user_id="owner")

        assert state.id == session_id

    def test_paths_are_rebound_to_the_new_id(self, tmp_path):
        from backend.config import DATA_BASE_DIR, get_session_dir, get_session_uploads
        from backend.routes.export_import import _import_tar_gz
        from backend.models.schemas import DataFile

        session_id = str(uuid.uuid4())
        get_session_dir(session_id).mkdir(parents=True, exist_ok=True)

        payload = _session_json(session_id)
        payload["data_files"] = [
            DataFile(
                filename="matrix.h5ad",
                file_path=str(get_session_uploads(session_id) / "matrix.h5ad"),
                section="sequencing",
                data_type="scrnaseq",
            ).model_dump(mode="json")
        ]
        archive = _build_archive(
            tmp_path / "rebind.tar.gz",
            session_id,
            {f"{session_id}/uploads/{session_id}/matrix.h5ad": b"X"},
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=payload,
        )

        state = _import_tar_gz(str(archive), user_id="importer")

        assert state.id != session_id
        assert state.data_files[0].file_path == str(
            get_session_uploads(state.id) / "matrix.h5ad"
        )
        assert Path(state.data_files[0].file_path).exists()

    def test_a_stale_pkl_does_not_make_the_session_look_resumable(self):
        """The pkl on disk must be one *this* import wrote, not any pkl at all."""
        from backend.config import get_session_states
        from backend.models.schemas import AnalysisState
        from backend.routes.export_import import _mark_runtime_availability_after_import
        from backend.utils.runtime_apply import is_forward_resume_blocked

        session_id = str(uuid.uuid4())
        states = get_session_states(session_id)
        states.mkdir(parents=True, exist_ok=True)
        (states / f"{session_id}_session.pkl").write_bytes(b"KERNEL-FROM-ANOTHER-RUN")
        state = AnalysisState(id=session_id, user_id="owner", current_round=7)

        _mark_runtime_availability_after_import(state, imported_pkl=False)

        assert state.runtime_restore_status == "unavailable"
        assert is_forward_resume_blocked(state.runtime_restore_status)


# --- An imported session must land in a status this installation can back up -


class TestAnImportedSessionAlwaysHasAWayForward:
    """The reachable control must lead somewhere, not just some control.

    ``POST /run`` arms the intent a checkpoint-less session needs, but the UI
    only renders that button while ``isRerunEnabled()`` is true — i.e. while the
    runtime is unavailable or intentionally kept. An import that carried a
    kernel under a reallocated id has its runtime marked ``complete``, so the
    user's only control is Resume, and Resume never arms anything.
    """

    def _occupy(self, session_id):
        from backend.config import get_session_dir

        get_session_dir(session_id).mkdir(parents=True, exist_ok=True)

    def test_kernel_without_its_checkpoint_arms_the_restore_intent(self, tmp_path):
        from backend.config import DATA_BASE_DIR, get_session_states
        from backend.routes.export_import import _import_tar_gz
        from backend.utils.runtime_apply import _live_pkl, is_forward_resume_blocked

        session_id = str(uuid.uuid4())
        self._occupy(session_id)
        archive = _build_archive(
            tmp_path / "kernel.tar.gz",
            session_id,
            {
                f"{session_id}/states/{session_id}_session.pkl": b"KERNEL",
                f"{session_id}/states/checkpoints.sqlite": b"SQLITE",
            },
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(session_id, current_round=3),
        )

        state = _import_tar_gz(str(archive), user_id="importer", trust=True)

        assert state.id != session_id
        # The kernel came across, so the runtime is fine and Re-run is hidden.
        assert _live_pkl(state.id).exists()
        assert not is_forward_resume_blocked(state.runtime_restore_status)
        # The checkpoint could not, so Resume needs an intent to act on.
        assert not (get_session_states(state.id) / "checkpoints.sqlite").exists()
        assert state.restore_start_intent == "explicit_start"

    def test_a_usable_checkpoint_is_not_diverted(self, tmp_path):
        """id kept + kernel + checkpoint: resume() must continue exactly."""
        from backend.config import DATA_BASE_DIR
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "exact.tar.gz",
            session_id,
            {
                f"{session_id}/states/{session_id}_session.pkl": b"KERNEL",
                f"{session_id}/states/checkpoints.sqlite": b"SQLITE",
            },
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(session_id, current_round=3),
        )

        state = _import_tar_gz(str(archive), user_id="importer", trust=True)

        assert state.id == session_id
        assert getattr(state, "restore_start_intent", None) is None

    def test_a_never_executed_session_is_left_alone(self, tmp_path):
        from backend.config import DATA_BASE_DIR
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "fresh.tar.gz",
            session_id,
            {},
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(session_id, current_round=0),
        )

        state = _import_tar_gz(str(archive), user_id="importer")

        assert getattr(state, "restore_start_intent", None) is None

    @pytest.mark.parametrize(
        "reason", ["llm_failure", "agent_disabled", "marker_db_missing"]
    )
    def test_a_gate_only_a_live_workflow_can_release_is_cleared(self, tmp_path, reason):
        """Otherwise the imported session offers a dialog whose buttons 404.

        ``POST /{id}/llm-failure-action`` is the only exit for these three
        reasons and it resolves them through ``workflows[session_id]``, which no
        import creates. The WebSocket then parks the session in its HITL-hold
        branch and returns *before* registering a workflow — after re-opening the
        dialog — so Retry/Skip/Modify raise ``404 Workflow not found`` forever.
        ``ensure_session_loaded`` clears such a gate on a disk load, but an import
        primes the cache, so nothing else reaches this session.
        """
        from backend.config import DATA_BASE_DIR
        from backend.models.schemas import TaskStatus
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / f"gate-{reason}.tar.gz",
            session_id,
            {},
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(
                session_id,
                current_round=2,
                status=TaskStatus.WAITING_INPUT,
                wait_input_reason=reason,
                llm_failure_info={"node_name": "bioinformatics", "generation": 3},
                agent_disabled_info={"agent": "pathology", "generation": 3},
            ),
        )

        state = _import_tar_gz(str(archive), user_id="importer")

        assert state.wait_input_reason is None
        assert state.llm_failure_info is None
        assert state.agent_disabled_info is None
        assert state.status == TaskStatus.PAUSED

    def test_a_species_gate_is_left_standing(self, tmp_path):
        """Only the workflow-bound gates are cleared; this one has its own route."""
        from backend.config import DATA_BASE_DIR
        from backend.models.schemas import TaskStatus
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "species.tar.gz",
            session_id,
            {},
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(
                session_id,
                current_round=2,
                status=TaskStatus.WAITING_INPUT,
                wait_input_reason="species_input",
            ),
        )

        state = _import_tar_gz(str(archive), user_id="importer")

        assert state.wait_input_reason == "species_input"
        assert state.status == TaskStatus.WAITING_INPUT


class TestTheRuntimeTravelsOnlyWhenItIsTrusted:
    """``X-Trust`` covers the whole live runtime, not just the dill residual.

    The h5ad sidecar set is the other half: a Snapshot V2 restore writes
    ``runtime_h5ad_manifest.json`` plus the ``<name>.h5ad`` files it lists, and the
    sandbox rehydrates from the manifest alone with no pkl at all. It carries no
    code, but it does bind identifier-named globals into the kernel, and the
    ``X-Trust`` contract already says "restore runtime objects (dill/h5ad)".
    """

    def _archive(self, tmp_path, session_id, extra=None):
        from backend.config import DATA_BASE_DIR

        members = {
            f"{session_id}/states/runtime_h5ad_manifest.json":
                b'{"version": 1, "entries": {}}',
            f"{session_id}/states/adata.h5ad": b"H5AD",
        }
        members.update(extra or {})
        return _build_archive(
            tmp_path / f"{session_id[:8]}-runtime.tar.gz",
            session_id,
            members,
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(session_id, current_round=3),
        )

    def test_an_untrusted_import_drops_the_h5ad_sidecars(self, tmp_path):
        from backend.config import get_session_states
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        state = _import_tar_gz(str(self._archive(tmp_path, session_id)), user_id="importer")

        states = get_session_states(state.id)
        assert not (states / "runtime_h5ad_manifest.json").exists()
        assert not (states / "adata.h5ad").exists()
        # No runtime arrived, so the session must say so.
        assert state.runtime_restore_status == "unavailable"

    def test_a_trusted_h5ad_only_runtime_counts_as_a_runtime(self, tmp_path):
        """A "complete" restore target with no dill residual leaves no pkl at all.

        The manifest is then the only carrier of ``adata``, so marking the session
        runtime-less would send a genuine migration through a re-run it does not
        need -- and would drop the checkpoints that still match it.
        """
        from backend.config import get_session_states
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = self._archive(
            tmp_path, session_id,
            {f"{session_id}/states/checkpoints.sqlite": b"SQLITE"},
        )

        state = _import_tar_gz(str(archive), user_id="importer", trust=True)

        states = get_session_states(state.id)
        assert (states / "runtime_h5ad_manifest.json").exists()
        assert (states / "adata.h5ad").exists()
        assert state.runtime_restore_status == "complete"
        assert (states / "checkpoints.sqlite").exists()


class TestImportDoesNotCarryBlockersItCannotAnswer:
    """Retired-subsystem gates travelled in session.json; import must clear them.

    The archive carries blockers whose answering machinery no longer exists, so
    a session imported with them left would sit at ``/run`` 409 and ``/resume``
    refused, pointing the user at snapshot actions that are gone.
    """

    def _blocked_payload(self, session_id):
        from backend.models.schemas import TaskStatus

        return _session_json(
            session_id,
            current_round=2,
            status=TaskStatus.RUNNING,
            wait_input_reason="snapshot_failure",
            snapshot_failure={
                "code": "SNAPSHOT_CAPTURE_FAILED",
                "retry_context": {"producer": "pre_restore", "previous_status": "running"},
            },
            pending_required_snapshot={"phase": "pre_restore"},
            skip_required_snapshot_phase="*",
            recovery_blocked=True,
        )

    def test_the_blockers_are_gone_before_the_session_is_listable(self, tmp_path):
        from backend.config import DATA_BASE_DIR, get_session_metadata_path
        from backend.models.schemas import TaskStatus
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "blocked.tar.gz",
            session_id,
            {},
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=self._blocked_payload(session_id),
        )

        state = _import_tar_gz(str(archive), user_id="importer")

        assert state.snapshot_failure is None
        assert state.pending_required_snapshot is None
        assert state.wait_input_reason is None
        assert state.skip_required_snapshot_phase is None
        assert state.recovery_blocked is False
        assert state.status != TaskStatus.RUNNING
        # The crash-safety write must carry the normalized state, not the archive's:
        # a process that dies before its own save would otherwise leave the gate.
        on_disk = json.loads(
            get_session_metadata_path(state.id).read_text(encoding="utf-8")
        )
        assert on_disk["snapshot_failure"] is None
        assert on_disk["status"] != "running"



class TestImportReportsWhatItDropped:
    """Every drop used to reach only the log, under a modal saying "Import successful"."""

    def _occupy(self, session_id):
        from backend.config import get_session_dir

        get_session_dir(session_id).mkdir(parents=True, exist_ok=True)

    def test_the_summary_reaches_the_caller(self, tmp_path):
        from backend.config import DATA_BASE_DIR
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        self._occupy(session_id)  # force a new id, so the thread-id drop fires too
        archive = _build_archive(
            tmp_path / "dropped.tar.gz",
            session_id,
            {
                f"{session_id}/v2/manifests/abc.json": b"{}",
                f"{session_id}/states/checkpoints.sqlite": b"SQLITE",
                f"{session_id}/states/{session_id}_session.pkl": b"KERNEL",
                f"{session_id}/app.db": b"PWNED",
            },
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(session_id, current_round=3),
        )
        outcome: dict = {}

        _import_tar_gz(str(archive), user_id="importer", outcome=outcome)

        skipped = outcome["skipped"]
        assert skipped["retired_snapshot"] == 1
        assert skipped["untrusted_pickle"] == 1
        assert skipped["off_layout"] == 1
        assert skipped["checkpoint_dropped"] >= 1
        assert outcome["skipped_examples"], "examples must name at least one member"
        assert len(outcome["skipped_examples"]) <= 20

    def test_a_clean_archive_reports_nothing(self, tmp_path):
        from backend.config import DATA_BASE_DIR
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "clean.tar.gz",
            session_id,
            {f"{session_id}/uploads/{session_id}/matrix.h5ad": b"DATA"},
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(session_id, current_round=1),
        )
        outcome: dict = {}

        _import_tar_gz(str(archive), user_id="importer", outcome=outcome)

        assert not any(outcome["skipped"].values())


class TestARefusalNamesTheRealConstraint:
    def test_a_snapshot_only_archive_does_not_blame_the_file(self, tmp_path):
        """Every member of a snapshot export is filtered; nothing can read them.

        The archive is intact; this build is the constraint. "Session state JSON
        not found" read as corruption.
        """
        from backend.config import DATA_BASE_DIR
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        # A snapshot export archive: flat object storage, no session state member.
        archive = _build_archive(
            tmp_path / "v2only.tar.gz",
            session_id,
            {f"{session_id}/objects/{'a' * 64}": b"OBJ"},
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=None,
        )
        # _build_archive always writes a state member; drop it so the archive is
        # genuinely V2-only.
        _strip_member(archive, f"{session_id}/sessions/{session_id}.json")

        with pytest.raises(Exception) as excinfo:
            _import_tar_gz(str(archive), user_id="importer")

        detail = str(getattr(excinfo.value, "detail", excinfo.value))
        assert "Snapshots were removed from this server" in detail
        assert "SCAI_ENABLE_SNAPSHOT_V2" not in detail, (
            "an end-user message must not carry operator env-var wording"
        )


def _strip_member(path: Path, drop: str) -> None:
    """Rewrite *path* without the member named *drop*."""
    kept = []
    with tarfile.open(path, mode="r:gz") as tf:
        for member in tf.getmembers():
            if member.name == drop:
                continue
            extracted = tf.extractfile(member)
            kept.append((member.name, extracted.read() if extracted else b""))
    with tarfile.open(path, mode="w:gz") as tf:
        for name, payload in kept:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))


class TestJsonImportCreatesASession:
    """The .json path was the only one without id allocation.

    It took the id straight from the uploaded document and then stamped the
    caller's user_id onto it, so posting another user's session id handed that
    session over.
    """

    def test_a_colliding_id_does_not_take_over_the_session(self):
        from backend.config import get_session_dir, get_session_metadata_path
        from backend.routes.export_import_parts import import_routes

        victim_id = str(uuid.uuid4())
        get_session_dir(victim_id).mkdir(parents=True, exist_ok=True)
        get_session_metadata_path(victim_id).write_text(
            json.dumps(_session_json(victim_id, name="victim")), encoding="utf-8"
        )

        payload = _session_json(victim_id, name="attacker", current_round=1)
        state = _run_json_import(import_routes, payload, user_id="attacker")

        assert state.id != victim_id, "json import must not reuse an occupied id"
        victim = json.loads(
            get_session_metadata_path(victim_id).read_text(encoding="utf-8")
        )
        assert victim["user_id"] == "owner"
        assert victim["name"] == "victim"

    def test_a_free_id_is_still_kept(self):
        from backend.routes.export_import_parts import import_routes

        session_id = str(uuid.uuid4())
        payload = _session_json(session_id, name="mine")

        state = _run_json_import(import_routes, payload, user_id="owner")

        assert state.id == session_id


def _run_json_import(import_routes, payload, *, user_id):
    """Drive POST /api/session/import with a .json body."""
    import asyncio

    class _Stream:
        def __init__(self, blob):
            self._blob = blob

        def __aiter__(self):
            async def gen():
                yield self._blob
            return gen()

    class _Req:
        def __init__(self, blob, uid):
            self.headers = {
                "x-filename": "session.json",
                "content-length": str(len(blob)),
            }
            self.state = type("S", (), {"user_id": uid})()
            self.client = None
            self._blob = blob

        def stream(self):
            return _Stream(self._blob).__aiter__()

    blob = json.dumps(payload).encode()

    saved = {}

    async def _save(session_id, state, *a, **k):
        saved["state"] = state

    async def _cache(session_id, state):
        saved["state"] = state

    orig_save = import_routes.async_save_session
    orig_cache = import_routes._update_session_cache
    import_routes.async_save_session = _save
    import_routes._update_session_cache = _cache
    try:
        asyncio.run(import_routes.import_state(_Req(blob, user_id), None))
    finally:
        import_routes.async_save_session = orig_save
        import_routes._update_session_cache = orig_cache
    return saved["state"]


class TestImportedStatusIsHonest:
    def test_a_session_exported_mid_run_lands_paused(self, tmp_path):
        """Nothing here owns that run: no workflow, no kernel, no WebSocket.

        The frontend reads the status literally and hides the controls that
        would start one — including the re-run button a runtime-less import
        depends on — so the session looks busy until the user stops it.
        """
        from backend.config import DATA_BASE_DIR
        from backend.models.schemas import TaskStatus
        from backend.routes.export_import import _import_tar_gz

        archive_name = "midrun.tar.gz"
        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / archive_name,
            session_id,
            {},
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(
                session_id, current_round=2, status=TaskStatus.RUNNING
            ),
        )

        state = _import_tar_gz(str(archive), user_id="importer")

        assert state.status == TaskStatus.PAUSED

    @pytest.mark.parametrize(
        "status",
        ["COMPLETED", "FAILED", "STOPPED", "PAUSED", "WAITING_INPUT"],
    )
    def test_other_statuses_survive_the_round_trip(self, tmp_path, status):
        """Only a claimed live run is rewritten; the rest carry their meaning."""
        from backend.config import DATA_BASE_DIR
        from backend.models.schemas import TaskStatus
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        expected = getattr(TaskStatus, status)
        archive = _build_archive(
            tmp_path / f"{status}.tar.gz",
            session_id,
            {},
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(session_id, current_round=2, status=expected),
        )

        state = _import_tar_gz(str(archive), user_id="importer")

        assert state.status == expected


# --- states/ carries its session id in the filename, not in a path segment --


class TestStatesMembersFollowTheDestinationId:
    """Everything under ``states/`` is looked up by ``<session id><suffix>``.

    ``uploads/``, ``workspace/`` and ``reports/`` carry the id as a path segment
    the resolver strips and replaces. ``states/`` does not, so an import that
    reallocated the id used to write these under the archive's name, where
    nothing would ever look for them.
    """

    def _occupy(self, session_id: str) -> None:
        """Make *session_id* unavailable so the import has to allocate a new one."""
        from backend.config import get_session_dir

        get_session_dir(session_id).mkdir(parents=True, exist_ok=True)

    def _archive(self, tmp_path, session_id, members, **kwargs):
        from backend.config import DATA_BASE_DIR

        return _build_archive(
            tmp_path / f"{session_id[:8]}.tar.gz",
            session_id,
            members,
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=_session_json(session_id, current_round=3),
            **kwargs,
        )

    def test_notebook_and_bioinfo_checkpoint_are_renamed(self, tmp_path):
        """The default export carries both, and both used to be lost silently.

        The bioinformatics checkpoint is only *kept* when the kernel it describes
        arrives with it (see the kernel-less case below), so this pins the rename
        on the trusted-runtime import where both survive.
        """
        from backend.config import get_session_states
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        self._occupy(session_id)
        archive = self._archive(tmp_path, session_id, {
            f"{session_id}/states/{session_id}.ipynb": b'{"cells":[]}',
            f"{session_id}/states/{session_id}_bioinfo_checkpoint.json": b"{}",
            f"{session_id}/states/{session_id}_session.pkl": b"KERNEL",
        })

        state = _import_tar_gz(str(archive), user_id="importer", trust=True)

        assert state.id != session_id
        states = get_session_states(state.id)
        assert (states / f"{state.id}.ipynb").read_bytes() == b'{"cells":[]}'
        assert (states / f"{state.id}_bioinfo_checkpoint.json").exists()
        assert not (states / f"{session_id}.ipynb").exists()

    def test_a_bioinfo_checkpoint_without_its_kernel_is_dropped(self, tmp_path):
        """It would make the agent resume a loop whose values no longer exist.

        The file matches neither ``is_sandbox_pickle`` nor ``_is_checkpoint_member``,
        so an ordinary export carries it with no runtime at all. Kept,
        ``has_resumable_checkpoint`` sees a non-completed checkpoint for the round
        the import preserved, and the bioinformatics node restores the old
        conversation and discards the freshly built instruction — including the
        kernel-is-empty notice ``runtime_recompute_pending`` exists to deliver.

        The report checkpoint is a different family and stays: report generation
        resumes from the round results in the state, not from the kernel.
        """
        from backend.config import get_session_states
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = self._archive(tmp_path, session_id, {
            f"{session_id}/states/{session_id}.ipynb": b'{"cells":[]}',
            f"{session_id}/states/{session_id}_bioinfo_checkpoint.json": b"{}",
            f"{session_id}/states/{session_id}_report_checkpoint.json": b"{}",
        })

        state = _import_tar_gz(str(archive), user_id="importer")

        states = get_session_states(state.id)
        assert not (states / f"{state.id}_bioinfo_checkpoint.json").exists()
        assert (states / f"{state.id}_report_checkpoint.json").exists()
        # The notebook is documentation, not runtime state, and must survive.
        assert (states / f"{state.id}.ipynb").exists()

    def test_names_are_left_alone_when_the_id_is_kept(self, tmp_path):
        from backend.config import get_session_states
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = self._archive(tmp_path, session_id, {
            f"{session_id}/states/{session_id}.ipynb": b'{"cells":[]}',
        })

        state = _import_tar_gz(str(archive), user_id="importer")

        assert state.id == session_id
        assert (get_session_states(session_id) / f"{session_id}.ipynb").exists()

    def test_an_unrelated_states_filename_is_not_rewritten(self, tmp_path):
        """Only the id prefix is rebound; other names pass through untouched."""
        from backend.routes.export_import import _rebind_states_member_name

        assert _rebind_states_member_name("notes.txt", "AAA", "BBB") == "notes.txt"
        assert _rebind_states_member_name("AAA.ipynb", "AAA", "BBB") == "BBB.ipynb"
        assert _rebind_states_member_name("AAA.ipynb", "AAA", "AAA") == "AAA.ipynb"
        # The suffix must begin at a separator: a longer id that merely starts
        # with this one is a different session's file, not this one's.
        assert _rebind_states_member_name("AAAd_session.pkl", "AAA", "BBB") == "AAAd_session.pkl"
        assert _rebind_states_member_name("AAA_session.pkl", "AAA", "BBB") == "BBB_session.pkl"

    def test_a_case_variant_checkpoint_is_still_recognised(self):
        """On a case-insensitive volume it would resolve to the real lookup."""
        from backend.routes.export_import import _is_checkpoint_member

        assert _is_checkpoint_member("states/checkpoints.sqlite")
        assert _is_checkpoint_member("states/checkpoints.sqlite-wal")
        assert _is_checkpoint_member("states/CHECKPOINTS.SQLITE")
        assert not _is_checkpoint_member("states/notebook.ipynb")
        assert not _is_checkpoint_member("uploads/checkpoints.sqlite")

    def test_live_pkl_lands_where_the_sandbox_looks_for_it(self, tmp_path):
        """A renamed pkl is the difference between a kernel and a false claim."""
        from backend.routes.export_import import _import_tar_gz
        from backend.utils.runtime_apply import _live_pkl, is_forward_resume_blocked

        session_id = str(uuid.uuid4())
        self._occupy(session_id)
        archive = self._archive(tmp_path, session_id, {
            f"{session_id}/states/{session_id}_session.pkl": b"KERNEL",
        })

        state = _import_tar_gz(str(archive), user_id="importer", trust=True)

        assert state.id != session_id
        assert _live_pkl(state.id).read_bytes() == b"KERNEL"
        # The kernel really is there, so forward resume must not be blocked.
        assert not is_forward_resume_blocked(state.runtime_restore_status)

    def test_a_snapshot_pkl_alone_does_not_claim_a_kernel(self, tmp_path):
        """``is_sandbox_pickle`` also matches per-snapshot ``.pkl.zst`` backups."""
        from backend.routes.export_import import _import_tar_gz
        from backend.utils.runtime_apply import _live_pkl, is_forward_resume_blocked

        session_id = str(uuid.uuid4())
        archive = self._archive(tmp_path, session_id, {
            f"{session_id}/snapshots/snap1_session.pkl.zst": b"ZST",
        })

        state = _import_tar_gz(str(archive), user_id="importer", trust=True)

        assert not _live_pkl(state.id).exists()
        assert state.runtime_restore_status == "unavailable"
        assert is_forward_resume_blocked(state.runtime_restore_status)

    def test_checkpoint_is_dropped_when_the_id_changes(self, tmp_path):
        """Its rows are keyed by the archive's thread_id and cannot be rebound.

        Keeping the file would satisfy the ``has_checkpoint`` test in
        ``lifecycle.run_analysis``, which then withholds the ``explicit_start``
        intent — and resume, finding no row for this thread, executes nothing.
        """
        from backend.config import get_session_states
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        self._occupy(session_id)
        archive = self._archive(tmp_path, session_id, {
            f"{session_id}/states/checkpoints.sqlite": b"SQLITE",
            f"{session_id}/states/checkpoints.sqlite-wal": b"WAL",
        })

        state = _import_tar_gz(str(archive), user_id="importer")

        assert state.id != session_id
        states = get_session_states(state.id)
        assert not (states / "checkpoints.sqlite").exists()
        assert not (states / "checkpoints.sqlite-wal").exists()

    def test_checkpoint_survives_when_the_id_is_kept_and_the_kernel_arrives(self, tmp_path):
        """Both halves are required: the thread id must match AND the pkl must land.

        This is the one combination that yields exact continuation, and the one
        the admin manual promises for a cross-instance runtime migration.
        """
        from backend.config import get_session_states
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = self._archive(tmp_path, session_id, {
            f"{session_id}/states/checkpoints.sqlite": b"SQLITE",
            f"{session_id}/states/{session_id}_session.pkl": b"KERNEL",
        })

        state = _import_tar_gz(str(archive), user_id="importer", trust=True)

        assert state.id == session_id
        assert (get_session_states(session_id) / "checkpoints.sqlite").read_bytes() == b"SQLITE"
        # With both halves present the import must not divert Resume into a
        # fresh round.
        assert state.restore_start_intent is None

    def test_checkpoint_is_dropped_when_the_kernel_does_not_arrive(self, tmp_path):
        """A checkpoint names a graph node; the kernel holds what that node made.

        ``include_runtime=true`` carries the checkpoint while the pkl additionally
        needs ``trust``, so this is what a runtime export imported without trust
        looks like — and on a free id the thread-id drop above does not fire.
        ``resume()`` calls ``aget_state()`` before it ever consults
        ``restore_start_intent``, so a surviving checkpoint replays the source
        machine's graph node against an empty kernel.
        """
        from backend.config import get_session_states
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = self._archive(tmp_path, session_id, {
            f"{session_id}/states/checkpoints.sqlite": b"SQLITE",
            f"{session_id}/states/checkpoints.sqlite-wal": b"WAL",
            f"{session_id}/states/{session_id}_session.pkl": b"KERNEL",
        })

        state = _import_tar_gz(str(archive), user_id="importer", trust=False)

        assert state.id == session_id
        states = get_session_states(session_id)
        assert not (states / "checkpoints.sqlite").exists()
        assert not (states / "checkpoints.sqlite-wal").exists()
        # ...and the session is routed through the fresh-round re-entry instead.
        assert state.restore_start_intent == "explicit_start"


# --- Members the old filters dropped, and members they should have ---------


class TestMemberNameFiltering:
    def test_double_dot_inside_a_filename_is_kept(self, tmp_path):
        """'..' as a substring is an ordinary filename; only a segment escapes."""
        from backend.config import DATA_BASE_DIR, get_session_uploads
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "dots.tar.gz",
            session_id,
            {f"{session_id}/uploads/{session_id}/DEG_A..B.csv": b"gene,lfc\n"},
            data_base_dir=str(DATA_BASE_DIR),
        )

        state = _import_tar_gz(str(archive), user_id="owner")

        assert (get_session_uploads(state.id) / "DEG_A..B.csv").exists()

    def test_traversal_segment_is_skipped(self, tmp_path):
        from backend.config import DATA_BASE_DIR
        from backend.routes.export_import import _import_tar_gz, _is_unsafe_member_name

        assert _is_unsafe_member_name("sid/workspace/../../etc/passwd") is True
        assert _is_unsafe_member_name("/etc/passwd") is True
        assert _is_unsafe_member_name("sid/uploads/DEG_A..B.csv") is False

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "traversal.tar.gz",
            session_id,
            {f"{session_id}/workspace/../../escaped.txt": b"nope"},
            data_base_dir=str(DATA_BASE_DIR),
        )

        _import_tar_gz(str(archive), user_id="owner")

        assert not (DATA_BASE_DIR.parent / "escaped.txt").exists()


# --- Rewriting paths without rewriting prose -------------------------------


class TestPathRemapIsAnchored:
    def test_a_short_data_root_does_not_rewrite_urls_or_other_words(self):
        from backend.routes.export_import import _remap_paths_in_text

        text = json.dumps({
            "file": "/data/sessions/sid/uploads/x.h5ad",
            "url": "https://example.org/data/reference.csv",
            "other": "/dataset/atlas.h5ad",
            "prose": "we stored metadata/ here",
        })

        out = json.loads(_remap_paths_in_text(text, "/data", "/srv/scai", "sid", "sid"))

        assert out["file"] == "/srv/scai/sessions/sid/uploads/x.h5ad"
        assert out["url"] == "https://example.org/data/reference.csv"
        assert out["other"] == "/dataset/atlas.h5ad"
        assert out["prose"] == "we stored metadata/ here"

    def test_a_relative_source_root_is_refused_rather_than_guessed(self):
        from backend.routes.export_import import _remap_paths_in_text

        text = json.dumps({"key": "data_files", "path": "./data/sessions/sid/x"})

        out = _remap_paths_in_text(text, "./data", "/srv/scai", "sid", "sid")

        assert json.loads(out)["key"] == "data_files"

    def test_blob_backed_strings_are_remapped_too(self, tmp_path):
        """Strings >= 1 KB live in blobs/*.b64, which the text remap never saw."""
        from backend.config import DATA_BASE_DIR, get_session_dir
        from backend.routes.export_import import _import_tar_gz
        from backend.utils.blob_store import externalize_blobs

        session_id = str(uuid.uuid4())
        old_root = "/previous-install/data"
        long_code = (
            f"adata = sc.read_h5ad('{old_root}/sessions/{session_id}/uploads/x.h5ad')\n"
            + "# padding " * 200
        )
        payload = _session_json(session_id)
        payload["custom_prompts"] = {"pi": long_code}

        blob_dir = get_session_dir(session_id) / "blobs"
        externalized = externalize_blobs(payload, blob_dir)
        assert long_code not in json.dumps(externalized)

        members = {
            f"{session_id}/blobs/{p.name}": p.read_bytes()
            for p in blob_dir.iterdir()
        }
        import shutil
        shutil.rmtree(get_session_dir(session_id))

        archive = _build_archive(
            tmp_path / "blob.tar.gz",
            session_id,
            members,
            data_base_dir=old_root,
            state_payload=externalized,
        )

        state = _import_tar_gz(str(archive), user_id="owner")

        assert old_root not in state.custom_prompts["pi"]
        assert str(DATA_BASE_DIR) in state.custom_prompts["pi"]


# --- A failed import must not leave a half-session behind ------------------


class TestFailedImportLeavesNothingListable:
    def test_session_json_is_only_written_after_validation(self, tmp_path):
        from backend.config import DATA_BASE_DIR, get_session_metadata_path
        from fastapi import HTTPException
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        broken = _session_json(session_id)
        broken["status"] = "not-a-real-status"
        archive = _build_archive(
            tmp_path / "broken.tar.gz",
            session_id,
            {f"{session_id}/uploads/{session_id}/x.bin": b"data"},
            data_base_dir=str(DATA_BASE_DIR),
            state_payload=broken,
        )

        with pytest.raises(Exception):
            _import_tar_gz(str(archive), user_id="owner")

        assert not get_session_metadata_path(session_id).exists(), (
            "a session.json that AnalysisState cannot load makes an entry that "
            "list_saved_sessions shows but load_session_from_disk refuses"
        )

    def test_the_worker_reports_which_session_it_created(self, tmp_path):
        """Cleanup on cancel/failure needs the id before the function returns."""
        from backend.config import DATA_BASE_DIR
        from backend.routes.export_import import _import_tar_gz

        session_id = str(uuid.uuid4())
        archive = _build_archive(
            tmp_path / "outcome.tar.gz", session_id, {}, data_base_dir=str(DATA_BASE_DIR),
        )
        outcome: dict = {}

        _import_tar_gz(str(archive), user_id="owner", outcome=outcome)

        assert outcome["session_id"] == session_id

    def test_cleanup_of_an_unknown_session_is_a_no_op(self):
        from backend.routes.export_import import _cleanup_partial_session

        _cleanup_partial_session(None)
        _cleanup_partial_session("")


# --- The export side ------------------------------------------------------


class TestExportProducesImportableArchives:
    def test_symlinks_and_hardlinks_become_regular_members(self, tmp_path):
        """Import rejects every link member, so export must not emit one."""
        from backend.config import get_session_workspace
        from backend.routes.export_import import _build_session_tar, _import_tar_gz

        session_id = str(uuid.uuid4())
        workspace = get_session_workspace(session_id)
        workspace.mkdir(parents=True, exist_ok=True)
        real = workspace / "real.txt"
        real.write_bytes(b"payload")
        os.symlink(real, workspace / "link.txt")
        os.link(real, workspace / "hard.txt")
        from backend.config import get_session_metadata_path
        get_session_metadata_path(session_id).write_text(
            json.dumps(_session_json(session_id)), encoding="utf-8"
        )

        archive = tmp_path / "links.tar.gz"
        _build_session_tar(str(archive), session_id)

        with tarfile.open(archive) as tf:
            assert all(m.isfile() for m in tf.getmembers()), [
                (m.name, m.type) for m in tf.getmembers() if not m.isfile()
            ]

        # And the archive is genuinely importable, which is the point.
        import shutil
        from backend.config import get_session_dir
        shutil.rmtree(get_session_dir(session_id))
        state = _import_tar_gz(str(archive), user_id="owner")
        assert (get_session_workspace(state.id) / "link.txt").read_bytes() == b"payload"

    def test_a_file_that_disappears_mid_pack_does_not_fail_the_export(self, tmp_path):
        from backend.routes.export_import import _safe_add_file

        with tarfile.open(tmp_path / "partial.tar.gz", mode="w:gz") as tf:
            assert _safe_add_file(tf, tmp_path / "missing.bin", "sid/uploads/x") is False

    def test_llm_logs_travel_with_the_session(self, tmp_path):
        from backend.config import get_session_logs, get_session_metadata_path
        from backend.routes.export_import import _build_session_tar

        session_id = str(uuid.uuid4())
        logs = get_session_logs(session_id)
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "llm.jsonl").write_text('{"agent": "pi"}\n', encoding="utf-8")
        get_session_metadata_path(session_id).write_text(
            json.dumps(_session_json(session_id)), encoding="utf-8"
        )

        archive = tmp_path / "logs.tar.gz"
        _build_session_tar(str(archive), session_id)

        with tarfile.open(archive) as tf:
            assert f"{session_id}/logs/llm.jsonl" in tf.getnames()

    def test_export_filename_is_capped_in_bytes_not_characters(self):
        from backend.routes.export_import import _build_export_filename

        name = "单细胞数据分析" * 30  # 210 characters, 630 UTF-8 bytes
        filename = _build_export_filename("abcdef12-3456", name)

        assert len(filename.encode("utf-8")) < 255

    def test_endpoints_do_not_travel_with_a_portable_archive(self):
        """A stripped api_key next to a live base_url is the exfiltration shape."""
        from backend.utils.secret_redaction import strip_analysis_state_secrets

        payload = {
            "agent_server_configs": {
                "pi": {"llm_base_url": "https://llm.internal.example/v1", "api_key": "sk-secret"},
            },
            "literature_config": {
                "deep_research_llm_base_url": "https://dr.internal.example/v1",
                "deep_research_api_key": "sk-dr",
            },
        }

        portable = strip_analysis_state_secrets(payload)

        assert portable["agent_server_configs"]["pi"]["api_key"] == ""
        assert portable["agent_server_configs"]["pi"]["llm_base_url"] == ""
        assert portable["literature_config"]["deep_research_api_key"] == ""
        assert portable["literature_config"]["deep_research_llm_base_url"] == ""


def test_runtime_export_contains_only_verified_standalone_checkpoint(tmp_path, monkeypatch):
    from backend.routes.export_import_parts import tar_builders

    session_id = "runtime-archive-checkpoint"
    session_root = tmp_path / session_id
    states = session_root / "states"
    states.mkdir(parents=True)
    source = states / "checkpoints.sqlite"
    connection = sqlite3.connect(source)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("CREATE TABLE checkpoint_values (value TEXT)")
    connection.execute("INSERT INTO checkpoint_values VALUES ('from-wal')")
    connection.commit()
    assert source.with_name("checkpoints.sqlite-wal").exists()

    monkeypatch.setattr(tar_builders, "get_session_states", lambda _sid: states)
    monkeypatch.setattr(tar_builders, "legacy_session_state_paths", lambda *_args: [])
    monkeypatch.setattr(tar_builders, "UPLOAD_TEMP_DIR", tmp_path / "export-tmp")

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tf:
        tar_builders._add_states_to_tar(tf, session_id, include_runtime=True)
    connection.close()

    archive.seek(0)
    with tarfile.open(fileobj=archive, mode="r") as tf:
        names = tf.getnames()
        checkpoint_name = f"{session_id}/states/checkpoints.sqlite"
        assert names.count(checkpoint_name) == 1
        assert f"{session_id}/states/checkpoints.sqlite-wal" not in names
        assert f"{session_id}/states/checkpoints.sqlite-shm" not in names
        payload = tf.extractfile(checkpoint_name).read()

    extracted = tmp_path / "extracted.sqlite"
    extracted.write_bytes(payload)
    check = sqlite3.connect(extracted)
    assert check.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert check.execute("SELECT value FROM checkpoint_values").fetchone()[0] == "from-wal"
    assert check.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    check.close()


def test_runtime_export_fails_if_verified_checkpoint_snapshot_cannot_be_added(
    tmp_path, monkeypatch
):
    from backend.routes.export_import_parts import tar_builders

    session_id = "runtime-archive-checkpoint-add-failure"
    states = tmp_path / session_id / "states"
    states.mkdir(parents=True)
    source = states / "checkpoints.sqlite"
    connection = sqlite3.connect(source)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("CREATE TABLE checkpoint_values (value TEXT)")
    connection.execute("INSERT INTO checkpoint_values VALUES ('from-wal')")
    connection.commit()
    assert source.with_name("checkpoints.sqlite-wal").exists()

    temp_dir = tmp_path / "export-tmp"
    monkeypatch.setattr(tar_builders, "get_session_states", lambda _sid: states)
    monkeypatch.setattr(tar_builders, "legacy_session_state_paths", lambda *_args: [])
    monkeypatch.setattr(tar_builders, "UPLOAD_TEMP_DIR", temp_dir)
    original_add = tar_builders._safe_add_file
    attempted = []

    def reject_checkpoint(tf, path, arcname):
        attempted.append((Path(path), arcname))
        if arcname == f"{session_id}/states/checkpoints.sqlite":
            return False
        return original_add(tf, path, arcname)

    monkeypatch.setattr(tar_builders, "_safe_add_file", reject_checkpoint)
    archive = io.BytesIO()
    with pytest.raises((OSError, RuntimeError), match="checkpoint"):
        with tarfile.open(fileobj=archive, mode="w") as tf:
            tar_builders._add_states_to_tar(tf, session_id, include_runtime=True)
    connection.close()

    checkpoint_name = f"{session_id}/states/checkpoints.sqlite"
    assert any(
        path.parent == temp_dir and arcname == checkpoint_name
        for path, arcname in attempted
    )
    live_paths = {
        source,
        source.with_name("checkpoints.sqlite-wal"),
        source.with_name("checkpoints.sqlite-shm"),
    }
    assert live_paths.isdisjoint(path for path, _arcname in attempted)
    assert f"{session_id}/states/checkpoints.sqlite-wal" not in {
        arcname for _path, arcname in attempted
    }
    assert f"{session_id}/states/checkpoints.sqlite-shm" not in {
        arcname for _path, arcname in attempted
    }
    assert not list(temp_dir.glob(".checkpoint-export-*"))


def test_archive_round_trip_preserves_restore_version_provenance(tmp_path):
    """A restored session keeps its own version and its source version apart."""
    from backend.constants import VERSION
    from backend.routes.export_import import _build_session_tar, _import_tar_gz

    session_id = str(uuid.uuid4())
    archive = tmp_path / "restore-version.tar.gz"
    _build_session_tar(
        str(archive),
        session_id,
        state_payload=_session_json(
            session_id,
            scai_version=VERSION,
            restore_source_scai_version="0.55.0",
        ),
    )

    state = _import_tar_gz(str(archive), user_id="owner")

    assert state.scai_version == VERSION
    assert state.restore_source_scai_version == "0.55.0"
