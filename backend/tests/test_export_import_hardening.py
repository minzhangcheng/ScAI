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


# --- Importing is creating a session, never overwriting one -----------------


# --- An imported session must land in a status this installation can back up -


# --- states/ carries its session id in the filename, not in a path segment --


# --- Members the old filters dropped, and members they should have ---------


# --- Rewriting paths without rewriting prose -------------------------------


# --- A failed import must not leave a half-session behind ------------------


# --- The export side ------------------------------------------------------


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


class TestImportStaysInsideTheSession:


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


class TestFailedImportLeavesNothingListable:
    def test_session_json_is_only_written_after_validation(self, tmp_path):
        from backend.config import DATA_BASE_DIR, get_session_metadata_path
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
