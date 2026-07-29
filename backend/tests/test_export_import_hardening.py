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
