"""Regression tests for the Snapshot V2 capacity settings.

Covers the operator-tunable per-session quota and free-space reserve, the
advisory floor, and the usage figures the listing endpoint reports so the UI can
advise a cleanup before the ceiling refuses a capture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

GIB = 1024 ** 3


# ── 1. Defaults ──────────────────────────────────────────────────────────────


def test_default_capacity_is_500gib_quota_and_50gib_reserve():
    from backend.config import default_config

    assert default_config.snapshot_v2_session_quota_bytes == 500 * GIB
    assert default_config.snapshot_v2_free_space_reserve_bytes == 50 * GIB


def test_capacity_defaults_mirror_system_config():
    """The two modules that cannot import config must not drift from it.

    ``SnapshotImportLimits`` takes a duck-typed config and ``proof.py`` holds a
    module-level reserve, so both carry a hand-copied default. A change to
    ``SystemConfig`` that forgets them would leave the import path and the
    runtime-capture path enforcing the old ceiling.
    """
    from backend.config import default_config
    from backend.services.snapshot_v2.object_store import SnapshotImportLimits
    from backend.services.snapshot_v2.runtime_parts.proof import (
        RUNTIME_CAPTURE_FREE_SPACE_RESERVE_BYTES,
    )

    limits = SnapshotImportLimits()
    assert limits.session_quota_bytes == default_config.snapshot_v2_session_quota_bytes
    assert (
        limits.free_space_reserve_bytes
        == default_config.snapshot_v2_free_space_reserve_bytes
    )
    assert (
        RUNTIME_CAPTURE_FREE_SPACE_RESERVE_BYTES
        == default_config.snapshot_v2_free_space_reserve_bytes
    )


def test_from_config_fallbacks_match_the_field_defaults():
    """A config object missing the fields resolves to the same ceilings."""
    from backend.services.snapshot_v2.object_store import SnapshotImportLimits

    class _Bare:
        pass

    limits = SnapshotImportLimits.from_config(_Bare())
    assert limits.session_quota_bytes == 500 * GIB
    assert limits.free_space_reserve_bytes == 50 * GIB


# ── 2. Operator overrides ────────────────────────────────────────────────────
#
# These exercise the parser and the advisory check directly rather than
# reloading the config module. A reload rebinds ``default_config``, and every
# consumer that captured it by reference at import time keeps the old object —
# so a reloading test leaks a different ceiling into the rest of the session
# (it did: it broke ``test_quota_config_injected`` two files away).


def test_operator_can_raise_both_ceilings(monkeypatch):
    from backend.config._config import _parse_capacity_gb_env

    monkeypatch.setenv("SCAI_SNAPSHOT_SESSION_QUOTA_GB", "800")
    monkeypatch.setenv("SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB", "80")

    assert _parse_capacity_gb_env("SCAI_SNAPSHOT_SESSION_QUOTA_GB", 500) == 800 * GIB
    assert (
        _parse_capacity_gb_env("SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB", 50) == 80 * GIB
    )


def test_the_configured_fields_are_wired_to_the_parsed_values():
    """The SystemConfig defaults must be the parsed overrides, not literals."""
    from backend.config import _config as cfg

    assert (
        cfg.SystemConfig().snapshot_v2_session_quota_bytes
        == cfg.SNAPSHOT_SESSION_QUOTA_BYTES
    )
    assert (
        cfg.SystemConfig().snapshot_v2_free_space_reserve_bytes
        == cfg.SNAPSHOT_FREE_SPACE_RESERVE_BYTES
    )


@pytest.mark.parametrize("bad", ["abc", "", "  ", "12.5", "-5", "0"])
def test_an_unusable_override_falls_back_instead_of_refusing_to_start(
    monkeypatch, bad,
):
    """Parsed at import: raising here would make a typo un-bootable."""
    from backend.config._config import _parse_capacity_gb_env

    monkeypatch.setenv("SCAI_SNAPSHOT_SESSION_QUOTA_GB", bad)
    assert _parse_capacity_gb_env("SCAI_SNAPSHOT_SESSION_QUOTA_GB", 500) == 500 * GIB


def test_an_absent_override_uses_the_default(monkeypatch):
    from backend.config._config import _parse_capacity_gb_env

    monkeypatch.delenv("SCAI_SNAPSHOT_SESSION_QUOTA_GB", raising=False)
    assert _parse_capacity_gb_env("SCAI_SNAPSHOT_SESSION_QUOTA_GB", 500) == 500 * GIB


def test_a_quota_below_the_advised_floor_is_warned_about_but_honoured(caplog):
    """The floor is advice, not a clamp: the operator's value still applies."""
    from backend.config._config import (
        SNAPSHOT_SESSION_QUOTA_ADVISORY_MINIMUM_GB,
        warn_if_quota_below_advisory,
    )

    assert SNAPSHOT_SESSION_QUOTA_ADVISORY_MINIMUM_GB == 100

    with caplog.at_level("WARNING"):
        warned = warn_if_quota_below_advisory(50 * GIB)

    assert warned is True
    messages = [r.getMessage() for r in caplog.records]
    assert any("below the advised minimum" in m for m in messages), (
        f"no advisory warning logged; saw {messages}"
    )
    assert any("100 GiB" in m for m in messages)


def test_the_advised_floor_itself_does_not_warn(caplog):
    from backend.config._config import warn_if_quota_below_advisory

    with caplog.at_level("WARNING"):
        assert warn_if_quota_below_advisory(100 * GIB) is False
    assert not [r for r in caplog.records if "advised minimum" in r.getMessage()]


def test_the_shipped_default_does_not_warn(caplog):
    from backend.config._config import (
        SNAPSHOT_SESSION_QUOTA_BYTES,
        warn_if_quota_below_advisory,
    )

    with caplog.at_level("WARNING"):
        assert warn_if_quota_below_advisory(SNAPSHOT_SESSION_QUOTA_BYTES) is False


# ── 3. The usage figures the panel advises from ───────────────────────────────


class _Paths:
    def __init__(self, root: Path):
        self.root = root / "v2"
        self.objects = root / "objects"
        self.staging = root / "staging"
        self.journals = root / "journals"
        self.integrity = root / "integrity"
        self.committed = root / "committed"

    def ensure_dirs(self):
        for d in (
            self.root, self.objects, self.staging,
            self.journals, self.integrity, self.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def v2_paths(tmp_path):
    paths = _Paths(tmp_path)
    paths.ensure_dirs()
    return paths


def _manifest(snapshot_id: str, objects: dict):
    """A minimally valid committed manifest declaring *objects*."""
    from datetime import datetime, timezone

    from backend.services.snapshot_v2.models import (
        IdleCapsule, ObjectRef, SnapshotManifestV2,
    )
    from backend.services.snapshot_v2.runtime import empty_complete_runtime

    stub = ObjectRef(sha256="0" * 64, size=1, media_type="application/json")
    return SnapshotManifestV2(
        snapshot_id=snapshot_id,
        session_id="s1",
        name=snapshot_id,
        phase="manual",
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        analysis_state_ref=stub,
        notebook_ref=stub,
        runtime=empty_complete_runtime(),
        continuation=IdleCapsule(graph_version="2.0"),
        logical_size_bytes=sum(o.size for o in objects.values()),
        physical_added_bytes=0,
        root_hash="c" * 64,
        objects=objects,
    )


def _entry(manifest):
    e = type("E", (), {})()
    e.status = "ok"
    e.manifest = manifest
    e.snapshot_id = manifest.snapshot_id
    return e


class _FakeManifestStore:
    """Enumerates one committed manifest holding a single object of *size*."""

    def __init__(self, size: int, sha: str = "a" * 64):
        self._size = size
        self._sha = sha

    def list_committed_with_diagnostics(self):
        from backend.services.snapshot_v2.models import ObjectRef

        objects = {
            self._sha: ObjectRef(
                sha256=self._sha,
                size=self._size,
                media_type="application/octet-stream",
            )
        }
        return [_entry(_manifest("snap-1", objects))]


def _status(v2_paths, physical: int, quota: int):
    from backend.services.snapshot_v2.management import compute_store_status

    return compute_store_status(
        v2_paths, _FakeManifestStore(physical), quota_bytes=quota,
    )


def test_usage_below_the_threshold_does_not_advise_a_cleanup(v2_paths):
    status = _status(v2_paths, physical=50 * GIB, quota=100 * GIB)
    assert status.quota_bytes == 100 * GIB
    assert status.quota_usage_ratio == pytest.approx(0.5)
    assert status.quota_warning is False


def test_reaching_ninety_percent_advises_a_cleanup(v2_paths):
    """The boundary is inclusive: 90% already warrants the advice."""
    status = _status(v2_paths, physical=90 * GIB, quota=100 * GIB)
    assert status.quota_usage_ratio == pytest.approx(0.9)
    assert status.quota_warning is True


def test_just_under_the_threshold_stays_quiet(v2_paths):
    status = _status(v2_paths, physical=89 * GIB, quota=100 * GIB)
    assert status.quota_warning is False


def test_a_store_over_quota_reports_how_far_over(v2_paths):
    """Not clamped: an operator who lowers the quota must see the real figure."""
    status = _status(v2_paths, physical=150 * GIB, quota=100 * GIB)
    assert status.quota_usage_ratio == pytest.approx(1.5)
    assert status.quota_warning is True


def test_no_known_quota_is_not_a_full_store(v2_paths):
    """A non-positive ceiling must not warn — the advice would be unactionable."""
    status = _status(v2_paths, physical=10 * GIB, quota=0)
    assert status.quota_bytes == 0
    assert status.quota_usage_ratio == 0.0
    assert status.quota_warning is False


def test_the_threshold_compares_physical_not_logical_bytes(v2_paths):
    """A well-deduping lineage must not be warned about early.

    ``_check_quota`` enforces physical bytes, so a manifest set whose logical
    total is far larger than what is on disk is not near its ceiling.
    """
    from backend.services.snapshot_v2.management import compute_store_status
    from backend.services.snapshot_v2.models import ObjectRef

    sha = "b" * 64
    (Path(str(v2_paths.objects)) / sha).write_bytes(b"x" * 1024)

    class _RepeatedStore:
        def list_committed_with_diagnostics(self):
            objects = {
                sha: ObjectRef(
                    sha256=sha, size=1024,
                    media_type="application/octet-stream",
                )
            }
            return [_entry(_manifest(f"snap-{i}", objects)) for i in range(20)]

    status = compute_store_status(
        v2_paths, _RepeatedStore(), quota_bytes=10 * 1024,
    )
    assert status.logical_bytes == 20 * 1024
    assert status.physical_bytes == 1024
    assert status.quota_warning is False, (
        "warned on logical bytes: 20 references to one deduped object are not "
        "20 KB of disk, and the quota that would refuse the capture counts disk"
    )


def test_service_reports_against_its_own_quota_override(v2_paths, tmp_path):
    """A per-instance ceiling must be the one the status measures."""
    from backend.services.snapshot_v2.service import SnapshotService

    service = SnapshotService("s1", v2_paths, session_quota_bytes=200 * GIB)
    service.manifest_store = _FakeManifestStore(190 * GIB)

    status = service.store_status()
    assert status.quota_bytes == 200 * GIB, (
        "the status reported the global default instead of this service's "
        "ceiling, so it would advise a cleanup at the wrong point"
    )
    assert status.quota_warning is True
