"""Regression tests for the Snapshot V2 storage-path defects (group: storage).

One test per defect, named for the failure it prevents rather than for the
function it calls. Every one of these was verified to fail against the tree
before the corresponding fix landed.

Covered:
  F03  the kernel capture templates leave their dataset-sized temporaries bound
       in the kernel's ``__main__`` namespace
  F04  ``atomic_replace_*`` leak their ``.<pid>.staged`` partial on a failed copy
  F13  ``ObjectStore.put_stream`` leaks its staging temp when ``os.replace`` fails
  F15  ``ObjectStore.put_file`` installs an object without proving the bytes it
       copied still hash to the name it stored them under
  F16  the pre-write runtime quota projection dedups against the whole object
       store, so it is zero exactly when it matters
"""

from __future__ import annotations

import errno
import hashlib
import io
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── shared minimal paths object (mirrors test_snapshot_v2_edge_hardening) ────


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


# ── F03. The kernel capture templates unbind what they built ────────────────
#
# These templates run at the kernel's ``__main__`` scope, so every name they
# bind is a kernel global for the life of the kernel. ``_to_write`` is a full
# copy of the captured AnnData and ``_X`` / ``_uns`` / ``_mapping`` / ``_frame``
# are references into it, so leaving any of them bound keeps a whole dataset
# resident. Nothing else reclaims them: ``save_session`` skips ``_``-prefixed
# names, so they are never even persisted, and only a kernel restart — a
# failure path — clears them.


_ANNDATA_TEMPLATE_TEMPORARIES = (
    "_to_write", "_raw_ad", "_raw_varm", "_raw", "_adata",
    "_X", "_uns", "_mapping", "_frame", "_slot", "_cleanup_name",
)


def _exec_anndata_template(adata, h5ad_path, namespace=None):
    """Render and exec the h5ad template in *namespace*, which it returns.

    ``globals()`` inside ``exec(code, namespace)`` resolves to *namespace*, so
    this reproduces the kernel's module-scope binding semantics exactly. The
    caller may pass the dict in so it stays inspectable when the exec raises.
    """
    from backend.services.snapshot_v2.runtime_parts.kernel_templates import (
        _SAVE_ANNDATA_TEMPLATE,
    )

    if namespace is None:
        namespace = {"__builtins__": __builtins__, "probe": adata}
    code = _SAVE_ANNDATA_TEMPLATE.format(
        var_name_repr="probe", h5ad_path_repr=str(h5ad_path),
    )
    exec(code, namespace)
    return namespace


def _assert_no_capture_temporaries(namespace):
    leaked = [n for n in _ANNDATA_TEMPLATE_TEMPORARIES if n in namespace]
    assert leaked == [], (
        f"the h5ad capture template left {leaked} bound in the kernel namespace; "
        f"each is a reference into the sanitized copy, so the kernel carries a "
        f"second full dataset for the rest of its life and every later capture "
        f"peaks at 3x the dataset instead of 2x"
    )


def test_anndata_template_unbinds_its_sanitized_copy(tmp_path):
    anndata = pytest.importorskip("anndata")
    import numpy as np

    adata = anndata.AnnData(np.ones((4, 3)))
    adata.raw = adata  # exercises the _raw_ad / _raw_varm bindings too
    adata.obsm["X_pca"] = np.ones((4, 2))

    namespace = _exec_anndata_template(adata, tmp_path / "o.h5ad")

    assert (tmp_path / "o.h5ad").exists()
    _assert_no_capture_temporaries(namespace)


def test_anndata_template_unbinds_its_copy_when_the_write_fails(tmp_path):
    """The cleanup must be a ``finally``, not a trailing statement.

    A failing ``.write()`` is exactly the case where the kernel can least
    afford to keep the copy: the host is about to retry the capture, which
    builds another one.
    """
    anndata = pytest.importorskip("anndata")
    import numpy as np

    adata = anndata.AnnData(np.ones((4, 3)))
    target = tmp_path / "as_a_directory"
    target.mkdir()  # h5py cannot create a file here, so .write() raises

    namespace = {"__builtins__": __builtins__, "probe": adata}
    with pytest.raises(OSError):
        _exec_anndata_template(adata, target, namespace=namespace)

    _assert_no_capture_temporaries(namespace)


def test_dill_template_unbinds_its_readback_duplicate(tmp_path):
    """``_verified_obj`` is a freshly deserialized copy of the last residual."""
    pytest.importorskip("dill")

    from backend.services.snapshot_v2.runtime_parts.kernel_templates import (
        _SAVE_DILL_TEMPLATE,
    )

    namespace = {"__builtins__": __builtins__, "keeper": {"a": list(range(64))}}
    code = _SAVE_DILL_TEMPLATE.format(
        names_repr=["keeper"], dill_path_repr=str(tmp_path / "residual.pkl"),
    )
    exec(code, namespace)

    leaked = [
        n for n in ("_obj", "_verified_obj", "_verified_name", "_cleanup_name")
        if n in namespace
    ]
    assert leaked == [], (
        f"the dill template left {leaked} bound in the kernel namespace; "
        f"_verified_obj is a second deserialized copy of the residual value"
    )
    # The markers the host parses must survive the cleanup.
    assert namespace["_verified"] == ["keeper"]


# ── F04. atomic_replace_* reclaim their own partial, and orphans get swept ───
#
# The temp is ``<dst>.<pid>.staged``, which neither states-dir sweeper can
# match — both glob ``*.h5ad``. A dataset-sized partial there is attributed to
# no quota (which sums only ``objects/``) and is packed verbatim into every v1
# export archive.


def _dead_pid() -> int:
    """Return a pid that is not running, for the orphan-sweep test."""
    pid = 999983
    for _ in range(4096):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except PermissionError:
            pid += 1
        else:
            pid += 1
    pytest.skip("could not find a dead pid to stand in for a crashed writer")


def test_atomic_replace_h5ad_reclaims_its_partial_when_the_copy_fails(
    tmp_path, monkeypatch,
):
    from backend.config import get_session_states as gss
    from backend.utils.runtime_apply import atomic_replace_h5ad

    session_id = "sess-g-storage-h5ad-enospc"
    states = gss(session_id)
    states.mkdir(parents=True, exist_ok=True)
    for stale in states.glob("*"):
        if stale.is_file():
            stale.unlink()

    live = states / "adata.h5ad"
    live.write_bytes(b"ORIGINAL_LIVE_H5AD")

    src = tmp_path / "src.h5ad"
    src.write_bytes(b"x" * 4096)

    def enospc(source, destination, *a, **kw):
        Path(destination).write_bytes(b"x" * 2048)  # a near-complete partial
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("backend.utils.runtime_apply.shutil.copy2", enospc)

    with pytest.raises(OSError):
        atomic_replace_h5ad(session_id, str(src), var_name="adata")

    leaked = sorted(p.name for p in states.glob("*.staged"))
    assert leaked == [], (
        f"atomic_replace_h5ad leaked {leaked} in the states dir; both sweepers "
        f"glob '*.h5ad' and cannot match it"
    )
    assert live.read_bytes() == b"ORIGINAL_LIVE_H5AD"


def test_atomic_replace_pkl_reclaims_its_partial_when_the_copy_fails(
    tmp_path, monkeypatch,
):
    from backend.config import get_session_states as gss
    from backend.utils.runtime_apply import atomic_replace_pkl

    session_id = "sess-g-storage-pkl-enospc"
    states = gss(session_id)
    states.mkdir(parents=True, exist_ok=True)
    for stale in states.glob("*"):
        if stale.is_file():
            stale.unlink()

    live = states / f"{session_id}_session.pkl"
    live.write_bytes(b"ORIGINAL_LIVE_PKL")

    src = tmp_path / "src.pkl"
    src.write_bytes(b"y" * 4096)

    def enospc(source, destination, *a, **kw):
        Path(destination).write_bytes(b"y" * 2048)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("backend.utils.runtime_apply.shutil.copy2", enospc)

    with pytest.raises(OSError):
        atomic_replace_pkl(session_id, str(src))

    leaked = sorted(p.name for p in states.glob("*.staged"))
    assert leaked == [], f"atomic_replace_pkl leaked {leaked} in the states dir"
    assert live.read_bytes() == b"ORIGINAL_LIVE_PKL"


def test_live_h5ad_sweep_reclaims_a_crashed_writers_partial():
    """A killed process cannot unlink its own temp, so the sweep must.

    The temp of a *live* pid is an in-flight copy — ours or another process's —
    and must survive.
    """
    from backend.config import get_session_states as gss
    from backend.utils.runtime_apply import finalize_live_runtime_h5ad_set

    session_id = "sess-g-storage-sweep"
    states = gss(session_id)
    states.mkdir(parents=True, exist_ok=True)
    for stale in states.glob("*"):
        if stale.is_file():
            stale.unlink()

    live = states / "adata.h5ad"
    live.write_bytes(b"LIVE")
    orphan = states / f"adata.h5ad.{_dead_pid()}.staged"
    orphan.write_bytes(b"partial")
    in_flight = states / f"adata.h5ad.{os.getpid()}.staged"
    in_flight.write_bytes(b"in flight")

    finalize_live_runtime_h5ad_set(session_id, {"adata": "ab" * 32})

    assert not orphan.exists(), (
        "the crashed writer's dataset-sized partial was left in the states dir, "
        "where nothing else will ever reclaim it"
    )
    assert in_flight.exists(), "the sweep unlinked a live process's in-flight temp"
    assert live.read_bytes() == b"LIVE"


# ── F13. put_stream cleans up its staging temp when os.replace fails ─────────


def test_put_stream_does_not_leak_its_staging_temp_on_error(v2_paths, monkeypatch):
    from backend.services.snapshot_v2.object_store import ObjectStore

    data = b"z" * 4096
    sha = hashlib.sha256(data).hexdigest()
    store = ObjectStore(v2_paths)

    real_replace = os.replace

    def boom(a, b):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("backend.services.snapshot_v2.object_store.os.replace", boom)
    with pytest.raises(OSError):
        store.put_stream(
            io.BytesIO(data),
            expected_sha256=sha,
            expected_size=len(data),
            media_type="application/octet-stream",
        )
    monkeypatch.setattr(
        "backend.services.snapshot_v2.object_store.os.replace", real_replace,
    )

    leaked = [p.name for p in v2_paths.staging.iterdir() if p.is_file()]
    assert leaked == [], (
        f"put_stream leaked {leaked} in staging/; recovery's orphan sweep and "
        f"reconcile() both deliberately skip dot-prefixed names and quota sums "
        f"only objects/, so nothing will ever reclaim it"
    )


# ── F15. put_file proves the bytes it copied ────────────────────────────────
#
# put_file hashes the source and then copies it in a second pass. A concurrent
# writer to the workspace (the chat-delegation sandbox holds a different lock
# than the capture) between the two passes gets its bytes installed under the
# earlier hash, i.e. an object that does not hash to its own name.


def _install_mutating_hash(monkeypatch, replacement: bytes):
    from backend.services.snapshot_v2 import object_store as object_store_mod

    real_hash_file = object_store_mod._hash_file

    def racing_hash_file(path):
        digest = real_hash_file(path)
        Path(path).write_bytes(replacement)  # the concurrent writer wins
        return digest

    monkeypatch.setattr(object_store_mod, "_hash_file", racing_hash_file)


@pytest.mark.parametrize("replacement", [
    pytest.param(b"b" * 4096, id="same-length"),
    pytest.param(b"b" * 100, id="different-length"),
])
def test_put_file_refuses_a_source_that_changed_under_it(
    v2_paths, monkeypatch, replacement,
):
    from backend.services.snapshot_v2.errors import SnapshotIntegrityError
    from backend.services.snapshot_v2.object_store import ObjectStore

    original = b"a" * 4096
    sha = hashlib.sha256(original).hexdigest()
    src = v2_paths.root.parent / "racing.bin"
    src.write_bytes(original)

    _install_mutating_hash(monkeypatch, replacement)
    store = ObjectStore(v2_paths)

    with pytest.raises(SnapshotIntegrityError) as excinfo:
        store.put_file(src, "application/octet-stream")
    assert excinfo.value.code == "OBJECT_HASH_MISMATCH"

    assert not (v2_paths.objects / sha).exists(), (
        "put_file installed an object whose bytes do not hash to its own name; "
        "the manifest that names it is committed and HMAC-signed, and only the "
        "restore rehash ever notices"
    )
    leaked = [p.name for p in v2_paths.staging.iterdir() if p.is_file()]
    assert leaked == []


def test_put_file_evidence_size_agrees_with_the_ref_it_returns(v2_paths):
    from backend.services.snapshot_v2.object_store import ObjectStore

    data = b"c" * 2048
    src = v2_paths.root.parent / "stable.bin"
    src.write_bytes(data)

    store = ObjectStore(v2_paths)
    result = store.put_file(src, "application/octet-stream")

    evidence = store._get_evidence(result.ref.sha256)
    assert evidence is not None
    assert evidence.ref.size == result.ref.size == len(data), (
        "the evidence records the post-copy size while the manifest ref carries "
        "the pre-hash size; verify()'s cached fast path compares only sizes, so "
        "a disagreement here is what makes it unsound"
    )


# ── F16. The pre-write runtime projection is not deduped away ───────────────
#
# The head's runtime refs model the bytes the *next* capture will write, not
# those bytes themselves. Deduping them against the object store is always
# vacuous in the steady state — GC marks the head's objects reachable, so they
# are present by construction — which made the 0.36.12 pre-I/O ceiling a no-op
# on exactly the captures it was written to bound.


@pytest.mark.asyncio
async def test_pre_write_estimate_counts_a_runtime_already_in_the_store(v2_paths):
    from backend.services.snapshot_v2.models import ObjectRef, RuntimeManifest
    from backend.services.snapshot_v2.service_parts import capture_support
    from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

    svc = capture_support.CaptureSupportMixin.__new__(
        capture_support.CaptureSupportMixin,
    )
    svc.session_id = "s"
    svc.paths = v2_paths

    # The head's runtime object is physically present, exactly as GC guarantees
    # for every committed lineage head.
    blob = b"big-adata"
    rt_h = hashlib.sha256(blob).hexdigest()
    (v2_paths.objects / rt_h).write_bytes(blob)

    ref = ObjectRef(sha256=rt_h, size=8 * 1024 ** 3, media_type="application/x-hdf5")
    previous = make_snapshot_v2_manifest(
        session_id="s",
        snapshot_id="snap-prev",
        runtime=RuntimeManifest(
            status="complete",
            dependencies={"required": {}, "incidental_skips": []},
            adata_refs={"adata": ref},
            residual_ref=ref,  # same sha twice: must still count once
            captured_residual_names=["results"],
        ),
    )
    svc._resolve_previous_manifest = lambda _state: previous

    request = MagicMock()
    request.session_id = "s"
    request.runtime = None
    # Pin the non-runtime halves to zero so the assertion is about the runtime.
    request.analysis_state_ref = object()
    request.notebook_ref = object()
    request.checkpoint_ref = object()
    sandbox = MagicMock()
    sandbox.km = object()

    estimate = await svc._estimate_capture_addition(
        request, None, MagicMock(), sandbox=sandbox,
    )

    assert estimate == 8 * 1024 ** 3, (
        "the projection deduped the head's runtime against the object store, "
        "where those very objects always are, so the pre-I/O ceiling stayed at "
        "a few MB and an over-quota capture was still only caught after the "
        "kernel had been driven through a full serialization"
    )
