"""S4: imported archives must NOT restore sandbox pickle (dill.load = RCE)."""
from backend.routes.export_import import is_sandbox_pickle


def test_matches_bare_session_pkl():
    assert is_sandbox_pickle("session.pkl") is True


def test_matches_prefixed_session_pkl():
    assert is_sandbox_pickle("abc-123_session.pkl") is True


def test_matches_with_path_prefix():
    assert is_sandbox_pickle("states/abc_session.pkl") is True
    assert is_sandbox_pickle("sess/session.pkl") is True


def test_does_not_match_other_files():
    assert is_sandbox_pickle("state.json") is False
    assert is_sandbox_pickle("notebook.ipynb") is False
    assert is_sandbox_pickle("session.pkl.txt") is False
    assert is_sandbox_pickle("my_session.pklx") is False


def test_blocks_compressed_snapshot_pkl():
    assert is_sandbox_pickle("abc-123_session.pkl.zst") is True
    assert is_sandbox_pickle("snapshots/foo/bar_session.pkl.zst") is True


def test_allows_unrelated_zst_files():
    assert is_sandbox_pickle("data.zst") is False
    assert is_sandbox_pickle("backup.tar.zst") is False


def test_blocks_nested_pkl_under_snapshot_blob_dir():
    # After rglob recursion the export tar may contain nested paths under snapshots/.
    # The S4 skip must match by basename regardless of path depth.
    assert is_sandbox_pickle("snapshots/abc_blobs/inner_session.pkl") is True
    assert is_sandbox_pickle("snapshots/abc_blobs/deep/inner_session.pkl.zst") is True


def test_a_retired_generation_is_still_that_generation():
    """`retire_live_runtime` renames the dump, it does not delete it.

    A results-only export skips the runtime by name and an untrusted import
    refuses it by name, so a predicate that stops at the live spelling let a
    whole retired dump travel in an archive that declares no runtime, and let
    one land on disk past the X-Trust grant that exists to stop exactly that.
    """
    from backend.routes.export_import_parts.core import (
        is_runtime_generation_name,
        is_runtime_h5ad_name,
    )

    for name in (
        "states/abc_session.pkl.quarantine",
        "states/abc_session.pkl.quarantine.3",
        "states/abc_session.pkl.zst.quarantine",
        "states/abc_session.runtime.json.quarantine",
        "states/abc_session.runtime.json.sig.quarantine.1",
    ):
        assert is_runtime_generation_name(name) is True, name
    assert is_runtime_h5ad_name("states/adata.h5ad.quarantine") is True
    assert is_runtime_h5ad_name("states/runtime_h5ad_manifest.json.quarantine.2") is True


def test_the_quarantine_suffix_does_not_widen_the_predicate():
    """Only the retired spelling of a generation file, not anything set aside."""
    from backend.routes.export_import_parts.core import is_runtime_generation_name

    assert is_runtime_generation_name("states/notes.txt.quarantine") is False
    assert is_runtime_generation_name("states/abc_session.pklx.quarantine") is False
    assert is_runtime_generation_name("states/quarantine") is False
