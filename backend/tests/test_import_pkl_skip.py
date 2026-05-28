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
