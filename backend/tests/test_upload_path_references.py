"""§5.3.4: an input reference must survive leaving the machine that made it.

A2-9 removed the workspace path from the executor prompt and that half worked --
the measured run's LLM traffic has zero `/data/sessions/<id>/workspace` mentions.
It has ten `uploads/` ones, from the file manifest, and one of them reached a
notebook cell.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def uploads_root(tmp_path, monkeypatch):
    """Bind a data directory and hand back this session's real uploads root.

    The reference is computed against that root, not against the first
    "/uploads/" in the string, so the test has to know where it actually is.
    """
    def _bind(base: str = "data", session_id: str = "31545a54"):
        import backend.config as config
        import backend.config._config as cfg

        monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / base))
        importlib.reload(cfg)
        importlib.reload(config)
        return session_id, config.get_session_uploads(session_id)

    yield _bind

    import backend.config as config
    import backend.config._config as cfg

    monkeypatch.undo()
    importlib.reload(cfg)
    importlib.reload(config)


def test_an_input_is_named_through_the_uploads_root(uploads_root):
    from backend.agents.bioinformatics.profiling import _upload_reference

    session_id, root = uploads_root()
    reference = _upload_reference(
        str(root / "sequencing/_ungrouped/merged_counts.csv"), session_id,
    )

    assert reference == "_UPLOADS_DIR / 'sequencing/_ungrouped/merged_counts.csv'"
    assert str(root) not in reference


def test_a_data_directory_of_its_own_does_not_split_the_path(uploads_root):
    """The bug this replaced: the first "/uploads/" was not this session's.

    An installation whose data directory already contains one -- /mnt/uploads/scai
    is the shape that found it -- had the path cut at the wrong component, and
    the model was handed `_UPLOADS_DIR / 'scai/sessions/<id>/uploads/...'`, a
    file that does not exist.
    """
    from backend.agents.bioinformatics.profiling import _upload_reference

    session_id, root = uploads_root(base="uploads/scai")
    assert "/uploads/" in str(root.parent.parent.parent)

    reference = _upload_reference(str(root / "sequencing/g1/matrix.h5"), session_id)

    assert reference == "_UPLOADS_DIR / 'sequencing/g1/matrix.h5'"


def test_a_path_outside_the_uploads_root_is_reported_as_it_is(uploads_root):
    """Mislabelling it would be worse than naming it.

    A staged or externally mounted input is not under the session's uploads
    directory, and rewriting it to `_UPLOADS_DIR / ...` would produce a cell
    that fails at read time instead of one that is merely unportable.
    """
    from backend.agents.bioinformatics.profiling import _upload_reference

    session_id, _root = uploads_root()
    assert _upload_reference("/mnt/shared/atlas.h5ad", session_id) == "/mnt/shared/atlas.h5ad"
    assert _upload_reference("", session_id) == ""


def test_the_listing_no_longer_hands_over_absolute_upload_paths(uploads_root):
    from types import SimpleNamespace

    from backend.agents.bioinformatics.profiling import _file_listing

    session_id, root = uploads_root(session_id="abc")
    state = SimpleNamespace(
        id=session_id,
        data_files=[
            SimpleNamespace(
                filename="matrix.h5",
                data_type="scRNA-seq",
                section="sequencing",
                group_id="g1",
                file_path=str(root / "sequencing/g1/matrix.h5"),
            )
        ],
        get_group_by_id=lambda _gid: SimpleNamespace(name="Con"),
    )

    listing = _file_listing(state)

    assert "_UPLOADS_DIR / 'sequencing/g1/matrix.h5'" in listing
    assert str(root) not in listing
    # And the instruction rides with the listing, where the model reads it.
    assert "_UPLOADS_DIR" in listing.splitlines()[0]


def test_the_bootstrap_binds_the_uploads_root():
    """It has to be in the notebook, or a cell written against it does not run.

    A2-9 made the bootstrap cells 0 and 1, so a name bound there travels with
    the notebook. Overridable by the same mechanism as the other roots, so a
    replay can point it at its own copy of the inputs.
    """
    from backend.utils.sandbox_parts.kernel_bootstrap import build_path_setup_source

    class _Sandbox:
        class _NotebookState:
            state_dir = "/data/sessions/abc/states"

        working_dir = __import__("pathlib").Path("/data/sessions/abc/workspace")
        notebook_state = _NotebookState()
        _session_tmp = "/tmp/scai-abc"

    source = build_path_setup_source(_Sandbox())

    assert "_UPLOADS_DIR = Path(" in source
    assert "SCAI_REPLAY_UPLOADS_DIR" in source
    assert "_WORKSPACE_DIR.parent / 'uploads'" in source
