"""§5.3.4: an input reference must survive leaving the machine that made it.

A2-9 removed the workspace path from the executor prompt and that half worked --
the measured run's LLM traffic has zero `/data/sessions/<id>/workspace` mentions.
It has ten `uploads/` ones, from the file manifest, and one of them reached a
notebook cell.
"""
from __future__ import annotations


def test_an_input_is_named_through_the_uploads_root():
    from backend.agents.bioinformatics.profiling import _upload_reference

    reference = _upload_reference(
        "/data/sessions/31545a54/uploads/sequencing/_ungrouped/merged_counts.csv"
    )

    assert reference == "_UPLOADS_DIR / 'sequencing/_ungrouped/merged_counts.csv'"
    assert "/data/sessions" not in reference


def test_a_path_outside_the_uploads_root_is_reported_as_it_is():
    """Mislabelling it would be worse than naming it.

    A staged or externally mounted input is not under the session's uploads
    directory, and rewriting it to `_UPLOADS_DIR / ...` would produce a cell
    that fails at read time instead of one that is merely unportable.
    """
    from backend.agents.bioinformatics.profiling import _upload_reference

    assert _upload_reference("/mnt/shared/atlas.h5ad") == "/mnt/shared/atlas.h5ad"
    assert _upload_reference("") == ""


def test_the_listing_no_longer_hands_over_absolute_upload_paths():
    from types import SimpleNamespace

    from backend.agents.bioinformatics.profiling import _file_listing

    state = SimpleNamespace(
        data_files=[
            SimpleNamespace(
                filename="matrix.h5",
                data_type="scRNA-seq",
                section="sequencing",
                group_id="g1",
                file_path="/data/sessions/abc/uploads/sequencing/g1/matrix.h5",
            )
        ],
        get_group_by_id=lambda _gid: SimpleNamespace(name="Con"),
    )

    listing = _file_listing(state)

    assert "_UPLOADS_DIR / 'sequencing/g1/matrix.h5'" in listing
    assert "/data/sessions/abc/uploads" not in listing
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
