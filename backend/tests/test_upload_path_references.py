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
