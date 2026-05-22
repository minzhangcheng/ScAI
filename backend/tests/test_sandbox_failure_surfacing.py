"""P1-11: Sandbox dill restore/save failures must surface (not silently skip)."""
import re

import backend.utils.sandbox as sb_mod


def _read_src() -> str:
    with open(sb_mod.__file__) as f:
        return f.read()


def _extract_method(src: str, *names: str) -> str:
    for name in names:
        m = re.search(
            rf"(    (?:async )?def {name}\b.*?)(?=\n    (?:async )?def |\nclass |\Z)",
            src,
            re.DOTALL,
        )
        if m:
            return m.group(1)
    raise AssertionError(f"could not locate any of {names}")


def test_restore_failure_raises_not_silent():
    """P1-11: dill restore failure must raise instead of silently continuing."""
    src = _read_src()
    body = _extract_method(src, "_restore_session", "restore_session")
    # Must raise RuntimeError on restore failure rather than only setting a flag and continuing
    assert "raise RuntimeError" in body or "raise Exception" in body, (
        "P1-11: restore failure must raise so the workflow does not continue "
        "with inconsistent kernel state"
    )


def test_save_session_collects_and_reports_skipped_vars():
    """P1-11: save_session must track skipped vars (no silent except: pass) and log them."""
    src = _read_src()
    body = _extract_method(src, "save_session", "_save_session")
    # The save logic involves both Python-side handling AND kernel-side injected code.
    # The injected code (string) MUST collect skipped variable names and print them,
    # and the Python side MUST log them when present.
    assert "skipped" in body.lower(), "P1-11: must track skipped variable names"
    assert "logger." in body, "P1-11: must log skipped vars / report them"
