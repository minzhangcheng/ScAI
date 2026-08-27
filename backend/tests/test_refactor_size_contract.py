from pathlib import Path

from backend.tests.refactor_size_contract import (
    BASELINE_LONG_PYTHON,
    BASELINE_LONG_SHELL,
    BASELINE_OVERSIZED_FILES,
    scan_python_and_shell_debt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_new_backend_or_tooling_size_debt():
    debt = scan_python_and_shell_debt(REPO_ROOT)
    assert debt.oversized_files <= BASELINE_OVERSIZED_FILES
    assert debt.long_python <= BASELINE_LONG_PYTHON
    assert debt.long_shell <= BASELINE_LONG_SHELL


def test_shell_scan_ignores_quoted_and_heredoc_braces(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    body = [
        "#!/usr/bin/env bash",
        "long_phase() {",
        "  printf '%s' '}'",
        "  cat <<'PAYLOAD'",
        "}",
        "PAYLOAD",
        *["  true" for _ in range(295)],
        "}",
    ]
    (scripts / "runner").write_text("\n".join(body) + "\n")

    debt = scan_python_and_shell_debt(tmp_path)

    assert debt.long_shell == frozenset({("scripts/runner", "long_phase")})
