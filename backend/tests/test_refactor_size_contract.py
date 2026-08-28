from pathlib import Path

from backend.tests.refactor_size_contract import scan_python_and_shell_debt


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repository_has_no_oversized_production_or_tooling_code():
    debt = scan_python_and_shell_debt(REPO_ROOT)
    assert debt.oversized_files == frozenset()
    assert debt.long_python == frozenset()
    assert debt.long_shell == frozenset()


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
