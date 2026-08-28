"""Executable size-debt inventory for behavior-preserving decomposition."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


MAX_FILE_LINES = 1_000
MAX_FUNCTION_LINES = 300

_PYTHON_ROOTS = ("backend", "clients")
_SOURCE_SUFFIXES = frozenset({".py", ".js", ".jsx", ".ts", ".tsx"})
_SCRIPT_SUFFIXES = frozenset({"", ".py", ".sh"})
_SHELL_FUNCTION = re.compile(
    r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(\)[ \t]*\{"
)
_HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")


@dataclass(frozen=True)
class FunctionDebt:
    path: str
    qualified_name: str
    first_line: int
    lines: int


@dataclass(frozen=True)
class SizeDebt:
    oversized_files: frozenset[str]
    long_python: frozenset[tuple[str, str]]
    long_shell: frozenset[tuple[str, str]]


def _is_frontend_test(path: Path) -> bool:
    name = path.name
    return (
        "__tests__" in path.parts
        or ".test." in name
        or ".spec." in name
        or name.endswith(".d.ts")
        or name in {"test-setup.ts", "test-setup.tsx", "setupTests.ts"}
    )


def _source_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory in _PYTHON_ROOTS:
        base = root / directory
        if not base.is_dir():
            continue
        paths.extend(
            path
            for path in base.rglob("*.py")
            if "__pycache__" not in path.parts
            and not (directory == "backend" and "tests" in path.relative_to(base).parts)
        )

    frontend = root / "frontend" / "src"
    if frontend.is_dir():
        paths.extend(
            path
            for path in frontend.rglob("*")
            if path.is_file()
            and path.suffix in _SOURCE_SUFFIXES
            and not _is_frontend_test(path)
        )

    scripts = root / "scripts"
    if scripts.is_dir():
        paths.extend(
            path
            for path in scripts.iterdir()
            if path.is_file()
            and path.suffix in _SCRIPT_SUFFIXES
            and not path.name.endswith((".lock", ".lock.json"))
        )
    return sorted(set(paths))


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.scope: list[str] = []
        self.functions: list[FunctionDebt] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified_name = ".".join((*self.scope, node.name))
        end_line = node.end_lineno or node.lineno
        self.functions.append(
            FunctionDebt(
                path=self.relative_path,
                qualified_name=qualified_name,
                first_line=node.lineno,
                lines=end_line - node.lineno + 1,
            )
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def _python_function_debt(root: Path, path: Path) -> list[FunctionDebt]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    collector = _FunctionCollector(path.relative_to(root).as_posix())
    collector.visit(tree)
    return collector.functions


def _shell_code_and_heredocs(line: str) -> tuple[str, list[tuple[str, bool]]]:
    """Return unquoted shell syntax and heredocs introduced by one source line."""
    code: list[str] = []
    heredocs: list[tuple[str, bool]] = []
    single = False
    double = False
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and not single:
            escaped = True
            index += 1
            continue
        if char == "'" and not double:
            single = not single
            index += 1
            continue
        if char == '"' and not single:
            double = not double
            index += 1
            continue
        if char == "#" and not single and not double:
            previous = line[index - 1] if index else " "
            if previous.isspace():
                break
        if not single and not double and line.startswith("<<", index):
            match = _HEREDOC.match(line, index)
            if match is not None:
                heredocs.append((match.group(3), bool(match.group(1))))
        if not single and not double:
            code.append(char)
        index += 1
    return "".join(code), heredocs


def _shell_function_debt(root: Path, path: Path) -> list[FunctionDebt]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    relative = path.relative_to(root).as_posix()
    functions: list[FunctionDebt] = []
    index = 0
    while index < len(lines):
        match = _SHELL_FUNCTION.match(lines[index])
        if match is None:
            index += 1
            continue

        first_line = index + 1
        depth = 0
        heredocs: list[tuple[str, bool]] = []
        while index < len(lines):
            line = lines[index]
            if heredocs:
                delimiter, strip_tabs = heredocs[0]
                candidate = line.lstrip("\t") if strip_tabs else line
                if candidate == delimiter:
                    heredocs.pop(0)
                index += 1
                continue

            syntax, introduced = _shell_code_and_heredocs(line)
            depth += syntax.count("{") - syntax.count("}")
            heredocs.extend(introduced)
            index += 1
            if depth == 0:
                functions.append(
                    FunctionDebt(
                        path=relative,
                        qualified_name=match.group(1),
                        first_line=first_line,
                        lines=index - first_line + 1,
                    )
                )
                break
    return functions


def _looks_like_shell(path: Path) -> bool:
    if path.suffix == ".sh":
        return True
    try:
        with path.open(encoding="utf-8") as handle:
            first_line = handle.readline()
    except (OSError, UnicodeDecodeError):
        return False
    return first_line.startswith("#!") and ("sh" in first_line or "bash" in first_line)


def scan_python_and_shell_debt(root: Path) -> SizeDebt:
    """Scan current production and executable-tooling size debt."""
    oversized_files: set[str] = set()
    long_python: set[tuple[str, str]] = set()
    long_shell: set[tuple[str, str]] = set()

    for path in _source_paths(root):
        relative = path.relative_to(root).as_posix()
        try:
            line_count = len(path.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError):
            continue
        if line_count > MAX_FILE_LINES:
            oversized_files.add(relative)

        if path.suffix == ".py":
            long_python.update(
                (item.path, item.qualified_name)
                for item in _python_function_debt(root, path)
                if item.lines > MAX_FUNCTION_LINES
            )
        if path.parent == root / "scripts" and _looks_like_shell(path):
            long_shell.update(
                (item.path, item.qualified_name)
                for item in _shell_function_debt(root, path)
                if item.lines > MAX_FUNCTION_LINES
            )

    return SizeDebt(
        oversized_files=frozenset(oversized_files),
        long_python=frozenset(long_python),
        long_shell=frozenset(long_shell),
    )
