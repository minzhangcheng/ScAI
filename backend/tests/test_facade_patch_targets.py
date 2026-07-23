"""Guards against patches that a facade/parts split silently turned into no-ops.

When a symbol moves from a facade into its split-out package, the moved code resolves
that symbol from the *new* module's globals. A test that keeps patching the facade then
rebinds a name nobody reads -- it still passes, while exercising the real dependency.
Nothing else in the suite catches that, so it is checked structurally here.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

import pytest

from backend.tests.split_facade import SPLIT_PACKAGES

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parent


def _dotted(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT)
    return str(relative.with_suffix("") if relative.suffix else relative).replace("/", ".")


def _package_dir(package: str) -> Path:
    return REPO_ROOT / package.replace(".", "/")


@functools.lru_cache(maxsize=None)
def _functions_referencing(path: Path, name: str) -> tuple[str, ...]:
    """Functions in ``path`` that read ``name`` as a bare global."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return ()
    return tuple(
        f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.name}()"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id == name
            for sub in ast.walk(node)
        )
    )


def _module_aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> dotted module path, for every import in the file."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _patch_sites() -> list[tuple[Path, int, str, str]]:
    """(test_file, line, facade, attr) for every patch aimed straight at a facade."""
    sites: list[tuple[Path, int, str, str]] = []

    for test_file in sorted(TESTS_DIR.rglob("*.py")):
        if test_file.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(test_file.read_text())
        except (OSError, SyntaxError):
            continue
        aliases = _module_aliases(tree)

        for node in ast.walk(tree):
            # patch("backend.pkg.facade.name", ...) -- but not a deeper Class.attr path,
            # which patches a shared class object and is unaffected by the split.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                facade, _, attr = node.value.rpartition(".")
                if facade in SPLIT_PACKAGES and attr.isidentifier():
                    sites.append((test_file, node.lineno, facade, attr))
                continue

            # monkeypatch.setattr(module, "name", ...) / patch.object(module, "name", ...)
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            call = node.func
            is_setattr = call.attr == "setattr" and isinstance(call.value, ast.Name)
            is_patch_object = call.attr == "object" and isinstance(call.value, ast.Name)
            if not (is_setattr or is_patch_object) or len(node.args) < 2:
                continue
            target, attr_node = node.args[0], node.args[1]
            if not (isinstance(target, ast.Name) and isinstance(attr_node, ast.Constant)):
                continue
            if not isinstance(attr_node.value, str):
                continue
            if aliases.get(target.id) in SPLIT_PACKAGES:
                sites.append((test_file, node.lineno, aliases[target.id], attr_node.value))

    return sorted(set(sites))


def _derived_split_packages() -> dict[str, str]:
    """Facade -> package pairs discovered from the re-export imports in the source tree.

    A facade marks its re-exports with ``# noqa: F401``, which is what distinguishes a
    split from an ordinary relative import. Deriving the set here means a new split is
    caught whatever it is named, not just the ``*_parts`` convention.
    """
    found: dict[str, str] = {}
    for path in (REPO_ROOT / "backend").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        try:
            source = path.read_text()
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        lines = source.splitlines()
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
                continue
            head = node.module.split(".")[0]
            if not (path.parent / head / "__init__.py").is_file():
                continue
            if "# noqa" not in lines[node.lineno - 1]:
                continue
            found[_dotted(path)] = _dotted(path.parent / head)
    return found


def test_split_packages_registry_is_complete():
    """Every facade/parts split on disk must be registered so the helpers reach it."""
    by_convention = {
        _dotted(path)
        for path in (REPO_ROOT / "backend").rglob("*_parts")
        if path.is_dir() and (path / "__init__.py").exists()
    }
    missing = by_convention - set(SPLIT_PACKAGES.values())
    assert not missing, (
        "split packages missing from SPLIT_PACKAGES in backend/tests/split_facade.py: "
        f"{sorted(missing)}"
    )

    # Catches splits that do not use the *_parts suffix, e.g. pi_agent.py -> pi/.
    derived = _derived_split_packages()
    unregistered = {
        facade: package
        for facade, package in derived.items()
        if SPLIT_PACKAGES.get(facade) != package
    }
    assert not unregistered, (
        "these modules re-export from a sibling package but are not registered in "
        f"SPLIT_PACKAGES: {unregistered}"
    )


@pytest.mark.parametrize("facade,package", sorted(SPLIT_PACKAGES.items()))
def test_registered_splits_exist(facade, package):
    """Catches stale entries, which the completeness check above cannot see."""
    assert (REPO_ROOT / (facade.replace(".", "/") + ".py")).is_file(), f"no facade {facade}"
    assert (_package_dir(package) / "__init__.py").is_file(), f"no package {package}"


def test_no_test_patches_a_name_the_facade_no_longer_resolves():
    """A facade patch is a no-op for code that moved into the split package."""
    sites = _patch_sites()
    moved_by_target = {
        (facade, attr): [
            location
            for module in sorted(_package_dir(SPLIT_PACKAGES[facade]).glob("*.py"))
            for location in _functions_referencing(module, attr)
        ]
        for facade, attr in {(facade, attr) for _, _, facade, attr in sites}
    }

    offenders = [
        f"\n  {test_file.relative_to(REPO_ROOT)}:{line} patches {facade}.{attr}"
        f"\n      but it is read by: {', '.join(moved_by_target[(facade, attr)][:4])}"
        + (
            f" (+{len(moved_by_target[(facade, attr)]) - 4} more)"
            if len(moved_by_target[(facade, attr)]) > 4
            else ""
        )
        for test_file, line, facade, attr in sites
        if moved_by_target[(facade, attr)]
    ]

    assert not offenders, (
        "These patches target a facade, but the code that reads the name lives in the "
        "split-out package, so the patch does nothing and the test silently exercises "
        "the real dependency. Use patch_facade()/setattr_facade() from "
        "backend/tests/split_facade.py instead:" + "".join(offenders)
    )
