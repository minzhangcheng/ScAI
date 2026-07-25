"""Guards against patches that a facade/parts split silently turned into no-ops.

When a symbol moves from a facade into its split-out package, the moved code resolves
that symbol from the *new* module's globals. A test that keeps patching the facade then
rebinds a name nobody reads -- it still passes, while exercising the real dependency.
Nothing else in the suite catches that, so it is checked structurally here.
"""

from __future__ import annotations

import ast
import functools
import importlib
from pathlib import Path

import pytest

from backend.tests.split_facade import SPLIT_PACKAGES, part_names, reload_order

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parent


def _dotted(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT)
    return str(relative.with_suffix("") if relative.suffix else relative).replace("/", ".")


def _package_dir(package: str) -> Path:
    return REPO_ROOT / package.replace(".", "/")


def _parse_file(path: Path) -> tuple[str, ast.AST] | None:
    """Source text and parsed tree of ``path``, or None if it cannot be read or parsed.

    One place decides that an unreadable or invalid file is skipped rather than failing
    the scan. Every collector below depends on that policy, and a copy of it that lost
    ``SyntaxError`` would make a whole sweep quietly stop finding offenders -- the exact
    failure mode this module exists to catch. Deliberately not cached: holding the trees
    for all ~500 test files costs ~90 MB to save ~0.2 s.
    """
    try:
        source = path.read_text()
        return source, ast.parse(source)
    except (OSError, SyntaxError):
        return None


@functools.lru_cache(maxsize=None)
def _functions_referencing(path: Path, name: str) -> tuple[str, ...]:
    """Functions in ``path`` that read ``name`` as a bare global."""
    parsed = _parse_file(path)
    if parsed is None:
        return ()
    _, tree = parsed
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
        parsed = _parse_file(test_file)
        if parsed is None:
            continue
        _, tree = parsed
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
        parsed = _parse_file(path)
        if parsed is None:
            continue
        source, tree = parsed
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


def _path_segments(expr: ast.AST, bindings: dict[str, ast.AST]) -> list[str]:
    """Path segments of a filesystem expression, as written.

    Two shapes hide the segments from a naive scan, and both were used by the guards
    this check exists to protect. ``p = ROOT / "agents" / "x.py"`` followed by
    ``p.read_text()`` puts every segment behind an ``ast.Name``, so one level of local
    binding is followed -- ``bindings`` is file-global and last-wins, so a name reused
    for two paths in one file resolves to whichever is assigned last.
    ``__import__("pathlib").Path("backend/x.py")`` splices the module name in as an
    ordinary string constant, so ``__import__`` arguments are dropped -- otherwise the
    candidate reads "pathlib/backend/x.py" and matches nothing.
    """
    if isinstance(expr, ast.Name):
        expr = bindings.get(expr.id)  # one level; a Name bound to a Name yields []
        if expr is None:
            return []

    module_names = {
        arg
        for sub in ast.walk(expr)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "__import__"
        for arg in sub.args
    }
    constants = sorted(
        (
            sub
            for sub in ast.walk(expr)
            if isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and sub not in module_names
        ),
        # ast.walk is breadth-first, so `a / "x" / "y.py"` would otherwise yield the
        # trailing segment before the ones that precede it.
        key=lambda sub: (sub.lineno, sub.col_offset),
    )
    return [part for sub in constants for part in sub.value.split("/") if part]


def _reads_facade(segments: list[str], relative: str) -> bool:
    """True when ``segments`` names ``relative``, comparing from the filename backwards.

    The shorter side must be a suffix of the longer, and at least two segments must
    agree. Anchoring at the front instead would miss a path that starts anywhere other
    than the repo root; matching the filename alone would flag any unrelated reports.py
    or service.py.
    """
    target = relative.split("/")
    shared = min(len(segments), len(target))
    return shared >= 2 and segments[-shared:] == target[-shared:]


def _raw_source_reads() -> list[tuple[Path, int, str]]:
    """(test_file, line, facade) for tests that read a split facade's file as raw text.

    A facade is only a re-export header now, so a structural assertion over its text
    scans almost nothing. A negative assertion -- "this pattern must not appear" -- then
    holds no matter what the split package contains, and the guard dies silently at the
    moment the code it guards moves out. ``split_source``/``source_for_path`` follow the
    split; a bare ``Path(...).read_text()`` does not.

    Scope is the literal-path shape only, and it is reported for every registered facade,
    thin or not: a literal path has a drop-in replacement (``source_for_path`` takes the
    same string), so there is no reason to write one. The ``module.__file__`` /
    ``inspect.getsource(module)`` shape is left to
    ``test_no_test_reads_an_emptied_facade_through_its_module`` below, which reports it
    only once the facade is actually emptied -- every such site today reads a facade that
    still holds its implementation, so reporting them unconditionally would be noise.

    Runtime ``hasattr(facade, name)`` absence checks go blind the same way for the same
    reason; ``split_facade.binds()`` is the fix there and nothing here detects them.
    """
    exempt = {"split_facade.py", Path(__file__).name}
    sites: list[tuple[Path, int, str]] = []

    for test_file in sorted(TESTS_DIR.rglob("*.py")):
        if test_file.name in exempt:
            continue
        parsed = _parse_file(test_file)
        if parsed is None:
            continue
        _, tree = parsed

        bindings: dict[str, ast.AST] = {
            target.id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "read_text",
                "read_bytes",
            }:
                path_expr = node.func.value
            elif isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
                path_expr = node.args[0]
            else:
                continue

            segments = _path_segments(path_expr, bindings)
            if not segments or not segments[-1].endswith(".py"):
                continue
            facade = next(
                (
                    name
                    for name in SPLIT_PACKAGES
                    if _reads_facade(segments, name.replace(".", "/") + ".py")
                ),
                None,
            )
            if facade is not None:
                sites.append((test_file, node.lineno, facade))

    return sorted(set(sites))


def test_no_test_reads_a_split_facade_as_raw_source():
    """Source-scanning guards must follow the split, or they assert against a header."""
    offenders = [
        f"\n  {test_file.relative_to(REPO_ROOT)}:{line} reads {facade.rpartition('.')[2]}.py "
        f"directly, but its implementation lives in {SPLIT_PACKAGES[facade]}"
        for test_file, line, facade in _raw_source_reads()
    ]

    assert not offenders, (
        "These tests read a facade's file as raw text. The facade is a re-export header, "
        "so the assertion scans none of the implementation -- a negative assertion passes "
        "unconditionally. Use split_source()/source_for_path() from "
        "backend/tests/split_facade.py instead:" + "".join(offenders)
    )


@functools.lru_cache(maxsize=None)
def _is_emptied_facade(facade: str) -> bool:
    """True when nothing is left at the facade's own path but re-exports.

    Several splits are partial: bioinformatics_agent.py is still ~3k lines of
    implementation behind its re-export header, so reading *that* file is how a source
    scan reaches the code, not a way of missing it. What makes such a read wrong is the
    day the last function moves out. Absence of any def/class is the signal -- a module
    with none of either has no implementation text left to assert on, so every source
    assertion aimed at it is scanning nothing.

    An unparseable facade counts as not emptied, matching ``_parse_file``: the whole
    backend is broken at that point and every other test says so louder.
    """
    parsed = _parse_file(REPO_ROOT / (facade.replace(".", "/") + ".py"))
    if parsed is None:
        return False
    _, tree = parsed
    return not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def _module_source_reads() -> list[tuple[Path, int, str]]:
    """(test_file, line, facade) for source reads routed through an emptied facade module.

    ``open(mod.__file__)`` and ``inspect.getsource(mod)`` reach the same file a literal
    path would, but the path is not in the source text, so ``_raw_source_reads`` cannot
    see them. They are how nearly every source-scanning test here gets its text, and most
    target modules that were never split, so this reports only the ones whose facade has
    been emptied -- no noise now, a named list of sites to retarget the moment a partial
    split is finished.

    Only the module form matters. ``inspect.getsource(some_function)`` resolves through
    the object's code, which points at whichever part it now lives in, so it follows a
    split for free; the alias table below leaves those unresolved and they drop out.
    """
    exempt = {"split_facade.py", Path(__file__).name}
    sites: list[tuple[Path, int, str]] = []

    for test_file in sorted(TESTS_DIR.rglob("*.py")):
        if test_file.name in exempt:
            continue
        parsed = _parse_file(test_file)
        if parsed is None:
            continue
        _, tree = parsed
        aliases = _module_aliases(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            # open(mod.__file__) / Path(mod.__file__)
            if isinstance(node.func, ast.Name) and node.func.id in {"open", "Path"}:
                target = node.args[0]
                if not (isinstance(target, ast.Attribute) and target.attr == "__file__"):
                    continue
                target = target.value
            # inspect.getsource(mod)
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "getsource":
                target = node.args[0]
            else:
                continue

            if not isinstance(target, ast.Name):
                continue
            facade = aliases.get(target.id)
            if facade in SPLIT_PACKAGES and _is_emptied_facade(facade):
                sites.append((test_file, node.lineno, facade))

    return sorted(set(sites))


def test_no_test_reads_an_emptied_facade_through_its_module():
    """Finishing a partial split must not leave source scans asserting on a header.

    The sibling check above catches a literal path. This catches the same mistake made
    through the module object, which is how nearly every source-scanning test in this
    suite gets its text -- and it stays quiet until the facade it reads actually runs out
    of implementation, so it costs nothing until it matters.
    """
    offenders = [
        f"\n  {test_file.relative_to(REPO_ROOT)}:{line} reads "
        f"{facade.rpartition('.')[2]}.py through its module object, but nothing is left "
        f"there -- the implementation is in {SPLIT_PACKAGES[facade]}"
        for test_file, line, facade in _module_source_reads()
    ]

    assert not offenders, (
        "A split was finished under these tests: the facade they read has no "
        "implementation left, so their source assertions now scan a re-export header. "
        "Switch them to split_source() from backend/tests/split_facade.py:"
        + "".join(offenders)
    )


@pytest.mark.parametrize("facade,package_name", sorted(SPLIT_PACKAGES.items()))
def test_reload_order_can_order_every_registered_split(facade, package_name):
    """Every part must be reloadable dependency-first.

    graphlib supplies the ordering; what this pins is that the packages on disk are a
    shape it can order at all -- no import cycle between parts (CycleError) and no
    sub-package (NotImplementedError). pkgutil yields alphabetically, which is not a
    reload order: llm_client_parts alone has 54 edges pointing the wrong way, and
    reloading in that order leaves modules bound to pre-reload objects.
    """
    package = importlib.import_module(package_name)
    assert sorted(reload_order(package)) == sorted(part_names(package)), (
        f"{package_name}: reload_order dropped or invented a module"
    )
