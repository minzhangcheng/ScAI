"""Reach a name on a facade module *and* on every module split out of it.

Several large modules were split into a sibling package, keeping the original file as a
facade that re-exports the pieces. A function that moved into one of those modules
resolves its globals from the module it now lives in, so patching only the facade
rebinds a name nobody reads: the test keeps passing while exercising the real
dependency. The helpers here bind the replacement on every module that holds the name,
and `split_source` reads the implementation back out of all of them.

Add an entry to SPLIT_PACKAGES whenever a new facade/parts split lands --
test_facade_patch_targets.py fails when one is missing.
"""

from __future__ import annotations

import ast
import contextlib
import graphlib
import importlib
import importlib.util
import pkgutil
from pathlib import Path
from types import ModuleType
from unittest.mock import DEFAULT
from unittest.mock import patch as _mock_patch

# facade module -> package its implementation was split into
SPLIT_PACKAGES: dict[str, str] = {
    "backend.agents.bioinformatics_agent": "backend.agents.bioinformatics",
    "backend.agents.literature_agent": "backend.agents.literature",
    "backend.agents.pi_agent": "backend.agents.pi",
    "backend.agents.supervisor_agent": "backend.agents.supervisor",
    "backend.models.schemas": "backend.models.schema_parts",
    "backend.routes.chat": "backend.routes.chat_parts",
    "backend.routes.export_import": "backend.routes.export_import_parts",
    "backend.routes.reports": "backend.routes.reports_parts",
    "backend.services.report.variants": "backend.services.report.variants_parts",
    "backend.services.snapshot_v2.restore": "backend.services.snapshot_v2.restore_parts",
    "backend.services.snapshot_v2.runtime": "backend.services.snapshot_v2.runtime_parts",
    "backend.services.snapshot_v2.service": "backend.services.snapshot_v2.service_parts",
    "backend.utils.llm_client": "backend.utils.llm_client_parts",
    "backend.utils.expression_contract": "backend.utils.expression_contract_parts",
    "backend.utils.sandbox": "backend.utils.sandbox_parts",
    "backend.utils.upload": "backend.utils.upload_parts",
    "backend.workflow.agents": "backend.workflow.agent_nodes",
    "backend.workflow.reporting": "backend.workflow.reporting_parts",
}


def part_names(package: ModuleType) -> list[str]:
    """Submodule names of a split package, rejecting shapes the helpers cannot handle.

    Everything here assumes a flat parts package. ``pkgutil.iter_modules`` does not
    recurse, so a sub-package would be seen as its ``__init__`` alone: ``holders`` would
    patch a re-export binding while the implementation underneath went untouched, and
    ``split_source`` would read only the ``__init__`` text -- the silent no-op this
    module exists to prevent. Refuse the shape rather than half-support it.
    """
    names = []
    for info in pkgutil.iter_modules(package.__path__):
        if info.ispkg:
            raise NotImplementedError(
                f"{package.__name__}.{info.name} is a sub-package; the split helpers "
                f"only support a flat parts package. Flatten the split, or teach "
                f"part_names, holders, split_source and reload_order to recurse together."
            )
        names.append(info.name)
    return names


def split_modules(facade: str) -> list[ModuleType]:
    """The facade module followed by every module of the package it was split into.

    A module with no split registered yields just itself.
    """
    modules = [importlib.import_module(facade)]
    package_name = SPLIT_PACKAGES.get(facade)
    if package_name:
        package = importlib.import_module(package_name)
        for name in part_names(package):
            modules.append(importlib.import_module(f"{package_name}.{name}"))
    return modules


def binds(facade: str, attr: str) -> list[ModuleType]:
    """Every module in the split whose namespace holds ``attr``.

    This is what an ABSENCE assertion needs. ``hasattr(facade, attr)`` sees only the
    facade's own namespace plus what it re-exports, so a name reintroduced inside a part
    is invisible and a guard forbidding it passes while the thing is back. An empty list
    means nothing in the split binds the name; use ``holders`` where that is an error.

    ``vars()`` does not distinguish a definition from an import, so a module that merely
    re-exports ``attr`` is reported too -- hence ``binds`` rather than ``defines``. That
    is the direction both callers need: an absence guard must fail on a re-export (the
    name is reachable again either way), and the patching helpers must rebind every
    namespace some caller might resolve the name through. Do not use this to find where
    something is *implemented*.
    """
    return [module for module in split_modules(facade) if attr in vars(module)]


def holders(facade: str, attr: str) -> list[ModuleType]:
    """``binds``, refusing to come back empty.

    Patching nothing is the silent failure this module exists to prevent, so the
    patching helpers below need the raising form.
    """
    found = binds(facade, attr)
    if not found:
        raise AttributeError(f"{attr!r} is not bound in {facade} or its split package")
    return found


def split_source(facade: str) -> str:
    """Concatenated source of the facade and every module split out of it.

    For structural tests that assert on implementation source: the facade alone is now
    just a re-export header, so reading it would silently assert against nothing. The
    package's own ``__init__.py`` is included too -- ``pkgutil`` does not report it, and
    a source scan that skipped it would quietly miss anything defined there.
    """
    modules = split_modules(facade)
    package_name = SPLIT_PACKAGES.get(facade)
    if package_name:
        modules.insert(1, importlib.import_module(package_name))
    # Force a line boundary between files: a module whose last line lacks a newline
    # would otherwise be fused onto the next module's first line, silently corrupting
    # the regex and ast.parse assertions built on this text.
    sources = (Path(module.__file__).read_text() for module in modules)
    return "".join(src if src.endswith("\n") else src + "\n" for src in sources)


def _sibling_imports(package_name: str, module_name: str, source: str) -> set[str]:
    """Names of same-package modules that ``source`` imports from.

    Parts reach a sibling as ``from .other import x`` but also as
    ``from ..package.other import x``, so every relative level has to be resolved --
    reading only ``level == 1`` would silently miss the second form.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            target = importlib.util.resolve_name(
                "." * node.level + (node.module or ""), f"{package_name}.{module_name}"
            )
        else:
            target = node.module or ""
        if target.startswith(f"{package_name}."):
            sibling = target[len(package_name) + 1:].split(".")[0]
            if sibling != module_name:
                found.add(sibling)
    return found


def reload_order(package: ModuleType) -> list[str]:
    """Submodule names of ``package``, every module after the ones it imports from.

    ``pkgutil`` yields alphabetically, which is not a reload order: reloading a module
    before a sibling it imports from re-binds that sibling's *old* objects, so the
    module comes out of the reload still stale. Sort by the import graph instead.

    Raises ``graphlib.CycleError``, naming the cycle, if the parts import each other
    circularly -- no reload order is correct then, and returning one anyway would hand
    back stale objects with no signal at the call site.
    """
    root = Path(package.__path__[0])
    names = part_names(package)
    deps = {
        name: _sibling_imports(package.__name__, name, (root / f"{name}.py").read_text())
        & set(names)
        for name in names
    }
    return list(graphlib.TopologicalSorter(deps).static_order())


def reload_split(facade: str) -> ModuleType:
    """Reload a facade *and* every module of its split package, parts first.

    Tests that re-read environment-derived constants call importlib.reload on a module.
    After a split those constants live in the package, so reloading the facade alone
    re-runs nothing but the re-export header and the stale values survive.

    Parts are reloaded in dependency order (see ``reload_order``), then the package
    ``__init__``, then the facade, so nothing is rebuilt from pre-reload objects.
    """
    package_name = SPLIT_PACKAGES.get(facade)
    if package_name:
        package = importlib.import_module(package_name)
        # Parts before the package: reloading __init__ re-runs its `from .part import name`
        # lines, so doing it first would bind the objects the parts are about to replace,
        # and a facade importing through the package (the mixin splits do) would then
        # compose its class from stale pieces.
        for name in reload_order(package):
            importlib.reload(importlib.import_module(f"{package_name}.{name}"))
        importlib.reload(package)
    return importlib.reload(importlib.import_module(facade))


def source_for_path(relative_path: str) -> str:
    """Source behind a repo-relative path, following the split if there is one.

    A module that was split keeps only a re-export header at its original path, so a
    structural test that read that file would quietly assert against nothing.
    """
    module = relative_path.removesuffix(".py").replace("/", ".")
    if module in SPLIT_PACKAGES:
        return split_source(module)
    return (Path(__file__).resolve().parents[2] / relative_path).read_text(encoding="utf-8")


@contextlib.contextmanager
def patch_facade(facade: str, attr: str, new=DEFAULT, **mock_kwargs):
    """Bind one replacement for ``attr`` across the whole split, mock.patch style.

    Pass ``new`` to bind a concrete value (a path, a stub function), or omit it to get
    whatever ``mock.patch`` would have made -- an ``AsyncMock`` for a coroutine
    function, a ``MagicMock`` otherwise. The first holder's patch creates the
    replacement and the rest reuse it, so call assertions hold no matter which module
    invoked it.
    """
    modules = holders(facade, attr)
    with contextlib.ExitStack() as stack:
        replacement = stack.enter_context(
            _mock_patch.object(modules[0], attr, new, **mock_kwargs)
        )
        for module in modules[1:]:
            stack.enter_context(_mock_patch.object(module, attr, replacement))
        yield replacement


def setattr_facade(monkeypatch, facade: str, attr: str, value) -> None:
    """monkeypatch.setattr equivalent that covers the whole split."""
    for module in holders(facade, attr):
        monkeypatch.setattr(module, attr, value)
