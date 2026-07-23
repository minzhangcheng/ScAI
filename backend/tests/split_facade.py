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

import contextlib
import importlib
import pkgutil
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
from unittest.mock import patch as _mock_patch

# facade module -> package its implementation was split into
SPLIT_PACKAGES: dict[str, str] = {
    "backend.agents.bioinformatics_agent": "backend.agents.bioinformatics",
    "backend.agents.literature_agent": "backend.agents.literature",
    "backend.agents.pi_agent": "backend.agents.pi",
    "backend.agents.supervisor_agent": "backend.agents.supervisor",
    "backend.models.schemas": "backend.models.schema_parts",
    "backend.routes.reports": "backend.routes.reports_parts",
    "backend.services.report.variants": "backend.services.report.variants_parts",
    "backend.services.snapshot_v2.restore": "backend.services.snapshot_v2.restore_parts",
    "backend.services.snapshot_v2.runtime": "backend.services.snapshot_v2.runtime_parts",
    "backend.services.snapshot_v2.service": "backend.services.snapshot_v2.service_parts",
    "backend.utils.sandbox": "backend.utils.sandbox_parts",
}


def _split_modules(facade: str) -> list[ModuleType]:
    """The facade module followed by every module of the package it was split into."""
    modules = [importlib.import_module(facade)]
    package_name = SPLIT_PACKAGES.get(facade)
    if package_name:
        package = importlib.import_module(package_name)
        for info in pkgutil.iter_modules(package.__path__):
            modules.append(importlib.import_module(f"{package_name}.{info.name}"))
    return modules


def holders(facade: str, attr: str) -> list[ModuleType]:
    """Every module in the split that binds ``attr`` in its own namespace.

    Raises rather than returning empty: patching nothing is the silent failure this
    module exists to prevent.
    """
    found = [module for module in _split_modules(facade) if attr in vars(module)]
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
    modules = _split_modules(facade)
    package_name = SPLIT_PACKAGES.get(facade)
    if package_name:
        modules.insert(1, importlib.import_module(package_name))
    return "".join(Path(module.__file__).read_text() for module in modules)


@contextlib.contextmanager
def patch_facade(facade: str, attr: str, **mock_kwargs):
    """Bind one MagicMock for ``attr`` across the whole split, mock.patch style.

    Yields the replacement, so call assertions hold no matter which module invoked it.
    """
    replacement = MagicMock(**mock_kwargs)
    with contextlib.ExitStack() as stack:
        for module in holders(facade, attr):
            stack.enter_context(_mock_patch.object(module, attr, replacement))
        yield replacement


def setattr_facade(monkeypatch, facade: str, attr: str, value) -> None:
    """monkeypatch.setattr equivalent that covers the whole split."""
    for module in holders(facade, attr):
        monkeypatch.setattr(module, attr, value)
