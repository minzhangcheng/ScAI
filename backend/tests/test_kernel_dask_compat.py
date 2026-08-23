"""Dask dataframe selection must follow the installed release."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import pytest

from backend.utils.sandbox import LocalJupyterSandbox
from backend.utils.sandbox_parts import kernel as kernel_module


@pytest.mark.parametrize(
    ("release", "expected"),
    (("2024.12.1", "false"), ("2025.1.0", "true"), ("2026.1.1", "true")),
)
def test_dask_query_planning_tracks_installed_release(
    monkeypatch, release, expected
):
    monkeypatch.setattr(
        kernel_module, "package_version", lambda _name: release, raising=False
    )
    assert kernel_module._dask_query_planning_setting() == expected


def test_dask_query_planning_fails_conservatively_for_missing_or_invalid_version(
    monkeypatch,
):
    monkeypatch.setattr(
        kernel_module,
        "package_version",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError("dask")),
        raising=False,
    )
    assert kernel_module._dask_query_planning_setting() == "false"

    monkeypatch.setattr(
        kernel_module, "package_version", lambda _name: "unknown", raising=False
    )
    assert kernel_module._dask_query_planning_setting() == "false"


def test_kernel_launch_environment_uses_version_selector(tmp_path, monkeypatch):
    monkeypatch.setattr(
        kernel_module, "_dask_query_planning_setting", lambda: "true", raising=False
    )
    sandbox = LocalJupyterSandbox(
        session_id="dask-environment-contract",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )

    environment = sandbox._kernel_launch_environment()

    assert environment["DASK_DATAFRAME__QUERY_PLANNING"] == "true"

