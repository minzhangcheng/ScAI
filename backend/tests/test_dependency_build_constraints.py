from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_legacy_build_backend_has_setuptools_constraint():
    build_constraints = (ROOT / "build-constraints.txt").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "setuptools<82" in build_constraints
    assert "setuptools<82" in requirements


def test_all_dependency_installers_apply_build_constraints():
    for relative_path in ("docker-entrypoint.sh", "start.sh"):
        script = (ROOT / relative_path).read_text(encoding="utf-8")
        pip_upgrade = script.index('install -q --upgrade "pip>=25.3,<27"')
        requirements_install = script.index("PIP_ARGS=(install -q -r")

        assert pip_upgrade < requirements_install
        assert "--build-constraint" in script
        assert "PIP_CONSTRAINT=" in script
        assert "build-constraints.txt" in script


def test_native_runtime_matches_docker_python_minor():
    start = (ROOT / "start.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert 'PYTHON_VERSION="3.11"' in start
    assert 'VENV_DIR="$DEP_DIR/venv-py311"' in start
    assert 'UV_PYTHON_INSTALL_DIR="$DEP_DIR/python"' in start
    assert '"UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"' in start
    assert "FROM python:3.11-slim" in dockerfile
    assert "python3.12 -m venv" not in start


def test_scientific_runtime_excludes_known_incompatible_releases():
    requirements = {
        line.split("#", 1)[0].strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    }

    assert "zarr>=2.18.7,<3.0" in requirements
    assert "dask-image>=2024.5.0" in requirements


def test_native_dependency_installer_fails_before_service_startup():
    script = (ROOT / "start.sh").read_text(encoding="utf-8")
    helper = script.index("run_dependency_install()")
    pip_call = script.index('run_dependency_install "pip resolver upgrade"')
    npm_call = script.index('run_dependency_install "locked frontend npm ci"')
    service = script.index('echo "Starting services')

    assert helper < pip_call < npm_call < service
    assert "services were not started" in script
