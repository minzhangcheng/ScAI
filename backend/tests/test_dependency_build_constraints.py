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
