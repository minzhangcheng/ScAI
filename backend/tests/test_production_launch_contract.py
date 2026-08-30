"""Static contracts for the single-port production launch path."""

import ast
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_backend_direct_entry_uses_production_port_3000():
    tree = ast.parse((ROOT / "backend" / "main.py").read_text(encoding="utf-8"))
    direct_runs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "uvicorn"
        and node.func.attr == "run"
    ]

    assert len(direct_runs) == 1
    port = next(
        keyword.value
        for keyword in direct_runs[0].keywords
        if keyword.arg == "port"
    )
    assert isinstance(port, ast.Constant)
    assert port.value == 3000


def test_docker_runtime_has_no_node_toolchain_and_uses_port_3000():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    stages = dockerfile.split("FROM ")
    assert stages[1].startswith("node:20.19-")
    runtime = "FROM " + stages[-1]

    assert runtime.startswith("FROM python:3.11-slim")
    assert "nodejs" not in runtime
    assert "curl build-essential pkg-config gosu" in runtime
    user_creation = runtime.index("RUN useradd -m -u 10001 scai")
    source_copy = runtime.index("COPY --chown=scai:scai . .")
    frontend_copy = runtime.index(
        "COPY --from=frontend-build --chown=scai:scai"
    )
    assert user_creation < source_copy < frontend_copy
    assert "mkdir -p /app" in runtime
    assert "chown scai:scai /app" in runtime
    assert "chown -R scai:scai /app" not in runtime
    assert "127.0.0.1:3000/api/health" in runtime
    assert "EXPOSE 3000" in runtime

    entrypoint = (ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8")
    assert "npm install" not in entrypoint
    assert "npm run build" not in entrypoint
    assert "npx vite" not in entrypoint
    assert "vite preview" not in entrypoint
    assert "backend.main:app --host 0.0.0.0 --port 3000" in entrypoint


def test_native_production_is_single_process_and_development_keeps_vite_proxy():
    start = (ROOT / "start.sh").read_text(encoding="utf-8")
    lock = json.loads(
        (ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8")
    )
    locked_vite = lock["packages"]["node_modules/vite"]

    assert "npx vite --host" in start
    assert "vite preview" not in start
    assert 'BACKEND_PORT="$PORT"' in start
    assert "npm ci" in start
    assert "package-lock.json" in start
    assert "urllib.request.urlopen" in start
    assert "APP_PORT=$PORT" in start
    assert "BACKEND_PORT=$BACKEND_PORT" in start
    assert "SCAI_PROBE_HOST" in start
    assert 'validate_port "--port" "$PORT"' in start
    assert locked_vite["version"].split(".", 1)[0] == "8"
    assert locked_vite["engines"]["node"] == "^20.19.0 || >=22.12.0"
    assert 'VITE_MAJOR="${VITE_VERSION%%.*}"' in start
    assert '(major === 20 && minor >= 19)' in start
    assert '(major === 22 && minor >= 12)' in start
    assert 'VITE_MAJOR" != "8"' in start

    for relative_path in (
        "README.md",
        "docs/admin-manual.md",
    ):
        documentation = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "Node.js 20.19+ or 22.12+" in documentation
        assert "Node.js 18.x or 20+" not in documentation
    localized_documentation = (ROOT / "docs/admin-manual-zh.md").read_text(
        encoding="utf-8"
    )
    assert "Node.js 20.19+ 或 22.12+" in localized_documentation
    assert "Node.js 18.x" not in localized_documentation

    stop = (ROOT / "stop.sh").read_text(encoding="utf-8")
    assert "APP_PORT=" in stop
    assert "BACKEND_PORT=" in stop
    assert 'release_port "$APP_PORT"' in stop


def test_frontend_manifest_uses_one_supported_patched_toolchain():
    package = json.loads(
        (ROOT / "frontend" / "package.json").read_text(encoding="utf-8")
    )
    dependencies = package["dependencies"]
    dev_dependencies = package["devDependencies"]

    assert package["engines"]["node"] == "^20.19.0 || >=22.12.0"
    assert dependencies["react-router-dom"].startswith("^7.18.")
    assert dependencies["react-syntax-highlighter"].startswith("^16.")
    assert "recharts" not in dependencies
    assert dev_dependencies["vite"].startswith("^8.")
    assert dev_dependencies["@vitejs/plugin-react"].startswith("^6.")
    assert dev_dependencies["vitest"].startswith("^4.")
    assert dev_dependencies["@types/node"].startswith("^20.19.")


def test_public_client_examples_default_to_production_port_3000():
    manager = (ROOT / "session_manager.py").read_text(encoding="utf-8")
    client_module = (
        ROOT / "clients" / "python" / "scai_client" / "client.py"
    ).read_text(encoding="utf-8")
    client_init = (
        ROOT / "clients" / "python" / "scai_client" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert 'default="http://localhost:3000"' in manager
    assert "localhost:8000" not in manager
    assert "localhost:8000" not in client_module
    assert "localhost:8000" not in client_init

    for relative_path in ("docs/admin-manual.md", "docs/admin-manual-zh.md"):
        documentation = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "http://localhost:3000" in documentation
        # Port 8000 remains documented only as the native development backend.
        assert "development mode only" in documentation or "仅开发模式" in documentation


def test_agent_guides_use_a_single_port_production_command():
    production_command = "./start.sh --prod --host 0.0.0.0 --port 3000"
    for relative_path in ("AGENTS.md", "CLAUDE.md"):
        guide = (ROOT / relative_path).read_text(encoding="utf-8")
        assert production_command in guide
        assert "./start.sh --dev" in guide
        assert "--port 3000 --backport 8000" not in guide


def test_legacy_node_modules_symlink_is_repointed_without_deleting_target(tmp_path):
    frontend = tmp_path / "frontend"
    old_modules = tmp_path / "dep" / "node_modules"
    expected_modules = tmp_path / "dep" / "frontend" / "node_modules"
    frontend.mkdir()
    old_modules.mkdir(parents=True)
    expected_modules.mkdir(parents=True)
    sentinel = old_modules / "preserved.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (frontend / "node_modules").symlink_to(old_modules, target_is_directory=True)

    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "link_frontend_node_modules.sh"),
            str(frontend),
            str(expected_modules),
        ],
        check=True,
    )

    assert (frontend / "node_modules").resolve() == expected_modules.resolve()
    assert sentinel.read_text(encoding="utf-8") == "keep"
