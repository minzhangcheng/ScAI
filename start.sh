#!/bin/bash

# ScAI - Single Cell Sequencing Analysis Driven by AI
# Startup script

# Default settings
HOST="127.0.0.1"
PORT=3000
BACKPORT=8000

# Default to production mode
MODE="prod"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            MODE="dev"
            shift
            ;;
        --prod)
            MODE="prod"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --backport)
            BACKPORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./start.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev              Start in development mode (default: prod)"
            echo "  --prod             Start in production mode"
            echo "  --host HOST        Listen address (default: 127.0.0.1)"
            echo "  --port PORT        Application port (default: 3000)"
            echo "  --backport PORT    Development backend port (default: 8000)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

validate_port() {
    local label=$1
    local value=$2
    if [[ ! "$value" =~ ^[0-9]+$ ]] \
       || [ "$((10#$value))" -lt 1 ] \
       || [ "$((10#$value))" -gt 65535 ]; then
        echo "Error: $label must be an integer from 1 to 65535 (received: $value)"
        exit 1
    fi
}

validate_port "--port" "$PORT"
validate_port "--backport" "$BACKPORT"
PORT=$((10#$PORT))
BACKPORT=$((10#$BACKPORT))

echo "Starting ScAI - Single Cell Sequencing Analysis Driven by AI ($MODE mode)..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$SCRIPT_DIR/.scai.pid"

# Dependencies directory (outside code directory)
DEP_DIR="$PARENT_DIR/dep"
PYTHON_VERSION="3.11"
VENV_DIR="$DEP_DIR/venv-py311"
FRONTEND_DEP_DIR="$DEP_DIR/frontend"
NODE_MODULES_DIR="$FRONTEND_DEP_DIR/node_modules"

# Data directory (outside code directory)
DATA_DIR="$PARENT_DIR/data"

# Check if already running
if [ -f "$PID_FILE" ]; then
    echo "Warning: Instance already running. Please run ./stop.sh first"
    echo "   or delete $PID_FILE and retry"
    exit 1
fi

# Create the dependency directory before resolving a direct-run interpreter.
# The provider check is deliberately before Node/package parsing, venv creation,
# dependency installation, PID creation, and service startup.
mkdir -p "$DEP_DIR"
echo "Dependencies directory: $DEP_DIR"

python_minor() {
    "$1" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null
}

require_python_minor() {
    local label="$1" interpreter="$2" minor
    [ -x "$interpreter" ] || {
        echo "Error: $label is not an executable Python $PYTHON_VERSION interpreter." >&2
        return 1
    }
    minor="$(python_minor "$interpreter")" || return 1
    [ "$minor" = "$PYTHON_VERSION" ] || {
        echo "Error: $label reports Python $minor; Python $PYTHON_VERSION is required." >&2
        return 1
    }
}

exit_with_python_error() {
    local operation="$1"
    echo "Error: $operation failed; services were not started." >&2
    exit 1
}

PYTHON_BIN=""
if command -v "python$PYTHON_VERSION" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "python$PYTHON_VERSION")"
    require_python_minor "system python$PYTHON_VERSION" "$PYTHON_BIN" || exit 1
elif command -v uv >/dev/null 2>&1; then
    UV_PYTHON_INSTALL_DIR="$DEP_DIR/python"
    mkdir -p "$UV_PYTHON_INSTALL_DIR"
    UV_PYTHON_CMD=(env "UV_CACHE_DIR=$DEP_DIR/uv-cache"
        "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR" uv)
    UV_FIND_ARGS=(python find --no-project --no-python-downloads
        --managed-python --no-config --resolve-links "$PYTHON_VERSION")
    if ! PYTHON_BIN="$("${UV_PYTHON_CMD[@]}" "${UV_FIND_ARGS[@]}" 2>/dev/null)"; then
        "${UV_PYTHON_CMD[@]}" python install --no-bin \
            --install-dir "$UV_PYTHON_INSTALL_DIR" --no-config "$PYTHON_VERSION" \
            || exit_with_python_error "uv Python $PYTHON_VERSION installation"
        PYTHON_BIN="$("${UV_PYTHON_CMD[@]}" "${UV_FIND_ARGS[@]}" 2>/dev/null)" \
            || exit_with_python_error "post-install uv Python lookup"
    fi
    UV_ROOT_REAL="$(realpath -e "$UV_PYTHON_INSTALL_DIR")" \
        || exit_with_python_error "uv Python install-root resolution"
    PYTHON_REAL="$(realpath -e "$PYTHON_BIN")" \
        || exit_with_python_error "uv Python interpreter resolution"
    case "$PYTHON_REAL" in
        "$UV_ROOT_REAL"/*) PYTHON_BIN="$PYTHON_REAL" ;;
        *)
            echo "Error: uv returned an interpreter outside $UV_ROOT_REAL." >&2
            exit 1
            ;;
    esac
    require_python_minor "uv-managed Python" "$PYTHON_BIN" || exit 1
else
    echo "Error: Python $PYTHON_VERSION is required. Install python$PYTHON_VERSION or uv." >&2
    exit 1
fi
echo "Python interpreter: $PYTHON_BIN"

# Production needs Node.js only when the checked-out frontend build is missing
# or stale. Packaged deployments can run the FastAPI-served dist without a
# Node.js runtime; development always needs the Vite server.
NEEDS_FRONTEND_BUILD=0
if [ "$MODE" = "prod" ]; then
    if [ ! -f "$SCRIPT_DIR/frontend/dist/index.html" ]; then
        NEEDS_FRONTEND_BUILD=1
    elif find "$SCRIPT_DIR/frontend" \
        \( -path "$SCRIPT_DIR/frontend/dist" \
           -o -path "$SCRIPT_DIR/frontend/node_modules" \
           -o -path "$SCRIPT_DIR/frontend/.pytest_cache" \
           -o -path "$SCRIPT_DIR/frontend/.benchmarks" \) -prune \
        -o -type f -newer "$SCRIPT_DIR/frontend/dist/index.html" -print -quit \
        | grep -q .; then
        NEEDS_FRONTEND_BUILD=1
    fi
fi

NEEDS_NODE=0
if [ "$MODE" = "dev" ] || [ "$NEEDS_FRONTEND_BUILD" -eq 1 ]; then
    NEEDS_NODE=1
fi

if [ "$NEEDS_NODE" -eq 1 ]; then
    VITE_VERSION="$("$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["packages"]["node_modules/vite"]["version"])' "$SCRIPT_DIR/frontend/package-lock.json" 2>/dev/null)"
    VITE_MAJOR="${VITE_VERSION%%.*}"
    if [ -z "$VITE_VERSION" ] || [ "$VITE_MAJOR" != "8" ]; then
        echo "Error: Unable to determine a supported Node.js range for locked Vite ${VITE_VERSION:-unknown}."
        exit 1
    fi
    if ! command -v node &> /dev/null; then
        echo "Error: Node.js is required for Vite $VITE_VERSION development/builds. Install Node.js 20.19+ or 22.12+."
        exit 1
    fi
    if ! node -e 'const [major, minor] = process.versions.node.split(".").map(Number); const ok = (major === 20 && minor >= 19) || (major === 22 && minor >= 12) || major > 22; process.exit(ok ? 0 : 1)'; then
        echo "Error: Locked Vite $VITE_VERSION requires Node.js 20.19+ or 22.12+ (current: $(node --version))."
        exit 1
    fi
fi

# Create data directory
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/uploads"
mkdir -p "$DATA_DIR/states"
mkdir -p "$DATA_DIR/workspace"
mkdir -p "$DATA_DIR/sessions"
mkdir -p "$DATA_DIR/reports"
echo "Data directory: $DATA_DIR"

# Create virtual environment if it doesn't exist.  A symlink, regular file, or
# other pre-existing non-directory is never followed, repaired, or replaced.
if [ ! -e "$VENV_DIR" ] && [ ! -L "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR" \
        || exit_with_python_error "Python $PYTHON_VERSION virtual environment creation"
elif [ -L "$VENV_DIR" ] || [ ! -d "$VENV_DIR" ] \
     || ! require_python_minor "existing virtual environment" "$VENV_DIR/bin/python"; then
    echo "Error: $VENV_DIR is not a usable Python $PYTHON_VERSION virtual environment." >&2
    echo "Move it aside and retry; start.sh will create a compatible environment." >&2
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install backend dependencies
echo "Installing backend dependencies..."
PIP_CMD=("$VENV_DIR/bin/python" -m pip)
run_dependency_install() {
    local label="$1"
    shift
    if ! "$@"; then
        echo "Error: $label failed; services were not started." >&2
        exit 1
    fi
}

# Upgrade the ensurepip-provided resolver before processing the large
# scientific dependency graph. pip 24.0 can fail with ResolutionTooDeep.
run_dependency_install "pip resolver upgrade" \
    "${PIP_CMD[@]}" install -q --upgrade "pip>=25.3,<27"
PIP_ARGS=(install -q -r "$SCRIPT_DIR/requirements.txt")
if "${PIP_CMD[@]}" install --help 2>/dev/null | grep -q -- '--build-constraint'; then
    run_dependency_install "backend requirements" \
        "${PIP_CMD[@]}" "${PIP_ARGS[@]}" \
        --build-constraint "$SCRIPT_DIR/build-constraints.txt"
else
    run_dependency_install "backend requirements" env \
        "PIP_CONSTRAINT=$SCRIPT_DIR/build-constraints.txt" \
        "${PIP_CMD[@]}" "${PIP_ARGS[@]}"
fi

# Packages whose declared constraints are provably tighter than their code
# requires must be installed after the main graph, without dependencies, or
# pip would downgrade a pin this project deliberately holds. Rationale for
# every entry lives in requirements-nodeps.txt.
if [ -f "$SCRIPT_DIR/requirements-nodeps.txt" ]; then
    run_dependency_install "no-deps requirements" \
        "${PIP_CMD[@]}" install -q --no-deps -r "$SCRIPT_DIR/requirements-nodeps.txt"
fi

if [ "$NEEDS_NODE" -eq 1 ]; then
    # Install the exact lockfile graph in a dependency directory outside the
    # checkout. Both package files are compared so a lock-only security update
    # cannot accidentally build against stale modules.
    NEEDS_NPM_CI=0
    if [ ! -f "$FRONTEND_DEP_DIR/.install-complete" ] \
       || [ ! -d "$NODE_MODULES_DIR" ] \
       || ! cmp -s "$SCRIPT_DIR/frontend/package.json" "$FRONTEND_DEP_DIR/package.json" \
       || ! cmp -s "$SCRIPT_DIR/frontend/package-lock.json" "$FRONTEND_DEP_DIR/package-lock.json"; then
        NEEDS_NPM_CI=1
    fi

    if [ "$NEEDS_NPM_CI" -eq 1 ]; then
        echo "Installing the locked frontend dependency graph..."
        prepare_frontend_inputs() {
            mkdir -p "$FRONTEND_DEP_DIR" \
                && rm -f "$FRONTEND_DEP_DIR/.install-complete" \
                && cp "$SCRIPT_DIR/frontend/package.json" "$FRONTEND_DEP_DIR/package.json" \
                && cp "$SCRIPT_DIR/frontend/package-lock.json" "$FRONTEND_DEP_DIR/package-lock.json"
        }
        run_dependency_install "locked frontend input preparation" prepare_frontend_inputs
        cd "$FRONTEND_DEP_DIR"
        run_dependency_install "locked frontend npm ci" npm ci
        run_dependency_install "locked frontend install marker" \
            touch "$FRONTEND_DEP_DIR/.install-complete"
        cd "$SCRIPT_DIR"
    else
        echo "Locked frontend dependencies are up to date."
    fi

    cd "$SCRIPT_DIR/frontend"
    bash "$SCRIPT_DIR/scripts/link_frontend_node_modules.sh" \
        "$SCRIPT_DIR/frontend" "$NODE_MODULES_DIR"
    cd "$SCRIPT_DIR"
fi

# ── Frontend build for production mode ───────────────────────────
if [ "$MODE" = "prod" ] && [ "$NEEDS_FRONTEND_BUILD" -eq 1 ]; then
    echo "Building frontend for FastAPI static serving..."
    cd "$SCRIPT_DIR/frontend"
    npm run build
    cd "$SCRIPT_DIR"
elif [ "$MODE" = "prod" ]; then
    echo "Frontend build up to date; Node.js is not required at runtime."
fi

# Optional: fetch marker databases for ORA priors (off by default; never blocks startup)
# The script writes to $SCAI_DATA_DIR/markers and returns early when the table is
# already at the current schema revision, so no on-disk check is needed here.
if [ "${SCAI_AUTO_FETCH_MARKERS:-0}" = "1" ]; then
    echo "Fetching marker databases (SCAI_AUTO_FETCH_MARKERS=1)..."
    "$VENV_DIR/bin/python" scripts/fetch_markers.py || echo "WARNING: marker DB fetch failed; DB priors will be skipped"
fi

# Optional: cache CellTypist models for the analysis kernels (off by default; never
# blocks startup). Files already present are skipped, so this is safe to repeat.
if [ "${SCAI_AUTO_FETCH_CELLTYPIST:-0}" = "1" ]; then
    echo "Fetching CellTypist models (SCAI_AUTO_FETCH_CELLTYPIST=1)..."
    "$VENV_DIR/bin/python" scripts/fetch_celltypist_models.py || echo "WARNING: CellTypist model fetch failed; kernels will download models on demand"
fi

echo ""
echo "Starting services..."
echo ""

# Development keeps Vite's API/WebSocket proxy. Production exposes one Uvicorn
# port and serves the prebuilt SPA, API, and WebSockets from that process.
if [ "$MODE" = "dev" ]; then
    BACKEND_PORT="$BACKPORT"
else
    BACKEND_PORT="$PORT"
fi

echo "   Backend: http://$HOST:$BACKEND_PORT"
cd "$SCRIPT_DIR"
if [ "$MODE" = "dev" ]; then
    "$VENV_DIR/bin/uvicorn" backend.main:app --host "$HOST" --port "$BACKEND_PORT" --reload &
else
    "$VENV_DIR/bin/uvicorn" backend.main:app --host "$HOST" --port "$BACKEND_PORT" &
fi
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for application health check..."
if [ "$HOST" = "0.0.0.0" ] || [ "$HOST" = "::" ] || [ "$HOST" = "[::]" ]; then
    PROBE_HOST="127.0.0.1"
else
    PROBE_HOST="$HOST"
fi
for i in $(seq 1 30); do
    if SCAI_PROBE_HOST="$PROBE_HOST" SCAI_PROBE_PORT="$BACKEND_PORT" \
       "$VENV_DIR/bin/python" -c \
        'import os, urllib.request; host = os.environ["SCAI_PROBE_HOST"]; host = f"[{host}]" if ":" in host and not host.startswith("[") else host; urllib.request.urlopen(f"http://{host}:{int(os.environ['"'"'SCAI_PROBE_PORT'"'"'])}/api/health", timeout=1).read()' \
        >/dev/null 2>&1; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Error: application did not become healthy within 30 seconds."
        kill "$BACKEND_PID" 2>/dev/null
        exit 1
    fi
    sleep 1
done

FRONTEND_PID=""
if [ "$MODE" = "dev" ]; then
    echo "   Vite frontend: http://$HOST:$PORT"
    cd "$SCRIPT_DIR/frontend"
    SCAI_HOST="$HOST" SCAI_BACKPORT="$BACKPORT" npx vite --host "$HOST" --port "$PORT" &
    FRONTEND_PID=$!
else
    echo "   Application: http://$HOST:$PORT"
fi

# Save PIDs and paths to file for stop.sh
echo "BACKEND_PID=$BACKEND_PID" > "$PID_FILE"
if [ -n "$FRONTEND_PID" ]; then
    echo "FRONTEND_PID=$FRONTEND_PID" >> "$PID_FILE"
fi
echo "MODE=$MODE" >> "$PID_FILE"
echo "APP_PORT=$PORT" >> "$PID_FILE"
echo "BACKEND_PORT=$BACKEND_PORT" >> "$PID_FILE"
echo "SCRIPT_DIR=$SCRIPT_DIR" >> "$PID_FILE"
echo "VENV_DIR=$VENV_DIR" >> "$PID_FILE"

echo ""
echo "Services started successfully!"
echo ""
echo "   Open http://$HOST:$PORT in your browser"
echo "   PID saved to $PID_FILE"
echo ""
echo "   Code directory: $SCRIPT_DIR"
echo "   Dependencies directory: $DEP_DIR"
echo "   Data directory: $DATA_DIR"
echo ""
echo "   To stop: ./stop.sh"
echo "   Or press Ctrl+C (foreground mode)"
echo ""

# Handle shutdown (for foreground mode)
cleanup() {
    echo ""
    echo "Stopping services..."
    kill "$BACKEND_PID" 2>/dev/null
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null
    fi
    # Also kill any child processes
    pkill -P "$BACKEND_PID" 2>/dev/null
    if [ -n "$FRONTEND_PID" ]; then
        pkill -P "$FRONTEND_PID" 2>/dev/null
    fi
    rm -f "$PID_FILE"
    deactivate 2>/dev/null
    echo "Stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for the production server, or either development process.
if [ "$MODE" = "dev" ]; then
    wait -n "$BACKEND_PID" "$FRONTEND_PID"
    cleanup
else
    wait "$BACKEND_PID"
    rm -f "$PID_FILE"
fi
