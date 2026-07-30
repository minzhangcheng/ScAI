#!/bin/bash
set -e
trap 'echo "Error on line $LINENO"; exit 1' ERR

# ── Privilege drop: if running as root, fix volume ownership and re-exec as scai ──
if [ "$(id -u)" = "0" ]; then
    mkdir -p /dep /data
    chown -R scai:scai /dep /data
    exec gosu scai "$0" "$@"
fi
# ---- from here on, we are always running as `scai` ----

# Default to production mode
MODE="${SCAI_MODE:-prod}"

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
        *)
            shift
            ;;
    esac
done

APP_DIR="/app"
DEP_DIR="/dep"
DATA_DIR="/data"
VENV_DIR="$DEP_DIR/venv"
NODE_MODULES_DIR="$DEP_DIR/node_modules"

# ── Data directories ──────────────────────────────────────────────
# Create directories before redirecting output (logs dir must exist first)
for sub in uploads sessions states workspace reports logs upload_temp; do
    mkdir -p "$DATA_DIR/$sub"
done

# ── Redirect all output to log file (append mode) ────────────────
LOG_FILE="$DATA_DIR/logs/scai.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo ""
echo "========== ScAI starting at $(date) =========="
echo "Mode: $MODE"

export SCAI_DATA_DIR="$DATA_DIR"

# ── Python venv & dependencies ────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "Installing/updating Python dependencies..."
PIP_CMD=("$VENV_DIR/bin/python" -m pip)
# Python 3.11's ensurepip currently seeds pip 24.0, whose resolver can exhaust
# its 200,000-round limit on this scientific dependency graph. Upgrade before
# resolving requirements; pip still honors PIP_INDEX_URL for this bootstrap.
"${PIP_CMD[@]}" install -q --upgrade "pip>=25.3,<27"
PIP_ARGS=(install -q -r "$APP_DIR/requirements.txt")
if [ -n "$PIP_INDEX_URL" ]; then
    PIP_ARGS+=(-i "$PIP_INDEX_URL")
fi

# pip 25.3+ has a dedicated build-constraint option. Older pip versions also
# apply PIP_CONSTRAINT inside isolated PEP 517 build environments, so retain
# that compatibility path for persisted /dep virtual environments.
if "${PIP_CMD[@]}" install --help 2>/dev/null | grep -q -- '--build-constraint'; then
    "${PIP_CMD[@]}" "${PIP_ARGS[@]}" \
        --build-constraint "$APP_DIR/build-constraints.txt"
else
    PIP_CONSTRAINT="$APP_DIR/build-constraints.txt" \
        "${PIP_CMD[@]}" "${PIP_ARGS[@]}"
fi

# Packages whose declared constraints are provably tighter than their code
# requires must be installed after the main graph, without dependencies, or
# pip would downgrade a pin this project deliberately holds. Rationale for
# every entry lives in requirements-nodeps.txt.
if [ -f "$APP_DIR/requirements-nodeps.txt" ]; then
    NODEPS_ARGS=(install -q --no-deps -r "$APP_DIR/requirements-nodeps.txt")
    if [ -n "$PIP_INDEX_URL" ]; then
        NODEPS_ARGS+=(-i "$PIP_INDEX_URL")
    fi
    "${PIP_CMD[@]}" "${NODEPS_ARGS[@]}"
fi

# ── Node.js dependencies ─────────────────────────────────────────
# Symlink node_modules from dep volume into frontend directory
mkdir -p "$NODE_MODULES_DIR"
if [ -e "$APP_DIR/frontend/node_modules" ] && [ ! -L "$APP_DIR/frontend/node_modules" ]; then
    rm -rf "$APP_DIR/frontend/node_modules"
fi
if [ ! -L "$APP_DIR/frontend/node_modules" ]; then
    ln -s "$NODE_MODULES_DIR" "$APP_DIR/frontend/node_modules"
fi

# Install if node_modules is empty or package.json is newer
if [ ! -f "$NODE_MODULES_DIR/package-lock.json" ] || \
   [ "$APP_DIR/frontend/package.json" -nt "$NODE_MODULES_DIR/package-lock.json" ]; then
    echo "Installing/updating Node.js dependencies..."
    cd "$APP_DIR/frontend"
    npm install
    cd "$APP_DIR"
else
    echo "Node.js dependencies up to date."
fi

# ── Frontend build for production mode ───────────────────────────
if [ "$MODE" = "prod" ]; then
    echo "Building frontend for production..."
    cd "$APP_DIR/frontend"
    npm run build
    cd "$APP_DIR"
fi

# ── Start backend ─────────────────────────────────────────────────
echo "Starting backend..."
cd "$APP_DIR"
"$VENV_DIR/bin/uvicorn" backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend..."
for i in $(seq 1 30); do
    if curl -sf -o /dev/null http://127.0.0.1:8000/api/health 2>/dev/null; then
        echo "Backend ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Warning: Backend did not respond within 30s, starting frontend anyway."
    fi
    sleep 1
done

# ── Start frontend ────────────────────────────────────────────────
echo "Starting frontend..."
cd "$APP_DIR/frontend"
if [ "$MODE" = "dev" ]; then
    SCAI_HOST=127.0.0.1 SCAI_BACKPORT=8000 npx vite --host 0.0.0.0 --port 3000 &
else
    SCAI_HOST=127.0.0.1 SCAI_BACKPORT=8000 npx vite preview --host 0.0.0.0 --port 3000 &
fi
FRONTEND_PID=$!

echo ""
echo "ScAI is running at http://localhost:3000"
echo ""

# ── Signal handling ───────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Stopping services..."
    kill "$FRONTEND_PID" 2>/dev/null
    kill "$BACKEND_PID" 2>/dev/null
    wait "$FRONTEND_PID" 2>/dev/null
    wait "$BACKEND_PID" 2>/dev/null
    echo "Stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for either process to exit
wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
echo "A process exited unexpectedly, shutting down..."
cleanup
