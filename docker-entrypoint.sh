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

if [ "$MODE" = "dev" ]; then
    echo "Error: this production image does not contain Node.js or npm."
    echo "Use ./start.sh --dev from a source checkout for the Vite development proxy."
    exit 2
fi

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

# ── Start the production application ─────────────────────────────
# FastAPI serves the prebuilt Vite assets, REST API, and WebSockets on one
# external port. Running Uvicorn in the foreground gives it PID 1 after gosu,
# so container stop signals reach the server directly.
echo "Starting ScAI at http://0.0.0.0:3000 ..."
cd "$APP_DIR"
exec "$VENV_DIR/bin/uvicorn" backend.main:app --host 0.0.0.0 --port 3000
