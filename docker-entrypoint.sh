#!/bin/bash
set -e

APP_DIR="/app"
DEP_DIR="/dep"
DATA_DIR="/data"
VENV_DIR="$DEP_DIR/venv"
NODE_MODULES_DIR="$DEP_DIR/node_modules"

# ── Data directories ──────────────────────────────────────────────
echo "Setting up data directories..."
for sub in uploads sessions states workspace reports logs upload_temp; do
    mkdir -p "$DATA_DIR/$sub"
done
export SCAI_DATA_DIR="$DATA_DIR"

# ── Python venv & dependencies ────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "Installing/updating Python dependencies..."
"$VENV_DIR/bin/pip" install -q -r "$APP_DIR/requirements.txt"

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
if [ ! -f "$NODE_MODULES_DIR/.package-lock.json" ] || \
   [ "$APP_DIR/frontend/package.json" -nt "$NODE_MODULES_DIR/.package-lock.json" ]; then
    echo "Installing/updating Node.js dependencies..."
    cd "$APP_DIR/frontend"
    npm install
    cd "$APP_DIR"
else
    echo "Node.js dependencies up to date."
fi

# ── Start backend ─────────────────────────────────────────────────
echo "Starting backend..."
cd "$APP_DIR"
"$VENV_DIR/bin/uvicorn" backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend..."
for i in $(seq 1 30); do
    if curl -s -o /dev/null http://127.0.0.1:8000/api/config/status 2>/dev/null; then
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
SCAI_HOST=127.0.0.1 SCAI_BACKPORT=8000 npx vite --host 0.0.0.0 --port 3000 &
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
