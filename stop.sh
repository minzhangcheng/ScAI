#!/bin/bash

# ScAI - Stop script

echo "Stopping ScAI services..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$SCRIPT_DIR/.scai.pid"
APP_PORT=3000
BACKEND_PORT=8000
MODE="unknown"

# Function to kill process and its children
kill_tree() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        # Kill child processes first
        pkill -P "$pid" 2>/dev/null
        # Then kill the main process
        kill "$pid" 2>/dev/null
        # Wait a bit and force kill if still running
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
        return 0
    fi
    return 1
}

# Method 1: Use PID file
if [ -f "$PID_FILE" ]; then
    echo "Reading PID file..."
    BACKEND_PID="$(grep -E '^BACKEND_PID=' "$PID_FILE" | head -1 | cut -d= -f2 | tr -dc '0-9')"
    FRONTEND_PID="$(grep -E '^FRONTEND_PID=' "$PID_FILE" | head -1 | cut -d= -f2 | tr -dc '0-9')"
    APP_PORT="$(grep -E '^APP_PORT=' "$PID_FILE" | head -1 | cut -d= -f2 | tr -dc '0-9')"
    BACKEND_PORT="$(grep -E '^BACKEND_PORT=' "$PID_FILE" | head -1 | cut -d= -f2 | tr -dc '0-9')"
    MODE="$(grep -E '^MODE=(dev|prod)$' "$PID_FILE" | head -1 | cut -d= -f2)"
    APP_PORT="${APP_PORT:-3000}"
    BACKEND_PORT="${BACKEND_PORT:-8000}"

    if [ -n "$BACKEND_PID" ]; then
        echo "   Stopping backend (PID: $BACKEND_PID)..."
        kill_tree "$BACKEND_PID"
    fi

    if [ -n "$FRONTEND_PID" ]; then
        echo "   Stopping frontend (PID: $FRONTEND_PID)..."
        kill_tree "$FRONTEND_PID"
    fi

    rm -f "$PID_FILE"
    echo "   PID file cleaned up"
fi

# Method 2: Kill by process name (backup method)
echo ""
echo "Checking for remaining processes..."

# Kill uvicorn processes running our app
UVICORN_PIDS=$(pgrep -f "uvicorn backend.main:app" 2>/dev/null)
if [ -n "$UVICORN_PIDS" ]; then
    echo "   Stopping uvicorn processes: $UVICORN_PIDS"
    for pid in $UVICORN_PIDS; do
        kill_tree "$pid"
    done
fi

# Kill the development Vite server on the recorded application port.
VITE_PIDS=$(pgrep -f "vite.*--port $APP_PORT" 2>/dev/null)
if [ -n "$VITE_PIDS" ]; then
    echo "   Stopping vite processes: $VITE_PIDS"
    for pid in $VITE_PIDS; do
        kill_tree "$pid"
    done
fi

# Kill node processes related to our frontend
NODE_PIDS=$(pgrep -f "node.*scai/frontend" 2>/dev/null)
if [ -n "$NODE_PIDS" ]; then
    echo "   Stopping node processes: $NODE_PIDS"
    for pid in $NODE_PIDS; do
        kill_tree "$pid"
    done
fi

# Method 3: Kill by port (last resort)
echo ""
echo "Checking port usage..."

release_port() {
    local port=$1
    local port_pids
    port_pids=$(lsof -ti:"$port" 2>/dev/null)
    if [ -n "$port_pids" ]; then
        echo "   Releasing port $port (PID: $port_pids)"
        kill $port_pids 2>/dev/null
        sleep 1
        kill -9 $port_pids 2>/dev/null
    fi
}

release_port "$APP_PORT"
if [ "$BACKEND_PORT" != "$APP_PORT" ]; then
    release_port "$BACKEND_PORT"
fi

echo ""
echo "ScAI stopped"
echo ""

# Verify
sleep 1
if lsof -ti:"$APP_PORT" &>/dev/null || \
   { [ "$BACKEND_PORT" != "$APP_PORT" ] && lsof -ti:"$BACKEND_PORT" &>/dev/null; }; then
    echo "Warning: Some processes may still be running"
    echo "   You can check manually: lsof -i:$APP_PORT and lsof -i:$BACKEND_PORT"
else
    echo "All services stopped successfully"
fi
