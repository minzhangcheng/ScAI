#!/bin/bash

# ScAI - Stop script

echo "Stopping ScAI services..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$SCRIPT_DIR/.scai.pid"

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
    source "$PID_FILE"

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

# Kill vite dev server on port 3000
VITE_PIDS=$(pgrep -f "vite.*--port 3000" 2>/dev/null)
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

# Check port 8000 (backend)
PORT_8000_PID=$(lsof -ti:8000 2>/dev/null)
if [ -n "$PORT_8000_PID" ]; then
    echo "   Releasing port 8000 (PID: $PORT_8000_PID)"
    kill $PORT_8000_PID 2>/dev/null
    sleep 1
    kill -9 $PORT_8000_PID 2>/dev/null
fi

# Check port 3000 (frontend)
PORT_3000_PID=$(lsof -ti:3000 2>/dev/null)
if [ -n "$PORT_3000_PID" ]; then
    echo "   Releasing port 3000 (PID: $PORT_3000_PID)"
    kill $PORT_3000_PID 2>/dev/null
    sleep 1
    kill -9 $PORT_3000_PID 2>/dev/null
fi

echo ""
echo "ScAI stopped"
echo ""

# Verify
sleep 1
if lsof -ti:8000 &>/dev/null || lsof -ti:3000 &>/dev/null; then
    echo "Warning: Some processes may still be running"
    echo "   You can check manually: lsof -i:8000 and lsof -i:3000"
else
    echo "All services stopped successfully"
fi
