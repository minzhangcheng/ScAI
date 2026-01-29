#!/bin/bash

# InteSc - Integrated and Intelligent Single Cell Sequencing Analysis
# Startup script

echo "Starting InteSc - Integrated Single Cell Analysis System..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$SCRIPT_DIR/.intesc.pid"

# Dependencies directory (outside code directory)
DEP_DIR="$PARENT_DIR/intesc_dep"
VENV_DIR="$DEP_DIR/venv"
NODE_MODULES_DIR="$DEP_DIR/node_modules"

# Data directory (outside code directory)
DATA_DIR="$PARENT_DIR/intesc_data"

# Check if already running
if [ -f "$PID_FILE" ]; then
    echo "Warning: Instance already running. Please run ./stop.sh first"
    echo "   or delete $PID_FILE and retry"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not installed. Please install Python3 first"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "Error: Node.js not installed. Please install Node.js first"
    exit 1
fi

# Create dependencies directory
mkdir -p "$DEP_DIR"
echo "Dependencies directory: $DEP_DIR"

# Create data directory
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/uploads"
mkdir -p "$DATA_DIR/states"
mkdir -p "$DATA_DIR/workspace"
mkdir -p "$DATA_DIR/sessions"
mkdir -p "$DATA_DIR/reports"
echo "Data directory: $DATA_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Setup frontend dependencies
# Create symlink for node_modules in frontend directory pointing to shared location
if [ ! -d "$NODE_MODULES_DIR" ]; then
    echo "Installing frontend dependencies to shared directory..."
    mkdir -p "$NODE_MODULES_DIR"

    # Install dependencies to the shared location
    cd "$SCRIPT_DIR/frontend"

    # Remove existing node_modules if it's a directory (not a symlink)
    if [ -d "node_modules" ] && [ ! -L "node_modules" ]; then
        rm -rf node_modules
    fi

    # Create symlink
    if [ ! -L "node_modules" ]; then
        ln -s "$NODE_MODULES_DIR" node_modules
    fi

    # Install dependencies
    npm install
    cd "$SCRIPT_DIR"
else
    echo "Using existing frontend dependencies..."
    cd "$SCRIPT_DIR/frontend"

    # Ensure symlink exists
    if [ ! -L "node_modules" ]; then
        # Remove if it's a regular directory
        if [ -d "node_modules" ]; then
            rm -rf node_modules
        fi
        ln -s "$NODE_MODULES_DIR" node_modules
    fi

    # Check if dependencies need update
    if [ "$SCRIPT_DIR/frontend/package.json" -nt "$NODE_MODULES_DIR/.package-lock.json" ] 2>/dev/null; then
        echo "Updating frontend dependencies..."
        npm install
    fi
    cd "$SCRIPT_DIR"
fi

echo ""
echo "Starting services..."
echo ""

# Start backend in background
echo "   Backend: http://localhost:8000"
cd "$SCRIPT_DIR"
"$VENV_DIR/bin/uvicorn" backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "   Frontend: http://localhost:3000"
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

# Save PIDs and paths to file for stop.sh
echo "BACKEND_PID=$BACKEND_PID" > "$PID_FILE"
echo "FRONTEND_PID=$FRONTEND_PID" >> "$PID_FILE"
echo "SCRIPT_DIR=$SCRIPT_DIR" >> "$PID_FILE"
echo "VENV_DIR=$VENV_DIR" >> "$PID_FILE"

echo ""
echo "Services started successfully!"
echo ""
echo "   Open http://localhost:3000 in your browser"
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
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    # Also kill any child processes
    pkill -P $BACKEND_PID 2>/dev/null
    pkill -P $FRONTEND_PID 2>/dev/null
    rm -f "$PID_FILE"
    deactivate 2>/dev/null
    echo "Stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
