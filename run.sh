#!/bin/bash
# StockArena - One-click startup script
# Usage: ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  StockArena - Multi-Stock Trading Simulator"
echo "=========================================="

# Auto-create virtual environment if needed
if [ ! -d "ml_trading_env" ]; then
    echo "[*] Creating Python virtual environment..."
    python3 -m venv ml_trading_env
    echo "[OK] Virtual environment created"
fi

# Activate virtual environment
source ml_trading_env/bin/activate
echo "[OK] Python virtual environment activated"

# Install dependencies (only if requirements file exists and deps not yet installed)
if [ -f "requirements_ml.txt" ]; then
    if ! python -c "import flask" 2>/dev/null; then
        echo "[*] Installing dependencies..."
        pip install -r requirements_ml.txt -q
        echo "[OK] Dependencies installed"
    else
        echo "[OK] Dependencies already installed"
    fi
fi

# Start Python ML backend
echo "[*] Starting ML trading backend (port 5001)..."
python ml_trading_api.py &
ML_PID=$!
echo "    ML backend PID: $ML_PID"

# Wait for backend to be ready
sleep 2

# Start frontend HTTP server
echo "[*] Starting frontend service (port 8000)..."
python -m http.server 8000 &
HTTP_PID=$!
echo "    Frontend service PID: $HTTP_PID"

echo ""
echo "=========================================="
echo "  StockArena is running!"
echo "  Open: http://localhost:8000"
echo "=========================================="
echo "  Press Ctrl+C to stop"
echo ""

# Trap exit signal to clean up child processes
cleanup() {
    echo ""
    echo "[*] Stopping services..."
    kill $ML_PID 2>/dev/null
    kill $HTTP_PID 2>/dev/null
    echo "[OK] Stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for child processes
wait