#!/bin/bash
# Stock Trading Simulator - Startup Script
# Usage: ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Stock Trading Simulator"
echo "=========================================="

# Activate virtual environment
if [ -d "ml_trading_env" ]; then
    source ml_trading_env/bin/activate
    echo "[OK] Python virtual environment activated"
else
    echo "[!] Virtual environment ml_trading_env/ not found"
    echo "    Please run: python3 -m venv ml_trading_env"
    echo "    Then: source ml_trading_env/bin/activate && pip install -r requirements_ml.txt"
    exit 1
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
echo "  Services started!"
echo "  Frontend page: http://localhost:8000"
echo "  ML API:        http://localhost:5001/api/ml/status"
echo "=========================================="
echo "  Press Ctrl+C to stop all services"
echo ""

# Trap exit signal to clean up child processes
cleanup() {
    echo ""
    echo "[*] Stopping services..."
    kill $ML_PID 2>/dev/null
    kill $HTTP_PID 2>/dev/null
    echo "[OK] Services stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for child processes
wait