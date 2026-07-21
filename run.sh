#!/bin/bash
# 股票交易模拟器 - 启动脚本
# 用法: ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  股票交易模拟器 (Stock Trading Simulator)"
echo "=========================================="

# 激活虚拟环境
if [ -d "ml_trading_env" ]; then
    source ml_trading_env/bin/activate
    echo "[OK] 已激活 Python 虚拟环境"
else
    echo "[!] 未找到虚拟环境 ml_trading_env/"
    echo "    请运行: python3 -m venv ml_trading_env"
    echo "    然后: source ml_trading_env/bin/activate && pip install -r requirements_ml.txt"
    exit 1
fi

# 启动 Python ML 后端
echo "[*] 启动 ML 交易后端 (端口 5001)..."
python ml_trading_api.py &
ML_PID=$!
echo "    ML 后端 PID: $ML_PID"

# 等待后端就绪
sleep 2

# 启动前端 HTTP 服务
echo "[*] 启动前端服务 (端口 8000)..."
python -m http.server 8000 &
HTTP_PID=$!
echo "    前端服务 PID: $HTTP_PID"

echo ""
echo "=========================================="
echo "  服务已启动!"
echo "  前端页面: http://localhost:8000"
echo "  ML API:   http://localhost:5001/api/ml/status"
echo "=========================================="
echo "  按 Ctrl+C 停止所有服务"
echo ""

# 捕获退出信号，清理子进程
cleanup() {
    echo ""
    echo "[*] 正在停止服务..."
    kill $ML_PID 2>/dev/null
    kill $HTTP_PID 2>/dev/null
    echo "[OK] 服务已停止"
    exit 0
}
trap cleanup SIGINT SIGTERM

# 等待子进程
wait