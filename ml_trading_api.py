#!/usr/bin/env python3
"""
ML Trading Bot Backend - Pure Python Standard Library
使用纯Python标准库实现强化学习，不依赖外部包
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# ==================== 配置区域 ====================
def _read_int_env(name: str, default: int, min_val: int, max_val: int) -> int:
    """读取整数环境变量，并限制在安全范围内。"""
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(min_val, min(max_val, value))


PORT = int(os.environ.get('PORT', 5001))
HOST = os.environ.get('ML_HOST') or os.environ.get('HOST', '127.0.0.1')
MAX_BODY_SIZE = int(os.environ.get('ML_MAX_BODY_SIZE', 1024 * 1024))
AUTO_TRAIN_ON_START = os.environ.get('ML_AUTO_TRAIN_ON_START', '1') != '0'
AUTO_TRAIN_EPISODES = _read_int_env('ML_AUTO_TRAIN_EPISODES', 600, 0, 50_000)
AUTO_TRAIN_SERIES_LENGTH = _read_int_env('ML_AUTO_TRAIN_SERIES_LENGTH', 72, 35, 300)
# CORS 严格收敛：默认不发送任何跨域头（loopback 部署不需要）。
# 如需把 ML 后端直接暴露给其它源，导出 ML_CORS_ORIGIN=具体来源（如 http://localhost:4173）。
ML_CORS_ORIGIN = os.environ.get('ML_CORS_ORIGIN', '').strip()

# 配置日志：同时输出到 stdout 和文件，便于线上排查
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ml_trading.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger("ml_trading")

# ==================== 线程安全锁 ====================
# 仅保留一个细粒度锁，保护 QLearningAgent 内部可变状态（q_table、memory、epsilon 等）
# 避免多锁嵌套导致的死锁风险
_agent_lock = threading.Lock()
_auto_train_lock = threading.Lock()
_auto_train_status: Dict[str, Any] = {
    "enabled": AUTO_TRAIN_ON_START,
    "running": False,
    "continuous": True,  # 持续训练模式：启动后一直训练，不停止
    "episodes_per_batch": AUTO_TRAIN_EPISODES,  # 每轮训练的 episode 数
    "episodes_done": 0,
    "updates": 0,
    "batches_completed": 0,
    "started_at": None,
    "error": "",
}


# ==================== 工具函数 ====================
def _validate_numeric_list(data: Any, name: str, min_len: int = 1) -> List[float]:
    """验证输入是否为数值列表，长度满足要求，并返回浮点列表。"""
    if not isinstance(data, list):
        raise ValueError(f"{name} 必须是列表类型")
    if len(data) < min_len:
        raise ValueError(f"{name} 长度不足，至少需要 {min_len} 条数据")
    try:
        return [float(v) for v in data]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 中包含非数值元素") from exc


def _validate_bool(data: Any, name: str, default: bool = True) -> bool:
    """验证并返回布尔值，非法时返回默认值。"""
    if isinstance(data, bool):
        return data
    return default


def _validate_int(data: Any, name: str, default: int = 0, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """验证并返回整数，支持范围限制。"""
    try:
        val = int(data)
    except (TypeError, ValueError):
        return default
    if min_val is not None and val < min_val:
        return min_val
    if max_val is not None and val > max_val:
        return max_val
    return val


def _validate_float(data: Any, name: str, default: float = 0.0) -> float:
    """验证并返回浮点数。"""
    try:
        return float(data)
    except (TypeError, ValueError):
        return default


# ==================== Q-Learning 智能体 ====================
class QLearningAgent:
    """
    Q-Learning 强化学习智能体。

    使用离散化状态空间 + Q 表实现无模型强化学习。
    经验回放缓冲区采用 deque(maxlen) 实现 O(1) 自动淘汰旧经验。
    Q 表引入 LRU 淘汰策略，防止内存无限增长。
    """

    def __init__(
        self,
        state_size: int = 12,
        action_size: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        max_q_table_size: int = 10000,
        memory_capacity: int = 10000,
    ) -> None:
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.lr: float = learning_rate
        self.gamma: float = discount_factor
        self.epsilon: float = 1.0
        self.epsilon_decay: float = 0.998  # 温和衰减：保留更久的探索，避免过早收敛到次优策略
        self.epsilon_min: float = 0.05    # 保持 5% 探索尾部，应对市场状态切换
        self.max_q_table_size: int = max_q_table_size
        self.min_train_batch: int = 4     # 渐进式训练：缓冲≥4 即开始更新，避免前 32 步全部静默
        self.train_batch: int = 32
        # 周期自动保存：每 N 次 train 调用后落盘，防止进程异常退出导致学习成果丢失
        self.auto_save_freq: int = 100
        self._save_counter: int = 0

        # Q 表：离散化状态 tuple -> 各动作 Q 值列表
        # 使用工厂函数确保访问新状态时自动创建零值列表
        self.q_table: Dict[Tuple[float, ...], List[float]] = defaultdict(lambda: [0.0] * action_size)
        # 目标网络：周期性同步主网络副本，用于计算目标 Q 值，降低过估计偏差
        self.target_q_table: Dict[Tuple[float, ...], List[float]] = {}
        self.target_update_freq: int = 50
        self._update_counter: int = 0
        # 记录每个状态的最近访问时间戳，用于 LRU 淘汰
        self.q_table_access: Dict[Tuple[float, ...], float] = {}
        # 经验回放缓冲区，自动丢弃最早经验
        self.memory: deque = deque(maxlen=memory_capacity)

    def get_state_key(self, state: List[float]) -> Tuple[float, ...]:
        """将连续状态向量离散化为可哈希的 tuple 键。

        针对不同特征维度使用不同精度进行 round，降低状态空间膨胀速度：
        - 价格：1 位小数
        - 价格变化：3 位小数
        - 技术指标：2 位小数
        - 资产比例：2 位小数
        - 其他：3 位小数
        """
        key: List[float] = []
        for i, val in enumerate(state):
            if i == 0:          # 价格 - 归一化
                key.append(round(val, 1))
            elif i in (1, 2, 3):  # 价格变化
                key.append(round(val, 3))
            elif i in (4, 5, 6, 7):  # 技术指标
                key.append(round(val, 2))
            elif i in (8, 9):  # 资产比例
                key.append(round(val, 2))
            else:
                key.append(round(val, 3))
        return tuple(key)

    def choose_action(self, state: List[float], training: bool = True) -> int:
        """根据 epsilon-greedy 策略选择动作。

        Args:
            state: 当前状态向量。
            training: 是否处于训练模式（训练时按 epsilon 概率随机探索）。

        Returns:
            动作索引（0 ~ action_size-1）。
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_key = self.get_state_key(state)
        with _agent_lock:
            self.q_table_access[state_key] = time.time()
            q_values = self.q_table[state_key]
            # 返回最大 Q 值对应的动作索引
            return q_values.index(max(q_values))

    def store_memory(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool) -> None:
        """存储单条经验到回放缓冲区（O(1) 复杂度，超限时自动覆盖）。"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size: int = 32) -> None:
        """从经验回放缓冲区随机采样并执行 Q-Learning 更新。

        使用目标网络（target_q_table）计算 Q 目标，降低最大化偏差；
        渐进式批量：缓冲≥ min_train_batch 即开始训练，加快冷启动收敛。
        训练结束后自动衰减 epsilon 并触发 LRU 淘汰。
        """
        if len(self.memory) < self.min_train_batch:
            return

        # 自适应批量：缓冲小时取全部数据，缓冲大时取上限 train_batch
        actual_batch = min(batch_size, len(self.memory))
        # 随机采样 batch，避免列表全量拷贝（random.sample 内部为 C 实现，效率较高）
        batch = random.sample(self.memory, actual_batch)

        with _agent_lock:
            # 首次训练时同步目标网络
            if not self.target_q_table:
                self.target_q_table = {k: list(v) for k, v in self.q_table.items()}

            for state, action, reward, next_state, done in batch:
                state_key = self.get_state_key(state)
                next_state_key = self.get_state_key(next_state)

                current_q = self.q_table[state_key][action]

                if done:
                    target_q = reward
                else:
                    # Double DQN：用主网络选动作，目标网络评估价值
                    # 主网络 argmax -> 目标网络取值，降低过估计
                    main_q_values = self.q_table[next_state_key]
                    best_action = main_q_values.index(max(main_q_values))
                    next_q_values = self.target_q_table.get(next_state_key, [0.0] * self.action_size)
                    target_q = reward + self.gamma * next_q_values[best_action]

                # Q 值更新：Q(s,a) += lr * (target - current)
                self.q_table[state_key][action] += self.lr * (target_q - current_q)
                self.q_table_access[state_key] = time.time()

            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # 周期性同步目标网络（每 target_update_freq 次训练）
            self._update_counter += 1
            if self._update_counter >= self.target_update_freq:
                self.target_q_table = {k: list(v) for k, v in self.q_table.items()}
                self._update_counter = 0

            # LRU 淘汰：Q 表过大时移除最久未访问的条目
            self._evict_if_needed()

        # 周期自动保存：在锁外调用 save()，避免 _agent_lock 重入死锁
        # save() 内部会重新获取锁做快照，外层 I/O 不阻塞推理请求
        self._save_counter += 1
        if self._save_counter >= self.auto_save_freq:
            self._save_counter = 0
            self.save()

    def _evict_if_needed(self) -> None:
        """Q 表超出上限时淘汰最久未访问的状态。

        注意：调用方必须已持有 _agent_lock，否则会出现竞态条件。
        """
        if len(self.q_table) <= self.max_q_table_size:
            return
        # 按访问时间升序排序，淘汰最早访问的条目
        # 使用 heapq.nsmallest 可进一步优化至 O(n log k)，但 k 接近 n 时收益有限
        entries = sorted(self.q_table_access.items(), key=lambda x: x[1])
        to_remove = len(self.q_table) - self.max_q_table_size
        for key, _ in entries[:to_remove]:
            del self.q_table[key]
            del self.q_table_access[key]

    def get_q_values(self, state: List[float]) -> List[float]:
        """获取指定状态下各动作的 Q 值列表（返回副本，避免外部修改）。"""
        state_key = self.get_state_key(state)
        with _agent_lock:
            return list(self.q_table[state_key])

    @staticmethod
    def _serialize_state_key(key: Tuple[float, ...]) -> str:
        """将 Q 表的 tuple key 压缩为 JSON 字符串，便于持久化存储。"""
        return json.dumps(list(key), separators=(",", ":"))

    @staticmethod
    def _deserialize_state_key(key: str) -> Tuple[float, ...]:
        """将持久化存储的 JSON 字符串还原为 tuple key。"""
        try:
            return tuple(json.loads(key))
        except (TypeError, json.JSONDecodeError):
            # 若解析失败，尝试将字符串按逗号分割并转为浮点数
            try:
                parts = key.strip("[]()").split(",")
                return tuple(float(p.strip()) for p in parts if p.strip() != "")
            except Exception:
                return tuple()

    def save(self, filepath: str = "q_model.json") -> None:
        """将 Q 表和 epsilon 保存到 JSON 文件。

        保存时先获取快照，减少锁持有时间，降低对在线推理的延迟影响。
        采用「写到临时文件 + os.replace 原子替换」保证崩溃时不留下半截文件，
        并把文件权限收紧到 0600（同机多用户场景下防止其它账户读取模型）。
        """
        with _agent_lock:
            q_table_snapshot = {
                self._serialize_state_key(key): values
                for key, values in self.q_table.items()
            }
            epsilon = self.epsilon
        data = {"q_table": q_table_snapshot, "epsilon": epsilon}
        tmp_path = f"{filepath}.tmp.{os.getpid()}"
        try:
            # 先写临时文件并设权限，再原子替换
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception:
                # 写入过程中异常，确保临时文件被清理
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            os.replace(tmp_path, filepath)
            logger.info("模型已保存到 %s", filepath)
        except OSError as exc:
            logger.error("保存模型失败: %s", exc)

    def load(self, filepath: str = "q_model.json") -> bool:
        """从 JSON 文件加载 Q 表和 epsilon。

        加载失败时保留当前默认状态，不会导致服务崩溃。
        对每条 Q 表条目做严格 schema 校验，丢弃非法项。
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.info("模型文件 %s 不存在，将使用新模型", filepath)
            return False
        except json.JSONDecodeError as exc:
            logger.warning("模型文件 %s 格式错误: %s，将使用新模型", filepath, exc)
            return False
        except OSError as exc:
            logger.error("加载模型时发生错误: %s", exc)
            return False

        if not isinstance(data, dict):
            logger.warning("模型文件顶层结构不是对象，已忽略")
            return False

        raw_q_table = data.get("q_table", {})
        if not isinstance(raw_q_table, dict):
            logger.warning("模型 q_table 不是对象，已忽略")
            return False

        # 限制单次加载条目数，避免恶意 / 损坏的 JSON 把内存吃光
        max_entries = 200_000
        if len(raw_q_table) > max_entries:
            logger.warning("模型 q_table 条目数 %d 超过上限 %d，截断", len(raw_q_table), max_entries)
            raw_q_table = dict(list(raw_q_table.items())[:max_entries])

        q_table: Dict[Tuple[float, ...], List[float]] = {}
        skipped = 0
        for key, values in raw_q_table.items():
            # 限制 key 长度，避免被恶意大字符串拖垮
            if not isinstance(key, str) or len(key) > 4096:
                skipped += 1
                continue
            if not isinstance(values, list) or len(values) != self.action_size:
                skipped += 1
                continue
            # values 中必须全部是有限浮点
            try:
                float_values = [float(v) for v in values]
                if not all(math.isfinite(v) for v in float_values):
                    skipped += 1
                    continue
            except (TypeError, ValueError):
                skipped += 1
                continue
            try:
                tup_key = self._deserialize_state_key(key)
                if not tup_key:
                    skipped += 1
                    continue
                # 拒绝含 NaN/Inf 的 state key（防止被恶意构造的 JSON 污染训练）
                if not all(math.isfinite(v) for v in tup_key):
                    skipped += 1
                    continue
                # 限制 state 维度，防止恶意大元组消耗内存
                if len(tup_key) > 1024:
                    skipped += 1
                    continue
                q_table[tup_key] = float_values
            except (TypeError, ValueError):
                skipped += 1
                continue

        # 限制 epsilon 在 [0, 1]
        try:
            new_epsilon = float(data.get("epsilon", self.epsilon))
        except (TypeError, ValueError):
            new_epsilon = self.epsilon
        new_epsilon = max(0.0, min(1.0, new_epsilon))

        with _agent_lock:
            # 使用 defaultdict 并预填充已加载数据
            self.q_table = defaultdict(lambda: [0.0] * self.action_size, q_table)
            self.epsilon = new_epsilon
            # 加载后清空目标网络与更新计数器，下次 train 时会重建并同步
            self.target_q_table = {}
            self._update_counter = 0
            self._save_counter = 0
        if skipped:
            logger.warning("模型加载时丢弃 %d 条非法条目", skipped)
        logger.info("模型已从 %s 加载，共 %d 条状态", filepath, len(q_table))
        return True


# ==================== 技术指标计算 ====================
class TechnicalIndicators:
    """技术指标计算（纯 Python 实现，零外部依赖）。"""

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """计算 RSI（相对强弱指数），使用 Wilder 平滑递推。

        Args:
            prices: 收盘价序列。
            period: 计算周期，默认 14。

        Returns:
            RSI 值，范围 0~100；数据不足时返回 50（中性）。
        """
        n = len(prices)
        if n < period + 1:
            return 50.0

        # 初始平均涨跌幅度（简单平均）
        gains = 0.0
        losses = 0.0
        for i in range(1, period + 1):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains += change
            else:
                losses += -change
        avg_gain = gains / period
        avg_loss = losses / period

        # Wilder 平滑递推：avg = (prev_avg * (n-1) + current) / n
        for i in range(period + 1, n):
            change = prices[i] - prices[i - 1]
            gain = change if change > 0 else 0.0
            loss = -change if change < 0 else 0.0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _ema(prices: List[float], period: int) -> float:
        """计算 EMA（指数移动平均）序列的最后一个值。

        使用 SMA 作为种子，后续递推更新。
        """
        if len(prices) < period:
            return prices[-1] if prices else 100.0

        multiplier = 2.0 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = price * multiplier + ema * (1.0 - multiplier)
        return ema

    @staticmethod
    def _ema_incremental(price: float, prev_ema: float, period: int) -> float:
        """增量 EMA：给定当前价格和上一刻 EMA，计算新的 EMA。

        避免重复扫描完整历史序列，适合实时流式计算。
        """
        multiplier = 2.0 / (period + 1)
        return price * multiplier + prev_ema * (1.0 - multiplier)

    @staticmethod
    def calculate_macd(
        prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[float, float, float]:
        """计算 MACD 指标（O(n) 增量递推）。

        Returns:
            (macd_line, signal_line, histogram)
            数据不足时返回 (0, 0, 0)。
        """
        n = len(prices)
        if n < slow + signal:
            return 0.0, 0.0, 0.0

        # 使用 SMA 作为 EMA 种子
        ema_fast = sum(prices[:fast]) / fast
        ema_slow = sum(prices[:slow]) / slow

        # 将 fast EMA 递推至 slow 位置，保证后续同步推进
        for i in range(fast, slow):
            ema_fast = TechnicalIndicators._ema_incremental(prices[i], ema_fast, fast)

        macd_history: List[float] = []
        for i in range(slow, n):
            ema_fast = TechnicalIndicators._ema_incremental(prices[i], ema_fast, fast)
            ema_slow = TechnicalIndicators._ema_incremental(prices[i], ema_slow, slow)
            macd_history.append(ema_fast - ema_slow)

        if len(macd_history) < signal:
            return (macd_history[-1] if macd_history else 0.0), 0.0, 0.0

        # 信号线 = MACD 序列的 EMA
        signal_line = sum(macd_history[:signal]) / signal
        for i in range(signal, len(macd_history)):
            signal_line = TechnicalIndicators._ema_incremental(macd_history[i], signal_line, signal)

        macd_line = macd_history[-1]
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(
        prices: List[float], period: int = 20, std_dev: int = 2
    ) -> Tuple[float, float, float]:
        """计算布林带（Bollinger Bands）。

        Returns:
            (sma, upper_band, lower_band)
            数据不足时返回基于最后价格的默认值。
        """
        if len(prices) < period:
            last = prices[-1] if prices else 100.0
            return last, last + 10.0, last - 10.0

        recent = prices[-period:]
        sma = sum(recent) / period
        variance = sum((p - sma) ** 2 for p in recent) / period
        std = math.sqrt(variance)

        upper = sma + std * std_dev
        lower = sma - std * std_dev
        return sma, upper, lower

    @staticmethod
    def calculate_atr(
        highs: List[float], lows: List[float], closes: List[float], period: int = 14
    ) -> float:
        """计算 ATR（平均真实波幅），使用 Wilder 平滑递推。

        Returns:
            ATR 值；数据不足时返回 5.0 作为默认估计。
        """
        n = len(closes)
        if n < period + 1:
            return 5.0

        # 初始 TR 简单平均
        tr_sum = 0.0
        for i in range(1, period + 1):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            tr_sum += max(high_low, high_close, low_close)
        atr = tr_sum / period

        # Wilder 平滑递推
        for i in range(period + 1, n):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            tr = max(high_low, high_close, low_close)
            atr = (atr * (period - 1) + tr) / period

        return atr


# ==================== ML 交易机器人 ====================
class MLTradingBot:
    """ML 交易机器人：整合状态构建、决策生成与模型更新。"""

    def __init__(self) -> None:
        self.agent = QLearningAgent(state_size=12, action_size=3)
        self.ti = TechnicalIndicators()
        self.action_names = ["买入", "持有", "卖出"]

        # 尝试加载已训练模型
        self.agent.load()

    def create_state(
        self, price_data: List[float], holdings: float, cash: float, total_asset: float
    ) -> List[float]:
        """根据市场与账户数据构建 12 维状态向量。

        状态维度说明：
        0. 归一化价格
        1. 1 日价格变化率
        2. 5 日价格变化率
        3. 10 日价格变化率
        4. RSI / 100
        5. MACD / 10
        6. MACD 柱状图 / 10
        7. 布林带位置
        8. 持仓比例
        9. 现金比例
        10. 短期趋势（1/0）
        11. 波动率
        """
        # 数据不足时向后填充最近价格，保证指标计算稳定
        if len(price_data) < 30:
            price_data = price_data + [price_data[-1]] * (30 - len(price_data))

        prices = price_data[-30:]
        current_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else current_price

        normalized_price = current_price / 100.0

        price_change = (current_price - prev_price) / prev_price if prev_price != 0 else 0.0
        price_change_5 = (current_price - prices[-6]) / prices[-6] if len(prices) > 5 and prices[-6] != 0 else 0.0
        price_change_10 = (current_price - prices[-11]) / prices[-11] if len(prices) > 10 and prices[-11] != 0 else 0.0

        rsi = self.ti.calculate_rsi(prices) / 100.0
        macd, signal, hist = self.ti.calculate_macd(prices)
        macd_norm = macd / 10.0 if macd != 0 else 0.0

        sma, upper, lower = self.ti.calculate_bollinger_bands(prices)
        bb_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5

        holding_ratio = holdings * current_price / total_asset if total_asset > 0 else 0.0
        cash_ratio = cash / total_asset if total_asset > 0 else 0.0

        # 短期趋势：当前价是否高于 5 日均价
        trend = 1.0 if current_price > sum(prices[-5:]) / 5.0 else 0.0

        # 10 日波动率（标准差 / 当前价）
        volatility = 0.0
        if len(prices) >= 10:
            recent_10 = prices[-10:]
            mean = sum(recent_10) / 10.0
            variance = sum((p - mean) ** 2 for p in recent_10) / 10.0
            volatility = math.sqrt(variance) / current_price if current_price != 0 else 0.0

        return [
            normalized_price,
            price_change,
            price_change_5,
            price_change_10,
            rsi,
            macd_norm,
            hist / 10.0 if hist else 0.0,
            bb_position,
            holding_ratio,
            cash_ratio,
            trend,
            volatility,
        ]

    def make_decision(
        self,
        price_data: List[float],
        holdings: float,
        cash: float,
        total_asset: float,
        training: bool = True,
    ) -> Dict[str, Any]:
        """根据当前市场状态做出交易决策。

        Returns:
            包含 action、action_idx、confidence、q_values、state 的字典。
        """
        state = self.create_state(price_data, holdings, cash, total_asset)
        action_idx = self.agent.choose_action(state, training)

        action = self.action_names[action_idx]
        q_values = self.agent.get_q_values(state)
        max_q = max(q_values) if q_values else 0.0
        # 置信度映射：将最大 Q 值的绝对值压缩到 [0,1]
        confidence = min(abs(max_q) / 10.0, 1.0)

        return {
            "action": action,
            "action_idx": action_idx,
            "confidence": confidence,
            "q_values": q_values,
            "state": state,
        }

    def update(
        self,
        state: List[float],
        action_idx: int,
        reward: float,
        next_state: List[float],
        done: bool,
    ) -> None:
        """存储经验并触发模型训练。"""
        self.agent.store_memory(state, action_idx, reward, next_state, done)
        self.agent.train()

    def save(self) -> None:
        """保存模型到磁盘，并更新全局保存时间戳。"""
        global _model_saved_at
        self.agent.save()
        _model_saved_at = time.time()


# ==================== 全局单例 ====================
ml_bot = MLTradingBot()
_model_saved_at: float = time.time()  # 模型最后一次保存的时间戳


def _reload_latest_model_if_newer() -> bool:
    """如果磁盘上的模型文件比上次保存的更新，则重新加载。"""
    global _model_saved_at
    try:
        mtime = os.path.getmtime("q_model.json")
    except OSError:
        return False
    if mtime > _model_saved_at:
        with _agent_lock:
            ok = ml_bot.agent.load()
        if ok:
            _model_saved_at = mtime
            logger.info("检测到模型已更新，已重新加载（q_table 大小: %d）", len(ml_bot.agent.q_table))
        return ok
    return False


# ==================== 启动自动训练 ====================
def get_auto_train_status() -> Dict[str, Any]:
    """返回自动训练状态快照，供 /api/ml/status 展示。"""
    with _auto_train_lock:
        return dict(_auto_train_status)


def _set_auto_train_status(**updates: Any) -> None:
    with _auto_train_lock:
        _auto_train_status.update(updates)


def _generate_synthetic_price_series(length: int) -> List[float]:
    """生成带趋势/震荡/下跌/冲击的合成价格，用于启动预训练。"""
    regimes = [
        {"drift": 0.0018, "vol": 0.012, "weight": 0.30},
        {"drift": -0.0014, "vol": 0.016, "weight": 0.22},
        {"drift": 0.0000, "vol": 0.008, "weight": 0.34},
        {"drift": -0.0040, "vol": 0.030, "weight": 0.14},
    ]
    price = random.uniform(60.0, 160.0)
    prices: List[float] = []
    regime = random.choices(regimes, weights=[r["weight"] for r in regimes], k=1)[0]
    steps_left = random.randint(12, 32)
    shock = 0.0

    for _ in range(length):
        if steps_left <= 0:
            regime = random.choices(regimes, weights=[r["weight"] for r in regimes], k=1)[0]
            steps_left = random.randint(12, 32)
        steps_left -= 1

        if random.random() < 0.025:
            shock += random.choice([-1.0, 1.0]) * random.uniform(0.012, 0.045)
        shock *= 0.78

        ret = regime["drift"] + random.gauss(0.0, regime["vol"]) + shock
        ret = max(-0.12, min(0.10, ret))
        price = max(1.0, price * (1.0 + ret))
        prices.append(round(price, 4))
    return prices


def _build_training_state(prices: List[float], end_index: int, holding_ratio: float) -> List[float]:
    window_start = max(0, end_index - 29)
    window = prices[window_start:end_index + 1]
    if len(window) < 5:
        window = [window[0]] * (5 - len(window)) + window
    current_price = window[-1]
    total_asset = 10_000.0
    holdings = (total_asset * holding_ratio) / current_price if current_price > 0 else 0.0
    cash = total_asset * (1.0 - holding_ratio)
    return ml_bot.create_state(window, holdings, cash, total_asset)


def _pretrain_once(prices: List[float]) -> int:
    updates = 0
    last_index = len(prices) - 2
    for idx in range(30, last_index):
        future_index = min(len(prices) - 1, idx + 4)
        current_price = prices[idx]
        future_return = (prices[future_index] - current_price) / current_price if current_price else 0.0
        next_state_flat = _build_training_state(prices, idx + 1, 0.0)
        next_state_held = _build_training_state(prices, idx + 1, 0.55)

        flat_state = _build_training_state(prices, idx, 0.0)
        if future_return > 0.004:
            ml_bot.update(flat_state, 0, min(5.0, future_return * 140.0), next_state_held, False)
        elif future_return < -0.004:
            ml_bot.update(flat_state, 1, min(3.0, abs(future_return) * 80.0), next_state_flat, False)
        else:
            ml_bot.update(flat_state, 1, 0.6, next_state_flat, False)
        updates += 1

        held_state = _build_training_state(prices, idx, 0.55)
        if future_return < -0.004:
            ml_bot.update(held_state, 2, min(5.0, abs(future_return) * 140.0), next_state_flat, False)
        elif future_return > 0.004:
            ml_bot.update(held_state, 1, min(4.0, future_return * 100.0), next_state_held, False)
        else:
            ml_bot.update(held_state, 1, 0.4, next_state_held, False)
        updates += 1
    return updates


def run_continuous_training() -> None:
    """后台持续训练：启动后一直生成合成数据训练，每轮结束保存模型，永不停止。"""
    if not AUTO_TRAIN_ON_START or AUTO_TRAIN_EPISODES <= 0:
        _set_auto_train_status(enabled=False, running=False)
        logger.info("持续训练已关闭")
        return

    _set_auto_train_status(
        enabled=True,
        running=True,
        episodes_per_batch=AUTO_TRAIN_EPISODES,
        episodes_done=0,
        updates=0,
        batches_completed=0,
        started_at=time.time(),
        error="",
    )
    logger.info("持续训练启动：每轮 %d episodes，序列长度 %d，将无限循环", AUTO_TRAIN_EPISODES, AUTO_TRAIN_SERIES_LENGTH)

    total_updates = 0
    old_auto_save_freq = ml_bot.agent.auto_save_freq
    old_save_counter = ml_bot.agent._save_counter
    try:
        # 持续训练期间禁用 agent 内部的 auto_save，改由每轮结束时统一保存
        ml_bot.agent.auto_save_freq = max(old_auto_save_freq, AUTO_TRAIN_EPISODES * AUTO_TRAIN_SERIES_LENGTH * 2 + 1)
        ml_bot.agent._save_counter = 0

        batch = 0
        while True:
            batch += 1
            batch_updates = 0
            for episode in range(1, AUTO_TRAIN_EPISODES + 1):
                prices = _generate_synthetic_price_series(AUTO_TRAIN_SERIES_LENGTH)
                batch_updates += _pretrain_once(prices)
                if episode == 1 or episode % 25 == 0 or episode == AUTO_TRAIN_EPISODES:
                    _set_auto_train_status(
                        episodes_done=episode,
                        updates=total_updates + batch_updates,
                        batches_completed=batch - 1,
                    )

            total_updates += batch_updates
            # 每轮结束后保存模型，让交易接口能读到最新模型
            ml_bot.save()
            _set_auto_train_status(
                episodes_done=AUTO_TRAIN_EPISODES,
                updates=total_updates,
                batches_completed=batch,
            )
            logger.info("持续训练第 %d 轮完成：%d episodes，累计更新 %d 次，模型已保存", batch, AUTO_TRAIN_EPISODES, total_updates)
    except Exception as exc:
        logger.exception("持续训练异常")
        _set_auto_train_status(running=False, error=str(exc))
    finally:
        ml_bot.agent.auto_save_freq = old_auto_save_freq
        ml_bot.agent._save_counter = old_save_counter


def start_continuous_training_thread() -> None:
    """以 daemon 线程启动持续训练，避免阻塞 HTTP 服务。"""
    thread = threading.Thread(target=run_continuous_training, name="continuous-train", daemon=True)
    thread.start()


# ==================== HTTP 请求处理器 ====================
class RequestHandler(BaseHTTPRequestHandler):
    """基于 BaseHTTPRequestHandler 的 REST API 处理器。

    每个请求运行在独立线程中，共享全局 ml_bot 实例。
    所有涉及智能体内部可变状态的操作均受 _agent_lock 保护。
    """

    def log_message(self, format: str, *args: Any) -> None:
        """自定义日志格式，使用 logging 模块替代 print。"""
        logger.info("%s - %s", self.address_string(), format % args)

    def send_cors_headers(self) -> None:
        """仅当显式配置 ML_CORS_ORIGIN 时才发送 CORS 头（避免默认 * 通配）。"""
        if not ML_CORS_ORIGIN:
            return
        self.send_header("Access-Control-Allow-Origin", ML_CORS_ORIGIN)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json_response(self, data: Dict[str, Any], status: int = 200) -> None:
        """发送 JSON 响应，自动处理编码。"""
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_cors_headers()
        self.end_headers()
        try:
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            logger.error("JSON 序列化失败: %s", exc)
            self.wfile.write('{"error":"内部序列化错误"}'.encode('utf-8'))

    def do_OPTIONS(self) -> None:
        """处理浏览器 CORS 预检请求。"""
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        """处理 GET 请求。"""
        parsed = urlparse(self.path)

        if parsed.path == "/api/ml/status":
            with _agent_lock:
                payload = {
                    "success": True,
                    "epsilon": ml_bot.agent.epsilon,
                    "memory_size": len(ml_bot.agent.memory),
                    "action_size": ml_bot.agent.action_size,
                    "state_size": ml_bot.agent.state_size,
                    "q_table_size": len(ml_bot.agent.q_table),
                }
            payload["auto_training"] = get_auto_train_status()
            self.send_json_response(payload)
        else:
            self.send_json_response({"error": "未找到端点"}, 404)

    def do_POST(self) -> None:
        """处理 POST 请求。"""
        parsed = urlparse(self.path)
        content_length = self.headers.get("Content-Length")
        try:
            length = int(content_length) if content_length is not None else 0
        except ValueError:
            length = 0

        if length > MAX_BODY_SIZE:
            self.send_json_response({"error": "请求体过大"}, 413)
            return

        try:
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        except UnicodeDecodeError:
            self.send_json_response({"error": "请求体编码错误，仅支持 UTF-8"}, 400)
            return

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("收到无效的 JSON 请求: %s", exc)
            self.send_json_response({"error": "无效的 JSON"}, 400)
            return

        try:
            if parsed.path == "/api/ml/decide":
                self.handle_decide(data)
            elif parsed.path == "/api/ml/update":
                self.handle_update(data)
            elif parsed.path == "/api/ml/indicators":
                self.handle_indicators(data)
            elif parsed.path == "/api/ml/save":
                ml_bot.save()
                self.send_json_response({"success": True})
            elif parsed.path == "/api/ml/reload":
                # 从磁盘重新加载最新模型
                with _agent_lock:
                    success = ml_bot.agent.load()
                self.send_json_response({"success": success, "q_table_size": len(ml_bot.agent.q_table)})
            else:
                self.send_json_response({"error": "未找到端点"}, 404)
        except Exception as exc:
            logger.exception("处理请求 %s 时发生未捕获异常", parsed.path)
            self.send_json_response({"error": f"服务器内部错误: {exc}"}, 500)

    def handle_decide(self, data: Dict[str, Any]) -> None:
        """处理交易决策请求。

        期望 JSON 字段：
        - price_data: 价格序列（必需，至少 5 条）
        - holdings: 持仓股数（默认 0）
        - cash: 现金（默认 100）
        - total_asset: 总资产（默认 100）
        - training: 是否处于训练模式（默认 True）
        """
        # 开始交易前，检查并加载最新模型
        _reload_latest_model_if_newer()

        try:
            price_data = _validate_numeric_list(data.get("price_data"), "price_data", min_len=5)
            holdings = _validate_float(data.get("holdings", 0), "holdings")
            cash = _validate_float(data.get("cash", 100), "cash")
            total_asset = _validate_float(data.get("total_asset", 100), "total_asset")
            training = _validate_bool(data.get("training", True), "training")
        except ValueError as exc:
            logger.warning("决策请求参数校验失败: %s", exc)
            self.send_json_response({"error": str(exc)}, 400)
            return

        decision = ml_bot.make_decision(price_data, holdings, cash, total_asset, training)
        with _agent_lock:
            epsilon = ml_bot.agent.epsilon

        self.send_json_response({
            "success": True,
            "decision": decision,
            "epsilon": epsilon,
        })

    def handle_update(self, data: Dict[str, Any]) -> None:
        """处理模型更新请求（经验回放与训练）。

        支持两种传参方式：
        1. 直接传入 state / next_state（12 维列表）
        2. 传入 price_data / next_price_data 等原始数据，由服务端自动构建状态
        """
        try:
            action_idx = _validate_int(data.get("action_idx", 0), "action_idx", default=0, min_val=0, max_val=2)
            reward = _validate_float(data.get("reward", 0), "reward")
            done = _validate_bool(data.get("done", False), "done")
        except ValueError as exc:
            logger.warning("更新请求参数校验失败: %s", exc)
            self.send_json_response({"error": str(exc)}, 400)
            return

        state = data.get("state", [])
        next_state = data.get("next_state", [])

        # 若未直接提供状态向量，则尝试从原始数据构建
        if not state and data.get("price_data"):
            try:
                pd_state = _validate_numeric_list(data.get("price_data"), "price_data", min_len=5)
                h_state = _validate_float(data.get("holdings", 0), "holdings")
                c_state = _validate_float(data.get("cash", 100), "cash")
                ta_state = _validate_float(data.get("total_asset", 100), "total_asset")
                state = ml_bot.create_state(pd_state, h_state, c_state, ta_state)
            except ValueError as exc:
                logger.warning("构建 state 失败: %s", exc)
                self.send_json_response({"error": f"构建 state 失败: {exc}"}, 400)
                return

        if not next_state and data.get("next_price_data"):
            try:
                pd_next = _validate_numeric_list(data.get("next_price_data"), "next_price_data", min_len=5)
                h_next = _validate_float(data.get("next_holdings", 0), "next_holdings")
                c_next = _validate_float(data.get("next_cash", 100), "next_cash")
                ta_next = _validate_float(data.get("next_total_asset", 100), "next_total_asset")
                next_state = ml_bot.create_state(pd_next, h_next, c_next, ta_next)
            except ValueError as exc:
                logger.warning("构建 next_state 失败: %s", exc)
                self.send_json_response({"error": f"构建 next_state 失败: {exc}"}, 400)
                return

        if not state or not next_state:
            self.send_json_response({"error": "更新数据不足，缺少 state 或 next_state"}, 400)
            return

        ml_bot.update(state, action_idx, reward, next_state, done)
        with _agent_lock:
            epsilon = ml_bot.agent.epsilon

        self.send_json_response({
            "success": True,
            "epsilon": epsilon,
        })

    def handle_indicators(self, data: Dict[str, Any]) -> None:
        """处理技术指标计算请求。

        期望 JSON 字段：
        - prices: 收盘价序列（必需，至少 20 条）
        """
        try:
            prices = _validate_numeric_list(data.get("prices"), "prices", min_len=20)
        except ValueError as exc:
            logger.warning("指标请求参数校验失败: %s", exc)
            self.send_json_response({"error": str(exc)}, 400)
            return

        # TechnicalIndicators 为纯函数，无内部状态，无需加锁
        ti = TechnicalIndicators()

        rsi = ti.calculate_rsi(prices)
        macd, signal, hist = ti.calculate_macd(prices)
        sma, upper, lower = ti.calculate_bollinger_bands(prices)

        buy_signal = rsi < 30 and macd > signal
        sell_signal = rsi > 70 and macd < signal

        self.send_json_response({
            "success": True,
            "indicators": {
                "rsi": rsi,
                "macd": macd,
                "signal": signal,
                "histogram": hist,
                "bollinger_sma": sma,
                "bollinger_upper": upper,
                "bollinger_lower": lower,
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
            },
        })


# ==================== 服务器入口 ====================
def run_server() -> None:
    """启动 HTTP 服务器。"""
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    logger.info("=" * 60)
    logger.info("ML Trading Bot Backend (Pure Python) 启动中...")
    logger.info("=" * 60)
    logger.info("API 服务地址: http://%s:%d", HOST, PORT)
    logger.info("可用端点:")
    logger.info("   GET  /api/ml/status     - 查看状态")
    logger.info("   POST /api/ml/decide     - 获取 ML 交易决策")
    logger.info("   POST /api/ml/update     - 更新 ML 模型")
    logger.info("   POST /api/ml/indicators - 获取技术指标")
    logger.info("   POST /api/ml/reload     - 重新加载最新模型")
    logger.info("   POST /api/ml/save       - 保存模型")
    logger.info("=" * 60)
    start_continuous_training_thread()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断信号，服务器正在关闭...")
    finally:
        server.server_close()
        logger.info("服务器已关闭")


if __name__ == "__main__":
    run_server()
