#!/usr/bin/env python3
"""
ML Trading Bot Backend - Pure Python Standard Library
Implements reinforcement learning using only Python standard library, no external dependencies.
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

# ==================== Configuration ====================
def _read_int_env(name: str, default: int, min_val: int, max_val: int) -> int:
    """Read an integer environment variable, clamped to a safe range."""
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
# Strict CORS: by default no cross-origin headers are sent (loopback deployments don't need them).
# To expose the ML backend directly to other origins, set ML_CORS_ORIGIN=<origin> (e.g. http://localhost:4173).
ML_CORS_ORIGIN = os.environ.get('ML_CORS_ORIGIN', '').strip()

# Configure logging: output to both stdout and file for online troubleshooting
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

# ==================== Thread Safety Locks ====================
# Single fine-grained lock protecting QLearningAgent mutable internal state (q_table, memory, epsilon, etc.)
# Avoids deadlock risk from nested locks
_agent_lock = threading.Lock()
_auto_train_lock = threading.Lock()
_auto_train_status: Dict[str, Any] = {
    "enabled": AUTO_TRAIN_ON_START,
    "running": False,
    "continuous": True,  # Continuous training mode: keep training after start, never stop
    "episodes_per_batch": AUTO_TRAIN_EPISODES,  # Number of episodes per training batch
    "episodes_done": 0,
    "updates": 0,
    "batches_completed": 0,
    "started_at": None,
    "error": "",
}


# ==================== Utility Functions ====================
def _validate_numeric_list(data: Any, name: str, min_len: int = 1) -> List[float]:
    """Validate that input is a numeric list meeting length requirements, returning a list of floats."""
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")
    if len(data) < min_len:
        raise ValueError(f"{name} length insufficient, at least {min_len} data points required")
    try:
        return [float(v) for v in data]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains non-numeric elements") from exc


def _validate_bool(data: Any, name: str, default: bool = True) -> bool:
    """Validate and return a boolean, falling back to default on invalid input."""
    if isinstance(data, bool):
        return data
    return default


def _validate_int(data: Any, name: str, default: int = 0, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Validate and return an integer, with optional range clamping."""
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
    """Validate and return a float."""
    try:
        return float(data)
    except (TypeError, ValueError):
        return default


# ==================== Q-Learning Agent ====================
class QLearningAgent:
    """
    Q-Learning reinforcement learning agent.

    Uses discretized state space + Q-table for model-free reinforcement learning.
    Experience replay buffer uses deque(maxlen) for O(1) automatic eviction of old experiences.
    Q-table uses LRU eviction policy to prevent unbounded memory growth.
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
        self.epsilon_decay: float = 0.998  # Gentle decay: preserve exploration longer to avoid premature convergence to suboptimal policy
        self.epsilon_min: float = 0.05    # Maintain 5% exploration tail to handle market regime shifts
        self.max_q_table_size: int = max_q_table_size
        self.min_train_batch: int = 4     # Incremental training: start updating once buffer reaches 4, avoiding silent first 32 steps
        self.train_batch: int = 32
        # Periodic auto-save: persist to disk every N train() calls to prevent loss from process crashes
        self.auto_save_freq: int = 100
        self._save_counter: int = 0

        # Q-table: discretized state tuple -> list of Q-values for each action
        # Uses factory function to auto-create zero-value lists when accessing new states
        self.q_table: Dict[Tuple[float, ...], List[float]] = defaultdict(lambda: [0.0] * action_size)
        # Target network: periodically synced copy of the main network, used for computing target Q-values to reduce overestimation bias
        self.target_q_table: Dict[Tuple[float, ...], List[float]] = {}
        self.target_update_freq: int = 50
        self._update_counter: int = 0
        # Track last access timestamp for each state, used for LRU eviction
        self.q_table_access: Dict[Tuple[float, ...], float] = {}
        # Experience replay buffer, auto-discards oldest experiences
        self.memory: deque = deque(maxlen=memory_capacity)

    def get_state_key(self, state: List[float]) -> Tuple[float, ...]:
        """Discretize a continuous state vector into a hashable tuple key.

        Uses different rounding precision for different feature dimensions to
        slow down state space explosion:
        - Price: 1 decimal place
        - Price changes: 3 decimal places
        - Technical indicators: 2 decimal places
        - Asset ratios: 2 decimal places
        - Other: 3 decimal places
        """
        key: List[float] = []
        for i, val in enumerate(state):
            if i == 0:          # Price - normalized
                key.append(round(val, 1))
            elif i in (1, 2, 3):  # Price changes
                key.append(round(val, 3))
            elif i in (4, 5, 6, 7):  # Technical indicators
                key.append(round(val, 2))
            elif i in (8, 9):  # Asset ratios
                key.append(round(val, 2))
            else:
                key.append(round(val, 3))
        return tuple(key)

    def choose_action(self, state: List[float], training: bool = True) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state: Current state vector.
            training: Whether in training mode (random exploration with epsilon probability).

        Returns:
            Action index (0 ~ action_size-1).
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_key = self.get_state_key(state)
        with _agent_lock:
            self.q_table_access[state_key] = time.time()
            q_values = self.q_table[state_key]
            # Return the action index with the highest Q-value
            return q_values.index(max(q_values))

    def store_memory(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool) -> None:
        """Store a single experience in the replay buffer (O(1) complexity, auto-overwrites when full)."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size: int = 32) -> None:
        """Sample from experience replay buffer and perform Q-Learning updates.

        Uses target network (target_q_table) to compute Q targets, reducing maximization bias.
        Incremental batch: starts training once buffer reaches min_train_batch, accelerating cold-start convergence.
        After training, automatically decays epsilon and triggers LRU eviction.
        """
        if len(self.memory) < self.min_train_batch:
            return

        # Adaptive batch size: use all data when buffer is small, cap at train_batch when large
        actual_batch = min(batch_size, len(self.memory))
        # Randomly sample batch; random.sample is C-implemented internally, reasonably efficient
        batch = random.sample(self.memory, actual_batch)

        with _agent_lock:
            # Sync target network on first training step
            if not self.target_q_table:
                self.target_q_table = {k: list(v) for k, v in self.q_table.items()}

            for state, action, reward, next_state, done in batch:
                state_key = self.get_state_key(state)
                next_state_key = self.get_state_key(next_state)

                current_q = self.q_table[state_key][action]

                if done:
                    target_q = reward
                else:
                    # Double DQN: use main network to select action, target network to evaluate value
                    # Main network argmax -> target network lookup, reduces overestimation
                    main_q_values = self.q_table[next_state_key]
                    best_action = main_q_values.index(max(main_q_values))
                    next_q_values = self.target_q_table.get(next_state_key, [0.0] * self.action_size)
                    target_q = reward + self.gamma * next_q_values[best_action]

                # Q-value update: Q(s,a) += lr * (target - current)
                self.q_table[state_key][action] += self.lr * (target_q - current_q)
                self.q_table_access[state_key] = time.time()

            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Periodically sync target network (every target_update_freq training steps)
            self._update_counter += 1
            if self._update_counter >= self.target_update_freq:
                self.target_q_table = {k: list(v) for k, v in self.q_table.items()}
                self._update_counter = 0

            # LRU eviction: remove least-recently-accessed entries when Q-table exceeds limit
            self._evict_if_needed()

        # Periodic auto-save: call save() outside the lock to avoid _agent_lock reentry deadlock
        # save() internally acquires the lock for a snapshot; external I/O does not block inference requests
        self._save_counter += 1
        if self._save_counter >= self.auto_save_freq:
            self._save_counter = 0
            self.save()

    def _evict_if_needed(self) -> None:
        """Evict least-recently-accessed states when Q-table exceeds the size limit.

        Note: Caller must already hold _agent_lock, otherwise a race condition occurs.
        """
        if len(self.q_table) <= self.max_q_table_size:
            return
        # Sort by access time ascending, evict the earliest-accessed entries
        # heapq.nsmallest could further optimize to O(n log k), but benefit is limited when k is close to n
        entries = sorted(self.q_table_access.items(), key=lambda x: x[1])
        to_remove = len(self.q_table) - self.max_q_table_size
        for key, _ in entries[:to_remove]:
            del self.q_table[key]
            del self.q_table_access[key]

    def get_q_values(self, state: List[float]) -> List[float]:
        """Get the Q-value list for each action in the given state (returns a copy to prevent external modification)."""
        state_key = self.get_state_key(state)
        with _agent_lock:
            return list(self.q_table[state_key])

    @staticmethod
    def _serialize_state_key(key: Tuple[float, ...]) -> str:
        """Compress a Q-table tuple key into a JSON string for persistent storage."""
        return json.dumps(list(key), separators=(",", ":"))

    @staticmethod
    def _deserialize_state_key(key: str) -> Tuple[float, ...]:
        """Restore a persistent JSON string back into a tuple key."""
        try:
            return tuple(json.loads(key))
        except (TypeError, json.JSONDecodeError):
            # If parsing fails, try splitting by comma and converting to floats
            try:
                parts = key.strip("[]()").split(",")
                return tuple(float(p.strip()) for p in parts if p.strip() != "")
            except Exception:
                return tuple()

    def save(self, filepath: str = "q_model.json") -> None:
        """Save Q-table and epsilon to a JSON file.

        Takes a snapshot first to minimize lock hold time and reduce latency impact on online inference.
        Uses "write to temp file + os.replace atomic swap" to guarantee no partial file on crash,
        and tightens file permissions to 0600 (prevent other accounts from reading the model on multi-user machines).
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
            # Write to temp file with restricted permissions, then atomically replace
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception:
                # Clean up temp file on write failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            os.replace(tmp_path, filepath)
            logger.info("Model saved to %s", filepath)
        except OSError as exc:
            logger.error("Failed to save model: %s", exc)

    def load(self, filepath: str = "q_model.json") -> bool:
        """Load Q-table and epsilon from a JSON file.

        On load failure, retains current default state without crashing the service.
        Performs strict schema validation on each Q-table entry, discarding invalid items.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.info("Model file %s not found, using new model", filepath)
            return False
        except json.JSONDecodeError as exc:
            logger.warning("Model file %s format error: %s, using new model", filepath, exc)
            return False
        except OSError as exc:
            logger.error("Error loading model: %s", exc)
            return False

        if not isinstance(data, dict):
            logger.warning("Model file top-level structure is not an object, ignored")
            return False

        raw_q_table = data.get("q_table", {})
        if not isinstance(raw_q_table, dict):
            logger.warning("Model q_table is not an object, ignored")
            return False

        # Limit the number of entries loaded at once to prevent malicious/corrupt JSON from exhausting memory
        max_entries = 200_000
        if len(raw_q_table) > max_entries:
            logger.warning("Model q_table entry count %d exceeds limit %d, truncating", len(raw_q_table), max_entries)
            raw_q_table = dict(list(raw_q_table.items())[:max_entries])

        q_table: Dict[Tuple[float, ...], List[float]] = {}
        skipped = 0
        for key, values in raw_q_table.items():
            # Limit key length to prevent being dragged down by malicious large strings
            if not isinstance(key, str) or len(key) > 4096:
                skipped += 1
                continue
            if not isinstance(values, list) or len(values) != self.action_size:
                skipped += 1
                continue
            # All values must be finite floats
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
                # Reject state keys containing NaN/Inf (prevents maliciously crafted JSON from polluting training)
                if not all(math.isfinite(v) for v in tup_key):
                    skipped += 1
                    continue
                # Limit state dimension to prevent malicious large tuples from consuming memory
                if len(tup_key) > 1024:
                    skipped += 1
                    continue
                q_table[tup_key] = float_values
            except (TypeError, ValueError):
                skipped += 1
                continue

        # Clamp epsilon to [0, 1]
        try:
            new_epsilon = float(data.get("epsilon", self.epsilon))
        except (TypeError, ValueError):
            new_epsilon = self.epsilon
        new_epsilon = max(0.0, min(1.0, new_epsilon))

        with _agent_lock:
            # Use defaultdict pre-populated with loaded data
            self.q_table = defaultdict(lambda: [0.0] * self.action_size, q_table)
            self.epsilon = new_epsilon
            # Clear target network and update counter after loading; they will be rebuilt and synced on next train()
            self.target_q_table = {}
            self._update_counter = 0
            self._save_counter = 0
        if skipped:
            logger.warning("Discarded %d invalid entries during model load", skipped)
        logger.info("Model loaded from %s, %d states total", filepath, len(q_table))
        return True


# ==================== Technical Indicators ====================
class TechnicalIndicators:
    """Technical indicator calculation (pure Python implementation, zero external dependencies)."""

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index) using Wilder smoothing.

        Args:
            prices: Closing price series.
            period: Calculation period, default 14.

        Returns:
            RSI value, range 0~100; returns 50 (neutral) when data is insufficient.
        """
        n = len(prices)
        if n < period + 1:
            return 50.0

        # Initial average gain/loss (simple average)
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

        # Wilder smoothing: avg = (prev_avg * (n-1) + current) / n
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
        """Calculate the last value of an EMA (Exponential Moving Average) series.

        Uses SMA as seed, then recursively updates.
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
        """Incremental EMA: given current price and previous EMA, compute the new EMA.

        Avoids rescanning the full history, suitable for real-time streaming computation.
        """
        multiplier = 2.0 / (period + 1)
        return price * multiplier + prev_ema * (1.0 - multiplier)

    @staticmethod
    def calculate_macd(
        prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD indicator (O(n) incremental recurrence).

        Returns:
            (macd_line, signal_line, histogram)
            Returns (0, 0, 0) when data is insufficient.
        """
        n = len(prices)
        if n < slow + signal:
            return 0.0, 0.0, 0.0

        # Use SMA as EMA seed
        ema_fast = sum(prices[:fast]) / fast
        ema_slow = sum(prices[:slow]) / slow

        # Advance fast EMA to the slow position so subsequent steps stay synchronized
        for i in range(fast, slow):
            ema_fast = TechnicalIndicators._ema_incremental(prices[i], ema_fast, fast)

        macd_history: List[float] = []
        for i in range(slow, n):
            ema_fast = TechnicalIndicators._ema_incremental(prices[i], ema_fast, fast)
            ema_slow = TechnicalIndicators._ema_incremental(prices[i], ema_slow, slow)
            macd_history.append(ema_fast - ema_slow)

        if len(macd_history) < signal:
            return (macd_history[-1] if macd_history else 0.0), 0.0, 0.0

        # Signal line = EMA of the MACD series
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
        """Calculate Bollinger Bands.

        Returns:
            (sma, upper_band, lower_band)
            Returns default values based on the last price when data is insufficient.
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
        """Calculate ATR (Average True Range) using Wilder smoothing.

        Returns:
            ATR value; returns 5.0 as default estimate when data is insufficient.
        """
        n = len(closes)
        if n < period + 1:
            return 5.0

        # Initial TR simple average
        tr_sum = 0.0
        for i in range(1, period + 1):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            tr_sum += max(high_low, high_close, low_close)
        atr = tr_sum / period

        # Wilder smoothing
        for i in range(period + 1, n):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            tr = max(high_low, high_close, low_close)
            atr = (atr * (period - 1) + tr) / period

        return atr


# ==================== ML Trading Bot ====================
class MLTradingBot:
    """ML Trading Bot: integrates state construction, decision generation, and model updates."""

    def __init__(self) -> None:
        self.agent = QLearningAgent(state_size=12, action_size=3)
        self.ti = TechnicalIndicators()
        self.action_names = ["Buy", "Hold", "Sell"]

        # Try to load a previously trained model
        self.agent.load()

    def create_state(
        self, price_data: List[float], holdings: float, cash: float, total_asset: float
    ) -> List[float]:
        """Build a 12-dimensional state vector from market and account data.

        State dimension descriptions:
        0. Normalized price
        1. 1-day price change rate
        2. 5-day price change rate
        3. 10-day price change rate
        4. RSI / 100
        5. MACD / 10
        6. MACD histogram / 10
        7. Bollinger Band position
        8. Holding ratio
        9. Cash ratio
        10. Short-term trend (1/0)
        11. Volatility
        """
        # Pad with the last price when data is insufficient to ensure stable indicator calculation
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

        # Short-term trend: whether current price is above the 5-period moving average
        trend = 1.0 if current_price > sum(prices[-5:]) / 5.0 else 0.0

        # 10-day volatility (standard deviation / current price)
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
        """Make a trading decision based on the current market state.

        Returns:
            Dictionary containing action, action_idx, confidence, q_values, and state.
        """
        state = self.create_state(price_data, holdings, cash, total_asset)
        action_idx = self.agent.choose_action(state, training)

        action = self.action_names[action_idx]
        q_values = self.agent.get_q_values(state)
        max_q = max(q_values) if q_values else 0.0
        # Confidence mapping: compress the absolute max Q-value into [0, 1]
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
        """Store experience and trigger model training."""
        self.agent.store_memory(state, action_idx, reward, next_state, done)
        self.agent.train()

    def save(self) -> None:
        """Save model to disk and update the global save timestamp."""
        global _model_saved_at
        self.agent.save()
        _model_saved_at = time.time()


# ==================== Global Singleton ====================
ml_bot = MLTradingBot()
_model_saved_at: float = time.time()  # Timestamp of the last model save


def _reload_latest_model_if_newer() -> bool:
    """Reload the model if the on-disk file is newer than the last save timestamp."""
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
            logger.info("Detected model update, reloaded (q_table size: %d)", len(ml_bot.agent.q_table))
        return ok
    return False


# ==================== Auto-Training ====================
def get_auto_train_status() -> Dict[str, Any]:
    """Return a snapshot of auto-training status for display on /api/ml/status."""
    with _auto_train_lock:
        return dict(_auto_train_status)


def _set_auto_train_status(**updates: Any) -> None:
    with _auto_train_lock:
        _auto_train_status.update(updates)


def _generate_synthetic_price_series(length: int) -> List[float]:
    """Generate synthetic prices with trend/oscillation/decline/shock regimes for startup pre-training."""
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
    """Background continuous training: generates synthetic data and trains indefinitely after startup, saving the model after each round."""
    if not AUTO_TRAIN_ON_START or AUTO_TRAIN_EPISODES <= 0:
        _set_auto_train_status(enabled=False, running=False)
        logger.info("Continuous training is disabled")
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
    logger.info("Continuous training started: %d episodes per batch, series length %d, will loop indefinitely", AUTO_TRAIN_EPISODES, AUTO_TRAIN_SERIES_LENGTH)

    total_updates = 0
    old_auto_save_freq = ml_bot.agent.auto_save_freq
    old_save_counter = ml_bot.agent._save_counter
    try:
        # Disable agent's internal auto_save during continuous training; save once per batch instead
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
            # Save the model after each batch so the trading API can see the latest model
            ml_bot.save()
            _set_auto_train_status(
                episodes_done=AUTO_TRAIN_EPISODES,
                updates=total_updates,
                batches_completed=batch,
            )
            logger.info("Continuous training batch %d complete: %d episodes, %d cumulative updates, model saved", batch, AUTO_TRAIN_EPISODES, total_updates)
    except Exception as exc:
        logger.exception("Continuous training exception")
        _set_auto_train_status(running=False, error=str(exc))
    finally:
        ml_bot.agent.auto_save_freq = old_auto_save_freq
        ml_bot.agent._save_counter = old_save_counter


def start_continuous_training_thread() -> None:
    """Start continuous training as a daemon thread to avoid blocking the HTTP server."""
    thread = threading.Thread(target=run_continuous_training, name="continuous-train", daemon=True)
    thread.start()


# ==================== HTTP Request Handler ====================
class RequestHandler(BaseHTTPRequestHandler):
    """REST API handler based on BaseHTTPRequestHandler.

    Each request runs in its own thread, sharing the global ml_bot instance.
    All operations on the agent's mutable internal state are protected by _agent_lock.
    """

    def log_message(self, format: str, *args: Any) -> None:
        """Custom log format using the logging module instead of print."""
        logger.info("%s - %s", self.address_string(), format % args)

    def send_cors_headers(self) -> None:
        """Only send CORS headers when ML_CORS_ORIGIN is explicitly configured (avoids default * wildcard)."""
        if not ML_CORS_ORIGIN:
            return
        self.send_header("Access-Control-Allow-Origin", ML_CORS_ORIGIN)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json_response(self, data: Dict[str, Any], status: int = 200) -> None:
        """Send a JSON response with automatic encoding handling."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_cors_headers()
        self.end_headers()
        try:
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            logger.error("JSON serialization failed: %s", exc)
            self.wfile.write('{"error":"Internal serialization error"}'.encode('utf-8'))

    def do_OPTIONS(self) -> None:
        """Handle browser CORS preflight requests."""
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
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
            self.send_json_response({"error": "Endpoint not found"}, 404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        content_length = self.headers.get("Content-Length")
        try:
            length = int(content_length) if content_length is not None else 0
        except ValueError:
            length = 0

        if length > MAX_BODY_SIZE:
            self.send_json_response({"error": "Request body too large"}, 413)
            return

        try:
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        except UnicodeDecodeError:
            self.send_json_response({"error": "Request body encoding error, only UTF-8 is supported"}, 400)
            return

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("Received invalid JSON request: %s", exc)
            self.send_json_response({"error": "Invalid JSON"}, 400)
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
                # Reload the latest model from disk
                with _agent_lock:
                    success = ml_bot.agent.load()
                self.send_json_response({"success": success, "q_table_size": len(ml_bot.agent.q_table)})
            else:
                self.send_json_response({"error": "Endpoint not found"}, 404)
        except Exception as exc:
            logger.exception("Uncaught exception handling request %s", parsed.path)
            self.send_json_response({"error": f"Internal server error: {exc}"}, 500)

    def handle_decide(self, data: Dict[str, Any]) -> None:
        """Handle trading decision requests.

        Expected JSON fields:
        - price_data: Price series (required, at least 5 data points)
        - holdings: Number of shares held (default 0)
        - cash: Cash amount (default 100)
        - total_asset: Total asset value (default 100)
        - training: Whether in training mode (default True)
        """
        # Check for and load the latest model before trading
        _reload_latest_model_if_newer()

        try:
            price_data = _validate_numeric_list(data.get("price_data"), "price_data", min_len=5)
            holdings = _validate_float(data.get("holdings", 0), "holdings")
            cash = _validate_float(data.get("cash", 100), "cash")
            total_asset = _validate_float(data.get("total_asset", 100), "total_asset")
            training = _validate_bool(data.get("training", True), "training")
        except ValueError as exc:
            logger.warning("Decision request parameter validation failed: %s", exc)
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
        """Handle model update requests (experience replay and training).

        Supports two parameter formats:
        1. Directly pass state / next_state (12-dimensional lists)
        2. Pass raw data like price_data / next_price_data, and the server builds the state automatically
        """
        try:
            action_idx = _validate_int(data.get("action_idx", 0), "action_idx", default=0, min_val=0, max_val=2)
            reward = _validate_float(data.get("reward", 0), "reward")
            done = _validate_bool(data.get("done", False), "done")
        except ValueError as exc:
            logger.warning("Update request parameter validation failed: %s", exc)
            self.send_json_response({"error": str(exc)}, 400)
            return

        state = data.get("state", [])
        next_state = data.get("next_state", [])

        # If state vector not provided directly, try to build from raw data
        if not state and data.get("price_data"):
            try:
                pd_state = _validate_numeric_list(data.get("price_data"), "price_data", min_len=5)
                h_state = _validate_float(data.get("holdings", 0), "holdings")
                c_state = _validate_float(data.get("cash", 100), "cash")
                ta_state = _validate_float(data.get("total_asset", 100), "total_asset")
                state = ml_bot.create_state(pd_state, h_state, c_state, ta_state)
            except ValueError as exc:
                logger.warning("Failed to build state: %s", exc)
                self.send_json_response({"error": f"Failed to build state: {exc}"}, 400)
                return

        if not next_state and data.get("next_price_data"):
            try:
                pd_next = _validate_numeric_list(data.get("next_price_data"), "next_price_data", min_len=5)
                h_next = _validate_float(data.get("next_holdings", 0), "next_holdings")
                c_next = _validate_float(data.get("next_cash", 100), "next_cash")
                ta_next = _validate_float(data.get("next_total_asset", 100), "next_total_asset")
                next_state = ml_bot.create_state(pd_next, h_next, c_next, ta_next)
            except ValueError as exc:
                logger.warning("Failed to build next_state: %s", exc)
                self.send_json_response({"error": f"Failed to build next_state: {exc}"}, 400)
                return

        if not state or not next_state:
            self.send_json_response({"error": "Insufficient update data: missing state or next_state"}, 400)
            return

        ml_bot.update(state, action_idx, reward, next_state, done)
        with _agent_lock:
            epsilon = ml_bot.agent.epsilon

        self.send_json_response({
            "success": True,
            "epsilon": epsilon,
        })

    def handle_indicators(self, data: Dict[str, Any]) -> None:
        """Handle technical indicator calculation requests.

        Expected JSON fields:
        - prices: Closing price series (required, at least 20 data points)
        """
        try:
            prices = _validate_numeric_list(data.get("prices"), "prices", min_len=20)
        except ValueError as exc:
            logger.warning("Indicator request parameter validation failed: %s", exc)
            self.send_json_response({"error": str(exc)}, 400)
            return

        # TechnicalIndicators is a pure function with no internal state, no lock needed
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


# ==================== Server Entry Point ====================
def run_server() -> None:
    """Start the HTTP server."""
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    logger.info("=" * 60)
    logger.info("ML Trading Bot Backend (Pure Python) starting...")
    logger.info("=" * 60)
    logger.info("API service address: http://%s:%d", HOST, PORT)
    logger.info("Available endpoints:")
    logger.info("   GET  /api/ml/status     - View status")
    logger.info("   POST /api/ml/decide     - Get ML trading decision")
    logger.info("   POST /api/ml/update     - Update ML model")
    logger.info("   POST /api/ml/indicators - Get technical indicators")
    logger.info("   POST /api/ml/reload     - Reload latest model")
    logger.info("   POST /api/ml/save       - Save model")
    logger.info("=" * 60)
    start_continuous_training_thread()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, server shutting down...")
    finally:
        server.server_close()
        logger.info("Server closed")


if __name__ == "__main__":
    run_server()