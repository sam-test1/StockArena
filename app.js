// ====================================================================
// 多股票交易模拟器 - 支持多只不同特征的股票，AI 可自由选择购买哪只
// ====================================================================

// ========== 股票池配置 ==========
// 每只股票有不同的波动率、漂移、相关性、起价，AI 可在不同股票间自由切换
const STOCKS = [
    {
        symbol: 'AAPL', name: '科技蓝筹', sector: '科技',
        startPrice: 1.00, color: '#1a56db', beta: 1.05, liquidity: 1.15,
        driftBias: 0.0006, volMult: 0.9, regimeWeights: { bull: 0.4, bear: 0.15, range: 0.4, crash: 0.05 },
        description: '稳健的科技龙头，趋势温和，波动较低。',
    },
    {
        symbol: 'MSFT', name: '稳健成长', sector: '科技',
        startPrice: 1.20, color: '#0ea5e9', beta: 0.95, liquidity: 1.2,
        driftBias: 0.0005, volMult: 0.85, regimeWeights: { bull: 0.35, bear: 0.15, range: 0.45, crash: 0.05 },
        description: '低波动、慢牛行情，适合长期持有。',
    },
    {
        symbol: 'TSLA', name: '高波动', sector: '汽车',
        startPrice: 0.85, color: '#ef4444', beta: 1.65, liquidity: 0.9,
        driftBias: 0.0008, volMult: 1.7, regimeWeights: { bull: 0.3, bear: 0.25, range: 0.25, crash: 0.2 },
        description: '高波动率，常出现暴涨暴跌，机会与风险并存。',
    },
    {
        symbol: 'NVDA', name: '热门赛道', sector: '芯片',
        startPrice: 1.50, color: '#9333ea', beta: 1.45, liquidity: 1.0,
        driftBias: 0.0012, volMult: 1.4, regimeWeights: { bull: 0.4, bear: 0.2, range: 0.25, crash: 0.15 },
        description: '高成长高波动，AI 风口赛道，趋势强。',
    },
    {
        symbol: 'COIN', name: '加密概念', sector: '金融',
        startPrice: 0.70, color: '#f59e0b', beta: 1.9, liquidity: 0.65,
        driftBias: 0.0000, volMult: 2.2, regimeWeights: { bull: 0.25, bear: 0.3, range: 0.2, crash: 0.25 },
        description: '跟随加密货币波动，肥尾分布，价格剧烈震荡。',
    },
    {
        symbol: 'KO', name: '防御消费', sector: '消费',
        startPrice: 1.00, color: '#10b981', beta: 0.45, liquidity: 1.4,
        driftBias: 0.0002, volMult: 0.55, regimeWeights: { bull: 0.25, bear: 0.15, range: 0.55, crash: 0.05 },
        description: '防御型蓝筹，波动小，震荡为主。',
    },
];

// ========== 全局状态 ==========
let isSimulationRunning = false;
let isMachineTradingEnabled = false;
let isMLRobotEnabled = false;
let usePythonML = true;
let mlBackendAvailable = false;
let simulationInterval;
let currentTimeRange = 5;
let startTime;

// 当前正在查看/手动交易的股票
let currentSymbol = STOCKS[0].symbol;

// 每只股票的价格历史与技术指标（按 symbol 存储）
const stockState = {};
// 每个交易者的每只股票持仓（user / machine / mlRobot）
const userHoldings = {};
const machineHoldings = {};
const mlRobotHoldings = {};

// 资产曲线（按时间汇总的总额）
let assetTimeData = [];
let userAssetData = [];
let machineAssetData = [];
let mlRobotAssetData = [];
let mlRobotPortfolioReturnWindow = [];
let mlRobotPortfolioSharpeHistory = [];
let lastMLPortfolioValue = null;

// 跨股票总览
const lastAssetRecordTime = { value: Date.now() };

// ========== 初始化每只股票的状态 ==========
function initStockStates() {
    for (const s of STOCKS) {
        stockState[s.symbol] = {
            symbol: s.symbol,
            currentPrice: s.startPrice,
            lastPrice: s.startPrice,
            priceData: [],
            highData: [],
            lowData: [],
            volumeData: [],
            timeData: [],
            rsiData: [],
            macdData: [],
            signalLineData: [],
            histogramData: [],
            ma5Data: [],
            ma10Data: [],
            marketRegime: 'range',
            regimeStepsLeft: 30,
            garchVariance: 0.0009,
            volumeARState: 1.0,
            // 这只股票的 startPrice 锚点（用于"买入持有"基准）
            startPriceRef: s.startPrice,
        };

        userHoldings[s.symbol] = {
            shares: 0, totalInvested: 0, txHistory: [],
        };
        machineHoldings[s.symbol] = {
            shares: 0, totalInvested: 0, txHistory: [],
            stopLoss: null, trailingHigh: null, entryTime: null,
            tookProfit1: false, tookProfit2: false,
        };
        mlRobotHoldings[s.symbol] = {
            shares: 0, totalInvested: 0, txHistory: [],
            qTable: {}, qTableAccess: {},
            stopLossPrice: null, takeProfitPrice: null,
            trailingPeak: null, lastBuyTime: null,
            returnLogWindow: [], returnWindow: [], downsideWindow: [],
            rewardHistory: [], accuracyHistory: [], sharpeHistory: [],
            assetData: [],
        };
    }
}

// 用户/Machine/ML 机器人的总现金（跨所有股票共享）
let userCash = 100.00;
let machineCash = 100.00;
let mlRobotCash = 100.00;

// 机器交易状态（跨股票）
let machinePeakAsset = 100;
let machineConsecutiveLosses = 0;
let machineLossCooldownUntil = 0;
let machineLastTradeProfit = 0;

// ML 机器人跨股票状态
const mlRobotStats = {
    startTime: Date.now(),
    maxDrawdown: 0,
    maxDrawdownAsset: 100,
    winTrades: 0,
    lossTrades: 0,
    grossProfit: 0,
    grossLoss: 0,
    biggestWin: 0,
    biggestLoss: 0,
    totalFees: 0,
    floatingPnL: 0,
    assetPeak: 100,
    totalReward: 0,
    correctDecisions: 0,
    totalDecisions: 0,
    learningRate: 0.1,
    discountFactor: 0.9,
    explorationRate: 0.3,
    explorationDecay: 0.995,
    isProcessing: false,
};

// ML 配置常量
const Q_TABLE_MAX_SIZE = 5000;
const EPSILON_DECAY_STEP = 10;
const REWARD_WINDOW = 50;
const SHARPE_WINDOW = 60;
const ML_AUTO_SAVE_INTERVAL_MS = 5 * 60 * 1000;

// 规则机器人交易参数
const MAX_DRAWDOWN_THRESHOLD = 0.15;
const MAX_CONSECUTIVE_LOSSES = 3;
const LOSS_COOLDOWN_MS = 60 * 1000;
const ENTRY_MIN_HOLD_MS = 30 * 1000;
const ENTRY_MAX_HOLD_MS = 5 * 60 * 1000;
const VOL_TARGET_PER_STEP = 0.0015;
const MAX_ENTRY_VOL = 0.05;
const DRAWDOWN_COOLDOWN_MULT = 1.5;
const MACHINE_MAX_POSITIONS = 4;
const MACHINE_MAX_POSITION_RATIO = 0.35;
const MACHINE_MAX_NEW_BUYS_PER_STEP = 2;
const MACHINE_CASH_RESERVE_RATIO = 0.05;

// 学习 AI 组合约束：允许同时持有多只股票，但控制单股集中度和现金垫。
const ML_MAX_POSITIONS = 4;
const ML_MAX_POSITION_RATIO = 0.35;
const ML_MAX_NEW_BUYS_PER_STEP = 3;
const ML_CASH_RESERVE_RATIO = 0.05;

// 全市场共同因子，让多股走势更像真实市场：个股不再完全独立随机游走。
const globalMarketState = {
    tick: 0,
    regime: 'range',
    stepsLeft: 45,
    lastReturn: 0,
    volatility: 0.010,
    eventShock: 0,
    sectorReturns: {},
};

// ML 后端
const ML_API_URL = window.location.origin;

// 周期自动保存
let mlAutoSaveTimer = null;
let mlAutoSaveInFlight = false;
let saveQTableTimer = null;
let saveQTablePending = false;

// 图表对象
let priceChart, rsiChart, macdChart, assetChart;
let rewardChart, accuracyChart, sharpeChart;

// DOM 元素引用
let startBtn, cashDisplay, totalAssetDisplay, currentPriceDisplay;
let priceChangeDisplay, priceIndicator, buyMessage;
let sellSharesInput, sellBtn, sellMessage, transactionHistoryTable;
let maCheckbox, rsiCheckbox, macdCheckbox;
let timeRange5, timeRange15, timeRange30, buySharesBtns;
let mlRobotToggle, learningRateSlider, learningRateDisplay;
let explorationRateSlider, explorationRateDisplay, resetMLModelBtn, mlRobotTransactionsBtn;
let currentSharpeDisplay;
let stockSelector, stockListEl, stockCardsEl, currentStockNameEl, currentStockSectorEl;
// 用户/规则机器人/学习机器人 现金与总值显示
let machineCashDisplay, machineTotalAssetDisplay;
let mlRobotCashDisplay, mlRobotTotalAssetDisplay;
let currentSharesDisplay, currentCostDisplay, currentValueDisplay;
let machineSharesDisplay, machineCostDisplay, machineValueDisplay;
let mlRobotSharesDisplay, mlRobotCostDisplay, mlRobotValueDisplay;
let userTransactionsBtn, machineTransactionsBtn;
let btTotalAsset, btReturnRate, btSharpe, btAccuracy, btDecisions;
let btTotalReward, btAvgReward, btWinRate, btTradeCount, btProfitFactor;
let btAvgPnL, btMaxDrawdown, btMaxDDAsset, btFloatingPnL, btFees;
let btAlphas, btMachine, btBuyHold, backtestSessionDuration, resetBacktestBtn;

// 预训练相关
let isPretraining = false;
let pretrainProgressContainer, pretrainProgressBar, pretrainProgressPercent;
let pretrainStatus, pretrainAccuracy, pretrainQTableSize;
let pretrainCancelRequested = false;

// 当前正在查看的交易历史类型
let currentTxView = 'user'; // 'user' | 'machine' | 'mlRobot'

// ====================================================================
// 工具函数
// ====================================================================

let _spareGaussian = null;
function gaussian() {
    if (_spareGaussian !== null) { const v = _spareGaussian; _spareGaussian = null; return v; }
    let u1 = 0, u2 = 0;
    while (u1 === 0) u1 = Math.random();
    while (u2 === 0) u2 = Math.random();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    _spareGaussian = mag * Math.sin(2.0 * Math.PI * u2);
    return mag * Math.cos(2.0 * Math.PI * u2);
}
function studentT(df) {
    const z = gaussian();
    let chi2 = 0;
    for (let i = 0; i < df; i++) { const g = gaussian(); chi2 += g * g; }
    return z / Math.sqrt(chi2 / df);
}

function getCurrentStock() { return stockState[currentSymbol]; }
function getCurrentStockCfg() { return STOCKS.find(s => s.symbol === currentSymbol); }

function getUserTotalAsset() {
    let v = userCash;
    for (const s of STOCKS) {
        v += userHoldings[s.symbol].shares * stockState[s.symbol].currentPrice;
    }
    return v;
}
function getMachineTotalAsset() {
    let v = machineCash;
    for (const s of STOCKS) {
        v += machineHoldings[s.symbol].shares * stockState[s.symbol].currentPrice;
    }
    return v;
}
function getMLRobotTotalAsset() {
    let v = mlRobotCash;
    for (const s of STOCKS) {
        v += mlRobotHoldings[s.symbol].shares * stockState[s.symbol].currentPrice;
    }
    return v;
}
function getHoldingValue(holdingsMap, symbol) {
    return holdingsMap[symbol].shares * stockState[symbol].currentPrice;
}
function countOpenPositions(holdingsMap) {
    return STOCKS.reduce((count, s) => count + (holdingsMap[s.symbol].shares > 0 ? 1 : 0), 0);
}
function calculateSharpeRatio(returns, minSamples = 2) {
    const clean = returns.filter((v) => Number.isFinite(v));
    if (clean.length < minSamples) return null;
    const mean = clean.reduce((a, b) => a + b, 0) / clean.length;
    const variance = clean.reduce((a, b) => a + (b - mean) ** 2, 0) / clean.length;
    const std = Math.sqrt(variance);
    if (std <= 1e-8) return mean >= 0 ? 0 : -0;
    // 以滚动 tick 为单位，乘 sqrt(60) 让读数更接近短周期年化/日内尺度。
    return (mean / std) * Math.sqrt(Math.min(60, clean.length));
}
function recordMLPortfolioMetrics(totalValue) {
    if (!Number.isFinite(totalValue) || totalValue <= 0) return;
    if (lastMLPortfolioValue !== null && lastMLPortfolioValue > 0) {
        const ret = Math.log(totalValue / lastMLPortfolioValue);
        if (Number.isFinite(ret)) {
            mlRobotPortfolioReturnWindow.push(ret);
            if (mlRobotPortfolioReturnWindow.length > SHARPE_WINDOW) mlRobotPortfolioReturnWindow.shift();
            const sharpe = calculateSharpeRatio(mlRobotPortfolioReturnWindow);
            if (sharpe !== null) {
                mlRobotPortfolioSharpeHistory.push(parseFloat(sharpe.toFixed(3)));
                if (mlRobotPortfolioSharpeHistory.length > 200) mlRobotPortfolioSharpeHistory.shift();
            }
        }
    }
    lastMLPortfolioValue = totalValue;
}

function calculateEMA(data, period) {
    if (data.length < period) return data.length > 0 ? data[data.length - 1] : 0;
    const k = 2 / (period + 1);
    let ema = data[data.length - period];
    for (let i = data.length - period + 1; i < data.length; i++) {
        ema = data[i] * k + ema * (1 - k);
    }
    return ema;
}
function calculateSMA(data, window) {
    if (data.length < window) return null;
    const values = data.slice(-window);
    return values.reduce((sum, value) => sum + value, 0) / window;
}

// ====================================================================
// 价格生成（每只股票独立）
// ====================================================================
const MARKET_REGIMES = {
    bull:  { drift: 0.0011, volMult: 0.95, weights: { bull: 0.58, range: 0.25, bear: 0.12, crash: 0.05 } },
    bear:  { drift: -0.0011, volMult: 1.25, weights: { bull: 0.14, range: 0.27, bear: 0.48, crash: 0.11 } },
    range: { drift: 0.0000, volMult: 0.70, weights: { bull: 0.25, range: 0.50, bear: 0.20, crash: 0.05 } },
    crash: { drift: -0.0045, volMult: 2.10, weights: { bull: 0.08, range: 0.22, bear: 0.45, crash: 0.25 } },
};

const SECTOR_BETA = {
    '科技': 1.15,
    '汽车': 1.35,
    '芯片': 1.30,
    '金融': 1.20,
    '消费': 0.55,
};

function pickWeighted(weights, fallback = 'range') {
    const r = Math.random();
    let acc = 0;
    for (const key of Object.keys(weights)) {
        acc += weights[key];
        if (r <= acc) return key;
    }
    return fallback;
}

function generateMarketStep() {
    globalMarketState.tick += 1;
    if (globalMarketState.stepsLeft <= 0) {
        const current = MARKET_REGIMES[globalMarketState.regime] || MARKET_REGIMES.range;
        globalMarketState.regime = pickWeighted(current.weights, 'range');
        globalMarketState.stepsLeft = 35 + Math.floor(Math.random() * 85);
    }
    globalMarketState.stepsLeft -= 1;

    const regime = MARKET_REGIMES[globalMarketState.regime] || MARKET_REGIMES.range;
    const phase = (globalMarketState.tick % 240) / 240;
    // U 型日内波动：开盘/尾盘更活跃，中段更平稳。
    const intradayVolMult = 0.75 + 0.55 * Math.abs(Math.cos(2 * Math.PI * phase));
    const shockDecay = 0.88;
    if (Math.random() < 0.012) {
        const direction = Math.random() < (globalMarketState.regime === 'crash' ? 0.72 : 0.48) ? -1 : 1;
        globalMarketState.eventShock += direction * (0.008 + Math.random() * 0.030);
    }
    globalMarketState.eventShock *= shockDecay;

    const marketNoise = studentT(6) / Math.sqrt(6 / 4);
    const marketVol = 0.009 * regime.volMult * intradayVolMult;
    const rawMarketReturn = regime.drift + marketNoise * marketVol + globalMarketState.eventShock;
    globalMarketState.lastReturn = Math.max(-0.09, Math.min(0.07, rawMarketReturn));
    globalMarketState.volatility = marketVol;

    const sectorReturns = {};
    for (const sector of Object.keys(SECTOR_BETA)) {
        const sectorNoise = gaussian() * marketVol * 0.45;
        sectorReturns[sector] = globalMarketState.lastReturn * (SECTOR_BETA[sector] - 1) * 0.35 + sectorNoise;
    }
    globalMarketState.sectorReturns = sectorReturns;
}

function generateNewPrice(symbol) {
    const st = stockState[symbol];
    const cfg = STOCKS.find(s => s.symbol === symbol);
    st.lastPrice = st.currentPrice;

    // 1) 个股状态切换（受全市场 regime 影响，但保留个股独立性）
    if (st.regimeStepsLeft <= 0) {
        const mixedWeights = { ...cfg.regimeWeights };
        mixedWeights[globalMarketState.regime] = (mixedWeights[globalMarketState.regime] || 0) + 0.20;
        const totalWeight = Object.values(mixedWeights).reduce((a, b) => a + b, 0);
        for (const key of Object.keys(mixedWeights)) mixedWeights[key] /= totalWeight;
        st.marketRegime = pickWeighted(mixedWeights, 'range');
        st.regimeStepsLeft = 25 + Math.floor(Math.random() * 70);
    }
    st.regimeStepsLeft = Math.max(0, st.regimeStepsLeft - 1);
    const regime = MARKET_REGIMES[st.marketRegime];

    // 2) GARCH 波动率：更低的基础波动 + 肥尾冲击，避免价格每秒过度跳动。
    const GARCH_OMEGA = 0.000006, GARCH_ALPHA = 0.07, GARCH_BETA = 0.91;
    let baseVol = 0.010 * regime.volMult * cfg.volMult;
    if (st.priceData.length >= 2) {
        const prevRet = (st.priceData[st.priceData.length - 1] - st.priceData[st.priceData.length - 2]) / st.priceData[st.priceData.length - 2];
        const prevRetSq = prevRet * prevRet;
        st.garchVariance = GARCH_OMEGA + GARCH_ALPHA * prevRetSq + GARCH_BETA * st.garchVariance;
        st.garchVariance = Math.max(1e-8, Math.min(0.012, st.garchVariance));
        const garchVol = Math.sqrt(st.garchVariance);
        baseVol = Math.min(0.045, Math.max(0.003, 0.55 * garchVol + 0.45 * baseVol));
    } else {
        st.garchVariance = 0.0001;
    }
    const volatility = baseVol;

    // 3) 漂移：市场共同因子 + 行业因子 + 个股 bias + 均值回归 + 温和动量
    let drift = regime.drift + cfg.driftBias;
    const marketComponent = (cfg.beta || 1) * globalMarketState.lastReturn;
    const sectorComponent = globalMarketState.sectorReturns[cfg.sector] || 0;
    const SOFT_CAP = 1000, SOFT_FLOOR = 0.05, REV_BAND = 200;
    if (st.currentPrice > SOFT_CAP - REV_BAND) {
        const overshoot = (st.currentPrice - (SOFT_CAP - REV_BAND)) / REV_BAND;
        drift -= 0.02 * overshoot * overshoot;
    }
    if (st.priceData.length >= 20) {
        const recent = st.priceData.slice(-60);
        const longMean = recent.reduce((a, b) => a + b, 0) / recent.length;
        const deviation = (longMean - st.currentPrice) / st.currentPrice;
        drift += 0.08 * Math.max(-0.02, Math.min(0.02, deviation));
    }
    if (st.priceData.length >= 6) {
        let momentum = 0;
        for (let i = st.priceData.length - 5; i < st.priceData.length; i++) {
            momentum += (st.priceData[i] - st.priceData[i - 1]) / st.priceData[i - 1];
        }
        drift += 0.10 * Math.max(-0.012, Math.min(0.012, momentum));
    }

    // 4) 噪声 + 情绪 + 个股新闻。用 log return 合成，涨跌幅更接近真实交易。
    const tNoise = studentT(7) / Math.sqrt(7 / 5);
    let sentiment = 0;
    if (Math.random() < 0.025) sentiment = (Math.random() - 0.5) * 0.018 * cfg.volMult;
    let news = 0;
    if (Math.random() < 0.006) {
        const newsSign = Math.random() < 0.52 ? 1 : -1;
        news = newsSign * (0.012 + Math.random() * 0.045) * cfg.volMult;
    }

    // 5) 合成新价
    let totalChange = drift + marketComponent + sectorComponent + tNoise * volatility + sentiment + news;
    const clipDown = globalMarketState.regime === 'crash' ? -0.13 : -0.08;
    const clipUp = globalMarketState.regime === 'bull' ? 0.09 : 0.07;
    totalChange = Math.max(clipDown, Math.min(clipUp, totalChange));
    let newPrice = st.currentPrice * Math.exp(totalChange);
    if (newPrice < SOFT_FLOOR) newPrice = SOFT_FLOOR;
    if (newPrice > SOFT_CAP) {
        const excess = newPrice - SOFT_CAP;
        newPrice = SOFT_CAP + Math.tanh(excess / 50) * 5;
    }
    st.currentPrice = parseFloat(newPrice.toFixed(2));

    // 6) OHLC 范围：high/low 包含上一价与收盘价，避免 K 线不可能形态。
    const open = st.lastPrice;
    const close = st.currentPrice;
    const intrabarRange = Math.max(0.003, close * (volatility + Math.abs(totalChange) * 0.30) * (0.7 + Math.random() * 0.8));
    const simulatedHigh = parseFloat((Math.max(open, close) + intrabarRange * (0.35 + Math.random() * 0.45)).toFixed(2));
    const simulatedLow = parseFloat(Math.max(0.01, Math.min(open, close) - intrabarRange * (0.35 + Math.random() * 0.45)).toFixed(2));

    // 7) 量能：和波动、新闻、下跌恐慌、流动性绑定，带 AR 平滑。
    const absChange = Math.abs(totalChange);
    const trendVolumeBoost = absChange * 18;
    const downsideBoost = totalChange < -0.015 ? 0.8 : 0;
    const regimeVolumeMult = globalMarketState.regime === 'crash' ? 2.2 : (globalMarketState.regime === 'range' ? 0.85 : 1.05);
    const newsVolumeBoost = news !== 0 ? 1.8 : 0;
    const volumeTarget = (cfg.liquidity || 1) * regimeVolumeMult * (1 + trendVolumeBoost + downsideBoost + newsVolumeBoost);
    st.volumeARState = 0.78 * st.volumeARState + 0.22 * volumeTarget + gaussian() * 0.08;
    st.volumeARState = Math.max(0.25, Math.min(5.0, st.volumeARState));
    const simulatedVolume = Math.max(800, Math.round(12000 * st.volumeARState * (0.85 + Math.random() * 0.3)));

    // 8) 推送序列
    const now = new Date();
    const timeLabel = now.getHours().toString().padStart(2, '0') + ':' +
                     now.getMinutes().toString().padStart(2, '0') + ':' +
                     now.getSeconds().toString().padStart(2, '0');
    st.priceData.push(st.currentPrice);
    st.highData.push(simulatedHigh);
    st.lowData.push(simulatedLow);
    st.volumeData.push(simulatedVolume);
    st.timeData.push(timeLabel);

    // 10) MA
    const len = st.priceData.length;
    if (len >= 5) {
        let sum = 0; for (let i = len - 5; i < len; i++) sum += st.priceData[i];
        st.ma5Data.push(parseFloat((sum / 5).toFixed(2)));
    } else st.ma5Data.push(null);
    if (len >= 10) {
        let sum = 0; for (let i = len - 10; i < len; i++) sum += st.priceData[i];
        st.ma10Data.push(parseFloat((sum / 10).toFixed(2)));
    } else st.ma10Data.push(null);
}

// ====================================================================
// 技术指标
// ====================================================================
function calculateTechnicalIndicators(symbol) {
    const st = stockState[symbol];
    const prices = st.priceData;
    if (prices.length >= 14) {
        const gains = [], losses = [];
        for (let i = 1; i < prices.length; i++) {
            const c = prices[i] - prices[i - 1];
            gains.push(c > 0 ? c : 0);
            losses.push(c < 0 ? Math.abs(c) : 0);
        }
        const avgGain = gains.slice(-14).reduce((a, b) => a + b, 0) / 14;
        const avgLoss = losses.slice(-14).reduce((a, b) => a + b, 0) / 14;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        const rsi = avgLoss === 0 ? 100 : 100 - (100 / (1 + rs));
        st.rsiData.push(parseFloat(rsi.toFixed(2)));
    } else st.rsiData.push(null);

    if (prices.length >= 26) {
        const ema12 = calculateEMA(prices, 12);
        const ema26 = calculateEMA(prices, 26);
        const macdLine = ema12 - ema26;
        let signalLine;
        if (st.macdData.length >= 9) {
            signalLine = calculateEMA(st.macdData.slice(-9), 9);
        } else signalLine = calculateEMA(st.macdData, 9);
        const histogram = macdLine - signalLine;
        st.macdData.push(parseFloat(macdLine.toFixed(4)));
        st.signalLineData.push(parseFloat(signalLine.toFixed(4)));
        st.histogramData.push(parseFloat(histogram.toFixed(4)));
    } else {
        st.macdData.push(null); st.signalLineData.push(null); st.histogramData.push(null);
    }
}

function calculateRSIFromPrices(prices, period = 21) {
    if (prices.length < period + 1) return null;
    const recent = prices.slice(-(period + 1));
    let gains = 0, losses = 0;
    for (let i = 1; i < recent.length; i++) {
        const c = recent[i] - recent[i - 1];
        if (c > 0) gains += c; else losses += Math.abs(c);
    }
    const avgGain = gains / period, avgLoss = losses / period;
    if (avgLoss === 0) return 100;
    return 100 - (100 / (1 + avgGain / avgLoss));
}
function calculateATRFromState(st, period = 14) {
    if (st.priceData.length < period + 1) return null;
    const trs = [];
    for (let i = st.priceData.length - period; i < st.priceData.length; i++) {
        const hl = st.highData[i] - st.lowData[i];
        const hc = Math.abs(st.highData[i] - st.priceData[i - 1]);
        const lc = Math.abs(st.lowData[i] - st.priceData[i - 1]);
        trs.push(Math.max(hl, hc, lc));
    }
    return trs.reduce((a, b) => a + b, 0) / trs.length;
}
function isVolumeConfirmed(st, period = 60, threshold = 1.2) {
    if (st.volumeData.length < period) return false;
    const last = st.volumeData[st.volumeData.length - 1];
    const avg = st.volumeData.slice(-period).reduce((a, b) => a + b, 0) / period;
    return last > avg * threshold;
}
function isMacdHistogramRising(st) {
    if (st.histogramData.length < 2) return false;
    const l = st.histogramData[st.histogramData.length - 1];
    const p = st.histogramData[st.histogramData.length - 2];
    if (l === null || p === null) return false;
    return l > 0 && l > p;
}

function calculateRecentVolatility(prices) {
    if (prices.length < 10) return 0;
    const recent = prices.slice(-10);
    let v = 0;
    for (let i = 1; i < recent.length; i++) v += Math.abs((recent[i] - recent[i - 1]) / recent[i - 1]);
    return v / (recent.length - 1);
}

// ====================================================================
// 模拟控制
// ====================================================================
function toggleSimulation() {
    if (isSimulationRunning) {
        clearInterval(simulationInterval);
        stopAutoSave();
        flushSaveQTable();
        saveMLModel();
        startBtn.innerHTML = '<i class="fa fa-play-circle mr-2"></i>开始训练';
        startBtn.classList.remove('bg-danger');
        startBtn.classList.add('bg-primary');
        buySharesBtns.forEach(btn => { btn.disabled = true; btn.classList.add('opacity-50', 'cursor-not-allowed'); });
        sellBtn.disabled = true;
        isSimulationRunning = false;
    } else {
        startTime = new Date();
        resetChartData();
        // 开始交易前，从 Python ML 后端加载最新模型
        reloadLatestMLModel();
        simulationInterval = setInterval(updateSimulation, 1000);
        startAutoSave();
        startBtn.innerHTML = '<i class="fa fa-stop-circle mr-2"></i>停止训练';
        startBtn.classList.remove('bg-primary');
        startBtn.classList.add('bg-danger');
        buySharesBtns.forEach(btn => { btn.disabled = false; btn.classList.remove('opacity-50', 'cursor-not-allowed'); });
        sellBtn.disabled = false;
        isSimulationRunning = true;
    }
}

function updateSimulation() {
    // 1) 先生成全市场共同因子，再让每只股票在同一市场环境中报价
    generateMarketStep();
    for (const s of STOCKS) {
        generateNewPrice(s.symbol);
        calculateTechnicalIndicators(s.symbol);
    }

    // 2) AI 交易：扫描所有股票并做组合级调仓
    if (isMachineTradingEnabled) {
        executeMachinePortfolioTrading();
    }
    if (isMLRobotEnabled) {
        executeMLRobotTrading();
    }

    updateCharts();
    updateAssetDisplay();
    updateMultiStockOverview();

    const now = Date.now();
    if (now - lastAssetRecordTime.value >= 5000) {
        recordAssetData();
        lastAssetRecordTime.value = now;
    }
    limitDataPoints();
}

// ====================================================================
// 图表更新
// ====================================================================
function updateCharts() {
    const st = getCurrentStock();
    const cfg = getCurrentStockCfg();
    const maxDisplayPoints = currentTimeRange * 10;
    const startIndex = Math.max(0, st.timeData.length - maxDisplayPoints);

    if (priceChart) {
        priceChart.data.labels = st.timeData.slice(startIndex);
        priceChart.data.datasets[0].data = st.priceData.slice(startIndex);
        priceChart.data.datasets[0].borderColor = cfg.color;
        priceChart.data.datasets[0].backgroundColor = cfg.color + '20';
        priceChart.data.datasets[1].data = st.ma5Data.slice(startIndex);
        priceChart.data.datasets[2].data = st.ma10Data.slice(startIndex);
        priceChart.data.datasets[1].hidden = !maCheckbox.checked;
        priceChart.data.datasets[2].hidden = !maCheckbox.checked;
        priceChart.update('none');
    }
    if (rsiChart) {
        rsiChart.data.labels = st.timeData.slice(startIndex);
        rsiChart.data.datasets[0].data = st.rsiData.slice(startIndex);
        rsiChart.data.datasets[1].data = Array(Math.min(maxDisplayPoints, st.timeData.length)).fill(70);
        rsiChart.data.datasets[2].data = Array(Math.min(maxDisplayPoints, st.timeData.length)).fill(30);
        rsiChart.update('none');
    }
    if (macdChart) {
        macdChart.data.labels = st.timeData.slice(startIndex);
        macdChart.data.datasets[0].data = st.histogramData.slice(startIndex);
        macdChart.data.datasets[1].data = st.macdData.slice(startIndex);
        macdChart.data.datasets[2].data = st.signalLineData.slice(startIndex);
        macdChart.update('none');
    }

    document.getElementById('rsiChart').parentNode.parentNode.style.display = rsiCheckbox.checked ? 'block' : 'none';
    document.getElementById('macdChart').parentNode.parentNode.style.display = macdCheckbox.checked ? 'block' : 'none';

    // 更新当前价格显示
    if (st.priceData.length > 0) {
        const lastIdx = st.priceData.length - 1;
        const priceChange = st.currentPrice - st.lastPrice;
        const priceChangePct = st.lastPrice > 0 ? (priceChange / st.lastPrice) * 100 : 0;
        currentPriceDisplay.textContent = '¥' + st.currentPrice.toFixed(2);
        priceChangeDisplay.textContent = priceChange.toFixed(2) + ' (' + priceChangePct.toFixed(2) + '%)';
        if (priceChange > 0) {
            priceIndicator.classList.remove('bg-danger');
            priceIndicator.classList.add('bg-success');
            priceChangeDisplay.classList.remove('text-danger');
            priceChangeDisplay.classList.add('text-success');
        } else if (priceChange < 0) {
            priceIndicator.classList.remove('bg-success');
            priceIndicator.classList.add('bg-danger');
            priceChangeDisplay.classList.remove('text-success');
            priceChangeDisplay.classList.add('text-danger');
        }
    }
}

function updateMLCharts() {
    if (!rewardChart || !accuracyChart) return;

    const maxDataPoints = 50;

    // 汇总所有 6 只股票的历史：每步取一个聚合值（平均）
    // 累计奖励：把 6 只股票每步的奖励求和
    let maxLen = 0;
    for (const s of STOCKS) {
        maxLen = Math.max(maxLen, mlRobotHoldings[s.symbol].rewardHistory.length);
    }
    const rewardData = [];
    const accuracyData = [];
    const perStockSharpeData = [];
    for (let i = 0; i < maxLen; i++) {
        let sumR = 0, cntR = 0;
        let sumA = 0, cntA = 0;
        let sumS = 0, cntS = 0;
        for (const s of STOCKS) {
            const rh = mlRobotHoldings[s.symbol];
            if (i < rh.rewardHistory.length) { sumR += rh.rewardHistory[i]; cntR++; }
            if (i < rh.accuracyHistory.length) { sumA += rh.accuracyHistory[i]; cntA++; }
            if (i < rh.sharpeHistory.length) { sumS += rh.sharpeHistory[i]; cntS++; }
        }
        // 累计奖励用"求和"——6 只股票的总奖励
        rewardData.push(cntR > 0 ? parseFloat((sumR).toFixed(2)) : 0);
        // 准确率用平均
        accuracyData.push(cntA > 0 ? parseFloat((sumA / cntA).toFixed(2)) : 0);
        // 单股夏普保留给兼容；优先展示组合级夏普
        perStockSharpeData.push(cntS > 0 ? parseFloat((sumS / cntS).toFixed(3)) : 0);
    }

    // 取最近 50 个数据点
    const labels = rewardData.map((_, i) => `${i + 1}`);
    const rSliced = rewardData.slice(-maxDataPoints);
    const aSliced = accuracyData.slice(-maxDataPoints);
    const sharpeSource = mlRobotPortfolioSharpeHistory.length > 0 ? mlRobotPortfolioSharpeHistory : perStockSharpeData;
    const sSliced = sharpeSource.slice(-maxDataPoints);
    const lSliced = labels.slice(-maxDataPoints);
    const sharpeLabels = sharpeSource.map((_, i) => `${i + 1}`).slice(-maxDataPoints);

    rewardChart.data.labels = lSliced;
    rewardChart.data.datasets[0].data = rSliced;
    rewardChart.update('none');

    accuracyChart.data.labels = lSliced;
    accuracyChart.data.datasets[0].data = aSliced;
    accuracyChart.update('none');

    if (sharpeChart) {
        sharpeChart.data.labels = sharpeLabels;
        sharpeChart.data.datasets[0].data = sSliced;
        sharpeChart.update('none');

        if (currentSharpeDisplay) {
            // 末尾夏普值
            const last = sSliced.length > 0 ? sSliced[sSliced.length - 1] : null;
            currentSharpeDisplay.textContent = last === null ? '--' : last.toFixed(2);
            currentSharpeDisplay.classList.remove('text-green-600', 'text-red-600', 'text-gray-500');
            currentSharpeDisplay.classList.add(last === null ? 'text-gray-500' : (last >= 0 ? 'text-green-600' : 'text-red-600'));
        }
    }
}

function recordAssetData() {
    const now = new Date();
    const timeLabel = now.getHours().toString().padStart(2, '0') + ':' + now.getMinutes().toString().padStart(2, '0');
    const userTotal = getUserTotalAsset();
    const machineTotal = getMachineTotalAsset();
    const mlTotal = getMLRobotTotalAsset();

    assetTimeData.push(timeLabel);
    userAssetData.push(parseFloat(userTotal.toFixed(2)));
    machineAssetData.push(parseFloat(machineTotal.toFixed(2)));
    mlRobotAssetData.push(parseFloat(mlTotal.toFixed(2)));
    recordMLPortfolioMetrics(mlTotal);

    if (assetChart) {
        assetChart.data.labels = assetTimeData;
        assetChart.data.datasets[0].data = userAssetData;
        assetChart.data.datasets[1].data = machineAssetData;
        assetChart.data.datasets[2].data = mlRobotAssetData;
        assetChart.data.datasets[1].hidden = !isMachineTradingEnabled;
        assetChart.data.datasets[2].hidden = !isMLRobotEnabled;
        assetChart.update('none');
    }

    updateMLCharts();
}

function updateAssetDisplay() {
    const st = getCurrentStock();
    const uh = userHoldings[currentSymbol];
    const mh = machineHoldings[currentSymbol];
    const rh = mlRobotHoldings[currentSymbol];

    // 用户
    cashDisplay.textContent = '¥' + userCash.toFixed(2);
    if (currentSharesDisplay) currentSharesDisplay.textContent = uh.shares + ' 股';
    const avgCost = uh.shares > 0 ? (uh.totalInvested / uh.shares) : 0;
    if (currentCostDisplay) currentCostDisplay.textContent = '¥' + avgCost.toFixed(2);
    const stockValue = uh.shares * st.currentPrice;
    if (currentValueDisplay) currentValueDisplay.textContent = '¥' + stockValue.toFixed(2);
    totalAssetDisplay.textContent = '¥' + getUserTotalAsset().toFixed(2);

    // 规则机器人
    if (machineCashDisplay) machineCashDisplay.textContent = '¥' + machineCash.toFixed(2);
    if (machineSharesDisplay) machineSharesDisplay.textContent = mh.shares + ' 股';
    const mAvgCost = mh.shares > 0 ? (mh.totalInvested / mh.shares) : 0;
    if (machineCostDisplay) machineCostDisplay.textContent = '¥' + mAvgCost.toFixed(2);
    const mStockValue = mh.shares * st.currentPrice;
    if (machineValueDisplay) machineValueDisplay.textContent = '¥' + mStockValue.toFixed(2);
    if (machineTotalAssetDisplay) machineTotalAssetDisplay.textContent = '¥' + getMachineTotalAsset().toFixed(2);

    // ML 机器人
    if (mlRobotCashDisplay) mlRobotCashDisplay.textContent = '¥' + mlRobotCash.toFixed(2);
    if (mlRobotSharesDisplay) mlRobotSharesDisplay.textContent = rh.shares + ' 股';
    const rAvgCost = rh.shares > 0 ? (rh.totalInvested / rh.shares) : 0;
    if (mlRobotCostDisplay) mlRobotCostDisplay.textContent = '¥' + rAvgCost.toFixed(2);
    const rStockValue = rh.shares * st.currentPrice;
    if (mlRobotValueDisplay) mlRobotValueDisplay.textContent = '¥' + rStockValue.toFixed(2);
    if (mlRobotTotalAssetDisplay) mlRobotTotalAssetDisplay.textContent = '¥' + getMLRobotTotalAsset().toFixed(2);

    updateBacktestInfo();
}

// ====================================================================
// 回测信息
// ====================================================================
function updateBacktestInfo() {
    if (!btTotalAsset) return;

    const mlTotal = getMLRobotTotalAsset();
    const machineTotal = getMachineTotalAsset();
    const startCash = 100;
    const returnRate = (mlTotal - startCash) / startCash;

    // 浮动盈亏
    let floating = 0;
    for (const s of STOCKS) {
        const rh = mlRobotHoldings[s.symbol];
        if (rh.shares > 0) {
            floating += rh.shares * stockState[s.symbol].currentPrice - rh.totalInvested;
        }
    }
    mlRobotStats.floatingPnL = floating;

    if (mlTotal > mlRobotStats.assetPeak) mlRobotStats.assetPeak = mlTotal;
    if (mlRobotStats.assetPeak > 0) {
        const dd = (mlRobotStats.assetPeak - mlTotal) / mlRobotStats.assetPeak;
        if (dd > mlRobotStats.maxDrawdown) {
            mlRobotStats.maxDrawdown = dd;
            mlRobotStats.maxDrawdownAsset = mlRobotStats.assetPeak;
        }
    }

    btTotalAsset.textContent = '¥' + mlTotal.toFixed(2);
    btReturnRate.textContent = (returnRate >= 0 ? '+' : '') + (returnRate * 100).toFixed(2) + '%';
    btReturnRate.className = 'text-xs mt-0.5 ' + (returnRate >= 0 ? 'text-blue-600' : 'text-red-600');

    const sharpe = calculateSharpeRatio(mlRobotPortfolioReturnWindow);
    if (sharpe !== null) {
        btSharpe.textContent = sharpe.toFixed(2);
        btSharpe.className = 'text-lg font-bold ' + (sharpe >= 0 ? 'text-green-700' : 'text-red-600');
    } else {
        btSharpe.textContent = '--';
        btSharpe.className = 'text-lg font-bold text-gray-500';
    }

    const accPct = mlRobotStats.totalDecisions > 0
        ? (mlRobotStats.correctDecisions / mlRobotStats.totalDecisions) * 100 : 0;
    btAccuracy.textContent = accPct.toFixed(1) + '%';
    btDecisions.textContent = mlRobotStats.totalDecisions + ' 次决策';

    const totalR = mlRobotStats.totalReward;
    let allRewards = [];
    for (const s of STOCKS) allRewards = allRewards.concat(mlRobotHoldings[s.symbol].rewardHistory);
    const avgR = allRewards.length > 0 ? allRewards.reduce((a, b) => a + b, 0) / allRewards.length : 0;
    btTotalReward.textContent = totalR.toFixed(2);
    btTotalReward.className = 'text-lg font-bold ' + (totalR >= 0 ? 'text-amber-700' : 'text-red-600');
    btAvgReward.textContent = '平均 ' + (avgR >= 0 ? '+' : '') + avgR.toFixed(2);

    const trades = mlRobotStats.winTrades + mlRobotStats.lossTrades;
    const winRate = trades > 0 ? (mlRobotStats.winTrades / trades) * 100 : null;
    btWinRate.textContent = winRate === null ? '--' : winRate.toFixed(1) + '%';
    btTradeCount.textContent = trades + ' 笔成交';

    if (trades === 0) {
        btProfitFactor.textContent = '--';
        btAvgPnL.textContent = '单笔 0.00';
    } else {
        const factor = mlRobotStats.grossLoss < 0
            ? mlRobotStats.grossProfit / Math.abs(mlRobotStats.grossLoss)
            : (mlRobotStats.grossProfit > 0 ? Infinity : 0);
        btProfitFactor.textContent = factor === Infinity ? '∞' : factor.toFixed(2);
        const totalPnL = mlRobotStats.grossProfit + mlRobotStats.grossLoss;
        const avgPnL = totalPnL / trades;
        btAvgPnL.textContent = '单笔 ' + (avgPnL >= 0 ? '+' : '') + avgPnL.toFixed(2);
    }

    btMaxDrawdown.textContent = (mlRobotStats.maxDrawdown * 100).toFixed(2) + '%';
    btMaxDDAsset.textContent = '峰值 ¥' + mlRobotStats.maxDrawdownAsset.toFixed(2);
    btFloatingPnL.textContent = (floating >= 0 ? '+' : '') + '¥' + floating.toFixed(2);
    btFloatingPnL.className = 'text-base font-semibold ' + (floating >= 0 ? 'text-green-600' : 'text-red-600');
    btFees.textContent = '手续费 ¥' + mlRobotStats.totalFees.toFixed(2);

    const fmtPct = (v) => (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%';
    btAlphas.textContent = fmtPct(returnRate);
    btAlphas.className = 'font-semibold ' + (returnRate >= 0 ? 'text-purple-600' : 'text-red-600');
    btMachine.textContent = fmtPct((machineTotal - startCash) / startCash);
    btMachine.className = 'font-semibold ' + (machineTotal >= startCash ? 'text-green-600' : 'text-red-600');

    // 买入持有基准：所有股票平均
    let buyHoldAvg = 0;
    for (const s of STOCKS) {
        buyHoldAvg += (stockState[s.symbol].currentPrice - s.startPrice) / s.startPrice;
    }
    buyHoldAvg /= STOCKS.length;
    btBuyHold.textContent = fmtPct(buyHoldAvg);
    btBuyHold.className = 'font-semibold ' + (buyHoldAvg >= 0 ? 'text-green-600' : 'text-red-600');

    if (backtestSessionDuration) {
        const minutes = Math.floor((Date.now() - mlRobotStats.startTime) / 60000);
        const seconds = Math.floor(((Date.now() - mlRobotStats.startTime) % 60000) / 1000);
        backtestSessionDuration.textContent = `会话 ${minutes} 分 ${seconds} 秒`;
    }
}

// ====================================================================
// 交易操作
// ====================================================================
function handleBuy(shares, opts = {}) {
    // opts.isMachine / opts.isMLRobot
    const { isMachine = false, isMLRobot = false } = opts;
    if (isNaN(shares) || shares <= 0) {
        if (!isMachine && !isMLRobot) {
            buyMessage.textContent = '请选择有效的股数';
            buyMessage.classList.add('text-danger');
        }
        return false;
    }

    const st = getCurrentStock();
    const cost = shares * st.currentPrice;
    const now = new Date();
    const timeLabel = now.getHours().toString().padStart(2, '0') + ':' +
                     now.getMinutes().toString().padStart(2, '0') + ':' +
                     now.getSeconds().toString().padStart(2, '0');

    if (isMLRobot) {
        if (cost > mlRobotCash) return false;
        const rh = mlRobotHoldings[currentSymbol];
        mlRobotCash -= cost;
        mlRobotCash = parseFloat(mlRobotCash.toFixed(2));
        rh.totalInvested += cost;
        rh.shares += shares;
        const fee = cost * 0.001;
        mlRobotStats.totalFees += fee;
        rh.txHistory.push({ time: timeLabel, timestamp: Date.now(), type: '买入', symbol: currentSymbol, price: st.currentPrice, shares, amount: cost, fee });
        return true;
    } else if (isMachine) {
        if (cost > machineCash) return false;
        const mh = machineHoldings[currentSymbol];
        machineCash -= cost;
        machineCash = parseFloat(machineCash.toFixed(2));
        mh.totalInvested += cost;
        mh.shares += shares;
        mh.txHistory.push({ time: timeLabel, timestamp: Date.now(), type: '买入', symbol: currentSymbol, price: st.currentPrice, shares, amount: cost });
        return true;
    } else {
        if (cost > userCash) {
            buyMessage.textContent = '资金不足';
            buyMessage.classList.add('text-danger');
            return false;
        }
        const uh = userHoldings[currentSymbol];
        userCash -= cost;
        userCash = parseFloat(userCash.toFixed(2));
        uh.totalInvested += cost;
        uh.shares += shares;
        uh.txHistory.push({ time: timeLabel, timestamp: Date.now(), type: '买入', symbol: currentSymbol, price: st.currentPrice, shares, amount: cost });
        buyMessage.textContent = '买入成功';
        buyMessage.classList.remove('text-danger');
        buyMessage.classList.add('text-success');
        setTimeout(() => { buyMessage.textContent = ''; }, 3000);
        updateAssetDisplay();
        updateTransactionHistory();
        return true;
    }
}

function handleSell(shares, opts = {}) {
    const { isMachine = false, isMLRobot = false } = opts;
    if (!isMachine && !isMLRobot) {
        shares = parseInt(sellSharesInput.value);
        if (isNaN(shares) || shares <= 0) {
            sellMessage.textContent = '请输入有效的股数';
            sellMessage.classList.add('text-danger');
            return false;
        }
    }
    const st = getCurrentStock();
    const revenue = shares * st.currentPrice;
    const now = new Date();
    const timeLabel = now.getHours().toString().padStart(2, '0') + ':' +
                     now.getMinutes().toString().padStart(2, '0') + ':' +
                     now.getSeconds().toString().padStart(2, '0');

    if (isMLRobot) {
        const rh = mlRobotHoldings[currentSymbol];
        if (shares <= 0 || shares > rh.shares) return false;
        const actual = Math.min(shares, rh.shares);
        mlRobotCash += revenue;
        mlRobotCash = parseFloat(mlRobotCash.toFixed(2));
        if (actual > 0 && rh.shares > 0) {
            rh.totalInvested -= (rh.totalInvested / rh.shares) * actual;
            rh.totalInvested = Math.max(0, rh.totalInvested);
        }
        rh.shares -= actual;
        const sellFee = revenue * 0.001;
        mlRobotStats.totalFees += sellFee;
        rh.txHistory.push({ time: timeLabel, timestamp: Date.now(), type: '卖出', symbol: currentSymbol, price: st.currentPrice, shares: actual, amount: revenue, fee: sellFee });
        return true;
    } else if (isMachine) {
        const mh = machineHoldings[currentSymbol];
        if (shares <= 0 || shares > mh.shares) return false;
        const actual = Math.min(shares, mh.shares);
        machineCash += revenue;
        machineCash = parseFloat(machineCash.toFixed(2));
        if (actual > 0 && mh.shares > 0) {
            mh.totalInvested -= (mh.totalInvested / mh.shares) * actual;
            mh.totalInvested = Math.max(0, mh.totalInvested);
        }
        mh.shares -= actual;
        if (mh.shares <= 0) { mh.shares = 0; mh.totalInvested = 0; mh.stopLoss = null; mh.trailingHigh = null; mh.entryTime = null; mh.tookProfit1 = false; mh.tookProfit2 = false; }
        mh.txHistory.push({ time: timeLabel, timestamp: Date.now(), type: '卖出', symbol: currentSymbol, price: st.currentPrice, shares: actual, amount: revenue });
        return true;
    } else {
        const uh = userHoldings[currentSymbol];
        if (shares <= 0 || shares > uh.shares) {
            sellMessage.textContent = shares <= 0 ? '请输入有效的股数' : '持股不足';
            sellMessage.classList.add('text-danger');
            return false;
        }
        const actual = Math.min(shares, uh.shares);
        userCash += revenue;
        userCash = parseFloat(userCash.toFixed(2));
        if (actual > 0 && uh.shares > 0) {
            uh.totalInvested -= (uh.totalInvested / uh.shares) * actual;
            uh.totalInvested = Math.max(0, uh.totalInvested);
        }
        uh.shares -= actual;
        uh.txHistory.push({ time: timeLabel, timestamp: Date.now(), type: '卖出', symbol: currentSymbol, price: st.currentPrice, shares: actual, amount: revenue });
        sellSharesInput.value = '';
        sellMessage.textContent = '卖出成功';
        sellMessage.classList.remove('text-danger');
        sellMessage.classList.add('text-success');
        setTimeout(() => { sellMessage.textContent = ''; }, 3000);
        updateAssetDisplay();
        updateTransactionHistory();
        return true;
    }
}

// ====================================================================
// 交易记录显示
// ====================================================================
function updateTransactionHistory() {
    if (!transactionHistoryTable) return;
    transactionHistoryTable.innerHTML = '';
    let history, title;
    if (currentTxView === 'mlRobot') {
        // 合并所有股票的 ML 交易
        history = [];
        for (const s of STOCKS) history = history.concat(mlRobotHoldings[s.symbol].txHistory);
        history.sort((a, b) => b.timestamp - a.timestamp);
        title = '学习AI';
    } else if (currentTxView === 'machine') {
        history = [];
        for (const s of STOCKS) history = history.concat(machineHoldings[s.symbol].txHistory);
        history.sort((a, b) => b.timestamp - a.timestamp);
        title = '规则AI';
    } else {
        history = [];
        for (const s of STOCKS) history = history.concat(userHoldings[s.symbol].txHistory);
        history.sort((a, b) => b.timestamp - a.timestamp);
        title = '您的';
    }

    if (history.length === 0) {
        transactionHistoryTable.innerHTML = `<tr><td colspan="6" class="px-4 py-4 text-center text-gray-500">暂无${title}交易记录</td></tr>`;
        return;
    }

    let html = '';
    for (const t of history) {
        const typeClass = t.type === '买入' ? 'text-success' : 'text-danger';
        const typeIcon = t.type === '买入' ? 'fa-arrow-up' : 'fa-arrow-down';
        html += `
            <tr>
                <td class="px-4 py-3 whitespace-nowrap">${t.time}</td>
                <td class="px-4 py-3 whitespace-nowrap">
                    <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${typeClass} bg-gray-100">
                        <i class="fa ${typeIcon} mr-1"></i>${t.type}
                    </span>
                </td>
                <td class="px-4 py-3 whitespace-nowrap"><span class="inline-block px-1.5 py-0.5 rounded text-xs font-mono bg-gray-100">${t.symbol}</span></td>
                <td class="px-4 py-3 whitespace-nowrap">¥${t.price.toFixed(2)}</td>
                <td class="px-4 py-3 whitespace-nowrap">${t.shares} 股</td>
                <td class="px-4 py-3 whitespace-nowrap font-medium">¥${t.amount.toFixed(2)}</td>
            </tr>`;
    }
    transactionHistoryTable.innerHTML = html;
}

// ====================================================================
// 规则机器人：扫描所有股票选最佳
// ====================================================================
function evaluateBuySignalForSymbol(symbol) {
    const st = stockState[symbol];
    if (st.priceData.length < 30) return { canBuy: false, score: 0, reason: '数据不足' };

    const ma5 = calculateSMA(st.priceData, 5);
    const ma20 = calculateSMA(st.priceData, 20);
    const ma60 = st.priceData.length >= 60 ? calculateSMA(st.priceData, 60) : null;
    const rsi21 = calculateRSIFromPrices(st.priceData, 21);
    if (ma5 === null || ma20 === null || rsi21 === null) return { canBuy: false, score: 0, reason: '指标不足' };

    let score = 0;
    const reasons = [];
    if (ma5 > ma20) { score += 2; reasons.push('金叉'); }
    if (ma60 !== null && st.currentPrice > ma60) { score += 2; reasons.push('趋势线上'); }
    if (isVolumeConfirmed(st, Math.min(60, st.volumeData.length), 1.2)) { score += 1; reasons.push('量能'); }
    if (isMacdHistogramRising(st)) { score += 1; reasons.push('MACD'); }
    if (rsi21 < 85) { score += 1; reasons.push('RSI安全'); }
    if (rsi21 < 40) { score += 2; reasons.push('超卖'); }

    const atr = calculateATRFromState(st, 14);
    return { canBuy: score >= 4, score, reason: reasons.join('+'), atr };
}

function executeMachineTradingForStock(symbol) {
    try {
        const mh = machineHoldings[symbol];
        const st = stockState[symbol];
        if (st.priceData.length < 2) return;

        // 先把当前股票切到这支以便 handleBuy/Sell 内部读 currentSymbol
        // 实际上我们显式传 symbol → 改用 executeBuy/SellBySymbol 函数
        machineTradingForSymbol(symbol);
    } catch (e) { /* ignore */ }
}

// 显式指定股票版本的交易
function machineBuy(symbol, shares) {
    if (shares <= 0) return;
    const st = stockState[symbol];
    const cost = shares * st.currentPrice;
    if (cost > machineCash) return;
    const mh = machineHoldings[symbol];
    machineCash -= cost;
    machineCash = parseFloat(machineCash.toFixed(2));
    mh.totalInvested += cost;
    mh.shares += shares;
    const now = new Date();
    const timeLabel = now.getHours().toString().padStart(2, '0') + ':' +
                     now.getMinutes().toString().padStart(2, '0') + ':' +
                     now.getSeconds().toString().padStart(2, '0');
    mh.txHistory.push({ time: timeLabel, timestamp: Date.now(), type: '买入', symbol, price: st.currentPrice, shares, amount: cost });
}
function machineSell(symbol, shares) {
    if (shares <= 0) return;
    const st = stockState[symbol];
    const mh = machineHoldings[symbol];
    if (shares > mh.shares) shares = mh.shares;
    if (shares <= 0) return;
    const revenue = shares * st.currentPrice;
    machineCash += revenue;
    machineCash = parseFloat(machineCash.toFixed(2));
    if (shares > 0 && mh.shares > 0) {
        mh.totalInvested -= (mh.totalInvested / mh.shares) * shares;
        mh.totalInvested = Math.max(0, mh.totalInvested);
    }
    mh.shares -= shares;
    if (mh.shares <= 0) {
        mh.shares = 0; mh.totalInvested = 0;
        mh.stopLoss = null; mh.trailingHigh = null; mh.entryTime = null;
        mh.tookProfit1 = false; mh.tookProfit2 = false;
    }
    const now = new Date();
    const timeLabel = now.getHours().toString().padStart(2, '0') + ':' +
                     now.getMinutes().toString().padStart(2, '0') + ':' +
                     now.getSeconds().toString().padStart(2, '0');
    mh.txHistory.push({ time: timeLabel, timestamp: Date.now(), type: '卖出', symbol, price: st.currentPrice, shares, amount: revenue });
}

function manageMachinePosition(symbol) {
    const mh = machineHoldings[symbol];
    const st = stockState[symbol];
    if (mh.shares <= 0 || st.priceData.length < 2) return false;

    const avgCost = mh.totalInvested / mh.shares;
    const profitPct = (st.currentPrice - avgCost) / avgCost;
    const atr = calculateATRFromState(st, 14);
    if (atr && !mh.stopLoss) {
        mh.stopLoss = parseFloat(Math.max(st.currentPrice - 2.2 * atr, st.currentPrice * 0.90).toFixed(2));
    }
    if (st.currentPrice > (mh.trailingHigh || avgCost)) {
        mh.trailingHigh = st.currentPrice;
        const trailPct = atr ? Math.min(0.08, Math.max(0.025, (atr / st.currentPrice) * 1.8)) : 0.04;
        if (profitPct > 0.04) {
            mh.stopLoss = Math.max(mh.stopLoss || 0, parseFloat((mh.trailingHigh * (1 - trailPct)).toFixed(2)));
        }
    }

    const hardStop = mh.stopLoss || avgCost * 0.94;
    if (st.currentPrice <= hardStop) {
        const pnlPerShare = st.currentPrice - avgCost;
        machineLastTradeProfit = pnlPerShare;
        if (pnlPerShare < 0) machineConsecutiveLosses++; else machineConsecutiveLosses = 0;
        if (machineConsecutiveLosses >= MAX_CONSECUTIVE_LOSSES) {
            machineLossCooldownUntil = Date.now() + LOSS_COOLDOWN_MS;
        }
        machineSell(symbol, mh.shares);
        return true;
    }

    // 分级止盈 + 留底仓，让规则 AI 不会一有利润就全部离场。
    if (profitPct >= 0.07 && !mh.tookProfit1) {
        const sellShares = Math.max(1, Math.floor(mh.shares * 0.30));
        machineSell(symbol, sellShares);
        mh.tookProfit1 = true;
        mh.stopLoss = Math.max(mh.stopLoss || 0, parseFloat((avgCost * 1.002).toFixed(2)));
        machineLastTradeProfit = profitPct;
        return true;
    }
    if (profitPct >= 0.14 && !mh.tookProfit2) {
        const sellShares = Math.max(1, Math.floor(mh.shares * 0.45));
        machineSell(symbol, sellShares);
        mh.tookProfit2 = true;
        machineLastTradeProfit = profitPct;
        return true;
    }
    return false;
}

function executeMachinePortfolioTrading() {
    if (Date.now() < machineLossCooldownUntil) {
        // 冷却期仍允许风控卖出，不允许新开仓。
        for (const s of STOCKS) manageMachinePosition(s.symbol);
        return;
    }

    for (const s of STOCKS) manageMachinePosition(s.symbol);

    const totalAsset = getMachineTotalAsset();
    if (totalAsset > machinePeakAsset) machinePeakAsset = totalAsset;
    const drawdown = machinePeakAsset > 0 ? (machinePeakAsset - totalAsset) / machinePeakAsset : 0;
    if (drawdown >= MAX_DRAWDOWN_THRESHOLD) return;

    const openPositions = countOpenPositions(machineHoldings);
    const slots = Math.max(0, MACHINE_MAX_POSITIONS - openPositions);
    if (slots <= 0 || machineCash <= totalAsset * MACHINE_CASH_RESERVE_RATIO) return;

    const candidates = [];
    for (const s of STOCKS) {
        const symbol = s.symbol;
        const st = stockState[symbol];
        const mh = machineHoldings[symbol];
        if (mh.shares > 0 || st.priceData.length < 30) continue;
        const sig = evaluateBuySignalForSymbol(symbol);
        if (!sig.canBuy) continue;
        const recentVol = calculateRecentVolatility(st.priceData);
        if (recentVol > MAX_ENTRY_VOL) continue;
        const ma20 = calculateSMA(st.priceData, 20);
        const trendBonus = ma20 && st.currentPrice > ma20 ? 0.8 : 0;
        const volumeBonus = isVolumeConfirmed(st, Math.min(60, st.volumeData.length), 1.15) ? 0.5 : 0;
        const riskPenalty = recentVol * 18;
        candidates.push({
            symbol,
            score: sig.score + trendBonus + volumeBonus - riskPenalty,
            signalScore: sig.score,
            atr: sig.atr,
            recentVol,
        });
    }

    candidates.sort((a, b) => b.score - a.score);
    let buys = 0;
    for (const candidate of candidates) {
        if (buys >= Math.min(slots, MACHINE_MAX_NEW_BUYS_PER_STEP)) break;
        const st = stockState[candidate.symbol];
        const mh = machineHoldings[candidate.symbol];
        const maxPositionValue = totalAsset * MACHINE_MAX_POSITION_RATIO;
        const cashReserve = totalAsset * MACHINE_CASH_RESERVE_RATIO;
        const availableCash = Math.max(0, machineCash - cashReserve);
        if (availableCash <= 0) break;

        const conviction = Math.max(0.18, Math.min(0.45, 0.12 + candidate.signalScore * 0.045));
        const volCap = Math.min(0.45, VOL_TARGET_PER_STEP / Math.max(candidate.recentVol, 0.004));
        const targetValue = Math.min(maxPositionValue, totalAsset * Math.min(conviction, volCap), availableCash);
        const shares = Math.floor(targetValue / st.currentPrice);
        if (shares <= 0) continue;

        machineBuy(candidate.symbol, shares);
        mh.stopLoss = candidate.atr ? parseFloat(Math.max(st.currentPrice - 2.2 * candidate.atr, st.currentPrice * 0.90).toFixed(2)) : parseFloat((st.currentPrice * 0.94).toFixed(2));
        mh.trailingHigh = st.currentPrice;
        mh.entryTime = Date.now();
        mh.tookProfit1 = false;
        mh.tookProfit2 = false;
        if (candidate.signalScore >= 6) machineConsecutiveLosses = 0;
        buys++;
    }
}

function machineTradingForSymbol(symbol) {
    const mh = machineHoldings[symbol];
    const st = stockState[symbol];
    if (st.priceData.length < 2) return;

    // 连败冷却
    if (Date.now() < machineLossCooldownUntil) return;

    // 持仓管理
    if (mh.shares > 0) {
        const avgCost = mh.totalInvested / mh.shares;
        const profitPct = (st.currentPrice - avgCost) / avgCost;
        const atr = calculateATRFromState(st, 14);
        if (atr && !mh.stopLoss) {
            mh.stopLoss = parseFloat(Math.max(st.currentPrice - 2 * atr, st.currentPrice * 0.9).toFixed(2));
        }
        if (st.currentPrice <= (mh.stopLoss || avgCost * 0.95)) {
            const pnlPerShare = st.currentPrice - avgCost;
            machineLastTradeProfit = pnlPerShare;
            if (pnlPerShare < 0) machineConsecutiveLosses++; else machineConsecutiveLosses = 0;
            if (machineConsecutiveLosses >= MAX_CONSECUTIVE_LOSSES) {
                machineLossCooldownUntil = Date.now() + LOSS_COOLDOWN_MS;
            }
            machineSell(symbol, mh.shares);
            return;
        }
        // 分级止盈
        if (profitPct >= 0.08 && !mh.tookProfit1) {
            const sellShares = Math.max(1, Math.floor(mh.shares / 3));
            machineSell(symbol, sellShares);
            mh.tookProfit1 = true;
            machineLastTradeProfit = profitPct;
            mh.stopLoss = Math.max(mh.stopLoss || 0, parseFloat(avgCost.toFixed(2)));
            return;
        }
        if (profitPct >= 0.15 && !mh.tookProfit2) {
            const sellShares = Math.max(1, Math.floor(mh.shares / 2));
            machineSell(symbol, sellShares);
            mh.tookProfit2 = true;
            machineLastTradeProfit = profitPct;
            return;
        }
        if (profitPct > 0.15) {
            if (st.currentPrice > (mh.trailingHigh || avgCost)) {
                mh.trailingHigh = st.currentPrice;
                const trailPct = atr ? Math.min(0.05, (atr / st.currentPrice) * 1.5) : 0.03;
                mh.stopLoss = parseFloat((mh.trailingHigh * (1 - trailPct)).toFixed(2));
            }
        }
        return;
    }

    // 资金管理
    const totalAsset = getMachineTotalAsset();
    if (totalAsset > machinePeakAsset) machinePeakAsset = totalAsset;
    const drawdown = machinePeakAsset > 0 ? (machinePeakAsset - totalAsset) / machinePeakAsset : 0;
    if (drawdown >= MAX_DRAWDOWN_THRESHOLD) return;

    // 评估这只股票的买入信号
    const sig = evaluateBuySignalForSymbol(symbol);
    if (!sig.canBuy) return;

    // 计算仓位
    const recentVol = calculateRecentVolatility(st.priceData);
    if (recentVol > MAX_ENTRY_VOL) return;
    const edge = Math.min(1, sig.score / 10);
    const variance = Math.max(0.005, recentVol * recentVol);
    const kelly = Math.max(0, edge - (1 - edge) / 2) / Math.max(variance * 100, 1);
    let positionRatio = Math.min(0.5, Math.max(0.15, kelly * 2));
    if (sig.score >= 8) positionRatio = Math.max(positionRatio, 0.5);
    else if (sig.score >= 6) positionRatio = Math.max(positionRatio, 0.33);
    const volTargetRatio = Math.min(0.5, VOL_TARGET_PER_STEP / Math.max(recentVol, 0.005));
    positionRatio = Math.min(positionRatio, volTargetRatio);

    // 信号强度：评分越高越优先买；评分相同则波动率高的优先买（机会）
    const shares = Math.floor(machineCash * positionRatio / st.currentPrice);
    if (shares <= 0) return;
    machineBuy(symbol, shares);
    mh.stopLoss = sig.atr ? parseFloat(Math.max(st.currentPrice - 2 * sig.atr, st.currentPrice * 0.9).toFixed(2)) : parseFloat((st.currentPrice * 0.95).toFixed(2));
    mh.trailingHigh = st.currentPrice;
    mh.entryTime = Date.now();
    mh.tookProfit1 = false; mh.tookProfit2 = false;
    if (sig.score >= 6) machineConsecutiveLosses = 0;
}

// ====================================================================
// ML 机器人：跨股票自由选择
// ====================================================================
function selectAction(qTable, state, explorationRate) {
    if (Math.random() < explorationRate) {
        const actions = ['buy', 'sell', 'hold'];
        return actions[Math.floor(Math.random() * actions.length)];
    }
    const qValues = qTable[state] || {};
    const buyQ = qValues['buy'] || 0;
    const sellQ = qValues['sell'] || 0;
    const holdQ = qValues['hold'] || 0;
    if (buyQ === 0 && sellQ === 0 && holdQ === 0) return 'hold';
    const temperature = 0.5;
    const qArr = [buyQ, sellQ, holdQ];
    const maxQ = Math.max(...qArr);
    const expValues = qArr.map(q => Math.exp((q - maxQ) / temperature));
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    const probs = expValues.map(e => e / sumExp);
    const rand = Math.random();
    const actions = ['buy', 'sell', 'hold'];
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
        cum += probs[i];
        if (rand < cum) return actions[i];
    }
    return 'hold';
}
function updateQTable(qTable, qTableAccess, state, action, reward, nextState, learningRate, discountFactor) {
    const currentQ = qTable[state]?.[action] || 0;
    const nextQValues = qTable[nextState] || {};
    const nextMaxQ = Math.max(nextQValues['buy'] || 0, nextQValues['sell'] || 0, nextQValues['hold'] || 0);
    const newQ = currentQ + learningRate * (reward + discountFactor * nextMaxQ - currentQ);
    if (!qTable[state]) qTable[state] = {};
    qTable[state][action] = newQ;
    qTableAccess[state] = Date.now();
    evictQTableIfNeeded(qTable, qTableAccess);
}
function evictQTableIfNeeded(qTable, qTableAccess) {
    const keys = Object.keys(qTable);
    if (keys.length <= Q_TABLE_MAX_SIZE) return;
    const entries = keys.map(k => ({ key: k, time: qTableAccess[k] || 0 }));
    entries.sort((a, b) => a.time - b.time);
    const toRemove = entries.slice(0, keys.length - Q_TABLE_MAX_SIZE);
    for (const e of toRemove) { delete qTable[e.key]; delete qTableAccess[e.key]; }
}
function calculateReward(oldAsset, newAsset, action, priceChange, extra = {}) {
    const volatility = extra.volatility || 0;
    const holdingProfit = extra.holdingProfit || 0;
    const trendDir = extra.trendDir || 0;
    let reward = (newAsset - oldAsset) / oldAsset * 100;
    if (action === 'buy' || action === 'sell') reward -= 0.3;
    if ((action === 'buy' && priceChange > 0.002) || (action === 'sell' && priceChange < -0.002)) reward += 3.0;
    if ((action === 'buy' && priceChange < -0.002) || (action === 'sell' && priceChange > 0.002)) reward -= 1.5;
    if (action === 'buy' && trendDir > 0) reward += 1.0;
    if (action === 'sell' && trendDir < 0) reward += 1.0;
    if (action === 'buy' && trendDir < 0) reward -= 1.5;
    if (action === 'sell' && trendDir > 0) reward -= 1.5;
    if (action === 'hold') {
        if (holdingProfit > 0.01) reward += 0.5;
        if (holdingProfit < -0.02) reward -= 0.3;
    }
    if (volatility > 0.04 && (action === 'buy' || action === 'sell')) reward -= 0.5;
    return Math.max(-15, Math.min(15, reward));
}
function evaluateDecision(action, priceChange) {
    if (action === 'buy') return priceChange > 0.003;
    if (action === 'sell') return priceChange < -0.003;
    return Math.abs(priceChange) <= 0.003;
}
function encodeSmartState(price, priceChange, rsi, macd, holdingStock, assetChange, marketState, holdingProfit, trendDir, extras = {}) {
    let pcBin = priceChange > 0.02 ? 4 : priceChange > 0.005 ? 3 : priceChange > -0.005 ? 2 : priceChange > -0.02 ? 1 : 0;
    let rsiBin = rsi < 20 ? 0 : rsi < 40 ? 1 : rsi < 60 ? 2 : rsi < 80 ? 3 : 4;
    const macdBucket = macd > 0.01 ? 2 : macd < -0.01 ? 0 : 1;
    const holdingBin = holdingStock ? 1 : 0;
    let acBin = assetChange > 0.01 ? 2 : assetChange < -0.01 ? 0 : 1;
    const ms = marketState === 'high_risk' ? 2 : (marketState === 'low_risk' ? 0 : 1);
    let profitBin = 1;
    if (holdingProfit !== undefined) { if (holdingProfit > 0.02) profitBin = 2; else if (holdingProfit < -0.02) profitBin = 0; }
    let trendBin = 1;
    if (trendDir !== undefined) { if (trendDir > 0) trendBin = 2; else if (trendDir < 0) trendBin = 0; }
    let volBin = 1;
    if (extras.volatility !== undefined) { if (extras.volatility > 0.03) volBin = 2; else if (extras.volatility < 0.01) volBin = 0; }
    let vqBin = 1;
    if (extras.volumeRatio !== undefined) { if (extras.volumeRatio > 1.5) vqBin = 2; else if (extras.volumeRatio < 0.7) vqBin = 0; }
    let hpBin = 0;
    if (extras.holdingPeriodSec !== undefined) { if (extras.holdingPeriodSec > 120) hpBin = 2; else if (extras.holdingPeriodSec > 30) hpBin = 1; }
    return `${pcBin},${rsiBin},${macdBucket},${holdingBin},${acBin},${ms},${profitBin},${trendBin},${volBin},${vqBin},${hpBin}`;
}
function assessMarketState(st) {
    if (st.priceData.length < 10) return 'normal';
    const vol = calculateRecentVolatility(st.priceData);
    const rsi = st.rsiData[st.rsiData.length - 1] || 50;
    const macd = st.macdData[st.macdData.length - 1] || 0;
    if (vol > 0.03 || rsi > 80 || rsi < 20 || Math.abs(macd) > 0.05) return 'high_risk';
    if (vol < 0.01 && rsi > 40 && rsi < 60 && Math.abs(macd) < 0.01) return 'low_risk';
    return 'normal';
}

function mlBuy(symbol, shares) {
    if (shares <= 0) return;
    const st = stockState[symbol];
    const cost = shares * st.currentPrice;
    if (cost > mlRobotCash) return;
    const rh = mlRobotHoldings[symbol];
    mlRobotCash -= cost;
    mlRobotCash = parseFloat(mlRobotCash.toFixed(2));
    rh.totalInvested += cost;
    rh.shares += shares;
    const fee = cost * 0.001;
    mlRobotStats.totalFees += fee;
    const now = new Date();
    const timeLabel = now.getHours().toString().padStart(2, '0') + ':' +
                     now.getMinutes().toString().padStart(2, '0') + ':' +
                     now.getSeconds().toString().padStart(2, '0');
    rh.txHistory.push({ time: timeLabel, timestamp: Date.now(), type: '买入', symbol, price: st.currentPrice, shares, amount: cost, fee });
}
function mlSell(symbol, shares) {
    if (shares <= 0) return null;
    const st = stockState[symbol];
    const rh = mlRobotHoldings[symbol];
    if (shares > rh.shares) shares = rh.shares;
    if (shares <= 0) return null;
    const revenue = shares * st.currentPrice;
    const avgCost = rh.shares > 0 ? (rh.totalInvested / rh.shares) : 0;
    const costPortion = avgCost * shares;
    const realizedPnL = revenue - costPortion;
    mlRobotCash += revenue;
    mlRobotCash = parseFloat(mlRobotCash.toFixed(2));
    const sellFee = revenue * 0.001;
    mlRobotStats.totalFees += sellFee;
    if (shares > 0 && rh.shares > 0) {
        rh.totalInvested -= (rh.totalInvested / rh.shares) * shares;
        rh.totalInvested = Math.max(0, rh.totalInvested);
    }
    rh.shares -= shares;
    const now = new Date();
    const timeLabel = now.getHours().toString().padStart(2, '0') + ':' +
                     now.getMinutes().toString().padStart(2, '0') + ':' +
                     now.getSeconds().toString().padStart(2, '0');
    rh.txHistory.push({ time: timeLabel, timestamp: Date.now(), type: '卖出', symbol, price: st.currentPrice, shares, amount: revenue, fee: sellFee });
    if (rh.shares <= 0) {
        rh.shares = 0; rh.totalInvested = 0;
        rh.stopLossPrice = null; rh.takeProfitPrice = null; rh.trailingPeak = null; rh.lastBuyTime = null;
    }
    // 更新 ML 机器人的盈亏统计
    if (realizedPnL > 0) {
        mlRobotStats.winTrades++;
        mlRobotStats.grossProfit += realizedPnL;
    } else if (realizedPnL < 0) {
        mlRobotStats.lossTrades++;
        mlRobotStats.grossLoss += realizedPnL;
    }
    return { realizedPnL, revenue, fee: sellFee };
}

function buildMLDecisionContext(symbol) {
    const st = stockState[symbol];
    const rh = mlRobotHoldings[symbol];
    if (st.priceData.length < 15) return null;
    const rsi = st.rsiData[st.rsiData.length - 1] || 50;
    const macd = st.macdData[st.macdData.length - 1] || 0;
    const priceChange = st.priceData.length >= 2
        ? (st.currentPrice - st.priceData[st.priceData.length - 2]) / st.priceData[st.priceData.length - 2]
        : 0;
    const ms = assessMarketState(st);
    const recentVol = calculateRecentVolatility(st.priceData);
    const ma5 = st.ma5Data[st.ma5Data.length - 1] || st.currentPrice;
    const ma10 = st.ma10Data[st.ma10Data.length - 1] || st.currentPrice;
    const trendDir = ma5 > ma10 ? 1 : (ma5 < ma10 ? -1 : 0);
    const avgCost = rh.shares > 0 ? rh.totalInvested / rh.shares : 0;
    const holdingProfit = rh.shares > 0 && avgCost > 0 ? (st.currentPrice - avgCost) / avgCost : 0;
    const holdingPeriodSec = rh.lastBuyTime ? (Date.now() - rh.lastBuyTime) / 1000 : 0;
    const avgVolume = st.volumeData.length >= 10 ? st.volumeData.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, st.volumeData.length) : 0;
    const volumeRatio = avgVolume > 0 ? st.volumeData[st.volumeData.length - 1] / avgVolume : 1;
    const state = encodeSmartState(st.currentPrice, priceChange, rsi, macd, rh.shares > 0, 0, ms, holdingProfit, trendDir, {
        volatility: recentVol,
        volumeRatio,
        holdingPeriodSec,
    });
    const qv = rh.qTable[state] || {};
    const buyQ = qv.buy || 0;
    const sellQ = qv.sell || 0;
    const holdQ = qv.hold || 0;
    return { symbol, st, rh, rsi, macd, priceChange, ms, recentVol, trendDir, holdingProfit, holdingPeriodSec, volumeRatio, state, buyQ, sellQ, holdQ };
}

async function executeMLRobotTrading() {
    if (mlRobotStats.isProcessing) return;
    mlRobotStats.isProcessing = true;
    try {
        if (!executeMLRobotTrading.counter) executeMLRobotTrading.counter = 0;
        executeMLRobotTrading.counter++;
        if (executeMLRobotTrading.counter % 3 !== 0) return;

        const beforeAsset = getMLRobotTotalAsset();
        if (beforeAsset > mlRobotStats.assetPeak) mlRobotStats.assetPeak = beforeAsset;
        const drawdown = mlRobotStats.assetPeak > 0 ? (mlRobotStats.assetPeak - beforeAsset) / mlRobotStats.assetPeak : 0;
        const actionBySymbol = {};

        // 组合风控：所有持仓都先更新移动止损，必要时卖出。
        for (const s of STOCKS) {
            const rh = mlRobotHoldings[s.symbol];
            const st = stockState[s.symbol];
            if (rh.shares <= 0) continue;
            const avgCost = rh.totalInvested / rh.shares;
            const atr = calculateATRFromState(st, 14) || (st.currentPrice * 0.02);
            rh.trailingPeak = Math.max(rh.trailingPeak || st.currentPrice, st.currentPrice);
            if (st.currentPrice > avgCost * 1.04) {
                rh.stopLossPrice = parseFloat(Math.max(rh.stopLossPrice || 0, avgCost * 1.002).toFixed(2));
            }
            if (st.currentPrice > avgCost * 1.09) {
                rh.stopLossPrice = Math.max(rh.stopLossPrice || 0, parseFloat((rh.trailingPeak - 1.7 * atr).toFixed(2)));
            }
            const stopHit = rh.stopLossPrice !== null && st.currentPrice <= rh.stopLossPrice;
            const takeHit = rh.takeProfitPrice !== null && st.currentPrice >= rh.takeProfitPrice;
            if (stopHit || takeHit || drawdown > 0.24) {
                mlSell(s.symbol, rh.shares);
                actionBySymbol[s.symbol] = 'sell';
            }
        }

        const contexts = STOCKS.map(s => buildMLDecisionContext(s.symbol)).filter(Boolean);
        const exploreProb = mlRobotStats.explorationRate;
        const actRandom = Math.random() < exploreProb;

        // 卖出：不是只卖一只，所有高风险/高卖出分的持仓都可被部分或全部处理。
        const sellCandidates = contexts
            .filter(ctx => ctx.rh.shares > 0)
            .map(ctx => {
                const profitExit = ctx.holdingProfit > 0.10 ? 1.2 : (ctx.holdingProfit > 0.06 ? 0.5 : 0);
                const lossExit = ctx.holdingProfit < -0.045 ? 1.5 : (ctx.holdingProfit < -0.025 ? 0.6 : 0);
                const trendExit = ctx.trendDir < 0 ? 0.5 : 0;
                const overboughtExit = ctx.rsi > 78 ? 0.5 : 0;
                return { ...ctx, sellScore: ctx.sellQ + profitExit + lossExit + trendExit + overboughtExit };
            })
            .sort((a, b) => b.sellScore - a.sellScore);

        for (const ctx of sellCandidates) {
            if (actionBySymbol[ctx.symbol] === 'sell') continue;
            const shouldSell = actRandom || ctx.sellScore > 0.85 || ctx.holdingProfit > 0.14 || ctx.holdingProfit < -0.055;
            if (!shouldSell) continue;
            const fullExit = ctx.holdingProfit < -0.055 || ctx.sellQ > 1.2 || drawdown > 0.20;
            const shares = fullExit ? ctx.rh.shares : Math.max(1, Math.floor(ctx.rh.shares * 0.45));
            mlSell(ctx.symbol, shares);
            actionBySymbol[ctx.symbol] = 'sell';
        }

        // 买入：允许每个决策周期买入多个候选，使学习 AI 能同时持有多只股票。
        const afterSellAsset = getMLRobotTotalAsset();
        const openPositions = countOpenPositions(mlRobotHoldings);
        let availableSlots = Math.max(0, ML_MAX_POSITIONS - openPositions);
        const cashReserve = afterSellAsset * ML_CASH_RESERVE_RATIO;
        const buyCandidates = contexts
            .filter(ctx => {
                if (actionBySymbol[ctx.symbol] === 'sell') return false;
                if (ctx.rh.shares <= 0 && availableSlots <= 0) return false;
                const currentValue = getHoldingValue(mlRobotHoldings, ctx.symbol);
                return currentValue < afterSellAsset * ML_MAX_POSITION_RATIO * 0.98;
            })
            .map(ctx => {
                const valueBias = ctx.rh.shares > 0 ? -0.25 : 0.15;
                const technicalBonus =
                    (ctx.rsi < 35 ? 0.9 : ctx.rsi < 55 ? 0.35 : ctx.rsi > 75 ? -0.6 : 0) +
                    (ctx.trendDir > 0 ? 0.55 : -0.25) +
                    (ctx.ms === 'low_risk' ? 0.35 : ctx.ms === 'high_risk' ? -0.5 : 0) +
                    (ctx.volumeRatio > 1.25 && ctx.priceChange > 0 ? 0.25 : 0);
                const riskPenalty = Math.max(0, ctx.recentVol - 0.018) * 16;
                return { ...ctx, buyScore: ctx.buyQ + technicalBonus + valueBias - riskPenalty };
            })
            .sort((a, b) => b.buyScore - a.buyScore);

        let buys = 0;
        for (const ctx of buyCandidates) {
            if (buys >= ML_MAX_NEW_BUYS_PER_STEP) break;
            if (mlRobotCash <= cashReserve) break;
            const hasPosition = ctx.rh.shares > 0;
            if (!hasPosition && availableSlots <= 0) break;
            if (!actRandom && ctx.buyScore < 0.20 && ctx.buyQ <= 0) continue;

            const currentValue = getHoldingValue(mlRobotHoldings, ctx.symbol);
            const maxPositionValue = afterSellAsset * ML_MAX_POSITION_RATIO;
            const room = Math.max(0, maxPositionValue - currentValue);
            if (room <= ctx.st.currentPrice) continue;

            const conviction = Math.max(0.10, Math.min(0.34, 0.12 + Math.max(0, ctx.buyScore) * 0.05));
            const volCap = ctx.recentVol > 0.04 ? 0.12 : (ctx.recentVol > 0.025 ? 0.20 : 0.30);
            const budget = Math.min(room, mlRobotCash - cashReserve, afterSellAsset * Math.min(conviction, volCap));
            const tradeShares = Math.floor(budget / ctx.st.currentPrice);
            if (tradeShares <= 0) continue;

            mlBuy(ctx.symbol, tradeShares);
            const rh = mlRobotHoldings[ctx.symbol];
            rh.lastBuyTime = Date.now();
            const atr = calculateATRFromState(ctx.st, 14) || (ctx.st.currentPrice * 0.02);
            rh.stopLossPrice = parseFloat(Math.max(ctx.st.currentPrice - 2.4 * atr, ctx.st.currentPrice * 0.91).toFixed(2));
            rh.takeProfitPrice = parseFloat((ctx.st.currentPrice + 4.2 * atr).toFixed(2));
            rh.trailingPeak = ctx.st.currentPrice;
            actionBySymbol[ctx.symbol] = 'buy';
            if (!hasPosition) availableSlots--;
            buys++;
        }

        // 训练：使用本轮真实执行/持有动作，而不是随机动作，奖励基于组合净值和价格方向。
        const afterAsset = getMLRobotTotalAsset();
        recordMLPortfolioMetrics(afterAsset);
        let stepTotalReward = 0;
        let stepCorrect = 0;
        let stepDecisions = 0;
        for (const ctx of contexts) {
            const action = actionBySymbol[ctx.symbol] || 'hold';
            const nextCtx = buildMLDecisionContext(ctx.symbol) || ctx;
            const reward = calculateReward(beforeAsset, afterAsset, action, ctx.priceChange, {
                volatility: ctx.recentVol,
                holdingProfit: nextCtx.holdingProfit,
                trendDir: ctx.trendDir,
            });
            updateQTable(ctx.rh.qTable, ctx.rh.qTableAccess, ctx.state, action, reward, nextCtx.state, mlRobotStats.learningRate, mlRobotStats.discountFactor);

            ctx.rh.rewardHistory.push(parseFloat(reward.toFixed(2)));
            if (ctx.rh.rewardHistory.length > 200) ctx.rh.rewardHistory.shift();
            stepTotalReward += reward;

            const isCorrect = evaluateDecision(action, ctx.priceChange);
            if (ctx.rh._correctCount === undefined) ctx.rh._correctCount = 0;
            if (ctx.rh._totalCount === undefined) ctx.rh._totalCount = 0;
            if (isCorrect) ctx.rh._correctCount++;
            ctx.rh._totalCount++;
            const runningAcc = ctx.rh._totalCount > 0 ? (ctx.rh._correctCount / ctx.rh._totalCount) * 100 : 0;
            ctx.rh.accuracyHistory.push(parseFloat(runningAcc.toFixed(2)));
            if (ctx.rh.accuracyHistory.length > 200) ctx.rh.accuracyHistory.shift();
            if (isCorrect) stepCorrect++;
            stepDecisions++;

            const symbolValue = getHoldingValue(mlRobotHoldings, ctx.symbol);
            ctx.rh.assetData.push(symbolValue);
            if (ctx.rh.assetData.length > SHARPE_WINDOW + 1) ctx.rh.assetData.shift();
            const symbolReturns = [];
            for (let i = 1; i < ctx.rh.assetData.length; i++) {
                const prev = ctx.rh.assetData[i - 1], cur = ctx.rh.assetData[i];
                if (prev > 0 && cur > 0) symbolReturns.push(Math.log(cur / prev));
            }
            const symbolSharpe = calculateSharpeRatio(symbolReturns, 3);
            if (symbolSharpe !== null) {
                ctx.rh.sharpeHistory.push(parseFloat(symbolSharpe.toFixed(3)));
                if (ctx.rh.sharpeHistory.length > 200) ctx.rh.sharpeHistory.shift();
            }
        }

        mlRobotStats.totalReward += stepTotalReward;
        mlRobotStats.correctDecisions += stepCorrect;
        mlRobotStats.totalDecisions += stepDecisions;

        if (mlRobotStats.totalDecisions > 0 && mlRobotStats.totalDecisions % EPSILON_DECAY_STEP === 0) {
            mlRobotStats.explorationRate *= mlRobotStats.explorationDecay;
            explorationRateSlider.value = mlRobotStats.explorationRate;
            explorationRateDisplay.textContent = mlRobotStats.explorationRate.toFixed(2);
        }
    } catch (e) {
        console.error('ML 错误:', e);
    } finally {
        mlRobotStats.isProcessing = false;
    }
}

// ====================================================================
// 持久化（每个股票一个 Q-table key）
// ====================================================================
function saveAllQTables() {
    saveQTablePending = true;
    if (saveQTableTimer !== null) return;
    saveQTableTimer = setTimeout(() => {
        saveQTableTimer = null;
        if (!saveQTablePending) return;
        saveQTablePending = false;
        try {
            const data = { perStock: {} };
            for (const s of STOCKS) {
                data.perStock[s.symbol] = {
                    qTable: mlRobotHoldings[s.symbol].qTable,
                    qTableAccess: mlRobotHoldings[s.symbol].qTableAccess,
                };
            }
            data.explorationRate = mlRobotStats.explorationRate;
            data.learningRate = mlRobotStats.learningRate;
            data.timestamp = Date.now();
            localStorage.setItem('ml_robot_qtable_multi', JSON.stringify(data));
        } catch {}
    }, 1000);
}
function flushSaveQTable() {
    if (saveQTableTimer !== null) { clearTimeout(saveQTableTimer); saveQTableTimer = null; }
    if (!saveQTablePending) return;
    saveQTablePending = false;
    try {
        const data = { perStock: {} };
        for (const s of STOCKS) {
            data.perStock[s.symbol] = {
                qTable: mlRobotHoldings[s.symbol].qTable,
                qTableAccess: mlRobotHoldings[s.symbol].qTableAccess,
            };
        }
        data.explorationRate = mlRobotStats.explorationRate;
        data.learningRate = mlRobotStats.learningRate;
        data.timestamp = Date.now();
        localStorage.setItem('ml_robot_qtable_multi', JSON.stringify(data));
    } catch {}
}
function loadQTable() {
    try {
        const saved = localStorage.getItem('ml_robot_qtable_multi');
        if (!saved) return false;
        const data = JSON.parse(saved);
        if (!data.perStock) return false;
        for (const s of STOCKS) {
            const sub = data.perStock[s.symbol];
            if (sub && sub.qTable) {
                mlRobotHoldings[s.symbol].qTable = sub.qTable;
                mlRobotHoldings[s.symbol].qTableAccess = sub.qTableAccess || {};
            }
        }
        if (typeof data.explorationRate === 'number') {
            mlRobotStats.explorationRate = data.explorationRate;
            explorationRateSlider.value = mlRobotStats.explorationRate;
            explorationRateDisplay.textContent = mlRobotStats.explorationRate.toFixed(2);
        }
        if (typeof data.learningRate === 'number') {
            mlRobotStats.learningRate = data.learningRate;
            learningRateSlider.value = mlRobotStats.learningRate;
            learningRateDisplay.textContent = mlRobotStats.learningRate.toFixed(2);
        }
        return true;
    } catch (e) {
        return false;
    }
}

async function saveMLModel() {
    if (!usePythonML || !mlBackendAvailable) return;
    try {
        const response = await fetch(`${ML_API_URL}/api/ml/save`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
        if (response.ok) {
            const result = await response.json();
            if (result.success) showToast('ML模型已自动保存', 'success', 2000);
        }
    } catch {}
}
function startAutoSave() {
    if (mlAutoSaveTimer !== null) return;
    if (!isMLRobotEnabled) return;
    mlAutoSaveTimer = setInterval(autoSaveMLModel, ML_AUTO_SAVE_INTERVAL_MS);
}
function stopAutoSave() {
    if (mlAutoSaveTimer !== null) { clearInterval(mlAutoSaveTimer); mlAutoSaveTimer = null; }
}
async function autoSaveMLModel() {
    if (mlAutoSaveInFlight) return;
    if (!isMLRobotEnabled || !usePythonML || !mlBackendAvailable) return;
    mlAutoSaveInFlight = true;
    try { await saveMLModel(); } finally { mlAutoSaveInFlight = false; }
}

async function checkPythonMLService() {
    if (!usePythonML) return;
    try {
        const response = await Promise.race([
            fetch(`${ML_API_URL}/api/ml/status`),
            new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 3000)),
        ]);
        if (response.ok) {
            const result = await response.json();
            if (result.success) mlBackendAvailable = true;
        }
    } catch (error) { mlBackendAvailable = false; }
}

async function reloadLatestMLModel() {
    if (!usePythonML || !mlBackendAvailable) return;
    try {
        const response = await fetch(`${ML_API_URL}/api/ml/reload`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                showToast(`ML 模型已重新加载 (${result.q_table_size} 状态)`, 'success', 2000);
            }
        }
    } catch (error) {
        console.warn('重新加载 ML 模型失败:', error);
    }
}

// ====================================================================
// 数据重置
// ====================================================================
function resetChartData() {
    globalMarketState.tick = 0;
    globalMarketState.regime = 'range';
    globalMarketState.stepsLeft = 45;
    globalMarketState.lastReturn = 0;
    globalMarketState.volatility = 0.010;
    globalMarketState.eventShock = 0;
    globalMarketState.sectorReturns = {};
    for (const s of STOCKS) {
        const st = stockState[s.symbol];
        st.priceData = []; st.highData = []; st.lowData = []; st.volumeData = []; st.timeData = [];
        st.rsiData = []; st.macdData = []; st.signalLineData = []; st.histogramData = [];
        st.ma5Data = []; st.ma10Data = [];
        st.garchVariance = 0.0001; st.volumeARState = 1.0;
        st.marketRegime = 'range'; st.regimeStepsLeft = 30;
    }
    assetTimeData = []; userAssetData = []; machineAssetData = []; mlRobotAssetData = [];
    mlRobotPortfolioReturnWindow = [];
    mlRobotPortfolioSharpeHistory = [];
    lastMLPortfolioValue = null;

    [priceChart, rsiChart, macdChart, assetChart, rewardChart, accuracyChart, sharpeChart].forEach(chart => {
        if (chart) {
            chart.data.labels = [];
            chart.data.datasets.forEach(dataset => dataset.data = []);
            chart.update();
        }
    });
}

function limitDataPoints() {
    const maxDataPoints = currentTimeRange * 60;
    for (const s of STOCKS) {
        const st = stockState[s.symbol];
        if (st.priceData.length > maxDataPoints + 10) {
            const excess = st.priceData.length - maxDataPoints;
            st.priceData.splice(0, excess); st.highData.splice(0, excess); st.lowData.splice(0, excess);
            st.volumeData.splice(0, excess); st.timeData.splice(0, excess); st.rsiData.splice(0, excess);
            st.macdData.splice(0, excess); st.signalLineData.splice(0, excess); st.histogramData.splice(0, excess);
            st.ma5Data.splice(0, excess); st.ma10Data.splice(0, excess);
        }
    }
    const maxAssetDataPoints = 30;
    if (userAssetData.length > maxAssetDataPoints + 5) {
        const excess = userAssetData.length - maxAssetDataPoints;
        userAssetData.splice(0, excess); machineAssetData.splice(0, excess);
        mlRobotAssetData.splice(0, excess); assetTimeData.splice(0, excess);
    }
}

function resetMLModel() {
    if (!confirm('确定要重置机器学习模型吗？')) return;
    mlRobotCash = 100.00;
    for (const s of STOCKS) {
        const rh = mlRobotHoldings[s.symbol];
        rh.shares = 0; rh.totalInvested = 0; rh.txHistory = [];
        rh.qTable = {}; rh.qTableAccess = {};
        rh.stopLossPrice = null; rh.takeProfitPrice = null;
        rh.trailingPeak = null; rh.lastBuyTime = null;
        rh.returnLogWindow = []; rh.returnWindow = []; rh.downsideWindow = [];
        rh.rewardHistory = []; rh.accuracyHistory = []; rh.sharpeHistory = []; rh.assetData = [];
    }
    mlRobotStats.totalReward = 0; mlRobotStats.correctDecisions = 0; mlRobotStats.totalDecisions = 0;
    mlRobotStats.explorationRate = 0.3;
    mlRobotStats.assetPeak = 100; mlRobotStats.maxDrawdown = 0; mlRobotStats.maxDrawdownAsset = 100;
    mlRobotStats.winTrades = 0; mlRobotStats.lossTrades = 0;
    mlRobotStats.grossProfit = 0; mlRobotStats.grossLoss = 0;
    mlRobotStats.totalFees = 0;
    mlRobotPortfolioReturnWindow = [];
    mlRobotPortfolioSharpeHistory = [];
    lastMLPortfolioValue = null;
    explorationRateSlider.value = mlRobotStats.explorationRate;
    explorationRateDisplay.textContent = mlRobotStats.explorationRate.toFixed(2);
    updateAssetDisplay();
    if (rewardChart) { rewardChart.data.datasets[0].data = []; rewardChart.update(); }
    if (accuracyChart) { accuracyChart.data.datasets[0].data = []; accuracyChart.update(); }
    if (sharpeChart) { sharpeChart.data.datasets[0].data = []; sharpeChart.update(); }
    localStorage.removeItem('ml_robot_qtable_multi');
    showToast('机器学习模型已重置', 'info');
}

function resetBacktest() {
    if (!confirm('确定要重置回测统计吗？')) return;
    mlRobotStats.startTime = Date.now();
    mlRobotStats.maxDrawdown = 0;
    mlRobotStats.maxDrawdownAsset = getMLRobotTotalAsset();
    mlRobotStats.winTrades = 0; mlRobotStats.lossTrades = 0;
    mlRobotStats.grossProfit = 0; mlRobotStats.grossLoss = 0;
    mlRobotStats.biggestWin = 0; mlRobotStats.biggestLoss = 0;
    mlRobotStats.totalFees = 0; mlRobotStats.floatingPnL = 0;
    mlRobotStats.assetPeak = getMLRobotTotalAsset();
    mlRobotPortfolioReturnWindow = [];
    mlRobotPortfolioSharpeHistory = [];
    lastMLPortfolioValue = getMLRobotTotalAsset();
    updateBacktestInfo();
    showToast('回测统计已重置', 'info');
}

// 预训练：在合成数据上快速预填 Q-table，让机器人一开始就具备一些先验知识
async function pretrainMLModel() {
    if (isPretraining) return;
    isPretraining = true;
    pretrainCancelRequested = false;

    // 显示进度容器
    if (pretrainProgressContainer) pretrainProgressContainer.classList.remove('hidden');
    const startBtn = document.getElementById('pretrainMLModelBtn');
    if (startBtn) { startBtn.disabled = true; startBtn.classList.add('opacity-50', 'cursor-not-allowed'); }

    const sizeSelect = document.getElementById('trainingDataSize');
    const totalEpisodes = parseInt(sizeSelect ? sizeSelect.value : '3000') || 3000;
    const episodesPerStock = Math.max(100, Math.floor(totalEpisodes / STOCKS.length));

    if (pretrainStatus) pretrainStatus.textContent = '初始化...';
    if (pretrainProgressBar) pretrainProgressBar.style.width = '0%';
    if (pretrainProgressPercent) pretrainProgressPercent.textContent = '0%';

    // 临时给 Q-table 一些先验：使用合成价格序列
    let correctTotal = 0, decisionTotal = 0;
    const lr = mlRobotStats.learningRate;
    const gamma = mlRobotStats.discountFactor;
    const batchSize = 50;

    try {
        for (let i = 0; i < episodesPerStock; i++) {
            if (pretrainCancelRequested) {
                if (pretrainStatus) pretrainStatus.textContent = '已取消';
                break;
            }

            const stockIdx = i % STOCKS.length;
            const s = STOCKS[stockIdx];
            const cfg = s;
            const rh = mlRobotHoldings[s.symbol];

            // 生成一段合成价格序列（~ 30 个 tick）
            const seqLen = 30;
            const prices = [];
            let p = cfg.startPrice;
            const vol = 0.025 * cfg.volMult;
            for (let k = 0; k < seqLen; k++) {
                const r = (gaussian() * vol) + cfg.driftBias;
                p = Math.max(0.5, p * (1 + r));
                prices.push(p);
            }

            // 沿序列遍历并更新 Q-table
            for (let t = 1; t < seqLen; t++) {
                const price = prices[t], prev = prices[t - 1];
                const priceChange = (price - prev) / prev;
                const rsi = Math.min(95, Math.max(5, 50 + priceChange * 800));
                const macd = priceChange * 10;
                const trendDir = t > 5 && prices[t] > prices[t - 5] ? 1 : -1;
                const vola = 0.02;
                const state = encodeSmartState(price, priceChange, rsi, macd, false, 0, 'neutral', 0, trendDir, { volatility: vola });
                const nextState = state; // 单步无后继

                // 评估三种行动并更新 Q
                const actions = ['buy', 'sell', 'hold'];
                for (const action of actions) {
                    let reward = 0;
                    let correct = false;
                    if (action === 'buy') {
                        const future = (prices[Math.min(seqLen - 1, t + 3)] - price) / price;
                        reward = future * 200;
                        correct = future > 0.002;
                    } else if (action === 'sell') {
                        const future = (price - prices[Math.min(seqLen - 1, t + 3)]) / price;
                        reward = future * 200;
                        correct = future > 0.002;
                    } else {
                        // hold 在震荡中正确
                        correct = Math.abs(priceChange) < 0.005;
                        reward = correct ? 1 : -1;
                    }
                    if (correct) correctTotal++;
                    decisionTotal++;
                    updateQTable(rh.qTable, rh.qTableAccess, state, action, reward, nextState, lr, gamma);
                }
            }

            if (i % batchSize === 0) {
                const pct = Math.floor((i / episodesPerStock) * 100);
                if (pretrainProgressBar) pretrainProgressBar.style.width = pct + '%';
                if (pretrainProgressPercent) pretrainProgressPercent.textContent = pct + '%';
                if (pretrainStatus) pretrainStatus.textContent = `训练中... ${i}/${episodesPerStock} 集 (${s.symbol})`;
                if (pretrainAccuracy) pretrainAccuracy.textContent = decisionTotal > 0 ? (correctTotal / decisionTotal * 100).toFixed(2) + '%' : '0.00%';
                // 让浏览器有机会刷新
                await new Promise(r => setTimeout(r, 0));
            }
        }

        // 完成
        if (pretrainProgressBar) pretrainProgressBar.style.width = '100%';
        if (pretrainProgressPercent) pretrainProgressPercent.textContent = '100%';
        if (pretrainStatus) pretrainStatus.textContent = '预训练完成 ✓';
        const acc = decisionTotal > 0 ? (correctTotal / decisionTotal * 100).toFixed(2) : '0.00';
        if (pretrainAccuracy) pretrainAccuracy.textContent = acc + '%';
        if (pretrainQTableSize) {
            let total = 0;
            for (const s of STOCKS) total += Object.keys(mlRobotHoldings[s.symbol].qTable).length;
            pretrainQTableSize.textContent = total.toString();
        }
        // 立刻保存
        flushSaveQTable();
        showToast(`预训练完成 ✓ (${episodesPerStock} 集 × ${STOCKS.length} 股)`, 'success', 3500);
    } catch (e) {
        if (pretrainStatus) pretrainStatus.textContent = '预训练失败: ' + e.message;
        console.error('pretrain error:', e);
    } finally {
        isPretraining = false;
        if (startBtn) { startBtn.disabled = false; startBtn.classList.remove('opacity-50', 'cursor-not-allowed'); }
        // 3s 后折叠进度条
        setTimeout(() => {
            if (!isPretraining && pretrainProgressContainer) pretrainProgressContainer.classList.add('hidden');
        }, 3000);
    }
}

// ====================================================================
// UI
// ====================================================================
let activeToast = null;
function showToast(message, type = 'info', duration = 3000) {
    if (activeToast) { try { document.body.removeChild(activeToast); } catch (e) {} activeToast = null; }
    const colors = { success: 'bg-green-600', error: 'bg-red-600', info: 'bg-purple-600', warning: 'bg-yellow-600' };
    const toast = document.createElement('div');
    toast.className = 'fixed bottom-4 right-4 ' + (colors[type] || colors.info) + ' text-white px-4 py-2 rounded-lg shadow-lg z-50 transition-all duration-300';
    toast.textContent = message;
    document.body.appendChild(toast);
    activeToast = toast;
    setTimeout(() => {
        toast.style.opacity = '0'; toast.style.transform = 'translateY(20px)';
        setTimeout(() => { if (document.body.contains(toast)) document.body.removeChild(toast); if (activeToast === toast) activeToast = null; }, 300);
    }, duration);
}

function setCurrentStock(symbol) {
    currentSymbol = symbol;
    const cfg = getCurrentStockCfg();
    if (currentStockNameEl) currentStockNameEl.textContent = cfg.name + ' (' + cfg.symbol + ')';
    if (currentStockSectorEl) currentStockSectorEl.textContent = cfg.sector + ' · ' + cfg.description;
    // 更新股票卡片高亮
    if (stockCardsEl) {
        stockCardsEl.querySelectorAll('[data-symbol]').forEach(el => {
            if (el.dataset.symbol === symbol) {
                el.classList.add('ring-2', 'ring-primary', 'bg-blue-50');
            } else {
                el.classList.remove('ring-2', 'ring-primary', 'bg-blue-50');
            }
        });
    }
    // 更新图表标题
    const chartStockNameEl = document.getElementById('chartStockName');
    if (chartStockNameEl) chartStockNameEl.textContent = cfg.symbol + ' ' + cfg.name;
    updateAssetDisplay();
    updateCharts();
    updateTransactionHistory();
    updateMultiStockOverview();
}

// 多股票总览表
function updateMultiStockOverview() {
    const tbody = document.getElementById('multiStockOverview');
    if (!tbody) return;
    let html = '';
    for (const s of STOCKS) {
        const st = stockState[s.symbol];
        const uh = userHoldings[s.symbol];
        const rh = mlRobotHoldings[s.symbol];
        if (st.priceData.length === 0) {
            html += `<tr><td colspan="7" class="px-2 py-2 text-gray-400">${s.symbol} ${s.name} 等待数据...</td></tr>`;
            continue;
        }
        const last = st.priceData[st.priceData.length - 1];
        const prev = st.priceData.length >= 2 ? st.priceData[st.priceData.length - 2] : last;
        const change = last - prev;
        const changePct = prev > 0 ? (change / prev) * 100 : 0;
        const color = change >= 0 ? 'text-red-600' : 'text-green-600';  // A股风格：红涨绿跌
        const rsi = st.rsiData[st.rsiData.length - 1];
        const rsiTxt = rsi === null || rsi === undefined ? '--' : rsi.toFixed(0);
        const ma5 = st.ma5Data[st.ma5Data.length - 1];
        const ma10 = st.ma10Data[st.ma10Data.length - 1];
        let trendTxt = '震荡';
        let trendColor = 'text-gray-500';
        if (ma5 && ma10) {
            if (ma5 > ma10 * 1.002) { trendTxt = '↑ 上升'; trendColor = 'text-red-600'; }
            else if (ma5 < ma10 * 0.998) { trendTxt = '↓ 下降'; trendColor = 'text-green-600'; }
        }
        const isActive = s.symbol === currentSymbol;
        const rowClass = isActive ? 'bg-blue-50 font-medium' : 'hover:bg-gray-50';
        const aiHasShares = rh.shares > 0 ? `持有 ${rh.shares}股` : (rh.qTable && Object.keys(rh.qTable).length > 0 ? '学习中' : '观察中');
        html += `
            <tr class="${rowClass} cursor-pointer" data-symbol="${s.symbol}">
                <td class="px-2 py-2">
                    <span class="inline-block w-2 h-2 rounded-full mr-1" style="background:${s.color}"></span>
                    <span class="font-medium">${s.symbol}</span>
                    <span class="text-gray-500 ml-1">${s.name}</span>
                </td>
                <td class="px-2 py-2 text-right font-mono">¥${last.toFixed(2)}</td>
                <td class="px-2 py-2 text-right ${color}">${change >= 0 ? '+' : ''}${changePct.toFixed(2)}%</td>
                <td class="px-2 py-2 text-right">${rsiTxt}</td>
                <td class="px-2 py-2 ${trendColor}">${trendTxt}</td>
                <td class="px-2 py-2 text-right">${uh.shares}</td>
                <td class="px-2 py-2 text-right text-xs text-purple-600">${aiHasShares}</td>
            </tr>`;
    }
    tbody.innerHTML = html;
    // 点击行切换股票
    tbody.querySelectorAll('[data-symbol]').forEach(el => {
        el.addEventListener('click', () => setCurrentStock(el.dataset.symbol));
    });
}

function setTimeRange(minutes) {
    currentTimeRange = minutes;
    timeRange5.classList.remove('bg-primary', 'text-white');
    timeRange15.classList.remove('bg-primary', 'text-white');
    timeRange30.classList.remove('bg-primary', 'text-white');
    timeRange5.classList.add('bg-gray-200', 'text-gray-700');
    timeRange15.classList.add('bg-gray-200', 'text-gray-700');
    timeRange30.classList.add('bg-gray-200', 'text-gray-700');
    if (minutes === 5) { timeRange5.classList.remove('bg-gray-200', 'text-gray-700'); timeRange5.classList.add('bg-primary', 'text-white'); }
    else if (minutes === 15) { timeRange15.classList.remove('bg-gray-200', 'text-gray-700'); timeRange15.classList.add('bg-primary', 'text-white'); }
    else if (minutes === 30) { timeRange30.classList.remove('bg-gray-200', 'text-gray-700'); timeRange30.classList.add('bg-primary', 'text-white'); }
    updateCharts();
    limitDataPoints();
}

function buildStockCards() {
    if (!stockCardsEl) return;
    stockCardsEl.innerHTML = '';
    for (const s of STOCKS) {
        const div = document.createElement('div');
        div.dataset.symbol = s.symbol;
        div.className = 'cursor-pointer rounded-lg p-2 border border-gray-200 hover:bg-blue-50 transition-all';
        div.innerHTML = `
            <div class="flex items-center justify-between">
                <div>
                    <div class="font-semibold text-sm" style="color:${s.color}">${s.symbol}</div>
                    <div class="text-xs text-gray-500">${s.name}</div>
                </div>
                <div class="text-xs text-gray-400">${s.sector}</div>
            </div>`;
        div.addEventListener('click', () => setCurrentStock(s.symbol));
        stockCardsEl.appendChild(div);
    }
}

// ====================================================================
// 初始化
// ====================================================================
function initialize() {
    initStockStates();

    // DOM 元素
    startBtn = document.getElementById('startBtn');
    cashDisplay = document.getElementById('cashDisplay');
    totalAssetDisplay = document.getElementById('totalAssetDisplay');
    currentPriceDisplay = document.getElementById('currentPriceDisplay');
    priceChangeDisplay = document.getElementById('priceChangeDisplay');
    priceIndicator = document.getElementById('priceIndicator');
    buyMessage = document.getElementById('buyMessage');
    sellSharesInput = document.getElementById('sellShares');
    sellBtn = document.getElementById('sellBtn');
    sellMessage = document.getElementById('sellMessage');
    transactionHistoryTable = document.getElementById('transactionHistory');
    maCheckbox = document.getElementById('maCheckbox');
    rsiCheckbox = document.getElementById('rsiCheckbox');
    macdCheckbox = document.getElementById('macdCheckbox');
    timeRange5 = document.getElementById('timeRange5');
    timeRange15 = document.getElementById('timeRange15');
    timeRange30 = document.getElementById('timeRange30');
    buySharesBtns = document.querySelectorAll('.buy-shares-btn');
    mlRobotToggle = document.getElementById('mlRobotToggle');
    learningRateSlider = document.getElementById('learningRateSlider');
    learningRateDisplay = document.getElementById('learningRateDisplay');
    explorationRateSlider = document.getElementById('explorationRateSlider');
    explorationRateDisplay = document.getElementById('explorationRateDisplay');
    resetMLModelBtn = document.getElementById('resetMLModelBtn');
    mlRobotTransactionsBtn = document.getElementById('mlRobotTransactionsBtn');
    currentSharpeDisplay = document.getElementById('currentSharpeDisplay');
    stockCardsEl = document.getElementById('stockCards');
    currentStockNameEl = document.getElementById('currentStockName');
    currentStockSectorEl = document.getElementById('currentStockSector');

    machineCashDisplay = document.getElementById('machineCashDisplay');
    machineTotalAssetDisplay = document.getElementById('machineTotalAssetDisplay');
    machineSharesDisplay = document.getElementById('machineSharesDisplay');
    machineCostDisplay = document.getElementById('machineCostDisplay');
    machineValueDisplay = document.getElementById('machineValueDisplay');
    mlRobotCashDisplay = document.getElementById('mlRobotCashDisplay');
    mlRobotTotalAssetDisplay = document.getElementById('mlRobotTotalAssetDisplay');
    mlRobotSharesDisplay = document.getElementById('mlRobotSharesDisplay');
    mlRobotCostDisplay = document.getElementById('mlRobotCostDisplay');
    mlRobotValueDisplay = document.getElementById('mlRobotValueDisplay');
    currentSharesDisplay = document.getElementById('currentSharesDisplay');
    currentCostDisplay = document.getElementById('currentCostDisplay');
    currentValueDisplay = document.getElementById('currentValueDisplay');
    userTransactionsBtn = document.getElementById('userTransactionsBtn');
    machineTransactionsBtn = document.getElementById('machineTransactionsBtn');

    btTotalAsset = document.getElementById('btTotalAsset');
    btReturnRate = document.getElementById('btReturnRate');
    btSharpe = document.getElementById('btSharpe');
    btAccuracy = document.getElementById('btAccuracy');
    btDecisions = document.getElementById('btDecisions');
    btTotalReward = document.getElementById('btTotalReward');
    btAvgReward = document.getElementById('btAvgReward');
    btWinRate = document.getElementById('btWinRate');
    btTradeCount = document.getElementById('btTradeCount');
    btProfitFactor = document.getElementById('btProfitFactor');
    btAvgPnL = document.getElementById('btAvgPnL');
    btMaxDrawdown = document.getElementById('btMaxDrawdown');
    btMaxDDAsset = document.getElementById('btMaxDDAsset');
    btFloatingPnL = document.getElementById('btFloatingPnL');
    btFees = document.getElementById('btFees');
    btAlphas = document.getElementById('btAlphas');
    btMachine = document.getElementById('btMachine');
    btBuyHold = document.getElementById('btBuyHold');
    backtestSessionDuration = document.getElementById('backtestSessionDuration');
    resetBacktestBtn = document.getElementById('resetBacktestBtn');

    // 预训练 DOM 绑定
    pretrainProgressContainer = document.getElementById('pretrainProgressContainer');
    pretrainProgressBar = document.getElementById('pretrainProgressBar');
    pretrainProgressPercent = document.getElementById('pretrainProgressPercent');
    pretrainStatus = document.getElementById('pretrainStatus');
    pretrainAccuracy = document.getElementById('pretrainAccuracy');
    pretrainQTableSize = document.getElementById('pretrainQTableSize');
    const pretrainMLModelBtn = document.getElementById('pretrainMLModelBtn');
    const cancelPretrainBtn = document.getElementById('cancelPretrainBtn');
    if (pretrainMLModelBtn) pretrainMLModelBtn.addEventListener('click', pretrainMLModel);
    if (cancelPretrainBtn) cancelPretrainBtn.addEventListener('click', () => { pretrainCancelRequested = true; });

    checkPythonMLService();

    startBtn.addEventListener('click', toggleSimulation);
    sellBtn.addEventListener('click', () => handleSell());

    buySharesBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            handleBuy(parseInt(this.dataset.shares));
        });
    });

    document.getElementById('machineTradingToggle').addEventListener('change', function() {
        isMachineTradingEnabled = this.checked;
        updateCharts();
    });
    mlRobotToggle.addEventListener('change', function() {
        isMLRobotEnabled = this.checked;
        updateCharts();
    });
    learningRateSlider.addEventListener('input', function() {
        mlRobotStats.learningRate = parseFloat(this.value);
        learningRateDisplay.textContent = mlRobotStats.learningRate.toFixed(2);
    });
    explorationRateSlider.addEventListener('input', function() {
        mlRobotStats.explorationRate = parseFloat(this.value);
        explorationRateDisplay.textContent = mlRobotStats.explorationRate.toFixed(2);
    });
    resetMLModelBtn.addEventListener('click', resetMLModel);
    if (resetBacktestBtn) resetBacktestBtn.addEventListener('click', resetBacktest);

    if (userTransactionsBtn) {
        userTransactionsBtn.addEventListener('click', () => {
            currentTxView = 'user';
            highlightTxBtn(userTransactionsBtn);
            updateTransactionHistory();
        });
    }
    if (machineTransactionsBtn) {
        machineTransactionsBtn.addEventListener('click', () => {
            currentTxView = 'machine';
            highlightTxBtn(machineTransactionsBtn);
            updateTransactionHistory();
        });
    }
    if (mlRobotTransactionsBtn) {
        mlRobotTransactionsBtn.addEventListener('click', () => {
            currentTxView = 'mlRobot';
            highlightTxBtn(mlRobotTransactionsBtn);
            updateTransactionHistory();
        });
    }
    function highlightTxBtn(activeBtn) {
        [userTransactionsBtn, machineTransactionsBtn, mlRobotTransactionsBtn].forEach(btn => {
            if (!btn) return;
            btn.classList.remove('bg-primary', 'text-white');
            btn.classList.add('bg-gray-200', 'text-gray-700');
        });
        if (activeBtn) {
            activeBtn.classList.remove('bg-gray-200', 'text-gray-700');
            activeBtn.classList.add('bg-primary', 'text-white');
        }
    }

    maCheckbox.addEventListener('change', updateCharts);
    rsiCheckbox.addEventListener('change', updateCharts);
    macdCheckbox.addEventListener('change', updateCharts);

    timeRange5.addEventListener('click', () => setTimeRange(5));
    timeRange15.addEventListener('click', () => setTimeRange(15));
    timeRange30.addEventListener('click', () => setTimeRange(30));

    initializeCharts();
    initializeMLCharts();
    loadQTable();
    buildStockCards();
    setCurrentStock(currentSymbol);

    buySharesBtns.forEach(btn => { btn.disabled = true; btn.classList.add('opacity-50', 'cursor-not-allowed'); });
    sellBtn.disabled = true;

    updateAssetDisplay();
    recordAssetData();
    updateBacktestInfo();

    window.addEventListener('beforeunload', function() {
        try { flushSaveQTable(); } catch (e) {}
        if (usePythonML && mlBackendAvailable) {
            try { navigator.sendBeacon(ML_API_URL + '/api/ml/save'); } catch (e) {}
        }
    });
}

function initializeCharts() {
    const priceCtx = document.getElementById('priceChart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: '价格', data: [], borderColor: '#1a56db', backgroundColor: 'rgba(26, 86, 219, 0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 4 },
                { label: 'MA5', data: [], borderColor: '#10b981', borderWidth: 1.5, fill: false, tension: 0.4, pointRadius: 0, hidden: false },
                { label: 'MA10', data: [], borderColor: '#f59e0b', borderWidth: 1.5, fill: false, tension: 0.4, pointRadius: 0, hidden: false }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    enabled: true, backgroundColor: 'rgba(255, 255, 255, 0.9)', titleColor: '#1e293b', bodyColor: '#1e293b',
                    borderColor: '#e2e8f0', borderWidth: 1, padding: 10, displayColors: true,
                    callbacks: { label: function(context) { return context.dataset.label + ': ¥' + context.parsed.y.toFixed(2); } }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
                y: { position: 'right', grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { callback: function(value) { return '¥' + value.toFixed(2); } } }
            },
            animation: { duration: 0 }
        }
    });

    const rsiCtx = document.getElementById('rsiChart').getContext('2d');
    rsiChart = new Chart(rsiCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'RSI', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 },
                { label: '超买线', data: Array(30).fill(70), borderColor: 'rgba(239, 68, 68, 0.5)', borderWidth: 1, borderDash: [5, 5], fill: false, pointRadius: 0 },
                { label: '超卖线', data: Array(30).fill(30), borderColor: 'rgba(16, 185, 129, 0.5)', borderWidth: 1, borderDash: [5, 5], fill: false, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: true, backgroundColor: 'rgba(255, 255, 255, 0.9)', titleColor: '#1e293b', bodyColor: '#1e293b', borderColor: '#e2e8f0', borderWidth: 1, padding: 10 } },
            scales: {
                x: { display: true, grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 6 } },
                y: { min: 0, max: 100, position: 'right', grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { stepSize: 20 } }
            },
            animation: { duration: 0 },
            interaction: { intersect: false, mode: 'index' }
        }
    });

    const macdCtx = document.getElementById('macdChart').getContext('2d');
    macdChart = new Chart(macdCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Histogram', data: [],
                    backgroundColor: function(context) { const v = context.dataset.data[context.dataIndex]; return v >= 0 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)'; },
                    borderColor: function(context) { const v = context.dataset.data[context.dataIndex]; return v >= 0 ? 'rgb(16, 185, 129)' : 'rgb(239, 68, 68)'; },
                    borderWidth: 1, barPercentage: 0.8
                },
                { label: 'MACD', data: [], type: 'line', borderColor: '#1a56db', borderWidth: 2, fill: false, tension: 0.4, pointRadius: 0, yAxisID: 'y1' },
                { label: 'Signal', data: [], type: 'line', borderColor: '#f59e0b', borderWidth: 2, fill: false, tension: 0.4, pointRadius: 0, yAxisID: 'y1' }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: true, backgroundColor: 'rgba(255, 255, 255, 0.9)', titleColor: '#1e293b', bodyColor: '#1e293b', borderColor: '#e2e8f0', borderWidth: 1, padding: 10 } },
            scales: {
                x: { display: false },
                y: { display: false, grid: { display: false } },
                y1: { position: 'right', grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { display: true } }
            },
            animation: { duration: 0 },
            interaction: { intersect: false, mode: 'index' }
        }
    });

    const assetCtx = document.getElementById('assetChart').getContext('2d');
    assetChart = new Chart(assetCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: '您的资产', data: [], borderColor: '#1a56db', backgroundColor: 'rgba(26, 86, 219, 0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 },
                { label: '规则AI资产', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0, hidden: false },
                { label: '学习AI资产', data: [], borderColor: '#9333ea', backgroundColor: 'rgba(147, 51, 234, 0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0, hidden: false }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: true, backgroundColor: 'rgba(255, 255, 255, 0.9)', titleColor: '#1e293b', bodyColor: '#1e293b', borderColor: '#e2e8f0', borderWidth: 1, padding: 10, callbacks: { label: function(context) { return '总资产: ¥' + context.parsed.y.toFixed(2); } } }
            },
            scales: {
                x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
                y: { position: 'right', grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { callback: function(value) { return '¥' + value.toFixed(2); } } }
            },
            animation: { duration: 0 },
            interaction: { intersect: false, mode: 'index' }
        }
    });
}

function initializeMLCharts() {
    const rewardCtx = document.getElementById('rewardChart').getContext('2d');
    rewardChart = new Chart(rewardCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: '累计奖励', data: [], borderColor: '#9333ea', backgroundColor: 'rgba(147, 51, 234, 0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: true, backgroundColor: 'rgba(255, 255, 255, 0.9)', titleColor: '#1e293b', bodyColor: '#1e293b', borderColor: '#e2e8f0', borderWidth: 1, padding: 10 } },
            scales: { x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 6 } }, y: { position: 'right', grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { callback: function(value) { return value.toFixed(1); } } } },
            animation: { duration: 0 }
        }
    });

    const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
    accuracyChart = new Chart(accuracyCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: '决策准确率', data: [], borderColor: '#9333ea', backgroundColor: 'rgba(147, 51, 234, 0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: true, backgroundColor: 'rgba(255, 255, 255, 0.9)', titleColor: '#1e293b', bodyColor: '#1e293b', borderColor: '#e2e8f0', borderWidth: 1, padding: 10, callbacks: { label: function(context) { return '准确率: ' + context.parsed.y.toFixed(2) + '%'; } } } },
            scales: { x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 6 } }, y: { position: 'right', grid: { color: 'rgba(0, 0, 0, 0.05)' }, min: 0, max: 100, ticks: { callback: function(value) { return value + '%'; } } } },
            animation: { duration: 0 }
        }
    });

    const sharpeCtx = document.getElementById('sharpeChart').getContext('2d');
    sharpeChart = new Chart(sharpeCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: '夏普比率', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: true, backgroundColor: 'rgba(255, 255, 255, 0.9)', titleColor: '#1e293b', bodyColor: '#1e293b', borderColor: '#e2e8f0', borderWidth: 1, padding: 10, callbacks: { label: function(context) { return '夏普: ' + context.parsed.y.toFixed(2); } } } },
            scales: { x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 6 } }, y: { position: 'right', grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { callback: function(value) { return value.toFixed(2); } } } },
            animation: { duration: 0 }
        }
    });
}

window.addEventListener('DOMContentLoaded', initialize);
