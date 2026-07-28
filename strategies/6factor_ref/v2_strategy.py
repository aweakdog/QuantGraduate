"""
================================================================================
 拓普集团(601689.SH) 量化策略 — Round 2: 权重优化+Kelly仓位+止损
================================================================================
 基于R1回测结果:
   - 调优因子权重 (基本面/动量高权重, 技术面/量能中权重, 资金面/筹码低权重)
   - Kelly公式动态仓位
   - -8%硬止损
   - 信号阈值优化

 SuperMind 兼容: init/handle_bar/history/order_target_percent
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# Round 2 策略参数 (基于R1因子表现调优)
# ============================================================
SYMBOL = '601689.SH'
SYMBOL_NAME = '拓普集团'

# R2权重优化 (基本面/动量权重提升, 因R1显示稳定性更高)
FACTOR_WEIGHTS_R2 = {
    'technical':    0.15,   # 技术面 (中等)
    'fund_flow':    0.10,   # 资金面 (信号噪声大,降权)
    'fundamental':  0.25,   # 基本面 (稳定,提权)
    'volume':       0.15,   # 量能 (辅助确认)
    'position':     0.10,   # 筹码 (降权)
    'momentum':     0.25,   # 动量 (趋势跟随,提权)
}

# 信号阈值
BUY_THRESHOLD = 62      # 略提高减少假信号
SELL_THRESHOLD = 38     # 降低卖出敏感性

# Kelly参数
MAX_POSITION = 0.70      # 最大仓位70%
MIN_POSITION = 0.0       # 最小仓位0%

# 止损
STOP_LOSS = -0.08        # -8%硬止损

# 因子计算参数
MA_SHORT = 5
MA_LONG = 20
RSI_PERIOD = 14
MOMENTUM_PERIOD = 20
VOL_PERIOD = 60


def init(context):
    """SuperMind 初始化"""
    context.symbol = SYMBOL
    context.position_pct = 0.0
    context.last_signal = 'HOLD'
    context.trade_count = 0
    context.win_count = 0
    context.trades = []
    context.equity_curve = []
    context.entry_price = 0.0
    context.highest_since_entry = 0.0  # 入场后最高价
    context.consecutive_signals = 0     # 连续同向信号确认

    context.subscribe(SYMBOL, '1d')

    print(f'[R2] 拓普集团 权重优化+Kelly仓位+止损策略 初始化完成')
    print(f'     因子权重: 基本面/动量各25%, 技术面/量能各15%, 资金面/筹码各10%')
    print(f'     仓位: Kelly动态, 最大{MAX_POSITION*100:.0f}%')
    print(f'     止损: {STOP_LOSS*100:.0f}%硬止损')
    print(f'     信号阈值: 买入>{BUY_THRESHOLD}, 卖出<{SELL_THRESHOLD}')


def handle_bar(context, bar):
    """SuperMind 每根K线调用"""
    symbol = context.symbol

    # 获取历史数据
    h_close = history(symbol, 'close', MA_LONG + MOMENTUM_PERIOD + 10, '1d')
    h_volume = history(symbol, 'volume', VOL_PERIOD + 5, '1d')
    h_high = history(symbol, 'high', RSI_PERIOD + 10, '1d')
    h_low = history(symbol, 'low', RSI_PERIOD + 10, '1d')

    if len(h_close) < MA_LONG:
        return

    close = np.array(h_close)
    volume = np.array(h_volume)
    high = np.array(h_high)
    low = np.array(h_low)

    latest_price = close[-1]

    # ---- 止损检查 (最高优先级) ----
    if context.position_pct > 0 and context.entry_price > 0:
        context.highest_since_entry = max(context.highest_since_entry, latest_price)
        current_pnl = (latest_price - context.entry_price) / context.entry_price

        if current_pnl <= STOP_LOSS:
            order_target_percent(symbol, 0.0)
            context.trades.append({
                'entry': context.entry_price, 'exit': latest_price,
                'pnl': current_pnl, 'reason': 'stop_loss'
            })
            context.trade_count += 1
            context.entry_price = 0.0
            context.highest_since_entry = 0.0
            context.consecutive_signals = 0
            context.equity_curve.append({
                'date': str(context.now), 'total_score': 0,
                'signal': 'STOP_LOSS', 'nav': context.portfolio.unit_net_value,
                'scores': {}, 'position': 0
            })
            return

    # ---- 计算6因子得分 ----
    scores = {}

    scores['technical'] = calc_technical(close)
    scores['fund_flow'] = calc_fund_flow(close, volume)
    scores['fundamental'] = calc_fundamental(close)
    scores['volume'] = calc_volume(volume)
    scores['position'] = calc_position(close, high, low)
    scores['momentum'] = calc_momentum(close, high, low)

    # ---- R2综合评分 (优化权重) ----
    total_score = sum(scores[k] * FACTOR_WEIGHTS_R2[k] for k in FACTOR_WEIGHTS_R2)

    # ---- Kelly仓位计算 ----
    kelly_position = calculate_kelly_position(scores, total_score, close)

    # ---- 交易信号 (增加确认机制) ----
    if total_score >= BUY_THRESHOLD:
        context.consecutive_signals = max(0, context.consecutive_signals) + 1
        if context.consecutive_signals >= 1:  # 至少1次确认
            order_target_percent(symbol, kelly_position)
            signal = 'BUY'
        else:
            signal = 'WAIT_BUY'
    elif total_score <= SELL_THRESHOLD:
        context.consecutive_signals = max(0, context.consecutive_signals) + 1
        if context.consecutive_signals >= 1:
            order_target_percent(symbol, 0.0)
            signal = 'SELL'
        else:
            signal = 'WAIT_SELL'
    else:
        context.consecutive_signals = 0
        signal = 'HOLD'

    # ---- 记录入场价 ----
    if signal == 'BUY' and context.position_pct > 0 and context.entry_price == 0:
        context.entry_price = latest_price
        context.highest_since_entry = latest_price
        context.trade_count += 1

    # ---- 记录 ----
    context.last_signal = signal
    context.equity_curve.append({
        'date': str(context.now),
        'total_score': total_score,
        'signal': signal,
        'nav': context.portfolio.unit_net_value,
        'scores': scores.copy(),
        'kelly_position': kelly_position,
        'actual_position': context.position_pct,
    })


def calculate_kelly_position(scores, total_score, close):
    """Kelly公式仓位计算

    f* = (p * W - q * L) / (W * L)
    其中 p=胜率估计, q=1-p, W=平均盈利, L=平均亏损
    简化为: f* = (2p - 1) * 0.5 (半凯利保守)
    """
    # 基于综合得分估计胜率
    if total_score >= 75:
        win_prob = 0.65
    elif total_score >= 65:
        win_prob = 0.55
    elif total_score >= 55:
        win_prob = 0.50
    elif total_score >= 45:
        win_prob = 0.42
    else:
        win_prob = 0.35

    # 半凯利: f* = (2p - 1) * 0.5
    kelly_f = max(0, (2 * win_prob - 1) * 0.5)

    # 基于波动率调整
    if len(close) >= 20:
        returns = np.diff(close[-20:]) / close[-21:-1]
        volatility = np.std(returns) * np.sqrt(252)
        vol_factor = min(1.0, 0.35 / max(volatility, 0.1))
        kelly_f *= vol_factor

    # 基于动量正向调整
    momentum_score = scores.get('momentum', 50)
    if momentum_score > 65:
        kelly_f *= 1.15
    elif momentum_score < 35:
        kelly_f *= 0.7

    # 基本面强时加仓
    fundamental_score = scores.get('fundamental', 50)
    if fundamental_score > 70:
        kelly_f *= 1.1

    # 限制范围
    kelly_f = max(0, min(MAX_POSITION, kelly_f))

    return round(kelly_f, 3)


# ============================================================
# 因子计算函数 (与R1相同)
# ============================================================

def calc_technical(close):
    score = 50
    if len(close) < MA_LONG:
        return score
    latest = close[-1]
    ma_short = np.mean(close[-MA_SHORT:])
    ma_long = np.mean(close[-MA_LONG:])
    if ma_short > 0:
        dev_short = (latest - ma_short) / ma_short * 100
        score += np.clip(dev_short * 2, -15, 15)
    if ma_long > 0:
        dev_long = (latest - ma_long) / ma_long * 100
        score += np.clip(dev_long * 1.5, -10, 10)
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    if len(ema12) > 0 and len(ema26) > 0:
        dif = ema12[-1] - ema26[-1]
        if latest > 0:
            dif_norm = dif / latest * 100
            score += np.clip(dif_norm * 3, -15, 15)
    if len(close) >= MA_LONG + 1:
        ma_s_prev = np.mean(close[-MA_SHORT-1:-1])
        ma_l_prev = np.mean(close[-MA_LONG-1:-1])
        if ma_s_prev <= ma_l_prev and ma_short > ma_long:
            score += 10
        elif ma_s_prev >= ma_l_prev and ma_short < ma_long:
            score -= 10
    return np.clip(score, 0, 100)


def calc_ema(data, period):
    result = []
    alpha = 2.0 / (period + 1)
    ema = data[0]
    result.append(ema)
    for i in range(1, len(data)):
        ema = alpha * data[i] + (1 - alpha) * ema
        result.append(ema)
    return np.array(result)


def calc_fund_flow(close, volume):
    score = 50
    if len(close) < 6 or len(volume) < 6:
        return score
    price_change = (close[-1] - close[-6]) / close[-6] * 100
    vol_ratio = np.mean(volume[-5:]) / max(np.mean(volume[-20:]), 1)
    if price_change > 0 and vol_ratio > 1.2:
        score += 15
    elif price_change > 0 and vol_ratio > 0.8:
        score += 8
    elif price_change < 0 and vol_ratio < 0.8:
        score += 5
    elif price_change < 0 and vol_ratio > 1.5:
        score -= 15
    obv = 0
    for i in range(1, min(20, len(close))):
        if close[-i] > close[-i-1]:
            obv += volume[-i]
        elif close[-i] < close[-i-1]:
            obv -= volume[-i]
    if obv > 0:
        score += max(0, min(10, obv / max(np.mean(volume), 1) / 5))
    else:
        score -= max(0, min(10, -obv / max(np.mean(volume), 1) / 5))
    return np.clip(score, 0, 100)


def calc_fundamental(close):
    score = 50
    if len(close) < 252:
        return score
    latest = close[-1]
    year_high = np.max(close[-252:])
    year_low = np.min(close[-252:])
    if year_high > year_low:
        pct_rank = (latest - year_low) / (year_high - year_low) * 100
        if pct_rank < 20:
            score += 20
        elif pct_rank < 40:
            score += 10
        elif pct_rank > 80:
            score -= 15
        elif pct_rank > 60:
            score -= 5
    if len(close) >= 120:
        ma_120 = np.mean(close[-120:])
        if ma_120 > 0:
            dev = (latest - ma_120) / ma_120 * 100
            score += np.clip(-dev * 1.5, -15, 15)
    return np.clip(score, 0, 100)


def calc_volume(volume):
    score = 50
    if len(volume) < VOL_PERIOD:
        return score
    avg_vol = np.mean(volume[-VOL_PERIOD:])
    if avg_vol <= 0:
        return score
    recent_vol = np.mean(volume[-5:])
    vol_ratio = recent_vol / avg_vol
    if 0.8 <= vol_ratio <= 1.5:
        score += 10
    elif 0.5 <= vol_ratio < 0.8:
        score += 5
    elif vol_ratio > 2.5:
        score -= 10
    elif vol_ratio < 0.3:
        score -= 8
    vol_trend = np.mean(volume[-10:]) / max(np.mean(volume[-30:]), 1)
    if 0.9 <= vol_trend <= 1.3:
        score += 8
    return np.clip(score, 0, 100)


def calc_position(close, high, low):
    score = 50
    if len(close) < 60:
        return score
    latest = close[-1]
    h60 = np.max(high[-60:])
    l60 = np.min(low[-60:])
    if h60 > l60:
        range_pct = (h60 - l60) / l60 * 100
        if range_pct < 20:
            score += 15
        elif range_pct < 30:
            score += 8
        elif range_pct < 40:
            score += 0
        elif range_pct > 60:
            score -= 10
        else:
            score -= 5
        pos_in_range = (latest - l60) / (h60 - l60)
        if pos_in_range < 0.3:
            score += 10
        elif pos_in_range > 0.7:
            score -= 5
    if len(close) >= 40:
        amp_20 = (np.max(high[-20:]) - np.min(low[-20:])) / np.mean(close[-20:]) * 100
        amp_40 = (np.max(high[-40:]) - np.min(low[-40:])) / np.mean(close[-40:]) * 100
        if amp_20 < amp_40 * 0.7:
            score += 8
    return np.clip(score, 0, 100)


def calc_momentum(close, high, low):
    score = 50
    if len(close) < MOMENTUM_PERIOD + RSI_PERIOD:
        return score
    rsi = calc_rsi(close, RSI_PERIOD)
    if rsi < 30:
        score += 15
    elif rsi < 40:
        score += 8
    elif rsi > 70:
        score -= 15
    elif rsi > 60:
        score -= 5
    if len(close) >= MOMENTUM_PERIOD + 1:
        mom = (close[-1] - close[-MOMENTUM_PERIOD-1]) / close[-MOMENTUM_PERIOD-1] * 100
        score += np.clip(mom * 1.5, -12, 12)
    if len(close) >= 10:
        short_trend = (close[-1] - close[-10]) / close[-10] * 100
        score += np.clip(short_trend * 2, -10, 10)
    return np.clip(score, 0, 100)


def calc_rsi(close, period=14):
    if len(close) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(-period, 0):
        diff = close[i] - close[i-1]
        if diff > 0:
            gains.append(diff); losses.append(0)
        else:
            gains.append(0); losses.append(-diff)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100
    return 100 - (100 / (1 + avg_gain / avg_loss))


# ============================================================
# 独立回测模拟
# ============================================================

class MockContext:
    def __init__(self):
        self.symbol = SYMBOL
        self.position_pct = 0.0
        self.last_signal = 'HOLD'
        self.trade_count = 0
        self.win_count = 0
        self.trades = []
        self.equity_curve = []
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.consecutive_signals = 0
        self.now = None
        self.portfolio = MockPortfolio()
        self._subscribed = []

    def subscribe(self, symbol, freq):
        self._subscribed.append((symbol, freq))


class MockPortfolio:
    def __init__(self):
        self.unit_net_value = 1.0
        self.cash = 1_000_000
        self.positions = {}


_mock_data = None
_mock_idx = 0
_mock_context = None


def history(symbol, fields, count, freq):
    global _mock_data, _mock_idx
    if _mock_data is None:
        return []
    end = min(_mock_idx + 1, len(_mock_data))
    start = max(0, end - count)
    if fields == 'close':
        return [b['close'] for b in _mock_data[start:end]]
    elif fields == 'volume':
        return [b['volume'] for b in _mock_data[start:end]]
    elif fields == 'high':
        return [b['high'] for b in _mock_data[start:end]]
    elif fields == 'low':
        return [b['low'] for b in _mock_data[start:end]]
    return []


def order_target_percent(symbol, target_pct):
    global _mock_context
    ctx = _mock_context
    prev_pct = ctx.position_pct
    ctx.position_pct = target_pct
    if target_pct > 0 and prev_pct == 0:
        ctx.entry_price = _mock_data[_mock_idx]['close']
        ctx.highest_since_entry = ctx.entry_price
    elif target_pct == 0 and prev_pct > 0:
        exit_price = _mock_data[_mock_idx]['close']
        pnl = (exit_price - ctx.entry_price) / ctx.entry_price
        ctx.trades.append({'entry': ctx.entry_price, 'exit': exit_price, 'pnl': pnl})
        if pnl > 0:
            ctx.win_count += 1
        ctx.entry_price = 0
        ctx.highest_since_entry = 0


def run_backtest(data_df):
    global _mock_data, _mock_idx, _mock_context

    if isinstance(data_df, pd.DataFrame):
        bars = data_df.to_dict('records')
    else:
        bars = list(data_df)

    _mock_data = bars
    _mock_context = MockContext()

    init(_mock_context)

    results = []
    start_idx = MA_LONG + MOMENTUM_PERIOD + 10
    for i in range(start_idx, len(bars)):
        _mock_idx = i
        bar = bars[i]
        _mock_context.now = bar.get('date', str(i))
        _mock_context.portfolio.unit_net_value = bar.get('close', 0) / bars[0].get('close', 1)

        handle_bar(_mock_context, bar)

        if len(_mock_context.equity_curve) > 0:
            results.append(_mock_context.equity_curve[-1])

    return results, _mock_context


if __name__ == '__main__':
    print('=' * 65)
    print('  拓普集团(601689.SH) Round 2 — 权重优化+Kelly仓位+止损')
    print('  兼容 SuperMind 研究环境 (init/handle_bar/history/order_target_percent)')
    print('=' * 65)
    print('  因子权重: 基本面/动量各25%, 技术面/量能各15%, 资金面/筹码各10%')
    print('  仓位管理: Kelly公式动态 (最大70%)')
    print('  止损: -8%硬止损')
    print('  信号阈值: 买入>62, 卖出<38')
    print('=' * 65)
