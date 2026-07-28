"""
================================================================================
 拓普集团(601689.SH) 量化策略 — Round 1: 6因子等权重基础策略
================================================================================
 SuperMind 回测框架兼容
 因子: 技术面 | 资金面 | 基本面 | 量能 | 筹码 | 动量
 权重: 等权重 各1/6
 仓位: 固定50%
 信号: 综合分>60买入, <40卖出
 回测: 2014-01-01 ~ 至今
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# 策略参数
# ============================================================
SYMBOL = '601689.SH'
SYMBOL_NAME = '拓普集团'
FIXED_POSITION = 0.50           # 固定仓位50%
BUY_THRESHOLD = 60              # 买入阈值
SELL_THRESHOLD = 40             # 卖出阈值
FACTOR_WEIGHTS = {              # 等权重
    'technical':    1/6,
    'fund_flow':    1/6,
    'fundamental':  1/6,
    'volume':       1/6,
    'position':     1/6,
    'momentum':     1/6,
}

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

    # 订阅标的
    context.subscribe(SYMBOL, '1d')

    print(f'[R1] 拓普集团 6因子等权重策略 初始化完成')
    print(f'     回测周期: 2014-01-01 ~ 至今')
    print(f'     因子权重: 6因子等权重(各{1/6*100:.1f}%)')
    print(f'     固定仓位: {FIXED_POSITION*100:.0f}%')
    print(f'     信号阈值: 买入>{BUY_THRESHOLD}, 卖出<{SELL_THRESHOLD}')


def handle_bar(context, bar):
    """SuperMind 每根K线调用"""
    symbol = context.symbol

    # 获取历史数据
    h_close = history(symbol, 'close', MA_LONG + MOMENTUM_PERIOD + 5, '1d')
    h_volume = history(symbol, 'volume', VOL_PERIOD + 5, '1d')
    h_high = history(symbol, 'high', RSI_PERIOD + 5, '1d')
    h_low = history(symbol, 'low', RSI_PERIOD + 5, '1d')

    if len(h_close) < MA_LONG:
        return

    close = np.array(h_close)
    volume = np.array(h_volume)
    high = np.array(h_high)
    low = np.array(h_low)

    # ---- 计算6因子得分 ----
    scores = {}

    # 因子1: 技术面 (均线偏离 + MACD)
    scores['technical'] = calc_technical(close)

    # 因子2: 资金面 (成交量价关系)
    scores['fund_flow'] = calc_fund_flow(close, volume)

    # 因子3: 基本面 (估值分位)
    scores['fundamental'] = calc_fundamental(close)

    # 因子4: 量能 (换手率活跃度)
    scores['volume'] = calc_volume(volume)

    # 因子5: 筹码 (价格分布集中度)
    scores['position'] = calc_position(close, high, low)

    # 因子6: 动量 (RSI + 趋势)
    scores['momentum'] = calc_momentum(close, high, low)

    # ---- 综合评分 ----
    total_score = sum(scores[k] * FACTOR_WEIGHTS[k] for k in FACTOR_WEIGHTS)

    # ---- 交易信号 ----
    if total_score >= BUY_THRESHOLD:
        order_target_percent(symbol, FIXED_POSITION)
        signal = 'BUY'
    elif total_score <= SELL_THRESHOLD:
        order_target_percent(symbol, 0.0)
        signal = 'SELL'
    else:
        signal = 'HOLD'

    # ---- 记录 ----
    context.last_signal = signal
    context.equity_curve.append({
        'date': str(context.now),
        'total_score': total_score,
        'signal': signal,
        'nav': context.portfolio.unit_net_value,
        'scores': scores.copy(),
    })


# ============================================================
# 因子计算函数
# ============================================================

def calc_technical(close):
    """技术面因子: 均线偏离度 + MACD趋势"""
    score = 50
    if len(close) < MA_LONG:
        return score

    latest = close[-1]
    ma_short = np.mean(close[-MA_SHORT:])
    ma_long = np.mean(close[-MA_LONG:])

    # 均线偏离度
    if ma_short > 0:
        dev_short = (latest - ma_short) / ma_short * 100
        score += np.clip(dev_short * 2, -15, 15)
    if ma_long > 0:
        dev_long = (latest - ma_long) / ma_long * 100
        score += np.clip(dev_long * 1.5, -10, 10)

    # MACD简易版
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    if len(ema12) > 0 and len(ema26) > 0:
        dif = ema12[-1] - ema26[-1]
        if latest > 0:
            dif_norm = dif / latest * 100
            score += np.clip(dif_norm * 3, -15, 15)

    # 金叉/死叉
    if len(close) >= MA_LONG + 1:
        ma_s_prev = np.mean(close[-MA_SHORT-1:-1])
        ma_l_prev = np.mean(close[-MA_LONG-1:-1])
        if ma_s_prev <= ma_l_prev and ma_short > ma_long:
            score += 10  # 金叉
        elif ma_s_prev >= ma_l_prev and ma_short < ma_long:
            score -= 10  # 死叉

    return np.clip(score, 0, 100)


def calc_ema(data, period):
    """计算EMA"""
    result = []
    alpha = 2.0 / (period + 1)
    ema = data[0]
    result.append(ema)
    for i in range(1, len(data)):
        ema = alpha * data[i] + (1 - alpha) * ema
        result.append(ema)
    return np.array(result)


def calc_fund_flow(close, volume):
    """资金面因子: 量价配合分析"""
    score = 50

    if len(close) < 6 or len(volume) < 6:
        return score

    # 近5日量价趋势
    price_change = (close[-1] - close[-6]) / close[-6] * 100
    vol_ratio = np.mean(volume[-5:]) / max(np.mean(volume[-20:]), 1)

    # 价升量增 = 资金流入
    if price_change > 0 and vol_ratio > 1.2:
        score += 15
    elif price_change > 0 and vol_ratio > 0.8:
        score += 8
    elif price_change < 0 and vol_ratio < 0.8:
        score += 5  # 缩量下跌可能见底
    elif price_change < 0 and vol_ratio > 1.5:
        score -= 15  # 放量下跌

    # OBV简易版
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
    """基本面因子: 基于价格位置的估值代理"""
    score = 50

    if len(close) < 252:
        return score

    latest = close[-1]

    # 价格在1年内的分位数 (低位=低估)
    year_high = np.max(close[-252:])
    year_low = np.min(close[-252:])
    if year_high > year_low:
        pct_rank = (latest - year_low) / (year_high - year_low) * 100
        if pct_rank < 20:
            score += 20  # 低位低估
        elif pct_rank < 40:
            score += 10
        elif pct_rank > 80:
            score -= 15  # 高位高估
        elif pct_rank > 60:
            score -= 5

    # 半年均线偏离
    if len(close) >= 120:
        ma_120 = np.mean(close[-120:])
        if ma_120 > 0:
            dev = (latest - ma_120) / ma_120 * 100
            score += np.clip(-dev * 1.5, -15, 15)

    return np.clip(score, 0, 100)


def calc_volume(volume):
    """量能因子: 成交量活跃度"""
    score = 50

    if len(volume) < VOL_PERIOD:
        return score

    avg_vol = np.mean(volume[-VOL_PERIOD:])
    if avg_vol <= 0:
        return score

    recent_vol = np.mean(volume[-5:])
    vol_ratio = recent_vol / avg_vol

    # 量能适中为佳
    if 0.8 <= vol_ratio <= 1.5:
        score += 10
    elif 0.5 <= vol_ratio < 0.8:
        score += 5
    elif vol_ratio > 2.5:
        score -= 10  # 异常放量
    elif vol_ratio < 0.3:
        score -= 8   # 极度缩量

    # 成交量趋势
    vol_trend = np.mean(volume[-10:]) / max(np.mean(volume[-30:]), 1)
    if 0.9 <= vol_trend <= 1.3:
        score += 8

    return np.clip(score, 0, 100)


def calc_position(close, high, low):
    """筹码因子: 价格波动范围代理筹码集中度"""
    score = 50

    if len(close) < 60:
        return score

    latest = close[-1]

    # 60日价格波动范围
    h60 = np.max(high[-60:])
    l60 = np.min(low[-60:])
    if h60 > l60:
        range_pct = (h60 - l60) / l60 * 100
        # 波动范围小 = 筹码集中
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

    # 当前价格在区间的位置
    if h60 > l60:
        pos_in_range = (latest - l60) / (h60 - l60)
        # 在区间下沿 = 筹码支撑
        if pos_in_range < 0.3:
            score += 10
        elif pos_in_range > 0.7:
            score -= 5

    # 20日振幅收窄 = 筹码集中信号
    if len(close) >= 40:
        amp_20 = (np.max(high[-20:]) - np.min(low[-20:])) / np.mean(close[-20:]) * 100
        amp_40 = (np.max(high[-40:]) - np.min(low[-40:])) / np.mean(close[-40:]) * 100
        if amp_20 < amp_40 * 0.7:
            score += 8  # 振幅收窄

    return np.clip(score, 0, 100)


def calc_momentum(close, high, low):
    """动量因子: RSI + 趋势强度"""
    score = 50

    if len(close) < MOMENTUM_PERIOD + RSI_PERIOD:
        return score

    # RSI
    rsi = calc_rsi(close, RSI_PERIOD)
    if rsi < 30:
        score += 15  # 超卖
    elif rsi < 40:
        score += 8
    elif rsi > 70:
        score -= 15  # 超买
    elif rsi > 60:
        score -= 5

    # N日动量
    if len(close) >= MOMENTUM_PERIOD + 1:
        mom = (close[-1] - close[-MOMENTUM_PERIOD-1]) / close[-MOMENTUM_PERIOD-1] * 100
        score += np.clip(mom * 1.5, -12, 12)

    # 短期趋势
    if len(close) >= 10:
        short_trend = (close[-1] - close[-10]) / close[-10] * 100
        score += np.clip(short_trend * 2, -10, 10)

    return np.clip(score, 0, 100)


def calc_rsi(close, period=14):
    """计算RSI"""
    if len(close) < period + 1:
        return 50
    gains = []
    losses = []
    for i in range(-period, 0):
        diff = close[i] - close[i-1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-diff)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ============================================================
# 独立回测模拟 (无SuperMind环境时使用)
# ============================================================

class MockContext:
    """模拟SuperMind context"""
    def __init__(self):
        self.symbol = SYMBOL
        self.position_pct = 0.0
        self.last_signal = 'HOLD'
        self.trade_count = 0
        self.win_count = 0
        self.trades = []
        self.equity_curve = []
        self.entry_price = 0.0
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


# 全局变量用于模拟
_mock_data = None
_mock_idx = 0
_mock_context = None


def history(symbol, fields, count, freq):
    """模拟SuperMind history"""
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
    """模拟SuperMind下单"""
    global _mock_context
    ctx = _mock_context
    prev_pct = ctx.position_pct
    ctx.position_pct = target_pct
    if target_pct > 0 and prev_pct == 0:
        ctx.trade_count += 1
        ctx.entry_price = _mock_data[_mock_idx]['close']
    elif target_pct == 0 and prev_pct > 0:
        # 平仓计算盈亏
        exit_price = _mock_data[_mock_idx]['close']
        pnl = (exit_price - ctx.entry_price) / ctx.entry_price
        ctx.trades.append({'entry': ctx.entry_price, 'exit': exit_price, 'pnl': pnl})
        if pnl > 0:
            ctx.win_count += 1
        ctx.entry_price = 0


def run_backtest(data_df, start_date='2014-01-01'):
    """模拟回测"""
    global _mock_data, _mock_idx, _mock_context

    if isinstance(data_df, pd.DataFrame):
        bars = data_df.to_dict('records')
    else:
        bars = list(data_df)

    _mock_data = bars
    _mock_context = MockContext()

    init(_mock_context)

    results = []
    for i in range(MA_LONG + MOMENTUM_PERIOD + 5, len(bars)):
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
    print('  拓普集团(601689.SH) Round 1 — 6因子等权重策略')
    print('  兼容 SuperMind 研究环境 (init/handle_bar/history/order_target_percent)')
    print('=' * 65)
    print('  因子配置: 技术面|资金面|基本面|量能|筹码|动量 (各1/6)')
    print('  仓位管理: 固定50%')
    print('  信号逻辑: 综合分>60买入, <40卖出')
    print('  回测周期: 2014-01-01 ~ 至今')
    print('=' * 65)
