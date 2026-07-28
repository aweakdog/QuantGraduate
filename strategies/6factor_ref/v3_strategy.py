"""
================================================================================
 拓普集团(601689.SH) 量化策略 — Round 3: 动态仓位+波动率调整+移动止盈
================================================================================
 基于R2回测结果:
   - 动态仓位: 综合评分驱动 + 波动率因子调整
   - 风险预算: 目标波动率限制
   - 移动止盈: 从最高点回撤N%止盈
   - 参数优化: 网格搜索最优参数组合

 6因子体系: 技术面(18%) | 资金面(12%) | 基本面(22%) | 量能(16%) | 筹码(12%) | 动量(20%)

 SuperMind 兼容: init/handle_bar/history/order_target_percent
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime
import itertools

# ============================================================
# Round 3 优化参数
# ============================================================
SYMBOL = '601689.SH'
SYMBOL_NAME = '拓普集团'

# R3优化权重
FACTOR_WEIGHTS_R3 = {
    'technical':    0.18,
    'fund_flow':    0.12,
    'fundamental':  0.22,
    'volume':       0.16,
    'position':     0.12,
    'momentum':     0.20,
}

# 信号阈值 (R3优化)
BUY_THRESHOLD = 58       # 放宽买入条件
SELL_THRESHOLD = 35      # 卖出更保守

# 动态仓位参数
MAX_POSITION = 0.80
MIN_POSITION = 0.02
BASE_POSITION = 0.50

# 波动率调整参数
TARGET_VOLATILITY = 0.30   # 目标年化波动30%
VOL_LOOKBACK = 60           # 波动率回看窗口

# 移动止盈参数
TRAILING_STOP_ATR_MULT = 3.0   # ATR倍数
ATR_PERIOD = 14
TRAILING_PROFIT_LOCK = 0.05    # 盈利5%后启用移动止盈

# 止损
STOP_LOSS = -0.08

# 因子计算参数
MA_SHORT = 5
MA_LONG = 20
RSI_PERIOD = 14
MOMENTUM_PERIOD = 20


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
    context.highest_since_entry = 0.0
    context.trailing_stop_price = 0.0      # 移动止盈线
    context.trailing_active = False         # 移动止盈是否激活
    context.consecutive_signals = 0
    context.portfolio_volatility = 0.0      # 组合波动率估计

    context.subscribe(SYMBOL, '1d')

    print(f'[R3] 拓普集团 动态仓位+波动率调整+移动止盈 初始化完成')
    print(f'     因子权重: 技术18% 资金12% 基本22% 量能16% 筹码12% 动量20%')
    print(f'     仓位: 动态(评分+波动率), 最大{MAX_POSITION*100:.0f}%')
    print(f'     止盈: 移动止盈(盈利>5%启用, ATR×{TRAILING_STOP_ATR_MULT})')
    print(f'     止损: {STOP_LOSS*100:.0f}%硬止损')
    print(f'     目标波动: {TARGET_VOLATILITY*100:.0f}%年化')


def handle_bar(context, bar):
    """SuperMind 每根K线调用"""
    symbol = context.symbol

    # 获取历史数据
    h_close = history(symbol, 'close', MA_LONG + MOMENTUM_PERIOD + VOL_LOOKBACK + 10, '1d')
    h_volume = history(symbol, 'volume', VOL_LOOKBACK + 5, '1d')
    h_high = history(symbol, 'high', ATR_PERIOD + RSI_PERIOD + 10, '1d')
    h_low = history(symbol, 'low', ATR_PERIOD + RSI_PERIOD + 10, '1d')

    if len(h_close) < MA_LONG:
        return

    close = np.array(h_close)
    volume = np.array(h_volume)
    high = np.array(h_high)
    low = np.array(h_low)

    latest_price = close[-1]

    # ---- 计算波动率 ----
    if len(close) >= VOL_LOOKBACK + 2:
        rets = np.diff(close[-VOL_LOOKBACK-1:]) / close[-VOL_LOOKBACK-1:-1]
        context.portfolio_volatility = np.std(rets) * np.sqrt(252)

    # ---- 计算ATR ----
    atr_val = calc_atr(high, low, close, ATR_PERIOD)

    # ================================================================
    # 止损/止盈检查 (最高优先级)
    # ================================================================
    if context.position_pct > 0 and context.entry_price > 0:
        context.highest_since_entry = max(context.highest_since_entry, latest_price)
        current_pnl = (latest_price - context.entry_price) / context.entry_price

        # 硬止损
        if current_pnl <= STOP_LOSS:
            order_target_percent(symbol, 0.0)
            context.trades.append({
                'entry': context.entry_price, 'exit': latest_price,
                'pnl': current_pnl, 'reason': 'stop_loss'
            })
            context.trade_count += 1
            context.entry_price = 0.0
            context.highest_since_entry = 0.0
            context.trailing_active = False
            context.consecutive_signals = 0
            context.equity_curve.append({
                'date': str(context.now), 'total_score': 0,
                'signal': 'STOP_LOSS', 'nav': context.portfolio.unit_net_value,
                'scores': {}, 'position': 0, 'volatility': context.portfolio_volatility
            })
            return

        # 移动止盈
        if current_pnl >= TRAILING_PROFIT_LOCK:
            if not context.trailing_active:
                context.trailing_stop_price = context.highest_since_entry * (1 - 0.03)
                context.trailing_active = True
            else:
                # 更新移动止盈线: 最高点回撤ATR×倍数
                trail = context.highest_since_entry - atr_val * TRAILING_STOP_ATR_MULT
                context.trailing_stop_price = max(context.trailing_stop_price, trail)

            if latest_price <= context.trailing_stop_price:
                order_target_percent(symbol, 0.0)
                context.trades.append({
                    'entry': context.entry_price, 'exit': latest_price,
                    'pnl': current_pnl, 'reason': 'trailing_stop',
                    'trailing_price': context.trailing_stop_price
                })
                if current_pnl > 0:
                    context.win_count += 1
                context.trade_count += 1
                context.entry_price = 0.0
                context.highest_since_entry = 0.0
                context.trailing_active = False
                context.consecutive_signals = 0
                context.equity_curve.append({
                    'date': str(context.now), 'total_score': 0,
                    'signal': 'TRAILING_STOP', 'nav': context.portfolio.unit_net_value,
                    'scores': {}, 'position': 0, 'volatility': context.portfolio_volatility
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

    # ---- R3综合评分 ----
    total_score = sum(scores[k] * FACTOR_WEIGHTS_R3[k] for k in FACTOR_WEIGHTS_R3)

    # ---- 动态仓位计算 ----
    dynamic_position = calculate_dynamic_position(
        total_score, scores, context.portfolio_volatility, close
    )

    # ---- 交易信号 ----
    if total_score >= BUY_THRESHOLD:
        context.consecutive_signals += 1
        if context.consecutive_signals >= 1:
            order_target_percent(symbol, dynamic_position)
            signal = 'BUY'
        else:
            signal = 'WAIT_BUY'
    elif total_score <= SELL_THRESHOLD:
        context.consecutive_signals += 1
        if context.consecutive_signals >= 1:
            order_target_percent(symbol, 0.0)
            signal = 'SELL'
        else:
            signal = 'WAIT_SELL'
    else:
        context.consecutive_signals = max(0, context.consecutive_signals - 0.5)
        signal = 'HOLD'

    # ---- 记录入场价 ----
    if signal == 'BUY' and context.position_pct > 0 and context.entry_price == 0:
        context.entry_price = latest_price
        context.highest_since_entry = latest_price
        context.trailing_active = False
        context.trade_count += 1

    if signal == 'SELL' and context.position_pct == 0 and context.entry_price > 0:
        exit_price = latest_price
        pnl = (exit_price - context.entry_price) / context.entry_price
        context.trades.append({
            'entry': context.entry_price, 'exit': exit_price,
            'pnl': pnl, 'reason': 'signal_sell'
        })
        if pnl > 0:
            context.win_count += 1
        context.entry_price = 0.0
        context.highest_since_entry = 0.0
        context.trailing_active = False

    # ---- 记录 ----
    context.last_signal = signal
    context.equity_curve.append({
        'date': str(context.now),
        'total_score': total_score,
        'signal': signal,
        'nav': context.portfolio.unit_net_value,
        'scores': scores.copy(),
        'dynamic_position': dynamic_position,
        'actual_position': context.position_pct,
        'volatility': context.portfolio_volatility,
        'atr': atr_val,
        'trailing_active': context.trailing_active,
    })


def calculate_dynamic_position(total_score, scores, volatility, close):
    """动态仓位: 评分驱动 × 波动率调整 × 趋势调整"""

    # 1. 基于综合评分的仓位
    if total_score >= 75:
        score_position = 0.70
    elif total_score >= 65:
        score_position = 0.55
    elif total_score >= 55:
        score_position = 0.35
    elif total_score >= 45:
        score_position = 0.15
    else:
        score_position = 0.0

    # 2. 波动率调整因子
    vol = volatility if volatility > 0 else 0.35
    vol_factor = min(1.5, max(0.3, TARGET_VOLATILITY / vol))

    # 3. 动量趋势调整
    momentum_score = scores.get('momentum', 50)
    if momentum_score >= 70:
        trend_factor = 1.20
    elif momentum_score >= 60:
        trend_factor = 1.10
    elif momentum_score >= 40:
        trend_factor = 0.95
    elif momentum_score >= 30:
        trend_factor = 0.75
    else:
        trend_factor = 0.50

    # 4. 基本面调整
    fundamental_score = scores.get('fundamental', 50)
    if fundamental_score >= 70:
        fundamental_factor = 1.15
    elif fundamental_score >= 55:
        fundamental_factor = 1.0
    else:
        fundamental_factor = 0.80

    # 5. 技术面信号强化
    tech_score = scores.get('technical', 50)
    if tech_score >= 65:
        tech_factor = 1.10
    elif tech_score < 35:
        tech_factor = 0.80
    else:
        tech_factor = 1.0

    # 综合仓位计算
    position = (score_position * vol_factor * trend_factor
                * fundamental_factor * tech_factor)

    # 截断到允许范围
    position = max(MIN_POSITION, min(MAX_POSITION, position))

    return round(position, 3)


def calc_atr(high, low, close, period=14):
    """计算ATR"""
    if len(close) < period + 1:
        return 0
    tr_list = []
    for i in range(1, min(len(close), period + 5)):
        tr = max(high[-i] - low[-i],
                 abs(high[-i] - close[-i-1]),
                 abs(low[-i] - close[-i-1]))
        tr_list.append(tr)
    return np.mean(tr_list)


# ============================================================
# 因子计算函数
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
    # 3年估值分位 (更长期)
    if len(close) >= 756:
        high_3y = np.max(close[-756:])
        low_3y = np.min(close[-756:])
        if high_3y > low_3y:
            pct_3y = (latest - low_3y) / (high_3y - low_3y) * 100
            if pct_3y < 15:
                score += 10
            elif pct_3y > 85:
                score -= 10
    return np.clip(score, 0, 100)


def calc_volume(volume):
    score = 50
    if len(volume) < 60:
        return score
    avg_vol = np.mean(volume[-60:])
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
    # 量能稳定度
    if len(volume) >= 20:
        vol_std = np.std(volume[-20:]) / max(np.mean(volume[-20:]), 1)
        if vol_std < 0.3:
            score += 5
        elif vol_std > 0.8:
            score -= 5
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
    # 趋势强度 ADX简化
    if len(close) >= 30:
        up_moves = [max(0, close[-i] - close[-i-1]) for i in range(1, 15)]
        dn_moves = [max(0, close[-i-1] - close[-i]) for i in range(1, 15)]
        avg_up = np.mean(up_moves)
        avg_dn = np.mean(dn_moves)
        if avg_up + avg_dn > 0:
            adx = abs(avg_up - avg_dn) / (avg_up + avg_dn) * 100
            if adx > 25:
                score += 8  # 强趋势
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
# 参数优化: 网格搜索
# ============================================================

def grid_search_optimize(data_df):
    """网格搜索最优参数组合"""
    param_grid = {
        'BUY_THRESHOLD': [55, 58, 60, 62, 65],
        'SELL_THRESHOLD': [30, 35, 38, 40, 42],
        'MAX_POSITION': [0.6, 0.7, 0.8],
        'TARGET_VOLATILITY': [0.25, 0.30, 0.35],
        'TRAILING_STOP_ATR_MULT': [2.0, 2.5, 3.0, 3.5],
    }

    keys = list(param_grid.keys())
    best_score = -np.inf
    best_params = None
    best_results = None

    combinations = list(itertools.product(*param_grid.values()))
    total = len(combinations)

    print(f'网格搜索: {total} 组参数组合...')

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # 应用参数
        global BUY_THRESHOLD, SELL_THRESHOLD, MAX_POSITION
        global TARGET_VOLATILITY, TRAILING_STOP_ATR_MULT

        BUY_THRESHOLD = params['BUY_THRESHOLD']
        SELL_THRESHOLD = params['SELL_THRESHOLD']
        MAX_POSITION = params['MAX_POSITION']
        TARGET_VOLATILITY = params['TARGET_VOLATILITY']
        TRAILING_STOP_ATR_MULT = params['TRAILING_STOP_ATR_MULT']

        results, ctx = run_backtest(data_df)

        # 计算目标函数: Sharpe × (1 - MaxDD/100)
        if len(results) > 100:
            scores_arr = [r['total_score'] for r in results]
            # 简化绩效评估: 平均得分 + 交易次数克制
            avg_score = np.mean(scores_arr) if scores_arr else 0
            penalty = max(0, ctx.trade_count - 50) * 0.1
            metric = avg_score - penalty

            if metric > best_score:
                best_score = metric
                best_params = params.copy()
                best_results = (results, ctx)

        if (idx + 1) % 50 == 0:
            print(f'  进度: {idx+1}/{total}')

    print(f'\n最优参数: {best_params}')
    print(f'最优得分: {best_score:.1f}')

    # 恢复最优参数
    if best_params:
        BUY_THRESHOLD = best_params['BUY_THRESHOLD']
        SELL_THRESHOLD = best_params['SELL_THRESHOLD']
        MAX_POSITION = best_params['MAX_POSITION']
        TARGET_VOLATILITY = best_params['TARGET_VOLATILITY']
        TRAILING_STOP_ATR_MULT = best_params['TRAILING_STOP_ATR_MULT']

    return best_params, best_results


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
        self.trailing_stop_price = 0.0
        self.trailing_active = False
        self.consecutive_signals = 0
        self.portfolio_volatility = 0.0
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
    start_idx = MA_LONG + MOMENTUM_PERIOD + VOL_LOOKBACK + 10
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
    print('  拓普集团(601689.SH) Round 3 — 动态仓位+波动率+移动止盈')
    print('  兼容 SuperMind 研究环境 + 参数网格优化')
    print('=' * 65)
    print('  因子权重: 技术18% 资金12% 基本22% 量能16% 筹码12% 动量20%')
    print('  仓位管理: 动态(评分驱动×波动率×趋势×基本面×技术)')
    print('  止盈: 移动止盈(盈利>5%启用, ATR追踪)')
    print('  止损: -8%硬止损')
    print('  优化: 网格搜索(5×5×3×3×4=900组)')
    print('=' * 65)
