"""网格交易回测器 (2026-10-22 研究用, 不进生产)。

经典等股数网格 (有回差): 中心价 c, 几何步长 s, 档价 L_j = c(1+s)^j; 记「参考档」b (上次成交所在档)。
  价格跌到 L_{b-1}: 买 1 份, b -= 1;  涨到 L_{b+1}: 卖 1 份, b += 1。
  每份 = unit_value / c 股 (unit_value = 资金*(1-base)/N), 所以每完成一次低买高卖赚一个步长。
  b 限制在 [-N, N]: 跌 N 档约满仓, 涨 N 档约空仓。起始建 base 比例底仓。

只有日线 OHLC, 成交规则写死并偏保守:
  * 日内路径: 开盘 -> 离开盘近的极值 -> 另一个极值 -> 收盘 (常用启发式)。开盘跳空越过的档位按【开盘价】成交,
    盘中穿越的档位按挂单价成交。path="close" 只在收盘价上判断, 给日内假设定悲观下界。
  * T+1: 当天买入的份额当天不能卖 (股票型 ETF); t0=True 用于黄金/债券等 T+0 品种。卖不出的那一格
    参考档不动, 第二天再按当时价格处理。
  * 成本: 佣金 max(费率*金额, 最低), 加每边滑点 (成交价比例)。ETF 无印花税。现金不够就只买得起的部分。
  * 中心价: center="fixed" 固定; reset=True 时收盘价越出 ±(N+1) 档就在收盘价重置中心并回到底仓
    (向上突破后会追回底仓——这是经典做法的代价, reset=False 则越界后不动)。center="ma" 每
    recenter_days 天把中心挪到 MA(ma_window), 并把仓位调到新网格下对应的档位。
  * active: 布尔序列, False 的日子收盘清仓离场; 重新 True 的那天以收盘价为中心建底仓 —— 用于
    「只在空仓期用闲钱做网格」。离场同样受 T+1 约束 (当天买的留到次日收盘)。
闲置资金按 cash_rate 年化计息 (默认 0; 对照货币基金时设 1.5%)。
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-9


@dataclass
class GridParams:
    step: float = 0.015
    levels: int = 10
    base: float = 0.5
    center: str = "fixed"          # fixed | ma
    reset: bool = True
    ma_window: int = 60
    recenter_days: int = 20
    path: str = "ohlc"             # ohlc | close
    t0: bool = False
    fee_rate: float = 1e-4
    min_fee: float = 5.0
    slippage: float = 5e-4
    cash_rate: float = 0.0


def band_of(price, center, p):
    return math.floor(math.log(price / center) / math.log(1 + p.step) + EPS)


def target_of_band(band, p):
    unit = (1 - p.base) / p.levels
    return min(1.0, max(0.0, p.base - band * unit))


class _Book:
    def __init__(self, capital, p):
        self.p, self.cash, self.shares, self.sellable = p, capital, 0.0, 0.0
        self.fees = self.turnover = 0.0
        self.n_trades = 0

    def equity(self, px):
        return self.cash + self.shares * px

    def trade(self, qty, price):
        """qty>0 买 (现金不足按可买量), qty<0 卖 (不超过可卖量)。返回实际成交份额。"""
        p = self.p
        if qty < 0:
            qty = -min(-qty, self.sellable)
            if qty > -EPS:
                return 0.0
            px = price * (1 - p.slippage)
            amt = -qty * px
            fee = max(amt * p.fee_rate, p.min_fee)
            self.cash += amt - fee
            self.sellable += qty
        else:
            px = price * (1 + p.slippage)
            amt = qty * px
            fee = max(amt * p.fee_rate, p.min_fee)
            if amt + fee > self.cash:
                qty = (self.cash - p.min_fee) / (px * (1 + p.fee_rate))
                if qty <= EPS:
                    return 0.0
                amt = qty * px
                fee = max(amt * p.fee_rate, p.min_fee)
            self.cash -= amt + fee
            if p.t0:
                self.sellable += qty
        self.shares += qty
        self.fees += fee
        self.n_trades += 1
        self.turnover += amt
        return qty

    def to_frac(self, frac, px, min_value):
        diff = frac * self.equity(px) / px - self.shares
        if abs(diff) * px >= min_value:
            self.trade(diff, px)


class _Grid:
    """参考档状态机。"""

    def __init__(self, book, center, p):
        self.book, self.p = book, p
        self.set_center(center)
        self.b = 0

    def set_center(self, center):
        self.center = center
        self.unit_shares = self.book.equity(center) * (1 - self.p.base) / self.p.levels / center

    def lvl(self, j):
        return self.center * (1 + self.p.step) ** j

    def walk(self, p_to, gap=False):
        """价格走到 p_to: 按「当前价相对参考档」补齐所有该成交的格 (不看来向 —— 昨天因 T+1 没卖成的格,
        今天价格不动也要补卖)。gap=True 时越过的档位都按 p_to 成交 (开盘跳空/悲观口径)。"""
        p, bk = self.p, self.book
        while self.b > -p.levels and p_to <= self.lvl(self.b - 1) * (1 + EPS):
            if bk.trade(self.unit_shares, p_to if gap else self.lvl(self.b - 1)) <= EPS:
                break
            self.b -= 1
        while self.b < p.levels and p_to >= self.lvl(self.b + 1) * (1 - EPS):
            if bk.trade(-self.unit_shares, p_to if gap else self.lvl(self.b + 1)) >= -EPS:
                break                                      # T+1 卖不出: 参考档不动, 之后再补
            self.b += 1

    def regrid(self, center, px, min_value):
        """换中心: 仓位调到新网格下当前价对应档位的目标, 参考档随之重定。"""
        self.set_center(center)
        self.b = max(-self.p.levels, min(self.p.levels, band_of(px, center, self.p)))
        self.book.to_frac(target_of_band(self.b, self.p), px, min_value)


def _slice(df, start, end):
    d = df
    if start:
        d = d[d.index >= pd.Timestamp(start)]
    if end:
        d = d[d.index <= pd.Timestamp(end)]
    return d


def _active(d, active):
    return (pd.Series(True, index=d.index) if active is None
            else active.reindex(d.index).fillna(False).astype(bool))


def run_grid(df, p: GridParams, capital=100000.0, active=None, start=None, end=None):
    """df: index=日期, 列 open/high/low/close (复权价)。返回 (逐日 DataFrame, 汇总 dict)。"""
    d = _slice(df, start, end)
    ma = df["close"].rolling(p.ma_window, min_periods=p.ma_window).mean().reindex(d.index)
    act = _active(d, active)
    min_value = capital * (1 - p.base) / p.levels * 0.5
    bk = _Book(capital, p)
    grid, last_recenter = None, 0
    daily_yield = (1 + p.cash_rate) ** (1 / 244) - 1
    rows = []
    for i, (dt, r) in enumerate(d.iterrows()):
        bk.cash *= 1 + daily_yield
        bk.sellable = bk.shares                            # 隔夜后全部可卖
        o, h, lo, c = (float(r[k]) for k in ("open", "high", "low", "close"))
        if not act.iloc[i]:
            if bk.shares > EPS:
                bk.trade(-bk.shares, c)
            grid = None
        elif grid is None:
            grid, last_recenter = _Grid(bk, c, p), i      # 进场: 收盘为中心, 收盘建底仓
            bk.to_frac(p.base, c, min_value)
            grid.set_center(c)
        else:
            if p.center == "ma" and i - last_recenter >= p.recenter_days and not np.isnan(ma.iloc[i]):
                grid.regrid(float(ma.iloc[i]), o, min_value)
                last_recenter = i
            if p.path == "close":
                grid.walk(c, gap=True)
            else:
                grid.walk(o, gap=True)                     # 跳空: 越过的档按开盘价
                first, second = (h, lo) if abs(h - o) <= abs(o - lo) else (lo, h)
                for y in (first, second, c):
                    grid.walk(y)
            if p.center == "fixed" and p.reset and abs(band_of(c, grid.center, p)) > p.levels:
                grid.regrid(c, c, min_value)
                grid.b = 0
                last_recenter = i
        eq = bk.equity(c)
        rows.append({"date": dt, "equity": eq, "exposure": bk.shares * c / eq if eq > 0 else 0.0,
                     "active": bool(act.iloc[i])})
    out = pd.DataFrame(rows).set_index("date")
    return out, summarize(out["equity"], capital, {"fees": round(bk.fees, 2), "n_trades": bk.n_trades,
                                                    "turnover_x": round(bk.turnover / capital, 2),
                                                    "avg_exposure": round(float(out["exposure"].mean()), 3)})


def constant_mix(df, frac=1.0, capital=100000.0, rebalance_days=0, active=None, start=None, end=None,
                 p: GridParams = None):
    """frac=1, rebalance_days=0: 买入持有; 0<frac<1 且定期再平衡 = 恒定比例 (网格平均仓位的公平对照)。
    active 同 run_grid: False 的日子收盘清仓。"""
    p = p or GridParams()
    d = _slice(df, start, end)
    act = _active(d, active)
    bk = _Book(capital, GridParams(**{**p.__dict__, "t0": False}))
    daily_yield = (1 + p.cash_rate) ** (1 / 244) - 1
    rows, held = [], 0
    for i, (dt, r) in enumerate(d.iterrows()):
        bk.cash *= 1 + daily_yield
        bk.sellable = bk.shares
        c = float(r["close"])
        if not act.iloc[i]:
            if bk.shares > EPS:
                bk.trade(-bk.shares, c)
            held = 0
        else:
            if held == 0 or (rebalance_days and held % rebalance_days == 0):
                bk.to_frac(frac, c, capital * 0.01)
            held += 1
        eq = bk.equity(c)
        rows.append({"date": dt, "equity": eq, "exposure": bk.shares * c / eq if eq > 0 else 0.0})
    out = pd.DataFrame(rows).set_index("date")
    return out, summarize(out["equity"], capital, {"fees": round(bk.fees, 2), "n_trades": bk.n_trades,
                                                    "avg_exposure": round(float(out["exposure"].mean()), 3)})


def summarize(equity, capital, extra=None):
    eq = equity.astype(float)
    ret = eq.pct_change().fillna(0.0)
    years = max(len(eq) / 244, 1e-9)
    total = eq.iloc[-1] / capital - 1
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    sd = float(ret.std())
    out = {"total_return_pct": round(total * 100, 2), "annual_pct": round(ann * 100, 2),
           "vol_pct": round(sd * np.sqrt(244) * 100, 2),
           "sharpe": round(float(ret.mean() / sd * np.sqrt(244)), 3) if sd > 0 else None,
           "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2), "days": len(eq)}
    out.update(extra or {})
    return out
