"""验证空仓择时是否有未来暴露 (前视泄漏)

做三件事:
  1. 逐日重算: 对若干抽样日 T, 只用 <=T 的行情从头重算 regime 状态,
     与"用全历史一次算完"的结果逐一比对。若完全一致 => 无未来信息参与。
  2. 广度成分核查: 每个日期参与广度计算的股票数, 确认只统计当日有 bar 的股票
     (退市股在退市后自动退出, 新股上市后自动进入)。
  3. 打印切换时点, 人工确认状态在信号日就已确定, 执行发生在 T+1。
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KLINE_DIR = ROOT / "data/raw/kline"
COL_MAP = {"时间": "date", "收盘价": "close", "开盘价": "open"}
MA, THRESH, CONFIRM = 20, 0.40, 2


def market_panel():
    rows = []
    for p in sorted(KLINE_DIR.glob("*.parquet")):
        try:
            kl = pd.read_parquet(p).rename(columns=COL_MAP)[["date", "close"]]
        except Exception:
            continue
        kl["date"] = pd.to_datetime(kl["date"])
        kl = kl.sort_values("date")
        kl["above_ma"] = (kl["close"] > kl["close"].rolling(MA).mean()).astype(float)
        kl["code"] = p.stem
        rows.append(kl[["date", "code", "close", "above_ma"]])
    return pd.concat(rows, ignore_index=True)


def agg(panel):
    return panel.groupby("date").agg(mkt_close=("close", "mean"),
                                     breadth=("above_ma", "mean"),
                                     n=("code", "size")).sort_index()


def regime(m):
    """与 wf_v35_breadth_alpha.build_regime_state 同逻辑 (breadth 判据)"""
    bad = (m["breadth"] < THRESH).fillna(False)
    off = bad.rolling(CONFIRM).min().fillna(0) == 1
    on = (~bad).rolling(CONFIRM).min().fillna(0) == 1
    state, out = False, {}
    for d in m.index:
        if not state and bool(off.loc[d]):
            state = True
        elif state and bool(on.loc[d]):
            state = False
        out[d] = state
    return pd.Series(out)


print("加载全市场K线 ...")
panel = market_panel()
full = agg(panel)
state_full = regime(full)
print(f"  {panel['code'].nunique()} 只股票, {len(full)} 个交易日")

# ── 1. 截断重算比对 ──
test_dates = [d for d in full.index if d >= pd.Timestamp("2023-09-20")]
samples = list(np.array(test_dates)[np.linspace(0, len(test_dates) - 1, 12).astype(int)])
print("\n=== 截断重算比对 (只用 <=T 的数据从头算) ===")
bad = 0
for T in samples:
    T = pd.Timestamp(T)
    m_trunc = agg(panel[panel["date"] <= T])
    s_trunc = regime(m_trunc)
    a, b = bool(s_trunc.loc[T]), bool(state_full.loc[T])
    ok = a == b
    bad += (not ok)
    print(f"  {T:%Y-%m-%d}  截断算={'空仓' if a else '满仓'}  "
          f"全历史算={'空仓' if b else '满仓'}  {'一致' if ok else '★不一致★'}")
print(f"-> {len(samples)-bad}/{len(samples)} 一致" +
      ("  无未来暴露" if bad == 0 else "  存在前视泄漏!"))

# ── 2. 广度成分核查 ──
print("\n=== 广度参与股票数 (应随上市/退市自然变化) ===")
for d in ["2019-01-15", "2021-06-15", "2023-09-20", "2024-09-30", "2026-07-27"]:
    d = pd.Timestamp(d)
    if d in full.index:
        print(f"  {d:%Y-%m-%d}  参与 {int(full.loc[d,'n'])} 只  "
              f"广度 {full.loc[d,'breadth']*100:.1f}%")

# ── 3. 切换时点 ──
s = state_full[state_full.index >= pd.Timestamp("2024-01-01")]
sw = s[s != s.shift()]
print(f"\n=== 2024 年起状态切换 ({len(sw)} 次) ===")
for d, v in list(sw.items())[:14]:
    print(f"  {d:%Y-%m-%d}  ->  {'空仓' if v else '满仓'}  "
          f"(当日广度 {full.loc[d,'breadth']*100:.1f}%)")
