"""网格研究的数据: ETF 前复权日线 + 历史空仓开关序列 (与 live_signal 同口径)。在研究根目录跑。

  python scripts/grid_data.py --root /home/yliog/quant-research-20260912-n2
产出 <root>/data/grid/etf_<code>.parquet、regime.parquet。
ETF 用腾讯源前复权 (041 访问东财被断开; 新浪只有不复权)。复权价把分红再投资算进去, 对网格的
百分比档位无影响。空仓开关: 全部 K 线文件每股 close > 自身 20 日均线 计 1, 当日均值 = 广度;
连续 2 天 < 40% 进空仓、连续 2 天 >= 40% 出空仓 (信号日收盘判, 次日收盘执行,
所以组合真正空仓的日子 = 信号状态滞后 2 天, 列 cash_day)。
"""
import argparse
from pathlib import Path

import pandas as pd

ETFS = {"510050": "上证50", "510300": "沪深300", "510500": "中证500", "512100": "中证1000",
        "159915": "创业板", "510880": "红利", "518880": "黄金"}
T0 = {"518880"}                    # 黄金 ETF 可 T+0


def fetch_etf(code, end):
    import akshare as ak
    sym = ("sz" if code.startswith("1") else "sh") + code
    df = ak.stock_zh_a_hist_tx(symbol=sym, start_date="20120101", end_date=end, adjust="qfq")
    df = df.rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["open", "high", "low", "close"]].astype(float).sort_index()


def regime_series(kline_dir, start="2016-06-01"):
    cols = {"时间": "date", "收盘价": "close"}
    series = {}
    for p in sorted(Path(kline_dir).glob("*.parquet")):
        try:
            k = pd.read_parquet(p).rename(columns=cols)
        except Exception:
            continue
        if "date" not in k or "close" not in k:
            continue
        s = pd.Series(k["close"].astype(float).values, index=pd.to_datetime(k["date"])).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        series[p.stem] = s[s.index >= start]
    px = pd.DataFrame(series).sort_index()
    ma = px.rolling(20, min_periods=20).mean()
    breadth = (px > ma).where(px.notna() & ma.notna()).mean(axis=1)
    bad = (breadth < 0.40).fillna(False)
    off = bad.rolling(2).min().fillna(0) == 1
    on = (~bad).rolling(2).min().fillna(0) == 1
    state, out = False, {}
    for d in breadth.index:
        if not state and off[d]:
            state = True
        elif state and on[d]:
            state = False
        out[d] = state
    reg = pd.DataFrame({"breadth": breadth, "state": pd.Series(out)})
    reg["cash_day"] = reg["state"].astype(bool).shift(2, fill_value=False)
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    ap.add_argument("--kline-dir", default=None, help="默认 <root>/data/raw/kline; 研究快照停更时指向生产目录(只读)")
    ap.add_argument("--skip-etf", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "data" / "grid"
    out.mkdir(parents=True, exist_ok=True)
    for code, name in ({} if a.skip_etf else ETFS).items():
        df = fetch_etf(code, a.end)
        df.to_parquet(out / f"etf_{code}.parquet")
        print(f"{code} {name}: {len(df)} 天 {df.index[0].date()} -> {df.index[-1].date()}")
    reg = regime_series(a.kline_dir or root / "data" / "raw" / "kline")
    reg.to_parquet(out / "regime.parquet")
    r = reg[reg.index >= "2017-01-01"]
    print(f"regime: {r.index[0].date()} -> {r.index[-1].date()}, 空仓日占比 {r['cash_day'].mean():.1%}")


if __name__ == "__main__":
    main()
