"""量化上交所侧 mtss 缺 融券余额 带来的误差, 并验证用 融券余量 x 收盘价 估算的效果

注: 本地K线为前复权, 但qfq以最新日为锚, 故【近期】qfq收盘价 == 实际收盘价,
    可直接用于估算融券余额。
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FF = ROOT / "data/raw/fund_flow_full/fundflow_history.parquet"
KL = ROOT / "data/raw/kline"

D = pd.Timestamp("2026-06-30")
DS = D.strftime("%Y%m%d")

loc = pd.read_parquet(FF)
loc["date"] = pd.to_datetime(loc["date"])
cur = loc[(loc["date"] == D) & loc["mtss_balance"].notna()][["code", "mtss_balance"]].copy()
cur["code"] = cur["code"].astype(str).str.zfill(6)
cur["mkt"] = cur["code"].str[0].map(lambda c: "SSE" if c == "6" else "SZSE")
print(f"本地 {D.date()} 有 mtss 的股票: {len(cur)} 只")
print(cur["mkt"].value_counts().to_string())

sse = ak.stock_margin_detail_sse(date=DS)
sse = sse.rename(columns={"标的证券代码": "code", "融资余额": "rz", "融券余量": "rq_vol"})
sse["code"] = sse["code"].astype(str).str.zfill(6)
for c in ("rz", "rq_vol"):
    sse[c] = pd.to_numeric(sse[c], errors="coerce")

# 用当日收盘价估算融券余额
px = []
for code in cur[cur["mkt"] == "SSE"]["code"]:
    p = KL / f"{code}.parquet"
    if not p.exists():
        continue
    k = pd.read_parquet(p, columns=["date", "close"])
    k["date"] = pd.to_datetime(k["date"])
    row = k[k["date"] == D]
    if len(row):
        px.append((code, float(row["close"].iloc[0])))
px = pd.DataFrame(px, columns=["code", "close"])
print(f"取到收盘价: {len(px)} 只")

m = cur[cur["mkt"] == "SSE"].merge(sse[["code", "rz", "rq_vol"]], on="code") \
                            .merge(px, on="code", how="left")
m["rq_est"] = m["rq_vol"].fillna(0) * m["close"].fillna(0)
m["est_a"] = m["rz"]                      # 仅融资余额
m["est_b"] = m["rz"] + m["rq_est"]        # 融资 + 估算融券
print(f"\n上交所侧重叠 {len(m)} 只")
for col, lbl in [("est_a", "仅融资余额        "), ("est_b", "融资+融券量x收盘价")]:
    dev = (m[col] / m["mtss_balance"] - 1).abs()
    print(f"  {lbl}  中位偏差 {dev.median()*100:7.4f}%  均值 {dev.mean()*100:7.4f}%  "
          f"<0.5%占比 {100*(dev<0.005).mean():5.1f}%  <2%占比 {100*(dev<0.02).mean():5.1f}%")

print("\n  样例对照 (前6只):")
print(m[["code", "mtss_balance", "est_a", "est_b"]].head(6).to_string(index=False))

# 深交所侧确认
szse = ak.stock_margin_detail_szse(date=DS)
cmap = {c: ("code" if "证券代码" in str(c) else
            "rz_rq" if str(c) == "融资融券余额" else str(c)) for c in szse.columns}
z = szse.rename(columns=cmap)
z["code"] = z["code"].astype(str).str.zfill(6)
z["rz_rq"] = pd.to_numeric(z["rz_rq"], errors="coerce")
mz = cur[cur["mkt"] == "SZSE"].merge(z[["code", "rz_rq"]], on="code")
dev = (mz["rz_rq"] / mz["mtss_balance"] - 1).abs()
print(f"\n深交所侧重叠 {len(mz)} 只 | 直接用'融资融券余额' 中位偏差 {dev.median()*100:.6f}%  "
      f"<0.01%占比 {100*(dev<0.0001).mean():.1f}%")
