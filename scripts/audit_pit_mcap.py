"""验证 kline 里的 outstanding_share 是否为逐日真实历史值(而非当前快照复制)

若为真实历史, 则 流通市值 = close_不复权 x outstanding_share 可作 PIT 规模指标。
注意: 本地 close 是前复权, 需确认是否另有不复权价, 否则市值口径会失真。
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KL = ROOT / "data/raw/kline"

samples = ["000063", "600519", "300750", "688981", "002594", "000001"]
print("=== outstanding_share 是否随时间变化 ===")
for c in samples:
    p = KL / f"{c}.parquet"
    if not p.exists():
        continue
    k = pd.read_parquet(p)
    k["date"] = pd.to_datetime(k["date"])
    os_ = k["outstanding_share"]
    nuniq = os_.nunique()
    print(f"  {c}: {len(k):>5}行 | outstanding_share 取值数 {nuniq:>4} | "
          f"首 {os_.iloc[0]:,.0f} -> 末 {os_.iloc[-1]:,.0f} | "
          f"{'✓随时间变化' if nuniq > 3 else '✗疑似常量快照'}")

print("\n=== 全样本抽查 300 只: outstanding_share 唯一值数分布 ===")
import random
random.seed(0)
files = sorted(KL.glob("*.parquet"))
pick = random.sample(files, min(300, len(files)))
nu = []
for p in pick:
    try:
        k = pd.read_parquet(p, columns=["outstanding_share"])
    except Exception:
        continue
    if len(k):
        nu.append(k["outstanding_share"].nunique())
s = pd.Series(nu)
print(f"  常量(=1个值): {(s<=1).sum()} 只 ({100*(s<=1).mean():.1f}%)")
print(f"  2-3个值     : {((s>1)&(s<=3)).sum()} 只")
print(f"  >3个值(真历史): {(s>3).sum()} 只 ({100*(s>3).mean():.1f}%)")
print(f"  中位唯一值数 : {s.median():.0f}")

print("\n=== close 是否为前复权 (与 amount/volume 推算的均价对比) ===")
for c in samples[:3]:
    p = KL / f"{c}.parquet"
    if not p.exists():
        continue
    k = pd.read_parquet(p)
    k["date"] = pd.to_datetime(k["date"])
    k["vwap"] = k["amount"] / k["volume"]
    k["ratio"] = k["close"] / k["vwap"]
    early, late = k.head(60)["ratio"].median(), k.tail(60)["ratio"].median()
    print(f"  {c}: close/vwap 早期 {early:.4f}  近期 {late:.4f}  "
          f"{'✗前复权(早期偏离1)' if abs(early-1) > 0.05 else '✓接近不复权'}")

print("\n  说明: 若 close 为前复权, 则 close x outstanding_share 不等于真实历史市值;")
print("        但 amount(成交额) 是原始值, 可安全用作 PIT 流动性/规模代理。")

print("\n=== PIT 日均成交额 示例 (2024-01-02 前60日) ===")
T = pd.Timestamp("2024-01-02")
rows = []
for p in pick[:150]:
    try:
        k = pd.read_parquet(p, columns=["date", "amount"])
    except Exception:
        continue
    k["date"] = pd.to_datetime(k["date"])
    h = k[k["date"] < T].tail(60)
    if len(h) >= 40:
        rows.append((p.stem, h["amount"].mean(), len(h)))
d = pd.DataFrame(rows, columns=["code", "adv60", "n"]).sort_values("adv60", ascending=False)
print(f"  可算 {len(d)} 只 | 前5:")
print(d.head(5).to_string(index=False))
print(f"  中位日均成交额 {d['adv60'].median():,.0f} 元")
