"""验证旧回测的超额收益是否来自"有基本面数据的那一小撮股票"

假设: 旧训练集里 revenue 只有 47/519 只有值, 模型把"revenue_ma5 非空"
当成了股票身份标记, 于是持仓高度集中在这 47 只上。
若成立 -> 旧回测成交里这 47 只的占比应远高于随机水平(约9%)。
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

OLD_TRAIN = PROC / "backup" / sys.argv[1]
OLD_BT = PROC / "wf_daily_em_t1close_s001_ts2022-09-01_te2026-07-27_cap20000.json"
NEW_BT = PROC / "wf_daily_em_t1close_s001_fundfix_ts2022-09-01_te2026-07-27_cap20000.json"

# ── 旧训练集里哪些股票有 revenue ──
old = pd.read_parquet(OLD_TRAIN, columns=["code", "revenue"])
has_rev = set(old.loc[old["revenue"].notna(), "code"].astype(str).str[:6].unique())
all_codes = set(old["code"].astype(str).str[:6].unique())
base = len(has_rev) / len(all_codes)
print(f"旧训练集: 共 {len(all_codes)} 只, 其中有 revenue 的 {len(has_rev)} 只 "
      f"= {base:.1%}  <- 随机基准线")

def buys(path):
    d = json.load(open(path, encoding="utf-8"))
    t = pd.DataFrame(d["trades"])
    if "action" in t.columns:
        t = t[t["action"].astype(str).str.upper().str.contains("BUY")]
    t["c6"] = t["code"].astype(str).str.zfill(6).str[:6]
    return t, d

print()
for label, path in [("旧回测(坏数据)", OLD_BT), ("新回测(修复后)", NEW_BT)]:
    t, d = buys(path)
    if t.empty:
        print(f"{label}: 无买入记录")
        continue
    inset = t["c6"].isin(has_rev)
    uniq = t["c6"].nunique()
    uniq_in = t.loc[inset, "c6"].nunique()
    amt = t["gross"].abs() if "gross" in t.columns else None
    print(f"{label}:")
    print(f"  买入 {len(t)} 笔, 涉及 {uniq} 只股票")
    print(f"  其中落在'有revenue的47只'里: {inset.sum()}/{len(t)} 笔 = {inset.mean():.1%}"
          f"   (随机应约 {base:.1%})")
    print(f"  去重后: {uniq_in}/{uniq} 只 = {uniq_in/uniq:.1%}")
    if amt is not None:
        print(f"  按买入金额: {amt[inset].sum()/amt.sum():.1%} 的钱投在这47只上")
    print(f"  倍数 = {inset.mean()/base:.1f}x 随机水平")
    print(f"  总收益 {d['summary']['total_return_pct']}%  IR {d['summary']['information_ratio']}")
    print()

# ── 旧回测最赚钱的股票是不是都在这47只里 ──
t, d = buys(OLD_BT)
tr = pd.DataFrame(d["trades"])
tr["c6"] = tr["code"].astype(str).str.zfill(6).str[:6]
print("旧回测买入次数最多的 12 只:")
top = tr[tr["action"].astype(str).str.upper().str.contains("BUY")]["c6"].value_counts().head(12)
for c, n in top.items():
    print(f"  {c}  买入{n:>3}次   有revenue数据: {'是 <<<' if c in has_rev else '否'}")
