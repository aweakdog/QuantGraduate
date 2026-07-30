"""审计: 实际选中的特征里, 有多少是对截面标签零信息的市场级常数?

前置结论(diag_feature_information.py): 421 个候选里 215 个是市场级常数
(同一天所有股票同值), 对按日期 demean 的标签单独预测能力恒为 0。

本脚本检查各个 feature_selection*.json 里选中的特征, 按信息类型分类计数。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from pipeline.config import settings  # noqa: E402

DATA_DIR = settings.DATA_DIR
PROC = DATA_DIR / "processed"
LABEL_RAW = "fwd_5d_ret"
CUTOFF = "2023-09-19"

print("重算市场级常数清单 ...")
df = pd.read_parquet(PROC / "training_data_pit_v24.parquet")
df["date"] = pd.to_datetime(df["date"])
s = df[df["date"] < pd.Timestamp(CUTOFF)]
num = [c for c in s.columns if pd.api.types.is_numeric_dtype(s[c])]
overall = s[num].std().replace(0, np.nan)
cs = s.groupby("date")[num].std().mean()
ratio = (cs / overall).fillna(0)
MARKET_WIDE = set(ratio[ratio < 0.01].index)
print(f"  市场级常数 {len(MARKET_WIDE)} 个 / 数值列 {len(num)} 个\n")

# 按来源给特征归类
def source_of(c: str) -> str:
    base = c.replace("_ma5", "").replace("_ma20", "")
    if base.startswith("tev_all_"):
        return "全市场事件(市场级)"
    if base.startswith("tev_"):
        return "板块事件"
    if base.startswith("ev_"):
        return "个股事件"
    if base.startswith("con_"):
        return "概念板块"
    if base.startswith(("marg_", "short_", "margin_")):
        return "融资融券"
    if base.startswith(("mf_", "dde_", "mtss_", "fund_flow")):
        return "资金流"
    if base.startswith(("ann_", "has_ann", "days_since")):
        return "公告"
    if base.startswith("leader_") or base.startswith("has_leader"):
        return "供应链"
    if base in ("cn_pmi", "us_ism_pmi"):
        return "宏观PMI(市场级)"
    if base.startswith(("usdind", "usdcnh", "usdjpy")):
        return "汇率(市场级)"
    if base.startswith(("cn2y", "cn5y", "us2y", "us5y")):
        return "国债(市场级)"
    if base.startswith(("sp_futures", "dj_futures", "nq_futures", "sox", "a50")):
        return "全球指数(市场级)"
    if base.startswith(("cn_commodity", "wf6", "lipf6", "eva", "phos", "soda",
                        "cn_gold", "cn_silver", "cn_copper", "cn_aluminum",
                        "cn_zinc", "cn_nickel", "cn_tin")):
        return "商品价格(市场级)"
    if base in ("pe", "pb", "roe", "mcap", "revenue", "profit", "eps", "bps",
                "total_assets", "debt_ratio", "gross_margin", "operate_cf"):
        return "基本面"
    return "技术面"


files = sorted(PROC.glob("feature_selection*.json"))
for f in files:
    d = json.loads(f.read_text(encoding="utf-8"))
    sel = d.get("selected_features") or d.get("features") or []
    if not sel:
        continue
    n_mw = sum(1 for c in sel if c in MARKET_WIDE)
    n_ma = sum(1 for c in sel if c.endswith(("_ma5", "_ma20")))
    print("=" * 78)
    print(f"{f.name}   共 {len(sel)} 个特征")
    print("=" * 78)
    print(f"  市场级常数(对截面标签零信息): {n_mw:>3} 个  ({n_mw/len(sel):>5.1%})")
    print(f"  _ma5/_ma20 派生             : {n_ma:>3} 个  ({n_ma/len(sel):>5.1%})")
    print(f"  真正有截面信息的原始特征     : {len(sel)-n_mw:>3} 个")

    cnt = {}
    for c in sel:
        k = source_of(c)
        cnt[k] = cnt.get(k, 0) + 1
    print("\n  按数据源分布:")
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        mark = " <- 零信息" if "市场级" in k else ""
        print(f"    {k:<22} {v:>3} 个 ({v/len(sel):>5.1%}){mark}")

    if n_mw:
        print(f"\n  被选中的市场级常数特征(前15个):")
        for c in [x for x in sel if x in MARKET_WIDE][:15]:
            print(f"    {c}")
    print()
