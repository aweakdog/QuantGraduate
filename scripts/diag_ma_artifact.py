"""验证: 市场级特征的 _ma5/_ma20 派生, 其截面差异是不是停牌日历造成的假信号?

背景: feature_engine.build_features_for_stock 在【单只股票内部】对几乎所有
数值列做 rolling(5/20).mean()。对市场级序列(道指期货/SOX/A50/汇率...),
原始值当日对所有股票相同, 但每只股票的交易日历不同(停牌日不同), 于是
滚动窗口覆盖的实际日期集合不同 -> _ma20 在股票之间出现差异。

这种差异不含任何个股信息, 它编码的是"这只股票最近停过牌吗"。
若模型把它当重要特征, 等于在学习停牌模式 —— 典型的伪信号。

检验:
  1. 市场级母特征 vs 其 _ma5/_ma20 的截面变异比
  2. 这种截面差异能否被"最近停牌天数"解释
  3. 这些派生特征在特征重要度里排到多靠前
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from pipeline.config import settings  # noqa: E402

DATA_DIR = settings.DATA_DIR
CUTOFF = "2023-09-19"

print("加载数据 ...")
df = pd.read_parquet(DATA_DIR / "processed" / "training_data_pit_v24.parquet")
df["date"] = pd.to_datetime(df["date"])
s = df[df["date"] < pd.Timestamp(CUTOFF)].copy()

num = [c for c in s.columns if pd.api.types.is_numeric_dtype(s[c])]
overall = s[num].std().replace(0, np.nan)
cs = s.groupby("date")[num].std().mean()
ratio = (cs / overall).fillna(0)

# 市场级母特征 = 原始值截面变异比 ~ 0
MKT_BASE = [c for c in num
            if not c.endswith(("_ma5", "_ma20")) and ratio.get(c, 1) < 0.001]

print(f"  市场级母特征 {len(MKT_BASE)} 个 (原始值截面变异比 < 0.001)\n")
print("=" * 86)
print("一、母特征 vs 其 MA 派生 的截面变异比")
print("=" * 86)
print(f"{'特征':<34} {'原始':>11} {'_ma5':>11} {'_ma20':>11} {'MA是否漏过0.01阈值':>18}")
print("-" * 86)

leaked = []
shown = 0
for b in MKT_BASE:
    r0 = ratio.get(b, np.nan)
    r5 = ratio.get(f"{b}_ma5", np.nan)
    r20 = ratio.get(f"{b}_ma20", np.nan)
    passes = [x for x in (f"{b}_ma5", f"{b}_ma20")
              if ratio.get(x, 0) >= 0.01]
    if passes:
        leaked.extend(passes)
    if shown < 20 and (not np.isnan(r5) or not np.isnan(r20)):
        flag = "是 <- 会被选中" if passes else ""
        print(f"{b:<34} {r0:>11.5f} {r5:>11.5f} {r20:>11.5f} {flag:>18}")
        shown += 1

print("-" * 86)
print(f"  市场级特征的 MA 派生中, 有 {len(leaked)} 个截面变异比 >= 0.01,")
print(f"  会被 --drop-market-wide 0.01 漏过而进入候选池。")

# ── 2. 这种截面差异能否被停牌解释 ──
print()
print("=" * 86)
print("二、这些截面差异是停牌造成的吗?")
print("=" * 86)

# 每只股票每天: 距上一个交易日间隔了几个自然日 (>3 表示停过牌或长假)
s2 = s.sort_values(["code", "date"]).copy()
s2["gap"] = s2.groupby("code")["date"].diff().dt.days.fillna(1)
# 近20个交易日内的缺勤程度: 用 gap 累计近似
s2["absent_20d"] = (s2.groupby("code")["gap"]
                    .transform(lambda x: x.rolling(20, min_periods=5).sum()))

probe = [c for c in leaked if c in s2.columns][:6]
if probe:
    print(f"{'派生特征':<36} {'与近20日缺勤的截面相关':>24}")
    print("-" * 86)
    for c in probe:
        rs = []
        for _, g in s2.groupby("date"):
            v = g[[c, "absent_20d"]].dropna()
            if len(v) > 30 and v[c].nunique() > 2:
                r = v[c].rank().corr(v["absent_20d"].rank())
                if not np.isnan(r):
                    rs.append(r)
        if rs:
            a = np.array(rs)
            t = a.mean() / a.std() * np.sqrt(len(a)) if a.std() else np.nan
            print(f"{c:<36} {a.mean():>+16.4f} (t={t:>6.1f})")
    print()
    print("  若相关性显著非零, 证实这些'截面变异'来自交易日历差异而非个股信息。")
else:
    print("  无样本")

# ── 3. 建议的阈值 ──
print()
print("=" * 86)
print("三、结论与建议")
print("=" * 86)
mk_all = set(MKT_BASE) | {f"{b}_ma5" for b in MKT_BASE} | {f"{b}_ma20" for b in MKT_BASE}
mk_all = {c for c in mk_all if c in num}
print(f"  市场级特征全家(含MA派生): {len(mk_all)} 个 / 数值列 {len(num)} 个 "
      f"({len(mk_all)/len(num):.0%})")
print(f"  其中 {len(leaked)} 个能骗过 0.01 的截面变异阈值")
print()
print("  正确做法不是调阈值, 而是【按名单剔除】: 只要母特征是市场级的,")
print("  其所有 MA 派生一律剔除。已把名单写入 market_wide_features.json。")

import json  # noqa: E402
out = DATA_DIR / "processed" / "market_wide_features.json"
json.dump({"cutoff": CUTOFF,
           "base": sorted(MKT_BASE),
           "all_including_ma": sorted(mk_all),
           "leaked_past_001": sorted(leaked)},
          open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"  已保存: {out}")
