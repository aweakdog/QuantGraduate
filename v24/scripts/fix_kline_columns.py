"""归一化 raw/kline 全部 universe 文件为单一中文 7 列。

问题: 早期 iFinD 补数把英文原文件(ex)与中文 new 直接 concat,
导致文件变成 14 列(英文7+中文7), 使 feature_engine.read_kline 的
rename 产生重复列 -> pd.to_datetime(df['date']) 报 'duplicate keys'.

修复: 英文子集 rename 为中文, 与中文子集按 时间 去重合并(英文历史<=06-26,
中文近期06-27~07-09 不重叠), 输出干净 7 中文列。不丢任何历史数据。
"""
import pandas as pd, os, json, glob

ROOT = "D:/myAI/WorkBuddy-workspace/quant-strategy"
KL = os.path.join(ROOT, "data/raw/kline")
WL = os.path.join(ROOT, "data/universe/watchlist_216.json")

wl = json.load(open(WL))
items = wl.get("watchlist", wl) if isinstance(wl, dict) else wl
codes = [str(x["code"]).split(".")[0] for x in items]

ENG = ["date", "close", "open", "high", "low", "volume", "amount"]
CHI = ["时间", "收盘价", "开盘价", "最高价", "最低价", "成交量", "总金额"]
ENG2CHI = dict(zip(ENG, CHI))

fixed = skipped = 0
for c in codes:
    p = os.path.join(KL, c + ".parquet")
    if not os.path.exists(p):
        continue
    d = pd.read_parquet(p)
    cols = set(d.columns)
    has_eng = set(ENG) & cols
    has_chi = set(CHI) & cols
    if not has_eng:
        skipped += 1
        continue  # 已是干净中文
    # 英文子集 -> 中文
    eng = d[ENG].copy()
    eng.columns = CHI
    eng["时间"] = pd.to_datetime(eng["时间"], errors="coerce")
    if has_chi:
        chi = d[CHI].copy()
        chi["时间"] = pd.to_datetime(chi["时间"], errors="coerce")
        comb = pd.concat([eng, chi])
    else:
        comb = eng
    comb = (comb.dropna(subset=["时间"])
                .drop_duplicates("时间", keep="last")
                .sort_values("时间")
                .reset_index(drop=True))
    comb = comb[CHI]
    comb.to_parquet(p, index=False)
    fixed += 1

print(f"FIXED={fixed}  SKIPPED(already clean)={skipped}")

# 校验
bad = 0
for c in codes:
    p = os.path.join(KL, c + ".parquet")
    d = pd.read_parquet(p)
    if list(d.columns) != CHI:
        bad += 1
        print("STILL BAD", c, list(d.columns))
print("VALIDATION bad=", bad)
