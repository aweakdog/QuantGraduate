"""从 Hermes 原始副本恢复 universe 216 只的日K历史, 与 WorkBuddy 近期回填合并.

背景:
- 早期 iFinD/thsdk 补数把"英文历史 + 中文近期"直接 concat -> 双列文件.
- fix_kline_columns.py 的 drop_duplicates('时间', keep='last') 误删历史行 -> 124 只只剩近期(~10行).
- Hermes raw/kline 是未动的原始副本, 含全部 216 只到 06-26 的历史(英文或中文).

恢复: 每只 = Hermes历史(<=06-26) + WorkBuddy近期(>06-26) 按时间合并.
  Hermes 全为历史, WB 近期全为 06-27~07-09, 日期不重叠 -> 无丢失.
"""
import pandas as pd, os, json, shutil

WB = "D:/myAI/WorkBuddy-workspace/quant-strategy"
HM = "D:/myAI/Hermes-Workspace/quant-strategy"
KL_WB = os.path.join(WB, "data/raw/kline")
KL_HM = os.path.join(HM, "data/raw/kline")
WL = os.path.join(WB, "data/universe/watchlist_216.json")

CHI = ["时间", "收盘价", "开盘价", "最高价", "最低价", "成交量", "总金额"]
ENG2CHI = {"date": "时间", "close": "收盘价", "open": "开盘价",
           "high": "最高价", "low": "最低价", "volume": "成交量", "amount": "总金额"}

wl = json.load(open(WL))
items = wl.get("watchlist", wl) if isinstance(wl, dict) else wl
codes = [str(x["code"]).split(".")[0] for x in items]

def to_chi(d):
    if "时间" in d.columns:
        return d[CHI].copy()
    if "date" in d.columns:
        return d.rename(columns=ENG2CHI)[CHI].copy()
    return d[CHI].copy()

CUT = pd.Timestamp("2026-06-26")
recovered = 0
for c in codes:
    pw = os.path.join(KL_WB, c + ".parquet")
    ph = os.path.join(KL_HM, c + ".parquet")
    if not os.path.exists(pw) or not os.path.exists(ph):
        print("SKIP missing", c); continue
    # Hermes 历史(到06-26) -> 中文
    h = to_chi(pd.read_parquet(ph))
    h["时间"] = pd.to_datetime(h["时间"], errors="coerce")
    h = h[h["时间"] <= CUT]
    # WorkBuddy 近期(>06-26) -> 中文
    w = to_chi(pd.read_parquet(pw))
    w["时间"] = pd.to_datetime(w["时间"], errors="coerce")
    w = w[w["时间"] > CUT]
    comb = pd.concat([h, w]).dropna(subset=["时间"]) \
              .drop_duplicates("时间", keep="last") \
              .sort_values("时间").reset_index(drop=True)
    comb = comb[CHI]
    comb.to_parquet(pw, index=False)
    recovered += 1

print(f"RECOVERED={recovered}/216")
# 校验
trunc = 0
for c in codes:
    d = pd.read_parquet(os.path.join(KL_WB, c + ".parquet"))
    if len(d) < 100:
        trunc += 1
        print("STILL TRUNC", c, len(d))
print("VALIDATION truncated(<100)=", trunc)
