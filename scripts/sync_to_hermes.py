"""把补全+重建后的数据同步到 Hermes 实时副本(Web 训练源).

同步内容:
1. data/processed/training_data_v22.parquet, training_data_v23.parquet
   -> Hermes data/processed/ (Web 的 latest_training_data 读版本最大者)
2. universe 216 只的 raw/kline/{code}.parquet
   -> Hermes data/raw/kline/ (Hermes feature_engine 未来重跑只需这 216 只)

不碰 Hermes 的 Web/data/web.db (用户实时库). 不同步其余 5300+ 只(与本策略无关).
"""
import os, json, shutil

SRC = "D:/myAI/WorkBuddy-workspace/quant-strategy"
DST = "D:/myAI/Hermes-Workspace/quant-strategy"

wl = json.load(open(os.path.join(SRC, "data/universe/watchlist_216.json")))
items = wl.get("watchlist", wl) if isinstance(wl, dict) else wl
codes = [str(x["code"]).split(".")[0] for x in items]

# 1) parquet
for n in ["v22", "v23"]:
    s = os.path.join(SRC, "data/processed", f"training_data_{n}.parquet")
    d = os.path.join(DST, "data/processed", f"training_data_{n}.parquet")
    if os.path.exists(s):
        shutil.copy2(s, d)
        print(f"SYNC parquet {n} -> {d}")

# 2) universe kline
src_kl = os.path.join(SRC, "data/raw/kline")
dst_kl = os.path.join(DST, "data/raw/kline")
os.makedirs(dst_kl, exist_ok=True)
cnt = 0
for c in codes:
    s = os.path.join(src_kl, c + ".parquet")
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(dst_kl, c + ".parquet"))
        cnt += 1
print(f"SYNC kline universe files = {cnt}/216")
print("DONE")
