import pandas as pd, glob, os, json, collections
BASE = r"D:\myAI\WorkBuddy-workspace\quant-strategy\data"
out = []

# 1. raw daily kline: max-date distribution
files = glob.glob(os.path.join(BASE, "raw", "kline", "*.parquet"))
maxdates = collections.Counter()
sample_recent = None
for f in files:
    try:
        d = pd.read_parquet(f, columns=["date"])
        md = pd.to_datetime(d["date"]).max()
        maxdates[str(md.date())] += 1
    except Exception:
        maxdates["ERR"] += 1
out.append(f"raw/kline files: {len(files)}")
for k in sorted(maxdates, reverse=True)[:8]:
    out.append(f"  maxdate {k}: {maxdates[k]} files")

# 2. fundflow_history
ff = pd.read_parquet(os.path.join(BASE, "raw", "fund_flow_full", "fundflow_history.parquet"))
out.append(f"fundflow_history: max={pd.to_datetime(ff['date']).max().date()} codes={ff['code'].nunique()} rows={len(ff)}")

# 3. training parquet v22/v23
for v in ["training_data_v22.parquet", "training_data_v23.parquet"]:
    p = os.path.join(BASE, "processed", v)
    df = pd.read_parquet(p, columns=["date", "code"])
    out.append(f"{v}: max={pd.to_datetime(df['date']).max().date()} codes={df['code'].nunique()} rows={len(df)}")

with open(os.path.join(BASE, "processed", "_audit_out.txt"), "w", encoding="utf-8") as fo:
    fo.write("\n".join(out))
print("\n".join(out))
