"""对比 origin/sswu 与当前的 fundflow_history.parquet, 判断是否有可合并的增量"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REL = "data/raw/fund_flow_full/fundflow_history.parquet"
TMP = Path("/tmp/sswu_fundflow.parquet")

blob = subprocess.run(["git", "show", f"origin/sswu:{REL}"], cwd=ROOT,
                      capture_output=True)
if blob.returncode != 0:
    print("提取失败:", blob.stderr.decode()[:200])
    sys.exit(1)
TMP.write_bytes(blob.stdout)
print(f"已提取 origin/sswu 版本: {len(blob.stdout):,} 字节")

head = "".join(TMP.read_bytes()[:4].decode("latin1"))
if head != "PAR1":
    print(f"[!] 不是 parquet 文件 (可能是 Git LFS 指针): {blob.stdout[:120]!r}")
    sys.exit(0)


def info(label, df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"\n[{label}]")
    print(f"  {len(df):,} 行 | {df['code'].nunique()} 只 | "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    for c in df.columns:
        if c in ("date", "code"):
            continue
        nn = df[c].notna().sum()
        print(f"    {c:18s} 非空 {nn:>7,} ({100*nn/len(df):5.1f}%)")
    return df


cur = info("当前 (HEAD/工作区)", pd.read_parquet(ROOT / REL))
oth = info("origin/sswu", pd.read_parquet(TMP))

print("\n=== 差异分析 ===")
ck = ["date", "code"]
cur_k = set(map(tuple, cur[ck].astype(str).values))
oth_k = set(map(tuple, oth[ck].astype(str).values))
print(f"  当前独有 (date,code): {len(cur_k - oth_k):,}")
print(f"  sswu 独有 (date,code): {len(oth_k - cur_k):,}")
print(f"  共有: {len(cur_k & oth_k):,}")

only = oth_k - cur_k
if only:
    od = pd.DataFrame(sorted(only), columns=["date", "code"])
    od["date"] = pd.to_datetime(od["date"])
    print(f"  sswu 独有部分日期范围: {od['date'].min().date()} ~ {od['date'].max().date()}")
    print(f"  按年份分布:\n{od['date'].dt.year.value_counts().sort_index().to_string()}")

# 非空数据量对比 (哪边信息更多)
print("\n=== 各字段非空数量对比 ===")
for c in cur.columns:
    if c in ck or c not in oth.columns:
        continue
    a, b = int(cur[c].notna().sum()), int(oth[c].notna().sum())
    flag = "  <- sswu 更多" if b > a else ""
    print(f"  {c:18s} 当前 {a:>7,} | sswu {b:>7,}{flag}")
