"""1) 定性 K线异常跳空是否为新股无涨跌幅期  2) 全数据域缺口清单"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KLINE = ROOT / "data" / "raw" / "kline"
TODAY = pd.Timestamp("2026-07-27")   # 最新交易日


def limit_pct(code):
    c = str(code)[:6]
    if c.startswith(("30", "68")):
        return 0.21
    if c.startswith(("8", "4", "9")):
        return 0.31
    return 0.11


# ═══ 1. 跳空定性 ═══
print("=" * 66)
print("【1. K线异常跳空定性】")
aud = pd.read_csv(ROOT / "data" / "processed" / "audit_kline_quality.csv")
sus = aud[aud["jumps"].fillna(0) > 0]["code"].astype(str).str.zfill(6).tolist()
print(f"  待查 {len(sus)} 只 ...")

cats = {"新股无涨跌幅期(上市前5日)": 0, "复牌后首日": 0, "其他(疑似复权断裂)": 0}
others = []
t0 = time.time()
for i, code in enumerate(sus, 1):
    f = KLINE / f"{code}.parquet"
    if not f.exists():
        continue
    d = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    c, v = d["close"].astype(float), d["volume"].astype(float)
    ret = c.pct_change()
    lim = limit_pct(code)
    mask = (ret.abs() > lim) & ~((v.shift(1) == 0) | (v == 0))
    listed_2021plus = d["date"].iloc[0] > pd.Timestamp("2021-01-05")
    for idx in np.where(mask)[0]:
        if listed_2021plus and idx <= 5:
            cats["新股无涨跌幅期(上市前5日)"] += 1
        elif (v.iloc[max(0, idx - 6):idx] == 0).any():
            cats["复牌后首日"] += 1
        else:
            cats["其他(疑似复权断裂)"] += 1
            others.append((code, str(d["date"].iloc[idx].date()), float(ret.iloc[idx])))
    if i % 40 == 0 or i == len(sus):
        print(f"\r  [{i}/{len(sus)}] {100*i/len(sus):5.1f}% | {time.time()-t0:.0f}s",
              end="", flush=True)
print("\n")
tot = sum(cats.values())
for k, n in cats.items():
    print(f"    {k:26s} {n:>4} 条 ({100*n/tot if tot else 0:5.1f}%)")
if others:
    others.sort(key=lambda x: -abs(x[2]))
    print(f"\n  疑似复权断裂 TOP10 (需人工确认):")
    for code, dt, r in others[:10]:
        print(f"    {code}  {dt}  {r*100:+7.1f}%")

# ═══ 2. 全域缺口清单 ═══
print("\n" + "=" * 66)
print("【2. 全数据域新鲜度 / 缺口清单】\n")


def scan(label, path, kind="dir", datecol=None, note=""):
    p = ROOT / path
    if not p.exists():
        return dict(域=label, 状态="缺失", 最新="-", 滞后="-", 规模="-", 备注=note or "路径不存在")
    try:
        if kind == "dir":
            fs = sorted(p.glob("*.parquet"))
            if not fs:
                return dict(域=label, 状态="空", 最新="-", 滞后="-", 规模="0 文件", 备注=note)
            mx, tot = [], 0
            for f in fs:
                try:
                    d = pd.read_parquet(f)
                except Exception:
                    continue
                dc = datecol or next((c for c in d.columns
                                      if "date" in str(c).lower() or "日期" in str(c)), None)
                if dc is None or dc not in d.columns:
                    continue
                tot += len(d)
                m = pd.to_datetime(d[dc], errors="coerce").max()
                if pd.notna(m):
                    mx.append(m)
            if not mx:
                return dict(域=label, 状态="无日期列", 最新="-", 滞后="-",
                            规模=f"{len(fs)} 文件", 备注=note)
            latest = max(mx)
            scale = f"{len(fs)} 文件/{tot:,} 行"
        elif kind == "filedate":
            fs = sorted(p.glob("*.parquet"))
            if not fs:
                return dict(域=label, 状态="空", 最新="-", 滞后="-", 规模="0 文件", 备注=note)
            ds = []
            for f in fs:
                digits = "".join(ch for ch in f.stem if ch.isdigit())
                if len(digits) >= 8:
                    ds.append(pd.to_datetime(digits[:8], format="%Y%m%d", errors="coerce"))
            ds = [x for x in ds if pd.notna(x)]
            if not ds:
                return dict(域=label, 状态="文件名无日期", 最新="-", 滞后="-",
                            规模=f"{len(fs)} 文件", 备注=note)
            latest = max(ds)
            scale = f"{len(fs)} 文件"
        else:  # single file
            d = pd.read_parquet(p)
            dc = datecol or next((c for c in d.columns
                                  if "date" in str(c).lower() or "日期" in str(c)), None)
            if dc is None:
                return dict(域=label, 状态="无日期列", 最新="-", 滞后="-",
                            规模=f"{len(d):,} 行", 备注=note)
            latest = pd.to_datetime(d[dc], errors="coerce").max()
            scale = f"{len(d):,} 行"
    except Exception as e:
        return dict(域=label, 状态=f"错误", 最新="-", 滞后="-", 规模="-",
                    备注=f"{type(e).__name__}")

    lag = (TODAY - latest).days
    st = "新鲜" if lag <= 3 else ("轻微滞后" if lag <= 10 else ("过期" if lag <= 60 else "严重过期"))
    return dict(域=label, 状态=st, 最新=str(latest.date()), 滞后=f"{lag}d",
                规模=scale, 备注=note)


items = [
    ("日K线", "data/raw/kline", "dir", None, "akshare新浪 已更新"),
    ("1分钟K线", "data/raw/kline_1min", "dir", None, "thsdk Windows-only"),
    ("公告", "data/raw/announcements", "dir", None, "东财np-anotice 已更新"),
    ("事件 events_v2", "data/raw/events_ifind/events_v2.parquet", "file", None, "公告关键词重建"),
    ("事件 events_daily", "data/raw/events_daily", "filedate", None, "iFinD"),
    ("资金流", "data/raw/fund_flow", "dir", None, "thsdk Windows-only"),
    ("基本面", "data/raw/fundamentals", "dir", None, ""),
    ("板块/概念", "data/raw/sectors", "dir", None, ""),
    ("融资融券", "data/raw/margin", "dir", None, ""),
    ("龙虎榜", "data/raw/lhb", "dir", None, ""),
    ("训练数据v24", "data/processed/training_data_v24.parquet", "file", None, "待重建"),
]
res = [scan(*x) for x in items]

# 宏观逐文件
mac = ROOT / "data" / "raw" / "macro"
if mac.exists():
    for f in sorted(mac.glob("*.parquet")):
        d = pd.read_parquet(f)
        dc = next((c for c in d.columns if "日期" in str(c) or "date" in str(c).lower()), None)
        if dc is None:
            continue
        latest = pd.to_datetime(d[dc], errors="coerce").max()
        lag = (TODAY - latest).days
        st = "新鲜" if lag <= 3 else ("轻微滞后" if lag <= 10 else ("过期" if lag <= 60 else "严重过期"))
        res.append(dict(域=f"  宏观·{f.stem}", 状态=st, 最新=str(latest.date()),
                        滞后=f"{lag}d", 规模=f"{len(d):,} 行", 备注=""))

out = pd.DataFrame(res)
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)
print(out.to_string(index=False))

print("\n" + "=" * 66)
print("【3. 汇总】")
for st in ["缺失", "严重过期", "过期", "轻微滞后", "新鲜"]:
    sub = out[out["状态"] == st]
    if len(sub):
        print(f"  {st:6s} ({len(sub):>2}): {', '.join(sub['域'].str.strip().tolist())}")
