"""可转债历史数据 (CB1): 东方财富 RPTA_WEB_KZZ_LS 每日收盘价 / 转股溢价率, 全部转债 (含已退市、已赎回)。

  python scripts/cb_data.py --out /home/yliog/quant-research-20260912-n2/data/cb
逐只缓存 <out>/raw/<code>.parquet (断点续跑), 最后合成 <out>/cb_panel.parquet (date, code, close, premium)。
akshare 的 bond_zh_cov_value_analysis 因接口多了两列而报错, 这里直接按字段名取。
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://datacenter-web.eastmoney.com/api/data/get"
TOKEN = "894050c76af8597a853f5b408b759f5d"


def parse(rows, code):
    df = pd.DataFrame(rows)
    if df.empty or "DATE" not in df:
        return pd.DataFrame(columns=["date", "code", "close", "premium"])
    return pd.DataFrame({"date": pd.to_datetime(df["DATE"]), "code": code,
                         "close": pd.to_numeric(df.get("FCLOSE"), errors="coerce"),
                         "premium": pd.to_numeric(df.get("SWAPOR"), errors="coerce")})


def fetch(code, session, tries=3):
    params = {"sty": "ALL", "token": TOKEN, "st": "date", "sr": "1", "source": "WEB", "type": "RPTA_WEB_KZZ_LS",
              "filter": f'(zcode="{code}")', "p": "1", "ps": "8000"}
    for i in range(tries):
        try:
            res = session.get(URL, params=params, timeout=30).json()
            return parse((res.get("result") or {}).get("data") or [], code)
        except (requests.RequestException, ValueError):
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{code} 拉取失败")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=0.3)
    a = ap.parse_args(argv)
    import akshare as ak
    out = Path(a.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    codes = sorted(ak.bond_zh_cov()["债券代码"].astype(str).str.zfill(6).unique())
    print(f"转债 {len(codes)} 只", flush=True)
    s = requests.Session()
    for i, c in enumerate(codes):
        p = raw / f"{c}.parquet"
        if p.exists():
            continue
        try:
            fetch(c, s).to_parquet(p)
        except RuntimeError as e:
            print(e, flush=True)
        if i % 100 == 0:
            print(f"  {i}/{len(codes)}", flush=True)
        time.sleep(a.sleep)
    panel = pd.concat([pd.read_parquet(p) for p in sorted(raw.glob("*.parquet"))], ignore_index=True)
    panel = panel.dropna(subset=["close"]).sort_values(["date", "code"])
    panel.to_parquet(out / "cb_panel.parquet")
    print(f"面板 {len(panel)} 行, {panel['code'].nunique()} 只, {panel['date'].min().date()} -> {panel['date'].max().date()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
