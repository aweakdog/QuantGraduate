# -*- coding: utf-8 -*-
"""
宏观数据日更 —— 替代已断供的 iFinD 源 (2026-08-13 上线)

源映射 (2024 年后重叠段逐日对账: 相关 / 中位比值):
  中国大宗商品价格指数   akshare macro_china_commodity_price_index (东财)   1.000000 / 1.000000
  全球半导体SOX         akshare index_us_stock_sina ".SOX" (新浪)          1.000000 / 1.000000
  CN2Y / CN5Y          akshare bond_china_yield (中债登官网)               1.000000 / 1.000000
  US2Y / US5Y          tushare us_tycr y2/y5 (美财政部)                    1.000000 / 1.000000
  USDCNH / USDJPY      tushare fx_daily bid_close (FXCM)                  0.988 / 1.0027 (收盘时点差)
  USDIND               tushare fx_daily 六币按 ICE 公式合成 DXY            0.992 / 1.0019 (收盘时点差)
  标普/道指/纳指/A50期货 akshare futures_foreign_hist ES/YM/NQ/CHA50CFD  0.9998+ / 1.0000 (2026-08-19 接管)

合并策略: 只 append 旧表末日之后的新行, 旧数据永不改写。
写盘前 no-regression 自检: 行数与"最新值"非空数都不得减少, 否则拒绝写入。
(教训来自 2026-08 资金流整表覆盖事故, 见 scripts/pull_fundflow_shard.py do_merge)

用法:
  python -m pipeline.pull_macro            # 增量: 每个文件从各自断点拉到今天
"""
from __future__ import annotations

import sys
import time
import shutil
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import settings  # noqa: E402

MACRO_DIR = Path(settings.MACRO_DIR)

TODAY = dt.date.today()

# ICE 美元指数权重 (合成 DXY)
_DXY_PAIRS = {"EURUSD": -0.576, "USDJPY": 0.136, "GBPUSD": -0.119,
              "USDCAD": 0.091, "USDSEK": 0.042, "USDCHF": 0.036}
_DXY_CONST = 50.14348112


def _pro():
    import tushare as ts
    if not settings.TUSHARE_TOKEN:
        raise RuntimeError(".env 里没有 TUSHARE_TOKEN")
    return ts.pro_api(settings.TUSHARE_TOKEN)


# ─── 各源 fetcher: 返回 DataFrame[日期(datetime64), 最新值(float)] ───────────

def fetch_commodity_idx(since: dt.date) -> pd.DataFrame:
    import akshare as ak
    d = ak.macro_china_commodity_price_index()
    d = d.rename(columns={d.columns[0]: "日期"})
    d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
    d["最新值"] = pd.to_numeric(d["最新值"], errors="coerce")
    return d[["日期", "最新值"]].dropna()


def fetch_sox(since: dt.date) -> pd.DataFrame:
    import akshare as ak
    d = ak.index_us_stock_sina(symbol=".SOX")
    out = pd.DataFrame({"日期": pd.to_datetime(d["date"], errors="coerce"),
                        "最新值": pd.to_numeric(d["close"], errors="coerce")})
    return out.dropna()


_cn_yield_cache: dict = {}


def _fetch_cn_yield(tenor: str, since: dt.date) -> pd.DataFrame:
    """中债登国债收益率, 单次查询范围有限 -> 按年分段; CN2Y/CN5Y 共用缓存"""
    import akshare as ak
    key = since.isoformat()
    if key not in _cn_yield_cache:
        segs = []
        start = since
        while start <= TODAY:
            end = min(start + dt.timedelta(days=360), TODAY)
            for attempt in range(3):
                try:
                    d = ak.bond_china_yield(start_date=start.strftime("%Y%m%d"),
                                            end_date=end.strftime("%Y%m%d"))
                    if "曲线名称" in d.columns and "5年" in d.columns:
                        segs.append(d[d["曲线名称"] == "中债国债收益率曲线"])
                        break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        print(f"    [cn_yield] 段 {start}~{end} 三次失败: {str(e)[:80]}")
                time.sleep(3)
            start = end + dt.timedelta(days=1)
            time.sleep(1)
        _cn_yield_cache[key] = pd.concat(segs) if segs else pd.DataFrame()
    cn = _cn_yield_cache[key]
    if not len(cn) or tenor not in cn.columns:
        return pd.DataFrame(columns=["日期", "最新值"])
    out = pd.DataFrame({"日期": pd.to_datetime(cn["日期"], errors="coerce"),
                        "最新值": pd.to_numeric(cn[tenor], errors="coerce")})
    return out.dropna()


# CN2Y 不修: 中债登 akshare 接口无 2 年期限(只有 3月/6月/1/3/5/7/10/30年),
# (1Y+3Y)/2 合成对账最大偏离 6.9% 不达标, 且 cn2y 不在 FB 特征集内。


def fetch_cn5y(since):
    return _fetch_cn_yield("5年", since)


def _fetch_us_tycr(col: str, since: dt.date) -> pd.DataFrame:
    d = _pro().us_tycr(start_date=since.strftime("%Y%m%d"), end_date=TODAY.strftime("%Y%m%d"))
    out = pd.DataFrame({"日期": pd.to_datetime(d["date"], errors="coerce"),
                        "最新值": pd.to_numeric(d[col], errors="coerce")})
    return out.dropna()


def fetch_us2y(since):
    return _fetch_us_tycr("y2", since)


def fetch_us5y(since):
    return _fetch_us_tycr("y5", since)


def _fetch_fx(code: str, since: dt.date) -> pd.DataFrame:
    d = _pro().fx_daily(ts_code=f"{code}.FXCM",
                        start_date=since.strftime("%Y%m%d"),
                        end_date=TODAY.strftime("%Y%m%d"))
    out = pd.DataFrame({"日期": pd.to_datetime(d["trade_date"], errors="coerce"),
                        "最新值": pd.to_numeric(d["bid_close"], errors="coerce")})
    return out.dropna().sort_values("日期")


def fetch_usdcnh(since):
    return _fetch_fx("USDCNH", since)


def fetch_usdjpy(since):
    return _fetch_fx("USDJPY", since)


def fetch_usdind(since: dt.date) -> pd.DataFrame:
    """ICE 公式六币合成美元指数"""
    pro = _pro()
    legs = {}
    for pair in _DXY_PAIRS:
        d = pro.fx_daily(ts_code=f"{pair}.FXCM",
                         start_date=since.strftime("%Y%m%d"),
                         end_date=TODAY.strftime("%Y%m%d"))
        if not len(d):
            raise RuntimeError(f"fx_daily {pair} 返回空, 放弃合成")
        d["日期"] = pd.to_datetime(d["trade_date"], errors="coerce")
        legs[pair] = d.set_index("日期")["bid_close"].astype(float)
        time.sleep(0.3)
    m = pd.concat(legs, axis=1).dropna()
    vals = _DXY_CONST * np.prod([m[p] ** w for p, w in _DXY_PAIRS.items()], axis=0)
    return pd.DataFrame({"日期": m.index, "最新值": vals}).dropna().sort_values("日期")


def _fetch_foreign_fut(sym: str, since: dt.date) -> pd.DataFrame:
    """外盘指数期货 (新浪). 接管断供的 iFinD 期货表 (2026-08-19):
    与遗留表重叠段 999 天对账 corr>0.9998 / 中位比值 1.0000 (ES/YM/NQ/CHA50CFD)"""
    import akshare as ak
    d = ak.futures_foreign_hist(symbol=sym)
    d = d.rename(columns={"date": "日期", "close": "最新值"})
    d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
    d["最新值"] = pd.to_numeric(d["最新值"], errors="coerce")
    return d[["日期", "最新值"]].dropna()


def fetch_sp_fut(since):
    return _fetch_foreign_fut("ES", since)


def fetch_dj_fut(since):
    return _fetch_foreign_fut("YM", since)


def fetch_nq_fut(since):
    return _fetch_foreign_fut("NQ", since)


def fetch_a50_fut(since):
    return _fetch_foreign_fut("CHA50CFD", since)


# ─── 合并与写盘 ──────────────────────────────

TARGETS = [
    ("中国大宗商品价格指数", fetch_commodity_idx),
    ("全球半导体SOX", fetch_sox),
    ("CN5Y", fetch_cn5y),
    ("US2Y", fetch_us2y),
    ("US5Y", fetch_us5y),
    ("USDCNH", fetch_usdcnh),
    ("USDJPY", fetch_usdjpy),
    ("USDIND", fetch_usdind),
    ("标普期货", fetch_sp_fut),
    ("道指期货", fetch_dj_fut),
    ("纳指期货", fetch_nq_fut),
    ("A50期货", fetch_a50_fut),
]


def update_one(name: str, fetcher) -> str:
    path = MACRO_DIR / f"{name}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["日期", "最新值"])
    old_dates = pd.to_datetime(old["日期"], errors="coerce") if len(old) else pd.Series([], dtype="datetime64[ns]")
    last = old_dates.max()
    since = (last.date() - dt.timedelta(days=5)) if pd.notna(last) else dt.date(2019, 1, 1)

    new = fetcher(since)
    if not len(new):
        return f"{name:14s} 源返回空 (旧末日 {last.date() if pd.notna(last) else 'N/A'})"

    # 只 append 旧末日之后的行, 旧数据永不改写
    add = new[new["日期"] > (last if pd.notna(last) else pd.Timestamp("1900-01-01"))]
    add = add.drop_duplicates(subset=["日期"]).sort_values("日期")
    if not len(add):
        return f"{name:14s} 无新增 (旧末日 {last.date()}, 源末日 {new['日期'].max().date()})"

    merged = old.copy()
    merged["日期"] = pd.to_datetime(merged["日期"], errors="coerce")
    add_aligned = add.reindex(columns=merged.columns)  # 旧表多余列(涨跌幅等)填 NaN
    merged = pd.concat([merged, add_aligned], ignore_index=True)
    merged = merged.drop_duplicates(subset=["日期"], keep="first").sort_values("日期").reset_index(drop=True)

    # no-regression 自检 (先自检后备份, 顺序不可反)
    n_old_val = pd.to_numeric(old["最新值"], errors="coerce").notna().sum() if len(old) else 0
    n_new_val = pd.to_numeric(merged["最新值"], errors="coerce").notna().sum()
    assert len(merged) >= len(old), f"{name} 合并后行数减少 {len(old)}->{len(merged)}"
    assert n_new_val >= n_old_val, f"{name} 最新值非空数减少 {n_old_val}->{n_new_val}"
    assert not merged["日期"].duplicated().any(), f"{name} 合并后日期重复"

    if path.exists():
        bak = path.with_suffix(f".parquet.bak")
        shutil.copy2(path, bak)
    merged.to_parquet(path, index=False)
    return f"{name:14s} +{len(add)} 行 -> 末日 {merged['日期'].max().date()} (原断点 {last.date() if pd.notna(last) else 'N/A'})"


def main():
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    fails = 0
    for name, fetcher in TARGETS:
        try:
            print("  " + update_one(name, fetcher), flush=True)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  {name:14s} FAIL: {str(e)[:150]}", flush=True)
        time.sleep(1)
    print(f"[pull_macro] 完成, 失败 {fails}/{len(TARGETS)}")
    # 单源偶发失败不算致命(feature_engine ffill 250 天), 全灭才报错
    if fails == len(TARGETS):
        sys.exit(1)


if __name__ == "__main__":
    main()
