"""回归测试: 入选特征不得靠"哪些股票有数据"来隐式选股

事故背景 (2026-08-14)
────────────────────
FBTR 实验里表现最好的那档 (20 种子中位总收益 +27.1%) 被证明全部来自一个掩码, 不是策略:

  旧资金流源(iFinD)只覆盖 data/universe/watchlist_216.json —— 一份 2026-07-18
  手工按题材挑的 216 只概念股。于是老矩阵里 fund_flow_* 只有 243 只有值, 其余是 NaN。
  LightGBM 只要在这列上切一刀 NaN / 非 NaN, 就把选股范围锁进了那批手挑票。

  实测这批票本身就是 alpha (scripts/coverage_bias_test.py):
    名单∩universe 130 只  2022-09~2026-08 等权 +174.0% (年化+30.6%)
    全 universe 630 只                        +56.2% (年化+12.5%)
    -> 年化超额 +16.7%, IR 1.42, 除 2022 外逐年跑赢

  所以把源换成 tushare(全覆盖, 掩码消失)塌 23.7pp, 直接剔掉资金流列(掩码同样消失)
  塌到 -16.2% —— 两条路都塌, 塌的不是资金流信息, 是掩码。

同一个病还有第二条通道: con_* / tev_* 由 watchlist_216 的 theme 映射派生, 所以
线上矩阵里 tev_* 只有 127/519 只有值, 且这 127 只 100% 属于 watchlist_216。
线上 F7 的 65 特征里就有 4 个这样的列 (tev_bull_5d / tev_bear_5d / tev_decay_*)。

为什么必须用测试锁住
──────────────────
这类偏差没有任何报错、IC 也不会变差 —— 它只是让回测数字虚高, 而虚高的来源是
"用 2026 年的后见之明挑 2022 年的票", 向前看没有任何理由继续成立。
判据不能只写死 watchlist_216: 任何"部分覆盖 + 覆盖集是某份人工名单"的列都是同一个病。
"""
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import pytest
except ModuleNotFoundError:
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "training_data_pit_v24.parquet"
# 覆盖率低于此值就认为该列在用"有没有数据"划分池子
MIN_STOCK_COVERAGE = 0.50
# 覆盖集有这么大比例落在人工名单里, 就认定是那份名单在当掩码。
# 定 0.85 而不是 0.95: 实测 ev_decay_n_5d 的覆盖集是 127/140 = 90.7% 落在
# watchlist_216 (剩下 13 只是后来入池的), 而它的偏差一点不小 ——
# 有值 140 只年化 +31.7% vs 无值 379 只 +9.1%, 年化差 +21.1%, IR 1.32。
# 阈值定在 0.95 就会漏掉这个真阳性。
CONTAINMENT = 0.85


class _Skip(Exception):
    pass


def _skip(msg):
    if pytest is not None:
        pytest.skip(msg)
    raise _Skip(msg)


def _c6(s):
    return s.astype(str).str.replace(r"\..*$", "", regex=True).str.zfill(6)


def _manual_lists():
    """所有人工挑选的名单 —— 它们都不该成为特征覆盖面的边界

    watchlist.json 目前与 watchlist_216.json 内容完全相同, 所以按集合去重,
    否则同一个问题会被报两遍。
    """
    out, seen = {}, []
    for fn in ("watchlist_216.json", "watchlist.json"):
        p = ROOT / "data" / "universe" / fn
        if not p.exists():
            continue
        wl = json.loads(p.read_text(encoding="utf-8"))
        items = wl.get("watchlist", wl) if isinstance(wl, dict) else wl
        codes = {str(x["code"] if isinstance(x, dict) else x).split(".")[0].zfill(6)
                 for x in items}
        if codes in seen:
            continue
        seen.append(codes)
        out[fn] = codes
    return out


def _live_features():
    """线上真正在用的那套特征 (live_config.FEATURES_FROM)"""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from live_config import FEATURES_FROM
    except Exception:
        return None, None
    p = ROOT / "data" / "processed" / FEATURES_FROM
    if not p.exists():
        return None, FEATURES_FROM
    return json.loads(p.read_text(encoding="utf-8")).get("selected_features"), FEATURES_FROM


def _coverage_table():
    import pyarrow.parquet as pq

    feats, src = _live_features()
    if not TRAIN.exists() or not feats:
        _skip(f"本地无训练集或特征集({src}), 该体检只在有数据的机器上跑")
    have = set(pq.ParquetFile(TRAIN).schema_arrow.names)
    cols = [c for c in feats if c in have]
    df = pd.read_parquet(TRAIN, columns=["code"] + cols)
    df["_c6"] = _c6(df["code"])
    pool = set(df["_c6"].unique())
    rows = []
    for c in cols:
        codes = set(df.loc[df[c].notna(), "_c6"])
        rows.append((c, codes, len(codes) / max(len(pool), 1)))
    return rows, pool, src


def test_no_selected_feature_is_masked_by_a_manual_list():
    """入选特征的覆盖面不得等于某份人工名单 —— 那等于让模型照名单选股。"""
    rows, pool, src = _coverage_table()
    lists = _manual_lists()
    if not lists:
        _skip("找不到人工名单文件, 无法做包含性判定")

    bad = []
    for col, codes, cov in rows:
        if cov >= MIN_STOCK_COVERAGE or not codes:
            continue
        for fn, wl in lists.items():
            inside = len(codes & wl) / len(codes)
            if inside >= CONTAINMENT:
                bad.append(f"{col}: 覆盖 {cov:.1%} 的池子, 其中 {inside:.1%} 落在 {fn}")
    assert not bad, (
        f"特征集 {src} 里有 {len(bad)} 个列在用人工名单当掩码, "
        f"回测数字会虚高且不可向前复制:\n  " + "\n  ".join(bad))


def test_no_selected_feature_has_tiny_stock_coverage():
    """更宽的护栏: 入选特征不该只覆盖池子的一小部分。

    即便覆盖集不对应任何已知名单, "有没有数据"本身也是一条与经济含义无关的
    分裂依据 —— 模型会拿它当选股规则, 而数据覆盖面将来一变, 策略就跟着变。
    """
    rows, pool, src = _coverage_table()
    thin = [f"{c}: {cov:.1%}" for c, _, cov in rows if cov < MIN_STOCK_COVERAGE]
    assert not thin, (
        f"特征集 {src} 里有 {len(thin)} 个列的股票覆盖率低于 "
        f"{MIN_STOCK_COVERAGE:.0%} (池子 {len(pool)} 只):\n  " + "\n  ".join(thin))


def _main():
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = []
    for n in names:
        try:
            globals()[n]()
            print(f"  PASS  {n}")
        except _Skip as e:
            print(f"  SKIP  {n}: {e}")
        except AssertionError as e:
            failed.append(n)
            print(f"  FAIL  {n}: {str(e)[:600]}")
    print(f"\n{len(names) - len(failed)}/{len(names)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
