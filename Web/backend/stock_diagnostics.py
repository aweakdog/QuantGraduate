"""自选股数据诊断: 数据起始 / 数据密集 / 建议训练起始.

权威源 = training_data_vXX (模型实际训练用的特征矩阵, 216 只, 满程 924 交易日).
兜底   = all_stock_list.parquet (全量 4987 只, 校验代码有效性 / 是否仅在全集但不在训练池).

设计原则:
- backend 模块不依赖 streamlit, 纯 functools.lru_cache, 可在 AppTest / 普通 python 下导入.
- 元数据按 (路径, mtime) 缓存, 训练集换版本 (v23→v24) 后自动失效刷新.
- 阈值对齐策略铁律: 新股<2年替换; 每日扩展重训.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import latest_training_data, data_dir, stock_name as _sn

# ── 阈值 (对齐策略铁律) ──────────────────────────────────────────────────
DENSITY_OK = 0.95        # 覆盖率 >= 95% 视为密集
DENSITY_WARN = 0.80      # 80–95% 黄; < 80% 红
MIN_HISTORY_DAYS = 365 * 2   # 历史 < 2 年 → 提示替换


# ── 元数据加载 (按路径+mtime 缓存) ──────────────────────────────────────
@functools.lru_cache(maxsize=4)
def _training_meta_cached(cache_key: tuple[str, float]) -> tuple[dict[str, dict[str, Any]], int, "np.ndarray"]:
    """返回 ({code: {first,last,n}}, total_distinct_days, 全局交易日序列)."""
    path, _ = cache_key
    df = pd.read_parquet(Path(path), columns=["date", "code"])
    total_days = int(df["date"].nunique())
    all_days = df["date"].drop_duplicates().sort_values().to_numpy()
    meta: dict[str, dict[str, Any]] = {}
    for code, sub in df.groupby("code"):
        d = sub["date"]
        meta[code] = {
            "first": d.min(),
            "last": d.max(),
            "n": int(len(d)),
        }
    return meta, total_days, all_days


def _training_meta() -> tuple[dict[str, dict[str, Any]], int, "np.ndarray"]:
    p = latest_training_data()
    return _training_meta_cached((str(p), float(p.stat().st_mtime)))


@functools.lru_cache(maxsize=1)
def _all_stock_set_cached(cache_key: tuple[str, float]) -> set[str]:
    p = data_dir() / "raw" / "all_stock_list.parquet"
    if not p.exists():
        return set()
    s = pd.read_parquet(p, columns=["code"])
    return set(s["code"].astype(str).tolist())


def _all_stock_set() -> set[str]:
    p = data_dir() / "raw" / "all_stock_list.parquet"
    if not p.exists():
        return set()
    return _all_stock_set_cached((str(p), float(p.stat().st_mtime)))


# ── 单票诊断 ─────────────────────────────────────────────────────────────
def diagnose_stock(code: str) -> dict[str, Any]:
    """返回单票诊断字典 (字段见 diagnose_self_selected 注释)."""
    code = str(code).strip().upper()
    meta, total_days, all_days = _training_meta()

    if code not in meta:
        all_set = _all_stock_set()
        if code in all_set:
            return {
                "code": code,
                "in_pool": False,
                "name": "",
                "first": None,
                "last": None,
                "n": 0,
                "density": 0.0,
                "history_days": 0,
                "status": "pool_only",
                "status_text": "🟠 不在训练池",
                "suggestion": "在全集但无特征数据 (不在 216 训练池), 需先扩展训练池才能训练",
            }
        return {
            "code": code,
            "in_pool": False,
            "name": "",
            "first": None,
            "last": None,
            "n": 0,
            "density": 0.0,
            "history_days": 0,
            "status": "unknown",
            "status_text": "⚫ 代码无效",
            "suggestion": "未在全量股票清单中找到, 代码可能错误或无行情数据",
        }

    m = meta[code]
    first = m["first"]
    last = m["last"]
    n = m["n"]
    # 分母 = 该股票上市后到最新日期之间的全局实际交易日数
    # (排除上市前无交易的日期, 避免新股被误判为缺口; 真停牌日仍计入缺口使 density 下降)
    eligible = int((all_days >= pd.Timestamp(first)).sum()) if len(all_days) else 0
    density = n / eligible if eligible else 0.0
    history_days = (last - first).days

    if density >= DENSITY_OK:
        status, status_text = "ok", "✅ 正常"
    elif density >= DENSITY_WARN:
        status, status_text = "warn_gap", "🟡 有缺口"
    else:
        status, status_text = "low", "🔴 稀疏"

    # 建议训练起始
    first_str = first.strftime("%Y-%m-%d")
    if status == "low":
        suggestion = f"数据缺口较多 ({n}/{total_days} 日), 建议起始 {first_str} 并排查停牌/退市"
    elif history_days < MIN_HISTORY_DAYS:
        suggestion = (
            f"历史不足 2 年 ({history_days} 天), 建议仅观察或替换老股; "
            f"若坚持训练, 建议起始 {first_str}"
        )
    else:
        suggestion = f"可全量训练, 建议起始 {first_str} (MA20 预热由特征工程自动处理)"

    return {
        "code": code,
        "in_pool": True,
        "name": "",
        "first": first,
        "last": last,
        "n": n,
        "density": density,
        "history_days": history_days,
        "status": status,
        "status_text": status_text,
        "suggestion": suggestion,
    }


# ── 批量诊断 (供 Tab1 展示) ───────────────────────────────────────────────
def diagnose_self_selected(wl_df: pd.DataFrame) -> pd.DataFrame:
    """对自选股 DataFrame (含 code/name/theme) 逐票诊断, 返回展示用 DataFrame.

    新增列:
        数据起始      str  YYYY-MM-DD (或 '—')
        数据密集      str  文本柱状图 + 百分比, e.g. '█████░░░░ 52%'
        建议训练起始  str  建议起始日期 + 风险提示
        状态          str  emoji + 文本
        _density_num  float (隐藏, 供排序/测试)
    """
    if wl_df is None or len(wl_df) == 0:
        return pd.DataFrame(columns=["code", "name", "数据起始", "数据密集", "建议训练起始", "状态"])

    name_map = {}
    if "name" in wl_df.columns:
        for _, r in wl_df.iterrows():
            code = str(r["code"]).strip().upper()
            # 优先用权威中文名覆盖 (JSON 中 name 可能为代码数字/空)
            name_map[code] = _sn(code) or str(r.get("name", "")).strip()

    rows: list[dict[str, Any]] = []
    for code in wl_df["code"].astype(str).str.strip().str.upper():
        if not code:
            continue
        d = diagnose_stock(code)
        d["name"] = name_map.get(code, d.get("name", ""))
        pct = int(round(d["density"] * 100))
        filled = max(0, min(10, pct // 10))
        bar = "█" * filled + "░" * (10 - filled)
        rows.append({
            "code": code,
            "name": d["name"],
            "数据起始": d["first"].strftime("%Y-%m-%d") if d["first"] is not None else "—",
            "数据密集": f"{bar} {pct}%",
            "建议训练起始": d["suggestion"],
            "状态": d["status_text"],
            "_density_num": d["density"],
            "_status": d["status"],
        })

    out = pd.DataFrame(rows)
    if len(out) and "_density_num" in out.columns:
        out = out.sort_values("_density_num", ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    # 自测: 用训练池里行数最少 (历史最短) 的几只验证诊断
    meta, total, _ = _training_meta()
    print(f"训练池股票数: {len(meta)} | 满程交易日: {total}")
    # 取密度最低的 5 只模拟自选股
    sample = pd.DataFrame(
        {"code": list(meta.keys())[:5], "name": ["测试"] * 5, "theme": ["x"] * 5}
    )
    diag = diagnose_self_selected(sample)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print(diag[["code", "数据起始", "数据密集", "状态", "建议训练起始"]].to_string(index=False))
    # 测一个无效代码
    bad = diagnose_stock("999999.SH")
    print("\n无效代码测试:", bad["status_text"], "|", bad["suggestion"])
