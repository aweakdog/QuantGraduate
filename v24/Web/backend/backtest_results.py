"""
backend/backtest_results.py — 预计算 Walk-Forward 回测结果发现器

职责:
  - 在 data/processed 下 glob 所有 wf_daily_v*.json (含 _reverse), 解析元信息
  - 不写死任何版本号/路径, 全靠 glob + 正则
  - 供「回测结果」Tab 直接渲染真实曲线, 无需等训练

JSON 结构 (由 wf_daily_expanding.py 产出):
  {
    "label","model","features","period","n_prediction_days",
    "summary": { ic_mean, ic_std, top3_excess_mean, cum_return_pct,
                 annualized_return_pct, sharpe, sharpe_raw, max_dd_pct,
                 win_rate_pct, hit_rate, total_cost_est_pct },
    "monthly_ic": { "2025-09": -0.0212, ... },
    "daily": [ {date, ic, top3_ret, ...}, ... ]   # 仅尾段采样, 不全量
  }
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from backend.paths import processed_dir

_FILENAME_RE = re.compile(r"wf_daily_v(\d+)(?:.*(reverse).*)?\.json$", re.IGNORECASE)
# Web 版 benchmark 系列 (wf_v23_web*.json): 独立版本号 230, 不与真实策略 v19~v23 混淆
# 后缀含连字符 (如 _ts2026-06-01), 必须用 [^.] 或 .* 而非 \w (连字符非 \w)
_FILENAME_RE_WEB = re.compile(r"wf_v23_web(?:_.*)?\.json$", re.IGNORECASE)
_WEB_VERSION = 230


def _parse_name(path: Path) -> tuple[int, str]:
    """从文件名解析 (版本号, 方向)。"""
    m = _FILENAME_RE.search(path.name)
    if m:
        version = int(m.group(1))
        direction = "反向" if m.group(2) else "正向"
        return version, direction
    mw = _FILENAME_RE_WEB.search(path.name)
    if mw:
        is_reverse = bool(re.search(r"reverse", path.name, re.IGNORECASE))
        return (_WEB_VERSION, "反向" if is_reverse else "正向")
    return (0, "正向")


def discover_backtests() -> list[dict[str, Any]]:
    """
    发现所有预计算回测结果, 返回按 (版本降序, 正向优先) 排序的列表。

    每项:
      path, name("v23 正向"), version, direction, period, features,
      n_days, summary(dict), monthly_ic(dict)
    支持两类文件:
      - wf_daily_vNN*.json     : 真实策略回测 (canonical v23 等)
      - wf_v23_web*.json       : Web 版 benchmark (版本号固定 230, 照妖镜独立配对)
    """
    out: list[dict[str, Any]] = []
    for p in sorted(list(processed_dir().glob("wf_daily_v*.json"))
                    + list(processed_dir().glob("wf_v23_web*.json"))):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = d.get("summary", {}) or {}
        monthly_ic = d.get("monthly_ic", {}) or {}
        version, direction = _parse_name(p)
        is_web = bool(_FILENAME_RE_WEB.search(p.name))
        # 标签优先用 JSON 内 period 字段 (含完整区间); 否则从文件名解析日期
        period_raw = (d.get("period") or "").strip()
        if period_raw:
            tag = f" ({period_raw})"
        else:
            dates = re.findall(r"\d{4}-\d{2}-\d{2}", p.name)
            tag = f" 〔{dates[0]}→{dates[-1]}〕" if dates else ""
        # canonical = 文件名不含实验区间标记 (_tr/_te/_ts...); 仅正/反配对主回测参与照妖镜。
        # 注意: 不能依赖 period 标签判 canonical, 因完整回测 JSON 本身含 period 字段。
        is_experiment = bool(re.search(r"_(tr|te|ts)\d{4}", p.name))
        label = "Web" if is_web else f"v{version}"
        name = f"{label} {direction}{tag}"
        out.append({
            "path": p,
            "name": name,
            "is_web": is_web,
            "canonical": not is_experiment,  # 主回测(canonical); 带区间后缀=实验变体
            "version": version,
            "direction": direction,
            "period": d.get("period", ""),
            "features": d.get("features", 0),
            "n_days": d.get("n_prediction_days", 0),
            "summary": summary,
            "monthly_ic": monthly_ic,
        })

    # 排序: 版本降序, 正向在前
    out.sort(key=lambda x: (-x["version"], 0 if x["direction"] == "正向" else 1))
    return out


def load_backtest(path: Path) -> Optional[dict[str, Any]]:
    """加载完整 JSON (含 daily 尾段)。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def directional_alpha(records: list[dict[str, Any]]) -> dict[int, float]:
    """
    方向性α = 正向Sharpe − 反向Sharpe, 按版本聚合。
    仅当某版本同时存在正/反时计入 (照妖镜: 反向亏钱才说明正向是真α)。
    只统计 canonical (无日期后缀) 主回测, 排除带 _tr/_te 的实验变体。
    """
    by_version: dict[int, dict[str, float]] = {}
    for r in records:
        if not r.get("canonical", True):
            continue
        if r["direction"] not in ("正向", "反向"):
            continue
        s = r["summary"].get("sharpe")
        if s is None:
            continue
        by_version.setdefault(r["version"], {})[r["direction"]] = float(s)
    alpha: dict[int, float] = {}
    for ver, dd in by_version.items():
        if "正向" in dd and "反向" in dd:
            alpha[ver] = round(dd["正向"] - dd["反向"], 3)
    return alpha


if __name__ == "__main__":
    recs = discover_backtests()
    print(f"发现 {len(recs)} 个回测结果:")
    for r in recs:
        s = r["summary"]
        print(
            f"  {r['name']:12s} | Sharpe={s.get('sharpe')} "
            f"IC={s.get('ic_mean')} cum%={s.get('cum_return_pct')} "
            f"MaxDD={s.get('max_dd_pct')}% win%={s.get('win_rate_pct')}"
        )
    print("方向性α:", directional_alpha(recs))
