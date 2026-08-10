"""数据源盘点: 拉下来的东西, 到底有没有进模型

背景: data/raw 下有 20+ 个来源, 但 feature_engine 只读其中几个。剩下的可能是
(a) 早期实验的遗留 (b) 被别的脚本用但没进特征 (c) 真的忘了接。
花钱花时间拉回来却没用上的数据, 要么接进去, 要么删掉别再维护。

判定方式: 在全仓库 .py 里搜这个目录/文件名被谁引用, 并区分
    进特征   —— 被 pipeline/feature_engine.py 引用 (真正进了模型)
    仅脚本   —— 只被 scripts/ 或其他地方引用 (做过分析, 没进模型)
    没人用   —— 全仓库找不到引用

    python scripts/audit_data_sources.py
"""
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FEATURE_ENGINE = "pipeline/feature_engine.py"

# 这些目录名太通用, 直接搜会命中无关代码, 需要更精确的模式
SKIP_NAMES = {"__pycache__"}


def du(path):
    try:
        out = subprocess.run(["du", "-sh", str(path)], capture_output=True,
                             text=True, timeout=60).stdout
        return out.split("\t")[0].strip()
    except Exception:
        return "?"


def newest_mtime(path):
    """最后一次更新时间 —— 判断这个源是不是还在维护"""
    if path.is_file():
        return datetime.fromtimestamp(path.stat().st_mtime)
    newest = None
    for p in path.rglob("*"):
        if p.is_file():
            t = datetime.fromtimestamp(p.stat().st_mtime)
            if newest is None or t > newest:
                newest = t
    return newest


def grep_refs(name):
    """全仓库搜引用, 返回引用它的文件列表(排除本脚本自己)"""
    try:
        out = subprocess.run(
            ["grep", "-rl", "--include=*.py", name, str(ROOT)],
            capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return []
    files = []
    for line in out.splitlines():
        rel = str(Path(line).relative_to(ROOT))
        if rel.startswith(("v24/", "scripts/audit_data_sources.py", ".venv")):
            continue          # v24 是旧版本快照, 不算现役引用
        files.append(rel)
    return sorted(set(files))


def main():
    items = sorted(RAW.iterdir(), key=lambda p: p.name)
    rows = []
    for p in items:
        if p.name in SKIP_NAMES:
            continue
        refs = grep_refs(p.name)
        in_feat = any(r == FEATURE_ENGINE for r in refs)
        mt = newest_mtime(p)
        rows.append({
            "name": p.name,
            "size": du(p),
            "n": sum(1 for _ in p.rglob("*") if _.is_file()) if p.is_dir() else 1,
            "mtime": mt,
            "refs": refs,
            "status": "进特征" if in_feat else ("仅脚本" if refs else "没人用"),
        })

    order = {"进特征": 0, "仅脚本": 1, "没人用": 2}
    rows.sort(key=lambda r: (order[r["status"]], r["name"]))

    print(f"{'数据源':<32}{'大小':>8}{'文件数':>8}{'最后更新':>13}  状态")
    print("-" * 78)
    cur = None
    for r in rows:
        if r["status"] != cur:
            cur = r["status"]
            print(f"── {cur} " + "─" * (72 - len(cur)))
        mt = r["mtime"].strftime("%Y-%m-%d") if r["mtime"] else "?"
        stale = ""
        if r["mtime"] and (datetime.now() - r["mtime"]).days > 30:
            stale = f"  (已 {(datetime.now() - r['mtime']).days} 天没更新)"
        print(f"{r['name']:<32}{r['size']:>8}{r['n']:>8}{mt:>13}{stale}")
        if r["status"] == "仅脚本":
            show = [x for x in r["refs"] if not x.startswith("data/")][:3]
            if show:
                print(f"{'':>4}被引用: {', '.join(show)}")

    n_unused = sum(1 for r in rows if r["status"] == "没人用")
    n_script = sum(1 for r in rows if r["status"] == "仅脚本")
    print("-" * 78)
    print(f"合计 {len(rows)} 个来源: 进特征 "
          f"{sum(1 for r in rows if r['status'] == '进特征')} 个, "
          f"仅脚本 {n_script} 个, 没人用 {n_unused} 个")


if __name__ == "__main__":
    main()
