"""
backend/paths.py — 文件系统路径的「唯一真相源」(single source of truth)

设计原则 (用户要求: 尽量不硬编码):
  - 所有路径均从本文件位置运行时推导, 绝不写死绝对路径
  - 训练数据自动选取版本号最大的 training_data_vXX.parquet (v19→v23→v24 无需改代码)
  - 数据源注册表以「相对 data 根的路径片段」声明, 不在 app.py / trainer.py 重复硬编码

调用方 (app.py / trainer.py) 只 import 本模块, 不直接拼路径。
"""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import pandas as pd

# ── 目录推导 (从本文件位置向上) ─────────────────────────────────────────────
#   本文件:  .../quant-strategy/Web/backend/paths.py
#   backend -> Web -> quant-strategy
_BACKEND_DIR = Path(__file__).resolve().parent
WEB_DIR = _BACKEND_DIR.parent                       # .../quant-strategy/Web
PROJECT_ROOT = WEB_DIR.parent                        # .../quant-strategy
DATA_DIR = PROJECT_ROOT / "data"
UNIVERSE_DIR = DATA_DIR / "universe"
PROCESSED_DIR = DATA_DIR / "processed"


def project_root() -> Path:
    return PROJECT_ROOT


def web_dir() -> Path:
    return WEB_DIR


def data_dir() -> Path:
    return DATA_DIR


def universe_dir() -> Path:
    return UNIVERSE_DIR


def processed_dir() -> Path:
    return PROCESSED_DIR


def latest_training_data() -> Path:
    """
    返回版本号最大的 training_data_vXX.parquet (例如当前 v23)。

    降级策略: 若没有 vXX 版本, 回退到 training_data.parquet; 再无则抛错。
    这样切换 v24 时本函数自动跟上, 业务代码零改动。
    """
    best: Path | None = None
    best_v = -1
    for p in PROCESSED_DIR.glob("training_data_v*.parquet"):
        m = re.search(r"training_data_v(\d+)\.parquet$", p.name)
        if m and p.is_file():
            v = int(m.group(1))
            if v > best_v:
                best_v = v
                best = p
    if best is None:
        plain = PROCESSED_DIR / "training_data.parquet"
        if plain.is_file():
            return plain
    if best is None:
        raise FileNotFoundError(
            f"未找到训练数据: 既无 training_data_v*.parquet 也无 training_data.parquet @ {PROCESSED_DIR}"
        )
    return best


def watchlist_path() -> Path:
    """
    股票池路径: 优先 universe/watchlist_216.json (真实训练池, 与 training_data_vXX 一致),
    回退 watchlist.json (旧版198池, 含金属股+脏代码, 仅作兜底)。
    2026-07-10 修正: 原优先 watchlist.json(198) 导致 Web 关注圈/训练池与 v23(216只)不一致。
    """
    canonical = UNIVERSE_DIR / "watchlist_216.json"
    if canonical.is_file():
        return canonical
    return UNIVERSE_DIR / "watchlist.json"


def self_selected_path() -> Path:
    """
    用户自定义自选股: 存于 Web/data/self_selected.json, 与训练池 (universe/watchlist*.json) 解耦。
    首次打开以默认训练池初始化, 编辑后独立保存, 不污染训练资产。
    """
    web_data = WEB_DIR / "data"
    web_data.mkdir(parents=True, exist_ok=True)
    return web_data / "self_selected.json"


def train_pool_path() -> Path:
    """
    用户指定的训练池: 存于 Web/data/train_pool.json, 是『关注圈』的子集 (用户自由指定)。

    语义 (2026-07-10 用户定义):
      - 关注圈 : 后台维护, 训练池中数据覆盖够+上市超2年的标的 (自动派生, 不可编辑)
      - 训练池 : 关注圈的用户子集, 模型训练/回测的实际范围 (用户可自由指定/缩减)
      - 自选股 : 用户(使用者)关心的标的, 独立维度
    默认不存在时回退到关注圈 (watchlist_216.json)。
    """
    web_data = WEB_DIR / "data"
    web_data.mkdir(parents=True, exist_ok=True)
    return web_data / "train_pool.json"


# ── 数据源注册表 ───────────────────────────────────────────────────────────
# 格式: 显示名 -> 相对 DATA_DIR 的路径片段列表。
# 声明即真相, app.py / trainer.py 不再各自硬编码。
_DATA_SOURCE_PARTS: dict[str, list[str]] = {
    "训练数据": ["__latest__"],                              # 特殊标记: 运行时解析为最新 training_data_vXX
    "资金流历史": ["raw", "fund_flow_full", "fundflow_history.parquet"],
    "融资融券历史": ["raw", "MainNetFlow", "margintrade_history.parquet"],
    "事件公告": ["raw", "events_ifind", "events_v2.parquet"],
    "季报特征": ["raw", "quarterly_features.parquet"],
    "股票池": ["universe", "watchlist.json"],
}


def data_sources() -> dict[str, Path]:
    """把注册表解析为 {显示名: 绝对 Path}。"""
    out: dict[str, Path] = {}
    for name, parts in _DATA_SOURCE_PARTS.items():
        if parts == ["__latest__"]:
            out[name] = latest_training_data()
        else:
            out[name] = DATA_DIR.joinpath(*parts)
    return out


# ── 训练集来源解析 ─────────────────────────────────────────────────────────
def _codes_from_watchlist() -> list[str]:
    """从默认股票池 (watchlist*.json) 解析代码列表 (大写)。"""
    p = watchlist_path()
    if not p.exists():
        return []
    try:
        data = json.load(open(p, encoding="utf-8")).get("watchlist", [])
    except Exception:
        return []
    return [str(r.get("code", "")).strip().upper() for r in data if str(r.get("code", "")).strip()]


def load_universe_codes(source: str, custom_codes: list[str] | None = None) -> list[str]:
    """解析训练集来源 → 代码列表.

    合并精简语义 (2026-07-10: 训练池与关注圈等价, 已合并为『关注圈』):
      - '关注圈' : Web/data/train_pool.json (用户可编辑训练池, 模型训练/回测实际范围);
                   不存在则回退默认股票池 (universe/watchlist_216.json, 216 只)
      - '自选股' : Web/data/self_selected.json (Tab 编辑); 不存在则回退默认池
      - 其它历史值 ("训练池"/"自定义"/空) : 一律按『关注圈』处理, 保证向后兼容
    返回大写代码列表; 空列表表示无有效股票。
    """
    source = (source or "关注圈").strip()
    if source == "自选股":
        p = self_selected_path()
        if p.exists():
            try:
                data = json.load(open(p, encoding="utf-8")).get("watchlist", [])
                codes = [str(r.get("code", "")).strip().upper() for r in data if str(r.get("code", "")).strip()]
                if codes:
                    return codes
            except Exception:
                pass
        return _codes_from_watchlist()  # 回退默认池
    # 默认: 关注圈 = 用户可编辑训练池 (train_pool.json), 回退 universe/watchlist_216.json
    p = train_pool_path()
    if p.exists():
        try:
            data = json.load(open(p, encoding="utf-8")).get("watchlist", [])
            codes = [str(r.get("code", "")).strip().upper() for r in data if str(r.get("code", "")).strip()]
            if codes:
                return codes
        except Exception:
            pass
    return _codes_from_watchlist()  # 关注圈 回退默认池


def earliest_train_date() -> str:
    """返回训练数据 parquet 的最早日期 (YYYY-MM-DD), 用于『全历史/训练时长不做限制』。"""
    try:
        p = latest_training_data()
        d = pd.read_parquet(p, columns=["date"])["date"].min()
        if hasattr(d, "date"):
            return str(d.date())
        return str(d)[:10]
    except Exception:
        return "2022-09-01"


def stock_name(code: str, fallback: str = "") -> str:
    """从权威全量股票清单 (all_stock_list.parquet) 查中文名。

    JSON 股票池 (watchlist/train_pool/self_selected) 的 name 字段可能为代码数字/空,
    一律以本函数返回的真实中文名覆盖, 保证 UI 显示正确。找不到时回退 fallback 或代码本身。
    """
    return _stock_name_map().get(str(code).strip().upper(), (fallback or str(code).strip().upper()))


@functools.lru_cache(maxsize=1)
def _stock_name_map() -> dict[str, str]:
    p = DATA_DIR / "raw" / "all_stock_list.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p, columns=["code", "name"])
    return {str(c).strip().upper(): str(n) for c, n in zip(df["code"], df["name"])}


if __name__ == "__main__":
    print("PROJECT_ROOT   :", PROJECT_ROOT)
    print("PROCESSED_DIR  :", PROCESSED_DIR)
    print("训练数据(最新) :", latest_training_data())
    print("股票池         :", watchlist_path())
    print("我的自选股     :", self_selected_path())
    for name, p in data_sources().items():
        print(f"  数据源 {name:8s} -> exists={p.exists()}  {p}")
