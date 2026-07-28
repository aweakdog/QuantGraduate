"""
chain_map_merger.py — 供应链数据源合并

将 supply_chain_map.json (详细绑定评分/证据等级)
与 chain_leader_universe.json (完整覆盖+交叉关系)
合并为统一的 chain_map_merged.json。

用法:
  python -m pipeline.chain_map_merger       # 执行合并
  python -m pipeline.chain_map_merger --dry  # 预览差异不写入
"""
import json
import os
import sys
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import Any

from pipeline.config import settings
from pipeline.logger import get_logger

log = get_logger("chain_map_merger")

SUPPLY_CHAIN_PATH = settings.SUPPLY_CHAIN_PATH
UNIVERSE_PATH = settings.CHAIN_LEADER_UNIVERSE_PATH
MERGED_PATH = settings.UNIVERSE_DIR / "chain_map_merged.json"


def load_json(path) -> dict:
    if not os.path.exists(path):
        log.warning("文件不存在: %s", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    log.info("已写入: %s (%d 条链)", path, len(data.get("chains", [])))


# ─── 工具 ────────────────────────────────────────────────────

def _binding_from_evidence(level: int, exposure: str) -> float:
    bmap = {1: 9.0, 2: 7.0, 3: 5.0, 4: 3.0}
    b = bmap.get(level, 3.0)
    if "核心" in exposure: b = min(b + 2.0, 10)
    elif "高" in exposure: b = min(b + 1.0, 10)
    return round(b, 1)


def _stock_to_supplier(stock: dict) -> dict:
    ev = stock.get("evidence", {})
    level = ev.get("level", 4)
    return {
        "name": stock.get("name", ""),
        "code": stock.get("code", ""),
        "role": stock.get("role", ""),
        "exposure": stock.get("exposure", ""),
        "evidence": {"level": level, "source": ev.get("source", ""),
                      "detail": ev.get("detail", ""), "revenue_pct": ev.get("revenue_pct", "")},
        "scoring": {"binding": _binding_from_evidence(level, stock.get("exposure", ""))},
    }


def _merge_supplier_lists(existing: list[dict], new: list[dict]) -> list[dict]:
    """按 code 去重合并供应商列表"""
    seen = {s["code"] for s in existing}
    merged = list(existing)
    for s in new:
        if s["code"] not in seen:
            merged.append(s)
            seen.add(s["code"])
    return merged


def universe_to_chain(leader: dict) -> dict:
    """将 universe leader 转为 chain 格式"""
    chain = {
        "theme": leader.get("theme", ""),
        "chain_leader": {
            "name": leader.get("name", ""), "code": leader.get("code", ""),
            "market": leader.get("market", ""), "desc": leader.get("rank", ""),
            "watch_events": leader.get("watch_events", []),
            "next_earnings": leader.get("next_earnings"),
            "universe_id": leader.get("id", ""),
        },
        "demand_links": [],
    }
    for ac in leader.get("a_share_chain", []):
        link = {"component": ac.get("component", ""), "direction": ac.get("direction", "正向"),
                "reaction_speed": ac.get("reaction_speed", ""), "elasticity": ac.get("elasticity", ""),
                "a_share_suppliers": [_stock_to_supplier(s) for s in ac.get("stocks", [])]}
        if link["a_share_suppliers"]:
            chain["demand_links"].append(link)
    return chain


# ─── 核心 ─────────────────────────────────────────────────────

def merge_chain_data(supply: dict, universe: dict) -> dict:
    """
    合并策略:
      1. 按链主名称分组合并 supply_chain_map 的 chains
         (处理同名链主出现在多个 theme 下的情况, 如 Corning×2)
      2. 合并为统一 chain 列表
      3. 追加 universe 中的缺失链主
      4. 对已有链主, 补齐 universe 中的额外供应商
    """

    # ── 1) 将 supply chains 按名称合并（处理同名多链情况）──
    supply_groups: OrderedDict[str, list[dict]] = OrderedDict()
    for c in supply.get("chains", []):
        name = c.get("chain_leader", {}).get("name", "")
        if name:
            key = name.lower()
            if key not in supply_groups:
                supply_groups[key] = []
            supply_groups[key].append(c)

    # 合并同名链: 保留第一个的 meta, 合并所有 demand_links
    merged_supply: list[dict] = []
    for name_lower, chains in supply_groups.items():
        base = json.loads(json.dumps(chains[0]))  # deep copy first
        seen_codes = set()
        for dl in base.get("demand_links", []):
            for s in dl.get("a_share_suppliers", []):
                seen_codes.add(s["code"])
        # 追加后续同名链的 supplier
        for extra_chain in chains[1:]:
            for dl in extra_chain.get("demand_links", []):
                target_component = dl.get("component", "")
                # 找匹配部件
                target = None
                for bdl in base["demand_links"]:
                    if bdl["component"] == target_component:
                        target = bdl
                        break
                if target is None:
                    target = {"component": target_component, "direction": dl.get("direction", ""),
                              "reaction_speed": dl.get("reaction_speed", ""),
                              "elasticity": dl.get("elasticity", ""), "a_share_suppliers": []}
                    base["demand_links"].append(target)
                for s in dl.get("a_share_suppliers", []):
                    if s["code"] not in seen_codes:
                        target["a_share_suppliers"].append(s)
                        seen_codes.add(s["code"])
        merged_supply.append(base)

    # ── 2) 构建索引 ──
    supply_map: dict[str, dict] = {}  # name_lower → chain dict
    for c in merged_supply:
        name = c["chain_leader"]["name"]
        if name:
            supply_map[name.lower()] = c

    universe_map: dict[str, dict] = {}
    for ldr in universe.get("chain_leaders", []):
        name = ldr.get("name", "")
        if name:
            universe_map[name.lower()] = ldr

    # ── 3) 处理 supply 已有链, 补齐新供应商 ──
    final_chains: list[dict] = []
    for name_lower, chain in supply_map.items():
        chain_copy = json.loads(json.dumps(chain))

        if name_lower in universe_map:
            supply_codes = {s["code"] for dl in chain_copy["demand_links"]
                            for s in dl["a_share_suppliers"]}

            uni_ldr = universe_map[name_lower]
            new_by_component: defaultdict[str, list] = defaultdict(list)
            for ac in uni_ldr.get("a_share_chain", []):
                for stock in ac.get("stocks", []):
                    code = stock.get("code", "")
                    if code not in supply_codes:
                        new_by_component[ac.get("component", "")].append(_stock_to_supplier(stock))
                        supply_codes.add(code)

            for comp, suppliers in new_by_component.items():
                target = next((dl for dl in chain_copy["demand_links"]
                               if dl["component"] == comp), None)
                if target is None:
                    target = {"component": comp, "direction": "", "reaction_speed": "",
                              "elasticity": "", "a_share_suppliers": []}
                    chain_copy["demand_links"].append(target)
                    log.info("  新增部件 %s -> %s (%d 供应商)",
                              chain["chain_leader"]["name"], comp, len(suppliers))
                else:
                    log.info("  补齐供应商 %s -> %s (%d 只)",
                              chain["chain_leader"]["name"], comp, len(suppliers))
                target["a_share_suppliers"].extend(suppliers)

        final_chains.append(chain_copy)

    # ── 4) 添加 universe 中有但 supply 没有的链主 ──
    existing_names = set(supply_map.keys())
    new_names = set(universe_map.keys()) - existing_names
    for name_lower in sorted(new_names):
        c = universe_to_chain(universe_map[name_lower])
        final_chains.append(c)
        n = sum(len(dl["a_share_suppliers"]) for dl in c["demand_links"])
        log.info("  新增链主: %s (%d 供应商)", c["chain_leader"]["name"], n)

    return {
        "meta": {
            "generated_by": "chain_map_merger.py",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "sources": ["supply_chain_map.json", "chain_leader_universe.json"],
        },
        "chains": final_chains,
        "cross_leader_relations": universe.get("cross_leader_relations", []),
    }


# ─── 加载器 ──────────────────────────────────────────────────

def load_unified_chain_map() -> dict:
    if os.path.exists(MERGED_PATH):
        return load_json(MERGED_PATH)
    return load_json(SUPPLY_CHAIN_PATH)


def find_suppliers_unified(leader_name: str) -> list[dict]:
    data = load_unified_chain_map()
    seen = set()
    suppliers = []
    for chain in data.get("chains", []):
        cl = chain["chain_leader"]
        if cl.get("name", "").lower() != leader_name.lower():
            continue
        for link in chain.get("demand_links", []):
            for s in link.get("a_share_suppliers", []):
                code = s.get("code", "")
                if code in seen:
                    continue
                seen.add(code)
                suppliers.append({
                    "code": code, "name": s.get("name", ""), "role": s.get("role", ""),
                    "exposure": s.get("exposure", ""), "component": link.get("component", ""),
                    "theme": chain.get("theme", ""),
                    "binding": s.get("scoring", {}).get("binding", 5.0),
                    "evidence_level": s.get("evidence", {}).get("level", 4),
                })
    return suppliers


def get_binding_for_supplier(leader_name: str, code6: str) -> float:
    for s in find_suppliers_unified(leader_name):
        if s["code"][:6] == code6[:6]:
            return s["binding"]
    return 5.0


# ─── 主入口 ──────────────────────────────────────────────────

def main():
    dry_run = "--dry" in sys.argv
    supply = load_json(SUPPLY_CHAIN_PATH)
    universe = load_json(UNIVERSE_PATH)
    if not supply or not universe:
        log.error("无法加载数据源"); sys.exit(1)
    if dry_run:
        print("=== 预览合并 运行 --dry 完成 ==="); return
    merged = merge_chain_data(supply, universe)
    n = sum(len(dl["a_share_suppliers"]) for c in merged["chains"] for dl in c["demand_links"])
    print(f"合并完成: {len(merged['chains'])} 链主, {n} 供应商, {len(merged['cross_leader_relations'])} 交叉关系")
    save_json(merged, MERGED_PATH)


if __name__ == "__main__":
    main()
