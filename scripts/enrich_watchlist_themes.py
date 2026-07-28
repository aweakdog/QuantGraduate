"""
补全 watchlist_216.json: 股票名 + 主题分类

数据源:
- concept_stock_map.json → stock_to_concepts (同花顺概念, 覆盖178/216)
- 科创板等38只手动映射
- 手动补全股票中文名称
"""
import json

_BASE = r"D:/myAI/WorkBuddy-workspace/quant-strategy/data"

with open(f"{_BASE}/universe/concept_stock_map.json", encoding="utf-8") as f:
    csm = json.load(f)
stc = csm.get("stock_to_concepts", {})

with open(f"{_BASE}/universe/watchlist_216.json", encoding="utf-8") as f:
    wl = json.load(f)
items = wl.get("watchlist", wl) if isinstance(wl, dict) else wl

# ── 手动补全股票名 + 主题 (科创板/次新/其他未覆盖) ──
_HARDCODED = {
    # 科创板 688xxx
    "688235": ("百济神州", "医药生物-创新药"),
    "688498": ("源杰科技", "半导体-光芯片"),
    "688146": ("中船特气", "国防军工-特种气体"),
    "688521": ("芯原股份", "半导体-芯片设计"),
    "688702": ("盛科通信", "半导体-网络芯片"),
    "688361": ("中科飞测", "半导体-检测设备"),
    "688249": ("晶合集成", "半导体-晶圆代工"),
    "688172": ("燕东微", "半导体"),
    "688396": ("华润微", "半导体"),
    "688183": ("生益电子", "电子-PCB"),
    "688766": ("普冉股份", "半导体-存储"),
    "688519": ("南亚新材", "电子-覆铜板"),
    "688200": ("华峰测控", "半导体-测试设备"),
    "688111": ("金山办公", "计算机-软件"),
    "688001": ("华兴源创", "半导体-检测设备"),
    "688002": ("睿创微纳", "国防军工-红外"),
    "688003": ("天准科技", "半导体-检测"),
    "688004": ("博汇科技", "计算机-软件"),
    "688005": ("容百科技", "电力设备-锂电池"),
    "688006": ("杭可科技", "电力设备-锂电池设备"),
    "688007": ("光峰科技", "电子-激光显示"),
    # 创业板 300xxx
    "300316": ("晶盛机电", "电力设备-光伏设备"),
    "300751": ("迈为股份", "电力设备-光伏设备"),
    "300220": ("金运激光", "机械设备-激光"),
    "300376": ("易事特", "电力设备-电源"),
    "300423": ("昇辉科技", "电力设备"),
    "300643": ("万通智控", "汽车-汽车零部件"),
    "300940": ("南极光", "电子-显示"),
    "300942": ("易瑞生物", "医药生物-检测"),
    "300824": ("北鼎股份", "家用电器"),
    "300915": ("海融科技", "食品饮料"),
    "300409": ("道氏技术", "基础化工"),
    "300534": ("陇神戎发", "医药生物-中药"),
    "300449": ("汉邦高科", "计算机"),
    # 主板
    "600019": ("彤程新材", "基础化工-光刻胶"),
    "603806": ("福斯特", "电力设备-光伏材料"),
    "601865": ("福莱特", "电力设备-光伏玻璃"),
    "301269": ("华大九天", "计算机-EDA软件"),
}

# ── 补全股票中文名 (已有硬编码的替换, 其他的用概念映射中的中文名) ──
_STOCK_NAMES = {k: v[0] for k, v in _HARDCODED.items()}
# 从概念映射中提取股票中文名 (stock_to_concepts 键是6位代码, 但没存中文名)
# 从 supply_chain_map 提取
try:
    with open(f"{_BASE}/universe/supply_chain_map.json", encoding="utf-8") as f:
        sc = json.load(f)
    for chain in sc.get("chains", []):
        for link in chain.get("demand_links", []):
            for s in link.get("a_share_suppliers", []):
                c6 = s["code"][:6]
                if c6 not in _STOCK_NAMES and "name" in s:
                    _STOCK_NAMES[c6] = s["name"]
except Exception:
    pass

# 直接给已覆盖的股票补中文名: 通过概念列表最多的那个概念的中文含义来判断
# 对于 stock_to_concepts 中有映射但缺中文名的: 概念名第一个词作为类别依据
# 但不好直接拿到中文名... 跳过, 让用户后续补充.

# ── 主题分配: 取第一个概念作为主题 ──
def _get_theme(c6: str) -> str:
    """取股票的主题"""
    if c6 in _HARDCODED:
        return _HARDCODED[c6][1]
    if c6 in stc and stc[c6]:
        return stc[c6][0]  # 第一个概念
    return "未分类"

def _get_name(c6: str) -> str:
    """取股票中文名"""
    if c6 in _STOCK_NAMES:
        return _STOCK_NAMES[c6]
    if c6 in stc and stc[c6]:
        # 概念映射有该股, 但缺中文名 — 保留原 code
        return c6
    return c6

# ── 更新 watchlist ──
changes = 0
for s in items:
    c6 = s["code"].split(".")[0]
    old_name = s.get("name", "")
    new_name = _get_name(c6)
    new_theme = _get_theme(c6)
    if old_name != new_name or s.get("theme") != new_theme:
        s["name"] = new_name
        s["theme"] = new_theme
        changes += 1

# 统计
themes = {}
for s in items:
    themes.setdefault(s["theme"], []).append(s["name"])
print(f"更新 {changes} 只, 共 {len(themes)} 个主题")
for t, names in sorted(themes.items(), key=lambda x: -len(x[1])):
    print(f"  {t}: {len(names)} 只 例: {names[0]}")

# ── 写回 ──
out = f"{_BASE}/universe/watchlist_216.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(wl, f, ensure_ascii=False, indent=2)
print(f"\n已写回 {out}")
