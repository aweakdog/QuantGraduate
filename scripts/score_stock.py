"""查询任意股票在模型最新截面里的评分与排名

别人报一个代码, 立刻回答"模型今天怎么看它"。不训练不算特征 ——
直接读每晚 live_signal 顺手落盘的全池预测缓存 (data/live/preds_cache*.json),
所以是秒级。缓存由 17:30 重建链每晚刷新, signal_date 会一并展示,
拿到手就知道是哪天收盘的观点。

用法:
  python scripts/score_stock.py 000725
  python scripts/score_stock.py 000725 600519 301308
输出 (人话):
  每个池子里的 排名/总数、预期5日收益、分位, 以及"够不够格进买入名单"。

也被 web_server 的 GET /api/score?code= 复用 (score_codes 函数)。
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
# (缓存文件, 人话名, 该池的板块约束说明)
POOLS = [
    ("preds_cache_mb.json", "主板池(线上7条线用)", "剔除创业板/科创板"),
    ("preds_cache.json", "全市场池(base纸面线用)", "含创业板/科创板"),
]


def _names() -> dict:
    m = {}
    pm = ROOT / "data" / "universe" / "pit_metadata.parquet"
    if pm.exists():
        d = pd.read_parquet(pm, columns=["code", "name"])
        m = dict(zip(d["code"].astype(str).str.zfill(6), d["name"]))
    p = ROOT / "data" / "raw" / "all_stock_list.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        m.update(dict(zip(d["code"].astype(str).str[:6], d["name"])))
    return m


def _pool_members() -> set:
    u = ROOT / "data" / "universe" / "universe_pit.parquet"
    if not u.exists():
        return set()
    d = pd.read_parquet(u, columns=["code"])
    return set(d["code"].astype(str).str.extract(r"(\d{6})")[0].dropna())


def score_codes(codes: list[str]) -> dict:
    """返回 {code: {name, pools: [{pool, signal_date, rank, n, pred_5d_pct,
    percentile, note}], reason}} 。reason 仅在完全查不到时给。"""
    names = _names()
    caches = []
    for fn, label, board_note in POOLS:
        p = LIVE / fn
        if not p.exists():
            continue
        c = json.loads(p.read_text(encoding="utf-8"))
        order = {str(k)[:6]: i for i, k in enumerate(c["codes"])}
        caches.append({"label": label, "board_note": board_note,
                       "date": str(c.get("signal_date", ""))[:10],
                       "order": order, "preds": c["preds"], "n": len(c["codes"])})
    members = _pool_members()
    out = {}
    for raw in codes:
        code = str(raw).strip()[:6]
        row = {"code": code, "name": names.get(code, ""), "pools": []}
        for c in caches:
            if code in c["order"]:
                i = c["order"][code]
                pct = (i + 1) / c["n"] * 100
                row["pools"].append({
                    "pool": c["label"], "signal_date": c["date"],
                    "rank": i + 1, "n": c["n"],
                    "pred_5d_pct": round(float(c["preds"][i]) * 100, 2),
                    "percentile": round(pct, 1),
                    "note": ("前5名, 在买入名单里" if i < 5 else
                             "前10%" if pct <= 10 else
                             "前30%" if pct <= 30 else
                             "后50%, 模型不看好" if pct > 50 else "中游"),
                })
        if not row["pools"]:
            if code not in members:
                row["reason"] = "不在股票池里(池子是 PIT 流动性筛选的 ~630 只), 模型从没给它打过分"
            elif code[:2] in ("30", "68"):
                row["reason"] = "创科板股票且当日无全市场缓存, 或当日特征不完整"
            else:
                row["reason"] = "在池子里但当日特征不完整(停牌/新股/数据缺), 这天没被打分"
        out[code] = row
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    res = score_codes(sys.argv[1:])
    for code, r in res.items():
        title = f"{code} {r['name']}".strip()
        print(f"\n== {title} ==")
        if not r["pools"]:
            print(f"  {r['reason']}")
            continue
        for p in r["pools"]:
            print(f"  [{p['pool']}] 信号日 {p['signal_date']}")
            print(f"    排名 {p['rank']}/{p['n']} (前 {p['percentile']}%)  "
                  f"预期5日收益 {p['pred_5d_pct']:+.2f}%  -> {p['note']}")


if __name__ == "__main__":
    main()
