"""把 PIT 股票池(universe_pit.parquet)导出为 feature_engine 可读的 watchlist json

产出 data/universe/watchlist_pit.json:
  {"watchlist": [{"code": "600519.SH", "name": "贵州茅台", "board": "主板"}, ...]}

包含所有生效期出现过的股票并集 —— 特征只需构建一次, 回测时再按生效日过滤成分。
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNI = ROOT / "data/universe/universe_pit.parquet"
META = ROOT / "data/universe/pit_metadata.parquet"
OUT = ROOT / "data/universe/watchlist_pit.json"


def suffix(code: str) -> str:
    if code.startswith("6"):
        return "SH"
    if code[:2] in ("43", "83", "87", "88", "92"):
        return "BJ"
    return "SZ"


def main():
    u = pd.read_parquet(UNI)
    meta = pd.read_parquet(META)
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    name_map = dict(zip(meta["code"], meta["name"]))
    board_map = dict(zip(meta["code"], meta["board"]))

    codes = sorted(set(u["code"].astype(str).str.zfill(6)))
    items = [{"code": f"{c}.{suffix(c)}",
              "name": name_map.get(c, c),
              "board": board_map.get(c, "")} for c in codes]

    OUT.write_text(json.dumps({"watchlist": items}, ensure_ascii=False, indent=1))
    print(f"已写入 {OUT}: {len(items)} 只")
    by_board = pd.Series([i["board"] for i in items]).value_counts()
    print(by_board.to_string())
    missing = [c for c in codes if not (ROOT / f"data/raw/kline/{c}.parquet").exists()]
    print(f"缺K线: {len(missing)}")


if __name__ == "__main__":
    main()
