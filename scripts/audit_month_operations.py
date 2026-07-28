import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "data" / "processed" / "wf_daily_v24_w100_ts2024-01-02_te2024-02-07_cap10000.json"


def main() -> None:
    result = json.loads(RESULT.read_text())
    daily = pd.DataFrame(result["daily"])
    portfolio = pd.DataFrame(result["portfolio"]["daily_values"])
    stocks = pd.read_parquet(ROOT / "data" / "raw" / "all_stock_list.parquet")
    names = dict(zip(stocks["code"].astype(str), stocks["name"]))
    concepts = json.loads((ROOT / "data" / "universe" / "concept_stock_map.json").read_text())["concept_to_stocks"]
    reverse = defaultdict(list)
    for concept, codes in concepts.items():
        for code in codes:
            reverse[str(code)[:6]].append(concept)

    for _, row in daily.iterrows():
        value = portfolio[portfolio["date"] == row["date"]].iloc[0]
        selected = []
        for code in row["holdings"]:
            code6 = code[:6]
            tags = "/".join(reverse[code6][:2]) if reverse[code6] else "未映射"
            selected.append(f"{code} {names.get(code, '?')} [{tags}]")
        print(
            f"{row['date']} raw={row['top3_raw_ret']:+.2%} "
            f"actual={value['daily_ret']:+.2%} value={value['value']:.0f} | "
            + "; ".join(selected)
        )

    frequency = defaultdict(int)
    for codes in daily["holdings"]:
        for code in codes:
            for concept in set(reverse[code[:6]]):
                frequency[concept] += 1
    print("\nConcept frequency among recommended stock-days:")
    for concept, count in sorted(frequency.items(), key=lambda item: -item[1])[:30]:
        print(count, concept)


if __name__ == "__main__":
    main()
