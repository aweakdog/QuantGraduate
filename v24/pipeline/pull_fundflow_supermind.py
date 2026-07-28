"""
SuperMind 批量拉取 DDE + 资金流数据 — 198只股票
用 query_iwencai 批量查询（5股/批），返回最新快照

输出: data/raw/fund_flow_full/supermind_dde_snapshot.parquet
"""
import asyncio, json, sys, time
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from supermind_executor import SuperMindExecutor

DATA = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")
WATCH_PATH = DATA / "universe" / "watchlist.json"
OUT_PATH = DATA / "raw" / "fund_flow_full" / "supermind_dde_snapshot.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(str(WATCH_PATH), encoding="utf-8") as f:
    stocks = json.load(f)["watchlist"]

# Batch 5 stocks per query
BATCH_SIZE = 5
INDICATORS = "DDE大单净量 DDE大单净额 DDE散户数量 DDE大单金额 主力资金流向 融资融券余额"

async def main():
    all_results = []
    
    async with SuperMindExecutor() as executor:
        await executor.connect()
        
        for i in range(0, len(stocks), BATCH_SIZE):
            batch = stocks[i:i+BATCH_SIZE]
            names = ",".join(s["name"] for s in batch)
            query = f"{names} {INDICATORS}"
            
            code = f"""
import pandas as pd
try:
    r = query_iwencai({json.dumps(query)}, df=True)
    if r is not None and hasattr(r, 'to_json'):
        print(r.to_json(orient='records', force_ascii=False))
    elif r is not None:
        print(str(r))
    else:
        print('[]')
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
            result = await executor.execute(code, timeout=30)
            
            if result.get("stdout"):
                try:
                    lines = result["stdout"].strip().split("\n")
                    json_str = lines[-1] if lines else "[]"
                    records = json.loads(json_str)
                    if isinstance(records, list):
                        all_results.extend(records)
                except json.JSONDecodeError:
                    pass
            
            batch_names = [s["name"] for s in batch]
            print(f"  [{i+BATCH_SIZE}/{len(stocks)}] {batch_names}")
            time.sleep(0.5)
    
    # Save
    if all_results:
        df = pd.DataFrame(all_results)
        df["pulled_at"] = pd.Timestamp.now()
        df.to_parquet(OUT_PATH, index=False)
        print(f"\nSaved: {OUT_PATH} ({len(df)} rows)")
    else:
        print("\nNo results!")

if __name__ == "__main__":
    asyncio.run(main())
