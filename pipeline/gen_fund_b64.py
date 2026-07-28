"""生成资金流数据 base64"""
import pandas as pd, json, base64, sys
from pathlib import Path
_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_default_csv = str(_DATA_ROOT / "fund_flow_6m.csv")
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else _default_csv,
                 index_col=0, parse_dates=True)
fund = {}
for date, row in df.iterrows():
    fund[date.strftime("%Y-%m-%d")] = {c: (None if pd.isna(row[c]) else row[c]) for c in df.columns}
b64 = base64.b64encode(json.dumps(fund, ensure_ascii=False).encode()).decode()
print(b64)
