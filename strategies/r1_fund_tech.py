"""
Round 1 — 资金面+技术面策略
"""
import sys, json, asyncio, subprocess
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SMLOGIN = _PROJECT_ROOT / "engine" / "smlogin"
if str(_SMLOGIN) not in sys.path:
    sys.path.insert(0, str(_SMLOGIN))
from smlogin import SuperMindSession

# 用ths venv生成CSV repr
THS_PY = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe"
CSV_PATH = str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv")
r = subprocess.run([THS_PY, "-c",
    "import pandas as pd; df=pd.read_csv(r'%s', index_col=0); print(repr(df.to_csv()))" % CSV_PATH],
    capture_output=True, text=True, timeout=30)
csv_repr = r.stdout.strip()
print(f"CSV repr: {len(csv_repr)} chars")

CODE = r'''
import pandas as pd, numpy as np, json
from datetime import datetime, date
from io import StringIO

stock = "601689.SH"
today = date.today().isoformat().replace("-", "")

FUND_CSV = %s

fund_df = pd.read_csv(StringIO(FUND_CSV), index_col=0, parse_dates=True)
print("=== ROUND 1 ===")
print("标的: " + stock + ", 天数: " + str(len(fund_df)))

data = get_price(stock, start_date="20251229", end_date=today,
                 fre_step="1d", fields=["close","high","low","open","volume"], fq="pre")
if len(data) < 20:
    print("数据不足"); raise SystemExit()

close = data["close"]; volume = data["volume"]
df = data.join(fund_df, how="left").fillna(method="ffill", limit=3)

df["ma5"] = df["close"].rolling(5).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ret"] = df["close"].pct_change()

df["fund_score"] = 50.0
df["fund_score"] += df["main_force_net"].apply(lambda x: min(20, x/1e7) if x>0 else max(-20, x/1e7))
df["fund_score"] += (((df["main_force_net"]>0) & (df["dde_net"]>0)).astype(int)) * 15
df["fund_score"] += ((df["ddx"] > df["ddx"].shift(1)).astype(int)) * 8
df["fund_score"] += ((df["main_force_net"]>0).astype(int).rolling(3).sum().apply(lambda x: x*5 if x>=2 else 0))
df["fund_score"] = df["fund_score"].clip(0, 100)

df["tech_score"] = 50.0
df.loc[df["close"] > df["ma5"], "tech_score"] += 15
df.loc[df["close"] > df["ma20"], "tech_score"] += 15
df.loc[df["volume"] > df["volume"].rolling(5).mean(), "tech_score"] += 10
df["tech_score"] = df["tech_score"].clip(0, 100)

df["total_score"] = df["fund_score"] * 0.70 + df["tech_score"] * 0.30

df["position"] = 0.0
df.loc[df["total_score"] >= 65, "position"] = 1.0
df.loc[df["total_score"] <= 40, "position"] = 0.0
df.loc[df["main_force_net"].rolling(3).sum() < -3e8, "position"] = 0.0

df["strategy_ret"] = df["ret"] * df["position"].shift(1)
sc = (1 + df["strategy_ret"]).cumprod()
bc = (1 + df["ret"]).cumprod()

ts = float(sc.iloc[-1] - 1); tb = float(bc.iloc[-1] - 1)
ann = 252.0 / len(df)
ann_s = (1 + ts) ** ann - 1
dd = float(((sc - sc.cummax()) / sc.cummax()).min())
sharpe = float(np.sqrt(252) * df["strategy_ret"].mean() / (df["strategy_ret"].std() + 1e-10))
beta = float(df["strategy_ret"].cov(df["ret"]) / (df["ret"].var() + 1e-10))
alpha = float(ann_s - beta * ((1 + tb) ** ann - 1))
wr = float((df["strategy_ret"] > 0).sum() / ((df["strategy_ret"] != 0).sum() + 1e-10))

result = {"round": 1, "stock": stock}
result["strategy_pct"] = round(ts*100, 2)
result["benchmark_pct"] = round(tb*100, 2)
result["excess_pct"] = round((ts-tb)*100, 2)
result["max_dd_pct"] = round(dd*100, 2)
result["sharpe"] = sharpe
result["alpha_pct"] = round(alpha*100, 2)
result["beta"] = beta
result["win_rate_pct"] = round(wr*100, 2)
result["trades"] = int((df["position"].diff()!=0).sum())
result["days"] = len(df)
print("===RESULT===")
print(json.dumps(result))
print("===END===")
''' % csv_repr

# 检查
if 'f"' in CODE:
    print("ERROR: 代码包含f-string!")
    exit(1)
print(f"策略代码: {len(CODE)} 字符")

async def main():
    async with SuperMindSession() as sm:
        r = await sm.execute(CODE, timeout=300)
    print(r["stdout"])
    if r["error"]:
        print("ERRORS:\n" + r["error"][:1500])

asyncio.run(main())
