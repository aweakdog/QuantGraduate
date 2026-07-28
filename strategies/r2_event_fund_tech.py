"""
Round 2 — 事件代理 + 多级止损（无%格式冲突）
"""
import sys, json, asyncio, subprocess
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SMLOGIN = _PROJECT_ROOT / "engine" / "smlogin"
if str(_SMLOGIN) not in sys.path:
    sys.path.insert(0, str(_SMLOGIN))
from smlogin import SuperMindSession

THS_PY = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe"
CSV_PATH = str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv")
r = subprocess.run([THS_PY, "-c",
    "import pandas as pd; df=pd.read_csv(r'%s', index_col=0); print(repr(df.to_csv()))" % CSV_PATH],
    capture_output=True, text=True, timeout=30)
csv_repr = r.stdout.strip()

CODE_TEMPLATE = r'''
import pandas as pd, numpy as np, json
from datetime import datetime, date
from io import StringIO

stock = "601689.SH"
today = date.today().isoformat().replace("-", "")

FUND_CSV = @CSV@
fund_df = pd.read_csv(StringIO(FUND_CSV), index_col=0, parse_dates=True)

print("=== ROUND 2: 事件代理 + 多级止损 ===")
print("标的: " + stock + ", 天数: " + str(len(fund_df)))

data = get_price(stock, start_date="20251229", end_date=today,
                 fre_step="1d", fields=["close","high","low","open","volume"], fq="pre")
if len(data) < 20: print("数据不足"); raise SystemExit()

close = data["close"]; high = data["high"]; low = data["low"]; volume = data["volume"]
df = data.join(fund_df, how="left").fillna(method="ffill", limit=3)

df["ma5"] = df["close"].rolling(5).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma60"] = df["close"].rolling(60).mean()
df["ret"] = df["close"].pct_change()

# ── 事件代理信号 ──
df["gap"] = (df["open"] / df["close"].shift(1) - 1) * 100
df["gap_up"] = (df["gap"] > 1.5).astype(int)
df["gap_down"] = (df["gap"] < -1.5).astype(int)
df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
df["vol_surge"] = (df["vol_ratio"] > 2.0).astype(int)
df["range"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
df["range_surge"] = (df["range"] > df["range"].rolling(20).mean() * 2).astype(int)

df["event_score"] = 50.0
df["event_score"] += df["gap_up"] * 20
df["event_score"] -= df["gap_down"] * 25
df["event_score"] += df["vol_surge"] * 10
df["event_score"] += df["range_surge"] * 8
df["event_score"] = df["event_score"].clip(0, 100)

# ── 资金面评分 ──
df["fund_score"] = 50.0
df["fund_score"] += df["main_force_net"].apply(lambda x: min(20, x/1e7) if x>0 else max(-20, x/1e7))
df["fund_score"] += (((df["main_force_net"]>0) & (df["dde_net"]>0)).astype(int)) * 15
df["fund_score"] += ((df["ddx"] > df["ddx"].shift(1)).astype(int)) * 8
df["fund_score"] += ((df["main_force_net"]>0).astype(int).rolling(3).sum().apply(lambda x: x*5 if x>=2 else 0))
df["fund_score"] = df["fund_score"].clip(0, 100)

# ── 技术面评分 ──
df["tech_score"] = 50.0
df.loc[df["close"] > df["ma5"], "tech_score"] += 10
df.loc[df["close"] > df["ma20"], "tech_score"] += 10
df.loc[df["close"] > df["ma60"], "tech_score"] += 10
df.loc[df["volume"] > df["volume"].rolling(5).mean(), "tech_score"] += 8
df["tech_score"] = df["tech_score"].clip(0, 100)

# ── 总分: 事件40% + 资金面40% + 技术面20% ──
df["total_score"] = df["event_score"] * 0.40 + df["fund_score"] * 0.40 + df["tech_score"] * 0.20

# ── 仓位信号 ──
df["position"] = 0.0
df.loc[df["total_score"] >= 65, "position"] = 1.0
df.loc[df["total_score"] <= 40, "position"] = 0.0
df.loc[df["main_force_net"].rolling(3).sum() < -3e8, "position"] = 0.0

# ── 多级止损 ──
in_position = False
entry_p = 0.0
peak_p = 0.0
days_held = 0

for i in range(len(df)):
    if df["position"].iloc[i] == 1.0 and not in_position:
        in_position = True; entry_p = df["close"].iloc[i]; peak_p = entry_p; days_held = 0
    elif in_position:
        peak_p = max(peak_p, df["close"].iloc[i])
        days_held += 1
        curr = df["close"].iloc[i]
        loss_entry = (curr - entry_p) / entry_p
        loss_peak = (curr - peak_p) / peak_p
        # 规则1: 硬止损 -6%
        # 规则2: 移动止盈 从峰值回撤-5%(盈利超2%后启用)
        # 规则3: 时间止损 持仓12天不赚钱
        stop1 = loss_entry < -0.06
        stop2 = loss_peak < -0.05 and loss_entry > 0.02
        stop3 = loss_entry < 0.005 and days_held > 12
        if stop1 or stop2 or stop3:
            df.loc[df.index[i], "position"] = 0.0
            in_position = False

# ── 回测 ──
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

result = {"round": 2, "stock": stock, "weights": "event40_fund40_tech20"}
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
'''.replace("@CSV@", csv_repr)

if 'f"' in CODE_TEMPLATE:
    print("ERROR: f-string found!"); exit(1)
print(f"策略代码: {len(CODE_TEMPLATE)} 字符")

async def main():
    async with SuperMindSession() as sm:
        r = await sm.execute(CODE_TEMPLATE, timeout=300)
    print(r["stdout"])
    if r["error"]:
        print("ERRORS:\n" + r["error"][:1500])

asyncio.run(main())
