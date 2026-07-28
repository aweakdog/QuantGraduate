"""
R3 Generalized — 通用化策略：动态阈值 + Z-score 资金面 + ATR 止损
自适应不同市值/波动率的股票，一套参数跑多股

用法:
  python strategies/r3_generalized.py 300342.SZ
  python strategies/r3_generalized.py 601689.SH
"""
import sys, json, asyncio, subprocess
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SMLOGIN = _PROJECT_ROOT / "engine" / "smlogin"
if str(_SMLOGIN) not in sys.path:
    sys.path.insert(0, str(_SMLOGIN))
from smlogin import SuperMindSession

STOCK = sys.argv[1] if len(sys.argv) > 1 else "601689.SH"
THS_PY = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe"
THS_VENV = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths"

# 根据股票代码确定数据文件路径
if STOCK == "601689.SH":
    CSV_PATH = str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv")
elif STOCK == "300342.SZ":
    CSV_PATH = str(_PROJECT_ROOT / "data" / "300342.SZ_fund_flow_6m.csv")
else:
    CSV_PATH = str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv")

print("R3 Generalized — 标的: %s" % STOCK)
print("数据: %s" % CSV_PATH)

# 用 ths venv 读 CSV 生成 repr
r = subprocess.run([THS_PY, "-c",
    "import pandas as pd; df=pd.read_csv(r'%s', index_col=0); print(repr(df.to_csv()))" % CSV_PATH],
    capture_output=True, text=True, timeout=30)
csv_repr = r.stdout.strip()
print("CSV repr: %d chars" % len(csv_repr))

CODE = r'''
import pandas as pd, numpy as np, json
from datetime import datetime, date
from io import StringIO

stock = "@STOCK@"
today = date.today().isoformat().replace("-", "")

FUND_CSV = @CSV_REPR@
fund_df = pd.read_csv(StringIO(FUND_CSV), index_col=0, parse_dates=True)

print("=== R3 通用化策略 ===")
print("标的: " + stock + ", 天数: " + str(len(fund_df)))

data = get_price(stock, start_date="20251229", end_date=today,
                 fre_step="1d", fields=["close","high","low","open","volume"], fq="pre")
if len(data) < 20:
    print("数据不足"); raise SystemExit()

close = data["close"]; high = data["high"]; low = data["low"]; volume = data["volume"]
df = data.join(fund_df, how="left").fillna(method="ffill", limit=3)

df["ma5"] = df["close"].rolling(5).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma60"] = df["close"].rolling(60).mean()
df["ret"] = df["close"].pct_change()
df["range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
df["gap_pct"] = (df["open"] / df["close"].shift(1) - 1) * 100

# 滚动参数 (20天窗口自适应)
W = 20

# 资金面 Z-score (替代固定阈值)
mf_mean = df["main_force_net"].rolling(W).mean()
mf_std = df["main_force_net"].rolling(W).std()
df["mf_z"] = (df["main_force_net"] - mf_mean) / (mf_std + 1)

# 事件信号: 标准差倍数阈值 (替代固定 %)
gap_std = df["gap_pct"].rolling(W).std()
df["gap_up"] = (df["gap_pct"] > 1.5 * gap_std).astype(int)
df["gap_down"] = (df["gap_pct"] < -1.5 * gap_std).astype(int)

vol_mean = df["volume"].rolling(W).mean()
vol_std = df["volume"].rolling(W).std()
df["vol_z"] = (df["volume"] - vol_mean) / (vol_std + 1)
df["vol_surge"] = (df["vol_z"] > 1.5).astype(int)

range_mean = df["range_pct"].rolling(W).mean()
range_std = df["range_pct"].rolling(W).std()
df["range_surge"] = (df["range_pct"] > range_mean + 1.5 * range_std).astype(int)

# ATR(14) 用于动态止损
df["atr14"] = df["range_pct"].rolling(14).mean()

# 事件代理评分 (40%)
df["event_score"] = 50.0
df["event_score"] += df["gap_up"] * 20
df["event_score"] -= df["gap_down"] * 25
df["event_score"] += df["vol_surge"] * 10
df["event_score"] += df["range_surge"] * 8
df["event_score"] = df["event_score"].clip(0, 100)

# 资金面评分 (40%) — Z-score 自适应
df["fund_score"] = 50.0
df["fund_score"] += df["mf_z"].clip(-2, 2) * 12.5
df["fund_score"] += (((df["main_force_net"]>0) & (df["dde_net"]>0)).astype(int)) * 10
df["fund_score"] += ((df["ddx"] > df["ddx"].shift(1)).astype(int)) * 5
df["fund_score"] += ((df["main_force_net"]>0).astype(int).rolling(3).sum().apply(lambda x: x*4 if x>=2 else 0))
df["fund_score"] = df["fund_score"].clip(0, 100)

# 技术面评分 (20%)
df["tech_score"] = 50.0
df.loc[df["close"] > df["ma5"], "tech_score"] += 10
df.loc[df["close"] > df["ma20"], "tech_score"] += 10
df.loc[df["close"] > df["ma60"], "tech_score"] += 10
df.loc[df["volume"] > df["volume"].rolling(5).mean(), "tech_score"] += 8
df["tech_score"] = df["tech_score"].clip(0, 100)

# 总分
df["total_score"] = df["event_score"] * 0.40 + df["fund_score"] * 0.40 + df["tech_score"] * 0.20

# 仓位信号
df["position"] = 0.0
df.loc[df["total_score"] >= 65, "position"] = 1.0
df.loc[df["total_score"] <= 40, "position"] = 0.0
# Z-score 资金面强制平仓 (替代固定 -3e8)
df.loc[df["mf_z"] < -1.5, "position"] = 0.0

# ATR 动态止损
in_position = False
entry_p = 0.0
peak_p = 0.0
days_held = 0

for i in range(len(df)):
    if df["position"].iloc[i] == 1.0 and not in_position:
        in_position = True
        entry_p = df["close"].iloc[i]
        peak_p = entry_p
        days_held = 0
    elif in_position:
        peak_p = max(peak_p, df["close"].iloc[i])
        days_held += 1
        curr = df["close"].iloc[i]
        loss_entry = (curr - entry_p) / entry_p
        loss_peak = (curr - peak_p) / peak_p
        atr = df["atr14"].iloc[i] / 100.0
        if atr < 0.01: atr = 0.01
        stop1 = loss_entry < -2.0 * atr
        stop2 = loss_peak < -1.5 * atr and loss_entry > 0.03
        stop3 = loss_entry < 0.005 and days_held > 15
        if stop1 or stop2 or stop3:
            df.loc[df.index[i], "position"] = 0.0
            in_position = False

# 回测
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

result = {"stock": stock, "version": "R3_generalized",
          "weights": "event40_fund40_tech20",
          "adaptive": "zscore_mf+1.5std_events+ATR_stop"}
result["strategy_pct"] = round(ts*100, 2)
result["benchmark_pct"] = round(tb*100, 2)
result["excess_pct"] = round((ts-tb)*100, 2)
result["max_dd_pct"] = round(dd*100, 2)
result["sharpe"] = round(sharpe, 3)
result["alpha_pct"] = round(alpha*100, 2)
result["beta"] = round(beta, 3)
result["win_rate_pct"] = round(wr*100, 2)
result["trades"] = int((df["position"].diff()!=0).sum())
result["days"] = len(df)
print("===RESULT===")
print(json.dumps(result))
print("===END===")
'''.replace("@STOCK@", STOCK).replace("@CSV_REPR@", csv_repr)

if 'f"' in CODE:
    print("ERROR: f-string found!"); exit(1)
print("策略代码: %d 字符" % len(CODE))

async def main():
    async with SuperMindSession() as sm:
        r = await sm.execute(CODE, timeout=300)
    print(r["stdout"])
    if r["error"]:
        print("ERRORS:\n" + r["error"][:1500])

asyncio.run(main())
