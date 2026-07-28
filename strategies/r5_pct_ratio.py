"""
R5 — 主力资金占比核心版
fund_score 基于 main_force_pct (主力净流入/成交额)
cross-market-cap 通用，不需要Z-score
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

if STOCK == "601689.SH":
    CSV_PATH = str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv")
elif STOCK == "300342.SZ":
    CSV_PATH = str(_PROJECT_ROOT / "data" / "300342.SZ_fund_flow_6m.csv")
else:
    CSV_PATH = str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv")

print("R5 main_force_pct — 标的: %s" % STOCK)

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

print("=== R5 主力占比核心版 ===")
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
df["range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
df["gap_pct"] = (df["open"] / df["close"].shift(1) - 1) * 100

W = 20
gap_std = df["gap_pct"].rolling(W).std()
vol_mean = df["volume"].rolling(W).mean(); vol_std = df["volume"].rolling(W).std()
range_mean = df["range_pct"].rolling(W).mean(); range_std = df["range_pct"].rolling(W).std()
vol_z = (df["volume"] - vol_mean) / (vol_std + 1)
df["atr14"] = df["range_pct"].rolling(14).mean()

# 事件信号
gap_up = (df["gap_pct"] > 1.5 * gap_std).astype(int)
gap_down = (df["gap_pct"] < -1.5 * gap_std).astype(int)
vol_surge = (vol_z > 1.5).astype(int)
range_surge = (df["range_pct"] > range_mean + 1.5 * range_std).astype(int)

event_score = 50.0
event_score += gap_up * 20; event_score -= gap_down * 25
event_score += vol_surge * 10; event_score += range_surge * 8
event_score = event_score.clip(0, 100)

# ── 资金评分: 基于 main_force_pct (主力占比 = 主力净流入/成交额) ──
# pct < |1%| → 噪音，不给分
# pct [-10, +10] → 分数 [-25, +25] 偏移
mfp = df["main_force_pct"].fillna(0)
fund_score = pd.Series(50.0, index=df.index)
# 只在pct绝对值≥1%时激活
active = mfp.abs() >= 1.0
fund_score[active] = fund_score[active] + mfp[active].clip(-10, 10) * 2.5
fund_score = fund_score.clip(0, 100)

# dde方向确认 (只在主力活跃时加分)
dde_confirm = (mfp > 0) & (df["dde_net"] > 0)
fund_score[dde_confirm & active] += 8
dde_deny = (mfp < 0) & (df["dde_net"] < 0)
fund_score[dde_deny & active] -= 8
fund_score = fund_score.clip(0, 100)

# 技术评分
tech_score = 50.0
tech_score += (df["close"] > df["ma5"]).astype(int) * 10
tech_score += (df["close"] > df["ma20"]).astype(int) * 12
tech_score += (df["close"] > df["ma60"]).astype(int) * 13
tech_score += (df["volume"] > df["volume"].rolling(5).mean()).astype(int) * 8
tech_score = tech_score.clip(0, 100)

# 总分: 事件35% + 资金35% + 技术30%
df["total_score"] = event_score * 0.35 + fund_score * 0.35 + tech_score * 0.30

# 仓位
df["position"] = 0.0
df.loc[df["total_score"] >= 65, "position"] = 1.0
df.loc[df["total_score"] <= 40, "position"] = 0.0
# 资金面滤网：主力占比<-3%强制平仓
df.loc[mfp < -3.0, "position"] = 0.0

# ATR止损
in_position = False; entry_p = 0.0; peak_p = 0.0; days_held = 0
for i in range(len(df)):
    if df["position"].iloc[i] == 1.0 and not in_position:
        in_position = True; entry_p = df["close"].iloc[i]; peak_p = entry_p; days_held = 0
    elif in_position:
        peak_p = max(peak_p, df["close"].iloc[i]); days_held += 1
        curr = df["close"].iloc[i]
        loss_entry = (curr - entry_p) / entry_p
        loss_peak = (curr - peak_p) / peak_p
        atr = df["atr14"].iloc[i] / 100.0
        if atr < 0.01: atr = 0.01
        stop1 = loss_entry < -2.0 * atr
        stop2 = loss_peak < -1.5 * atr and loss_entry > 0.03
        stop3 = loss_entry < 0.005 and days_held > 15
        if stop1 or stop2 or stop3:
            df.loc[df.index[i], "position"] = 0.0; in_position = False

# 回测
df["strategy_ret"] = df["ret"] * df["position"].shift(1)
sc = (1 + df["strategy_ret"]).cumprod(); bc = (1 + df["ret"]).cumprod()
ts = float(sc.iloc[-1] - 1); tb = float(bc.iloc[-1] - 1)
ann = 252.0 / len(df)
dd = float(((sc - sc.cummax()) / sc.cummax()).min())
sharpe = float(np.sqrt(252) * df["strategy_ret"].mean() / (df["strategy_ret"].std() + 1e-10))
beta = float(df["strategy_ret"].cov(df["ret"]) / (df["ret"].var() + 1e-10))
alpha = float(((1+ts)**ann-1) - beta * ((1+tb)**ann-1))
wr = float((df["strategy_ret"] > 0).sum() / ((df["strategy_ret"] != 0).sum() + 1e-10))

result = {"stock": stock, "version": "R5_pct_ratio",
          "weights": "event35_fund35_pct_tech30"}
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

if 'f"' in CODE: print("ERROR: f-string"); exit(1)
print("策略代码: %d 字符" % len(CODE))

async def main():
    async with SuperMindSession() as sm:
        r = await sm.execute(CODE, timeout=300)
    print(r["stdout"])
    if r["error"]: print("ERRORS:\n" + r["error"][:1500])

asyncio.run(main())
