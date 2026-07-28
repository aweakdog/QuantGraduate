"""
R4 Adaptive — 按标的特性动态调参
根据市值自动适配参数：大盘股重事件+资金，小盘股重技术+趋势

用法:
  python strategies/r4_adaptive.py 300342.SZ
  python strategies/r4_adaptive.py 601689.SH
"""
import sys, json, asyncio, subprocess, re
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SMLOGIN = _PROJECT_ROOT / "engine" / "smlogin"
if str(_SMLOGIN) not in sys.path:
    sys.path.insert(0, str(_SMLOGIN))
from smlogin import SuperMindSession

STOCK = sys.argv[1] if len(sys.argv) > 1 else "601689.SH"
THS_PY = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe"

# 确定数据路径
if STOCK == "601689.SH":
    CSV_PATH = str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv")
elif STOCK == "300342.SZ":
    CSV_PATH = str(_PROJECT_ROOT / "data" / "300342.SZ_fund_flow_6m.csv")
else:
    CSV_PATH = str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv")

# ── 查市值 ──
r_mcap = subprocess.run([THS_PY, "-c",
    "from thsdk import THS; ths=THS(); ths.connect(); "
    "r=ths.wencai_nlp('%s 总市值'); "
    "print(r.df.iloc[0,0] if r.success and r.data is not None else '0')" % STOCK],
    capture_output=True, text=True, timeout=20)
mcap_str = r_mcap.stdout.strip().split("\n")[-1]
try:
    mcap = float(mcap_str)
except (ValueError, TypeError):
    mcap = 0.0
mcap_yi = mcap / 1e8

# 按市值分类
if mcap > 300e8:     # >300亿
    profile = "large"
elif mcap > 50e8:    # 50-300亿
    profile = "mid"
else:                 # <50亿
    profile = "small"

print("R4 Adaptive — 标的: %s, 市值: %.0f亿, 分类: %s" % (STOCK, mcap_yi, profile))

# ── 查概念标签 ──
r_conc = subprocess.run([THS_PY, "-c",
    "from thsdk import THS; ths=THS(); ths.connect(); "
    "r=ths.wencai_nlp('%s 所属概念'); "
    "if r.success and r.data is not None: print(str(r.df.iloc[0,1]))" % STOCK],
    capture_output=True, text=True, timeout=20)
concepts = r_conc.stdout.strip().split("\n")[-1] if r_conc.stdout.strip() else ""
print("概念: %s" % concepts[:80])

# 读CSV嵌入
r_csv = subprocess.run([THS_PY, "-c",
    "import pandas as pd; df=pd.read_csv(r'%s', index_col=0); print(repr(df.to_csv()))" % CSV_PATH],
    capture_output=True, text=True, timeout=30)
csv_repr = r_csv.stdout.strip()
print("CSV repr: %d chars" % len(csv_repr))

# ── 参数模板（按分类动态设置） ──
if profile == "large":
    params = {
        "gap_mult": 1.5, "vol_mult": 1.5, "range_mult": 1.5,
        "buy_thresh": 65, "sell_thresh": 40,
        "w_event": 0.40, "w_fund": 0.40, "w_tech": 0.20,
        "mf_exit_z": -1.5,
        "hard_stop_atr": 2.0, "trail_stop_atr": 1.5, "trail_activate": 0.03,
        "time_stop_days": 15,
        "event_gap_up": 20, "event_gap_dn": -25, "event_vol": 10, "event_range": 8,
    }
elif profile == "mid":
    params = {
        "gap_mult": 2.0, "vol_mult": 2.0, "range_mult": 2.0,
        "buy_thresh": 68, "sell_thresh": 38,
        "w_event": 0.30, "w_fund": 0.35, "w_tech": 0.35,
        "mf_exit_z": -2.0,
        "hard_stop_atr": 1.5, "trail_stop_atr": 1.2, "trail_activate": 0.04,
        "time_stop_days": 12,
        "event_gap_up": 15, "event_gap_dn": -20, "event_vol": 8, "event_range": 6,
    }
else:  # small
    params = {
        "gap_mult": 2.5, "vol_mult": 2.5, "range_mult": 2.5,
        "buy_thresh": 72, "sell_thresh": 35,
        "w_event": 0.20, "w_fund": 0.30, "w_tech": 0.50,
        "mf_exit_z": -2.5,
        "hard_stop_atr": 1.2, "trail_stop_atr": 1.0, "trail_activate": 0.05,
        "time_stop_days": 8,
        "event_gap_up": 10, "event_gap_dn": -15, "event_vol": 5, "event_range": 4,
    }

print("参数: buy=%d, w_event=%.0f%%, w_fund=%.0f%%, w_tech=%.0f%%, gap=%.1f-sigma"
      % (params["buy_thresh"], params["w_event"]*100, params["w_fund"]*100,
         params["w_tech"]*100, params["gap_mult"]))

# 构建策略代码
CODE = r'''
import pandas as pd, numpy as np, json
from datetime import datetime, date
from io import StringIO

stock = "@STOCK@"
today = date.today().isoformat().replace("-", "")

FUND_CSV = @CSV_REPR@
fund_df = pd.read_csv(StringIO(FUND_CSV), index_col=0, parse_dates=True)

# 动态参数（由外层按市值设定）
P = @PARAMS@
W = 20

print("=== R4 Adaptive ===")
print("标的: " + stock + ", 天数: " + str(len(fund_df)))
print("参数: " + json.dumps(P))

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

# 滚动参数
mf_mean = df["main_force_net"].rolling(W).mean()
mf_std = df["main_force_net"].rolling(W).std()
df["mf_z"] = (df["main_force_net"] - mf_mean) / (mf_std + 1)

# 事件信号: 标准差倍数 (按市值动态)
gap_std = df["gap_pct"].rolling(W).std()
df["gap_up"]   = (df["gap_pct"] >  P["gap_mult"] * gap_std).astype(int)
df["gap_down"] = (df["gap_pct"] < -P["gap_mult"] * gap_std).astype(int)

vol_mean = df["volume"].rolling(W).mean()
vol_std = df["volume"].rolling(W).std()
df["vol_z"] = (df["volume"] - vol_mean) / (vol_std + 1)
df["vol_surge"] = (df["vol_z"] > P["vol_mult"]).astype(int)

range_mean = df["range_pct"].rolling(W).mean()
range_std = df["range_pct"].rolling(W).std()
df["range_surge"] = (df["range_pct"] > range_mean + P["range_mult"] * range_std).astype(int)

# ATR(14)
df["atr14"] = df["range_pct"].rolling(14).mean()

# 事件评分 (权重按市值调整)
event_raw = 50.0
event_raw += df["gap_up"] * P["event_gap_up"]
event_raw -= df["gap_down"] * abs(P["event_gap_dn"])
event_raw += df["vol_surge"] * P["event_vol"]
event_raw += df["range_surge"] * P["event_range"]
df["event_score"] = event_raw.clip(0, 100)

# 资金评分
df["fund_score"] = 50.0
df["fund_score"] += df["mf_z"].clip(-2, 2) * 12.5
df["fund_score"] += (((df["main_force_net"]>0) & (df["dde_net"]>0)).astype(int)) * 10
df["fund_score"] += ((df["ddx"] > df["ddx"].shift(1)).astype(int)) * 5
df["fund_score"] += ((df["main_force_net"]>0).astype(int).rolling(3).sum().apply(lambda x: x*4 if x>=2 else 0))
df["fund_score"] = df["fund_score"].clip(0, 100)

# 技术评分
df["tech_score"] = 50.0
df.loc[df["close"] > df["ma5"], "tech_score"] += 10
df.loc[df["close"] > df["ma20"], "tech_score"] += 12
df.loc[df["close"] > df["ma60"], "tech_score"] += 13
df.loc[df["volume"] > df["volume"].rolling(5).mean(), "tech_score"] += 8
df["tech_score"] = df["tech_score"].clip(0, 100)

# 总分 (权重按市值动态)
df["total_score"] = df["event_score"] * P["w_event"] + df["fund_score"] * P["w_fund"] + df["tech_score"] * P["w_tech"]

# 仓位
df["position"] = 0.0
df.loc[df["total_score"] >= P["buy_thresh"], "position"] = 1.0
df.loc[df["total_score"] <= P["sell_thresh"], "position"] = 0.0
df.loc[df["mf_z"] < P["mf_exit_z"], "position"] = 0.0

# ATR动态止损
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
        stop1 = loss_entry < -P["hard_stop_atr"] * atr
        stop2 = loss_peak < -P["trail_stop_atr"] * atr and loss_entry > P["trail_activate"]
        stop3 = loss_entry < 0.005 and days_held > P["time_stop_days"]
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

result = {"stock": stock, "version": "R4_adaptive", "profile": "@PROFILE@",
          "mcap_yi": @MCAP_YI@}
for k, v in P.items():
    if isinstance(v, float):
        result["p_" + k] = round(v, 3)
    else:
        result["p_" + k] = v
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
'''.replace("@STOCK@", STOCK).replace("@CSV_REPR@", csv_repr)\
   .replace("@PARAMS@", json.dumps(params))\
   .replace("@PROFILE@", profile)\
   .replace("@MCAP_YI@", "%.0f" % mcap_yi)

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
