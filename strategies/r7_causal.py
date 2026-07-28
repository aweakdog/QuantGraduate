"""
R7 — 事件×资金 因果相乘 + 小盘自适应窗口
核心理念：
- main_force_pct 归一化主力占比
- 事件×资金 = 原因×验证（无资金配合的事件=半个信号）
- 小盘股窗口自适应缩短
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

CSV_PATH = {"601689.SH": str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv"),
            "300342.SZ": str(_PROJECT_ROOT / "data" / "300342.SZ_fund_flow_6m.csv")}.get(STOCK, str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv"))

print("R7 — 因果相乘 标的: %s" % STOCK)

r = subprocess.run([THS_PY, "-c",
    "import pandas as pd; df=pd.read_csv(r'%s', index_col=0); print(repr(df.to_csv()))" % CSV_PATH],
    capture_output=True, text=True, timeout=30)
csv_repr = r.stdout.strip()

CODE = r'''
import pandas as pd, numpy as np, json
from datetime import datetime, date
from io import StringIO

stock = "@STOCK@"
today = date.today().isoformat().replace("-", "")
FUND_CSV = @CSV_REPR@
fund_df = pd.read_csv(StringIO(FUND_CSV), index_col=0, parse_dates=True)

print("=== R7 因果相乘 ===")
data = get_price(stock, start_date="20251229", end_date=today,
                 fre_step="1d", fields=["close","high","low","open","volume"], fq="pre")
if len(data) < 20: print("数据不足"); raise SystemExit()

close = data["close"]; high = data["high"]; low = data["low"]; volume = data["volume"]
df = data.join(fund_df, how="left").fillna(method="ffill", limit=3)
df["ret"] = df["close"].pct_change()
df["range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
df["gap_pct"] = (df["open"] / df["close"].shift(1) - 1) * 100

# ── 小盘自适应窗口 ──
# 用 range_pct 的波动率判断股性活跃度
vol_regime = df["range_pct"].rolling(60).std().median()
if vol_regime > 3.0:       # 高波动 = 小盘活跃股
    W = 10; ATR_P = 8; MA_S = 5; MA_M = 10; MA_L = 20
else:                      # 低波动 = 大盘稳健股
    W = 20; ATR_P = 14; MA_S = 5; MA_M = 20; MA_L = 60

print("窗口: W=%d, ATR=%d, MA=(%d,%d,%d)" % (W, ATR_P, MA_S, MA_M, MA_L))

df["ma_s"] = df["close"].rolling(MA_S).mean()
df["ma_m"] = df["close"].rolling(MA_M).mean()
df["ma_l"] = df["close"].rolling(MA_L).mean()

# 滚动统计（自适应窗口）
gap_std = df["gap_pct"].rolling(W).std()
vol_mean = df["volume"].rolling(W).mean(); vol_std = df["volume"].rolling(W).std()
range_mean = df["range_pct"].rolling(W).mean(); range_std = df["range_pct"].rolling(W).std()
vol_z = (df["volume"] - vol_mean) / (vol_std + 1)
df["atr"] = df["range_pct"].rolling(ATR_P).mean()

# ── 事件信号 (因) ──
gap_up = (df["gap_pct"] > 1.5 * gap_std).astype(int)
gap_down = (df["gap_pct"] < -1.5 * gap_std).astype(int)
vol_surge = (vol_z > 1.5).astype(int)
range_surge = (df["range_pct"] > range_mean + 1.5 * range_std).astype(int)

event_raw = 50.0
event_raw += gap_up * 18; event_raw -= gap_down * 22
event_raw += vol_surge * 10; event_raw += range_surge * 8
event_score = event_raw.clip(0, 100)

# event_direction: -1 (利空) 到 +1 (利好)
event_dir = (event_score - 50) / 50.0

# ── 主力资金确认 (果) ──
mfp = df["main_force_pct"].fillna(0)
# 归一化：pct本身已是主力净流入/成交额
# fund_confidence: 0(无主力) 到 1(强主力)
fund_conf = mfp.abs() / 5.0
fund_conf = fund_conf.clip(0, 1)
# fund_direction: -1(流出) 0(中性) +1(流入)
fund_dir = np.sign(mfp)

# ── 乘法交互: 事件 × 资金确认 ──
# 事件方向 × 资金方向 × 资金置信度
# 同向加强，反向减弱，无资金=事件无效
combined_dir = event_dir * fund_dir * fund_conf

# ── 技术趋势 (辅助) ──
tech_score = 50.0
tech_score += (df["close"] > df["ma_s"]).astype(int) * 8
tech_score += (df["close"] > df["ma_m"]).astype(int) * 10
tech_score += (df["close"] > df["ma_l"]).astype(int) * 12
tech_score = tech_score.clip(0, 100)
tech_dir = (tech_score - 50) / 50.0
# 技术权重要低，只是辅助
tech_weight = 0.15

# ── 总分 ──
# combined_dir: -1 到 +1
# 映射到 0-100 分数
total_score = 50.0 + combined_dir * 35.0 + tech_dir * 35.0 * tech_weight
total_score = total_score.clip(0, 100)

# 仓位
df["position"] = 0.0
df.loc[total_score >= 60, "position"] = 1.0
df.loc[total_score <= 40, "position"] = 0.0
df.loc[mfp < -3.0, "position"] = 0.0  # 主力大幅流出强制平仓

# ATR止损
inp = False; ep = 0.0; pp = 0.0; dh = 0
for i in range(len(df)):
    if df["position"].iloc[i] == 1.0 and not inp:
        inp = True; ep = df["close"].iloc[i]; pp = ep; dh = 0
    elif inp:
        pp = max(pp, df["close"].iloc[i]); dh += 1
        c = df["close"].iloc[i]; le = (c - ep) / ep; lp = (c - pp) / pp
        atr = df["atr"].iloc[i] / 100.0
        if atr < 0.01: atr = 0.01
        stop1 = le < -2.0 * atr
        stop2 = lp < -1.5 * atr and le > 0.03
        stop3 = le < 0.005 and dh > 12
        if stop1 or stop2 or stop3:
            df.loc[df.index[i], "position"] = 0.0; inp = False

# 回测
df["sr"] = df["ret"] * df["position"].shift(1)
sc = (1 + df["sr"]).cumprod(); bc = (1 + df["ret"]).cumprod()
ts = float(sc.iloc[-1] - 1); tb = float(bc.iloc[-1] - 1)
dd = float(((sc - sc.cummax()) / sc.cummax()).min())
sharpe = float(np.sqrt(252) * df["sr"].mean() / (df["sr"].std() + 1e-10))
beta = float(df["sr"].cov(df["ret"]) / (df["ret"].var() + 1e-10))
alpha = float(((1+ts)**(252/len(df))-1) - beta * ((1+tb)**(252/len(df))-1))
wr = float((df["sr"] > 0).sum() / ((df["sr"] != 0).sum() + 1e-10))

print("波动率评估: range_pct_60d_median=%.2f" % vol_regime)
result = {"stock": stock, "version": "R7_causal_multiply",
          "windows": "W%d_ATR%d_MA(%d,%d,%d)" % (W, ATR_P, MA_S, MA_M, MA_L)}
result["strategy_pct"] = round(ts*100, 2)
result["benchmark_pct"] = round(tb*100, 2)
result["excess_pct"] = round((ts-tb)*100, 2)
result["max_dd_pct"] = round(dd*100, 2)
result["sharpe"] = round(sharpe, 3)
result["alpha_pct"] = round(alpha*100, 2)
result["beta"] = round(beta, 3)
result["win_rate_pct"] = round(wr*100, 2)
result["trades"] = int((df["position"].diff()!=0).sum())
print("===RESULT===")
print(json.dumps(result))
print("===END===")
'''.replace("@STOCK@", STOCK).replace("@CSV_REPR@", csv_repr)

import re
if re.search(r'(?<![a-zA-Z0-9_.%])f"', CODE):
    print("ERROR: f-string"); exit(1)

async def main():
    async with SuperMindSession() as sm:
        r = await sm.execute(CODE, timeout=300)
    print(r["stdout"])
    if r["error"]: print("ERRORS:\n" + r["error"][:1500])

asyncio.run(main())
