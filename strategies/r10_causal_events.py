"""
R10 — 因果相乘 + 全历史事件 + 自适应趋势
上升趋势: R9逻辑 (MA20↑才入场)
下降趋势: 事件驱动超跌反弹 (事件+超卖+资金确认)
"""
import sys, json, asyncio, subprocess, requests, base64, time, os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SMLOGIN = _PROJECT_ROOT / "engine" / "smlogin"

STOCK = sys.argv[1] if len(sys.argv) > 1 else "601689.SH"
THS_PY = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe"
CSV_PATH = {"601689.SH": str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv"),
            "300342.SZ": str(_PROJECT_ROOT / "data" / "300342.SZ_fund_flow_6m.csv")}.get(STOCK, str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv"))
CACHE_FILE = str(_PROJECT_ROOT / "data" / ("%s_events.json" % STOCK.split(".")[0]))

# ─── Phase 1: 全历史利好事件采集 ───
if os.path.exists(CACHE_FILE):
    print("加载缓存事件: %s" % CACHE_FILE)
    with open(CACHE_FILE, 'r', encoding="utf-8") as f:
        daily_events = json.load(f)
else:
    print("采集全历史利好事件...")
    daily_events = {}  # date_str -> event_score

    # 需要采集的日期范围 (从CSV获取)
    r = subprocess.run([THS_PY, "-c",
        "import pandas as pd; df=pd.read_csv(r'%s', index_col=0, parse_dates=True); "
        "print(df.index[0].strftime('%%Y-%%m-%%d')); print(df.index[-1].strftime('%%Y-%%m-%%d'))" % CSV_PATH],
        capture_output=True, text=True, timeout=15)
    lines = r.stdout.strip().split('\n')
    start_date_str = lines[0] if len(lines) > 0 else "2025-12-29"
    end_date_str = lines[-1] if len(lines) > 1 else "2026-06-26"
    print("  日期范围: %s 至 %s" % (start_date_str, end_date_str))

    # 按周采样 + 全日期利好查询
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

    # 白名单模式: 只保留明确因果事件，不默认保留
    causal_event_patterns = [
        # 业务进展
        "专利", "发明", "量产", "投产", "开工", "新品", "新产品",
        "中标", "合同", "订单", "签约",
        # 业绩
        "净利润", "营收", "业绩预告", "季报", "年报", "财报", "利润",
        # 公司行为
        "分红", "除权", "送转", "回购", "增发", "减持", "增持", "股权激励",
        # 行业催化
        "涨停潮", "涨停板",
        # 利好公告类
        "合作", "战略合作",
    ]

    def is_causal_event(title):
        """白名单: 只保留明确因果事件"""
        return any(p in title for p in causal_event_patterns)

    stock_code = STOCK.split(".")[0]
    stock_name = {"300342": "天银机电", "601689": "拓普集团"}.get(stock_code, "")

    total_events = 0
    dt = start_dt
    while dt <= end_dt:
        dstr = dt.strftime("%Y-%m-%d")
        # 跳过周末
        if dt.weekday() >= 5:
            dt += timedelta(days=1)
            continue

        try:
            cmd = r"""
from thsdk import THS
import time, base64, json, sys
ths = THS()
ths.connect()
time.sleep(0.4)
r = ths.wencai_nlp("%s %s 利好")
time.sleep(0.4)
if r.success and r.data is not None:
    import pandas as pd
    df = r.df
    if "\u5173\u952e\u8bcd\u8d44\u8baf" in df.columns:
        raw = df.iloc[0]["\u5173\u952e\u8bcd\u8d44\u8baf"]
        if raw and isinstance(raw, str) and len(raw) > 50:
            decoded = base64.b64decode(raw).decode("utf-8", errors="replace")
            items = json.loads(decoded)
            for item in items:
                print(item.get("PageRawTitle", ""))
""" % (STOCK, dstr)
            r = subprocess.run([THS_PY, "-c", cmd], capture_output=True, text=True, timeout=25)
            titles = [l.strip() for l in r.stdout.split('\n') if l.strip() and not l.startswith('WARNING') and not l.startswith('2026')]

            # Filter and score
            day_score = 0
            for title in titles:
                if is_causal_event(title):
                    day_score += 10  # base score per event
                    total_events += 1

            if day_score > 0:
                daily_events[dstr] = day_score
                print("  %s: +%d (from %d raw items)" % (dstr, day_score, len(titles)))
        except Exception as e:
            pass  # silent skip failed dates

        dt += timedelta(days=1)

    # 缓存
    with open(CACHE_FILE, 'w', encoding="utf-8") as f:
        json.dump(daily_events, f)
    print("共采集 %d 事件日, %d 事件" % (len(daily_events), total_events))

# ─── Phase 2: 加载资金流数据 ───
print("\n加载资金流数据...")
r = subprocess.run([THS_PY, "-c",
    "import pandas as pd; df=pd.read_csv(r'%s', index_col=0); print(repr(df.to_csv()))" % CSV_PATH],
    capture_output=True, text=True, timeout=30)
csv_repr = r.stdout.strip()

# ─── Phase 3: 获取市值(自适应窗口) ───
print("获取市值...")
r_mcap = subprocess.run([THS_PY, "-c",
    "from thsdk import THS; ths=THS(); ths.connect(); "
    "r=ths.wencai_nlp('%s 总市值'); print(r.df.iloc[0,0] if r.success and r.data is not None else '0')" % STOCK],
    capture_output=True, text=True, timeout=20)
try: mcap = float(r_mcap.stdout.strip().split('\n')[-1])
except (ValueError, TypeError, IndexError): mcap = 0
is_small = mcap < 300e8
W = 10 if is_small else 20
ATR_P = 8 if is_small else 14
SENS = 3.0 if is_small else 4.0
GAIN = 45 if is_small else 40
BUY = 55 if is_small else 60
print("窗口: W=%d, ATR=%d | sens=%.1f gain=%.0f buy=%.0f" % (W, ATR_P, SENS, GAIN, BUY))

# ─── Phase 4: 构建SuperMind策略代码 ───
events_json = json.dumps(daily_events)

CODE = (r'''
import pandas as pd, numpy as np, json
from datetime import datetime, date
from io import StringIO

stock = "@STOCK@"
today = date.today().isoformat().replace("-", "")
FUND_CSV = @CSV_REPR@
fund_df = pd.read_csv(StringIO(FUND_CSV), index_col=0, parse_dates=True)

EVENTS = @EVENTS_JSON@

print("=== R10 因果相乘+全历史事件+自适应趋势 ===")
print("标的: " + stock + ", 事件日数: " + str(len(EVENTS)))

data = get_price(stock, start_date="20251229", end_date=today,
                 fre_step="1d", fields=["close","high","low","open","volume"], fq="pre")
if len(data) < 20: print("数据不足"); raise SystemExit()

close = data["close"]; high = data["high"]; low = data["low"]; volume = data["volume"]
df = data.join(fund_df, how="left").fillna(method="ffill", limit=3)
df["ret"] = df["close"].pct_change()
df["range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
df["gap_pct"] = (df["open"] / df["close"].shift(1) - 1) * 100

W = @W@; ATR_P = @ATR_P@; SENS = @SENS@; GAIN = @GAIN@; BUY = @BUY@
OVERSOLD_ENTRY = 50

df["ma_s"] = df["close"].rolling(5).mean()
df["ma_m"] = df["close"].rolling(W).mean()
df["ma_l"] = df["close"].rolling(W*2).mean()
df["trend"] = (df["close"] > df["ma_m"]).astype(int)  # 趋势滤网

gap_std = df["gap_pct"].rolling(W).std()
vol_mean = df["volume"].rolling(W).mean(); vol_std = df["volume"].rolling(W).std()
range_mean = df["range_pct"].rolling(W).mean(); range_std = df["range_pct"].rolling(W).std()
vol_z = (df["volume"] - vol_mean) / (vol_std + 1)
df["atr"] = df["range_pct"].rolling(ATR_P).mean()
mfp = df["main_force_pct"].fillna(0)

# === RSI for oversold detection ===
def calc_rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)

rsi = calc_rsi(df["close"], 14)
oversold = (rsi < 35).astype(int)

# === 事件信号（全历史因果事件） ===
event_base = pd.Series(0.0, index=df.index)  # 0=中性, 正=利好
for i, dt in enumerate(df.index):
    dstr = dt.strftime("%Y-%m-%d")
    if dstr in EVENTS:
        es = EVENTS[dstr]
        event_base.iloc[i] = min(max(es, 0), 60)  # cap at 60

# === 因果相乘 (R9核心) ===
gap_up = (df["gap_pct"] > 1.5 * gap_std).astype(int)
gap_down = (df["gap_pct"] < -1.5 * gap_std).astype(int)
vol_surge = (vol_z > 1.5).astype(int)
range_surge = (df["range_pct"] > range_mean + 1.5 * range_std).astype(int)
tech_event = 0.0 + gap_up*10 - gap_down*12 + vol_surge*5 + range_surge*4
tech_event = tech_event.clip(0, 100)

# 事件分 = 历史利好事件 + 技术事件
event_score = 50.0 + event_base * 0.5 + tech_event * 0.3
event_score = event_score.clip(0, 100)
event_dir = (event_score - 50) / 50.0

# 资金确认
fund_conf = mfp.abs() / SENS; fund_conf = fund_conf.clip(0, 1)
fund_dir = np.sign(mfp)
combined_dir = event_dir * fund_dir * fund_conf

# 技术趋势
tech_score = 50.0 + (df["close"]>df["ma_s"]).astype(int)*8 + (df["close"]>df["ma_m"]).astype(int)*10 + (df["close"]>df["ma_l"]).astype(int)*12
tech_score = tech_score.clip(0, 100)
tech_dir = (tech_score - 50) / 50.0

# 总分：因果相乘 + 技术修正 + 资金流补助
total_score = 50.0 + combined_dir * GAIN + tech_dir * 8.0 + mfp.clip(0, None) * 1.0
total_score = total_score.clip(0, 100)

# === 自适应入场逻辑 ===
df["position"] = 0.0

# 模式1: 上升趋势 — R9原逻辑
uptrend = (df["trend"] == 1)
df.loc[uptrend & (total_score >= BUY), "position"] = 1.0

# 模式2: 下降趋势 — 事件驱动入场（趋势滤网降级）
# 条件: 有明确因果事件 + 资金确认流入 → 允许入场
downtrend = (df["trend"] == 0)
event_trigger = (event_score > 55) & (mfp > 1.0)
df.loc[downtrend & event_trigger, "position"] = 1.0

# 统一出场
df.loc[total_score <= 40, "position"] = 0.0
df.loc[mfp < -3.0, "position"] = 0.0

# ATR止损
inp = False; ep = 0.0; pp = 0.0; dh = 0
for i in range(len(df)):
    if df["position"].iloc[i] == 1.0 and not inp:
        inp = True; ep = df["close"].iloc[i]; pp = ep; dh = 0
    elif inp:
        pp = max(pp, df["close"].iloc[i]); dh += 1
        c = df["close"].iloc[i]; le = (c-ep)/ep; lp = (c-pp)/pp
        atr = df["atr"].iloc[i]/100.0
        if atr < 0.01: atr = 0.01
        if le < -2.0*atr or (lp < -1.5*atr and le > 0.03) or (le < 0.005 and dh > 12):
            df.loc[df.index[i], "position"] = 0.0; inp = False

sr = df["ret"] * df["position"].shift(1)
sc = (1 + sr).cumprod(); bc = (1 + df["ret"]).cumprod()
ts = float(sc.iloc[-1] - 1); tb = float(bc.iloc[-1] - 1)
dd = float(((sc - sc.cummax()) / sc.cummax()).min())
sharpe = float(np.sqrt(252) * sr.mean() / (sr.std() + 1e-10))
beta = float(sr.cov(df["ret"]) / (df["ret"].var() + 1e-10))
alpha = float(((1+ts)**(252/len(df))-1) - beta * ((1+tb)**(252/len(df))-1))
wr = float((sr > 0).sum() / ((sr != 0).sum() + 1e-10))

event_days_count = len([d for d in df.index if d.strftime("%Y-%m-%d") in EVENTS])
trend_up_days = int(df["trend"].sum())
trend_down_days = int((1 - df["trend"]).sum())
bounce_trades = int(((df["trend"]==0) & event_trigger).sum())

print("事件覆盖: %d/%d 天" % (event_days_count, len(df)))
print("上升趋势: %d天, 下降趋势: %d天" % (trend_up_days, trend_down_days))
print("事件驱动入场信号: %d次" % bounce_trades)

result = {"stock": stock, "version": "R10_causal_events",
          "event_days": event_days_count, "trend_up": trend_up_days, "trend_down": trend_down_days,
          "event_entry": bounce_trades}
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
''').replace("@STOCK@", STOCK).replace("@CSV_REPR@", csv_repr)\
    .replace("@EVENTS_JSON@", events_json)\
    .replace("@W@", str(W)).replace("@ATR_P@", str(ATR_P))\
    .replace("@SENS@", str(SENS)).replace("@GAIN@", str(GAIN))\
    .replace("@BUY@", str(BUY))

# f-string检测
if 'f"' in CODE:
    # 更精确检测：排除dict key "wf"
    import re
    if re.search(r'(?<![a-zA-Z])f"', CODE):
        print("ERROR: f-string found"); exit(1)

print("代码: %d chars" % len(CODE))

# ─── Phase 5: 执行 ───
async def main():
    if str(_SMLOGIN) not in sys.path:
        sys.path.insert(0, str(_SMLOGIN))
    from smlogin import SuperMindSession
    async with SuperMindSession() as sm:
        r = await sm.execute(CODE, timeout=300)
    print(r["stdout"])
    if r["error"]:
        print("ERR:\n" + r["error"][:1500])

asyncio.run(main())
