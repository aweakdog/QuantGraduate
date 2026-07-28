"""
R9 — 因果相乘 + 趋势滤网
当价格在MA20上方才允许开仓（顺势），防止逆势交易
"""
import sys, json, asyncio, subprocess, requests
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
            "300342.SZ": str(_PROJECT_ROOT / "data" / "300342.SZ_fund_flow_6m.csv"),
            "601857.SH": str(_PROJECT_ROOT / "data" / "601857.SH_fund_flow_6m.csv")}.get(STOCK, str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv"))

r = subprocess.run([THS_PY, "-c",
    "import pandas as pd; df=pd.read_csv(r'%s', index_col=0); print(repr(df.to_csv()))" % CSV_PATH],
    capture_output=True, text=True, timeout=30)
csv_repr = r.stdout.strip()

# 拉大量7x24快讯覆盖回测期
print("拉取7x24快讯..."); all_news = []; headers = {'User-Agent': 'Mozilla/5.0'}
for page in range(1, 31):
    try:
        url = 'https://news.10jqka.com.cn/tapp/news/push/stock/?page={}&tag=A%E8%82%A1&track=website&pagesize=30'.format(page)
        r = requests.get(url, headers=headers, timeout=10)
        items = r.json()['data']['list']
        if not items: break
        all_news.extend(items)
    except Exception as e:
        print("  快讯API page %d error: %s" % (page, str(e)[:80]))
        break
print("共 %d 条快讯" % len(all_news))

# 查概念
r_conc = subprocess.run([THS_PY, "-c",
    "from thsdk import THS; ths=THS(); ths.connect(); "
    "r=ths.wencai_nlp('%s 所属概念');\n"
    "if r.success and r.data is not None:\n"
    "  print(str(r.df.iloc[0,0]))" % STOCK],
    capture_output=True, text=True, timeout=20)
concepts = [c.strip() for c in r_conc.stdout.strip().split("\n")[-1].split(";") if c.strip()]
print("概念: %s" % concepts)

stock_code = STOCK.split(".")[0]
stock_name = {"300342": "天银机电", "601689": "拓普集团", "601857": "中国石油"}.get(stock_code, "")

# 事件评分
daily_events = {}
for news in all_news:
    score = 0
    title = news.get("title", "")
    if stock_name and stock_name in title:
        score += 30
    for s in news.get("stock", []):
        if s.get("stockCode", "") == stock_code:
            score += 40
    for tag in news.get("tagInfo", []):
        if tag.get("name", "") in concepts:
            score += int(20 * float(tag.get("score", 0.5)))
    if score > 0:
        day = news.get("ctime", "0")
        day_date = __import__('datetime').datetime.fromtimestamp(int(day)).strftime("%Y-%m-%d") if day != "0" else ""
        if day_date:
            daily_events[day_date] = daily_events.get(day_date, 0) + score

print("有事件的交易日: %d" % len(daily_events))

events_json = json.dumps(daily_events)

# 自适应窗口
r_mcap = subprocess.run([THS_PY, "-c",
    "from thsdk import THS; ths=THS(); ths.connect(); "
    "r=ths.wencai_nlp('%s 总市值'); print(r.df.iloc[0,0] if r.success and r.data is not None else '0')" % STOCK],
    capture_output=True, text=True, timeout=20)
try: mcap = float(r_mcap.stdout.strip().split("\n")[-1])
except (ValueError, TypeError, IndexError): mcap = 0
is_small = mcap < 300e8
W = 10 if is_small else 20
ATR_P = 8 if is_small else 14
SENS = 3.0 if is_small else 4.0
GAIN = 45 if is_small else 40
BUY = 55 if is_small else 60

print("窗口: W=%d, ATR=%d | sens=%.1f gain=%.0f buy=%.0f" % (W, ATR_P, SENS, GAIN, BUY))

CODE = (r'''
import pandas as pd, numpy as np, json
from datetime import datetime, date
from io import StringIO

stock = "@STOCK@"
today = date.today().isoformat().replace("-", "")
FUND_CSV = @CSV_REPR@
fund_df = pd.read_csv(StringIO(FUND_CSV), index_col=0, parse_dates=True)

EVENTS = @EVENTS_JSON@

print("=== R9 因果相乘+趋势滤网 ===")
print("标的: " + stock + ", 事件天数: " + str(len(EVENTS)))

data = get_price(stock, start_date="20251229", end_date=today,
                 fre_step="1d", fields=["close","high","low","open","volume"], fq="pre")
if len(data) < 20: print("数据不足"); raise SystemExit()

close = data["close"]; high = data["high"]; low = data["low"]; volume = data["volume"]
df = data.join(fund_df, how="left").fillna(method="ffill", limit=3)
df["ret"] = df["close"].pct_change()
df["range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
df["gap_pct"] = (df["open"] / df["close"].shift(1) - 1) * 100

W = @W@; ATR_P = @ATR_P@; SENS = @SENS@; GAIN = @GAIN@; BUY = @BUY@

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

# 事件信号
event_base = pd.Series(50.0, index=df.index)
for i, dt in enumerate(df.index):
    dstr = dt.strftime("%Y-%m-%d")
    if dstr in EVENTS:
        es = EVENTS[dstr]
        event_base.iloc[i] = 50 + min(max(es, -30), 30)

gap_up = (df["gap_pct"] > 1.5 * gap_std).astype(int)
gap_down = (df["gap_pct"] < -1.5 * gap_std).astype(int)
vol_surge = (vol_z > 1.5).astype(int)
range_surge = (df["range_pct"] > range_mean + 1.5 * range_std).astype(int)
tech_event = 50.0 + gap_up*10 - gap_down*12 + vol_surge*5 + range_surge*4
tech_event = tech_event.clip(0, 100)

event_score = event_base * 0.70 + tech_event * 0.30
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
# tech_dir * 8.0: 技术全多头贡献约4.8分
# mfp.clip(0,None)*1.0: 主力资金正向流入加成(0~max+3)
total_score = 50.0 + combined_dir * GAIN + tech_dir * 8.0 + mfp.clip(0, None) * 1.0
total_score = total_score.clip(0, 100)

# 趋势滤网: 只在上升趋势中开仓
df["position"] = 0.0
df.loc[(total_score >= BUY) & (df["trend"] == 1), "position"] = 1.0
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

print("事件覆盖: %d/%d 天" % (len([d for d in df.index if d.strftime("%Y-%m-%d") in EVENTS]), len(df)))
result = {"stock": stock, "version": "R9_trend_filter",
          "event_days": len([d for d in df.index if d.strftime("%Y-%m-%d") in EVENTS])}
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

if 'f"' in CODE: print("ERROR: f-string"); exit(1)
print("代码: %d chars" % len(CODE))

async def main():
    from smlogin import SuperMindSession
    async with SuperMindSession() as sm:
        r = await sm.execute(CODE, timeout=240)
    print(r["stdout"])
    if r["error"]: print("ERR:\n" + r["error"][:1500])

asyncio.run(main())
