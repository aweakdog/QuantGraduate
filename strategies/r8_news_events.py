"""
R8 — 真实事件信号版
从同花顺7x24快讯拉新闻，匹配股票概念标签，生成事件分数
"""
import sys, json, asyncio, subprocess, requests
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SMLOGIN = _PROJECT_ROOT / "engine" / "smlogin"
if str(_SMLOGIN) not in sys.path:
    sys.path.insert(0, str(_SMLOGIN))
from smlogin import SuperMindSession

STOCK = sys.argv[1] if len(sys.argv) > 1 else "300342.SZ"
THS_PY = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe"

CSV_PATH = {"601689.SH": str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv"),
            "300342.SZ": str(_PROJECT_ROOT / "data" / "300342.SZ_fund_flow_6m.csv")}.get(STOCK, str(_PROJECT_ROOT / "data" / "fund_flow_6m.csv"))

# ── 1. 查股票概念标签 ──
r_conc = subprocess.run([THS_PY, "-c",
    "from thsdk import THS; ths=THS(); ths.connect(); "
    "r=ths.wencai_nlp('%s 所属概念');\n"
    "if r.success and r.data is not None:\n"
    "  print(str(r.df.iloc[0,0]))" % STOCK],
    capture_output=True, text=True, timeout=20)
concepts_str = r_conc.stdout.strip().split("\n")[-1] if r_conc.stdout.strip() else ""
concepts = [c.strip() for c in concepts_str.split(";") if c.strip()]
print("股票: %s  概念: %s" % (STOCK, concepts))

# ── 2. 查市值 ──
r_mcap = subprocess.run([THS_PY, "-c",
    "from thsdk import THS; ths=THS(); ths.connect(); "
    "r=ths.wencai_nlp('%s 总市值'); print(r.df.iloc[0,0] if r.success and r.data is not None else '0')" % STOCK],
    capture_output=True, text=True, timeout=20)
try: mcap = float(r_mcap.stdout.strip().split("\n")[-1])
except (ValueError, TypeError, IndexError): mcap = 0
print("市值: %.0f亿" % (mcap/1e8))

# ── 3. 拉7x24快讯 ──
headers = {'User-Agent': 'Mozilla/5.0'}
all_news = []
for page in [1, 2]:
    url = 'https://news.10jqka.com.cn/tapp/news/push/stock/?page=%d&tag=A股&track=website&pagesize=30' % page
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        all_news.extend(data['data']['list'])
    except (KeyError, json.JSONDecodeError, requests.RequestException): pass
print("拉取新闻: %d 条" % len(all_news))

# ── 4. 事件评分：每条新闻匹配概念标签 ──
stock_code = STOCK.split(".")[0]
stock_name = {"300342": "天银机电", "601689": "拓普集团"}.get(stock_code, "")

event_scores = []
for news in all_news:
    score = 0
    matched = []
    
    # 4a. 直接命中股票名称或代码
    if stock_name and stock_name in news.get("title", ""):
        score += 30
        matched.append("名称命中")
    
    # 4b. stock字段直接关联
    for s in news.get("stock", []):
        if s.get("stockCode", "") == stock_code:
            score += 40
            matched.append("代码命中")
    
    # 4c. tagInfo匹配概念标签
    for tag in news.get("tagInfo", []):
        tname = tag.get("name", "")
        tscore = float(tag.get("score", 0.5))
        if tname in concepts:
            score += int(20 * tscore)
            matched.append(tname)
    
    if score > 0:
        event_scores.append({
            "title": news["title"][:60],
            "score": score,
            "matched": matched,
            "ctime": news.get("ctime", "0"),
        })

event_scores.sort(key=lambda x: x["score"], reverse=True)
print("相关事件: %d 条" % len(event_scores))
for e in event_scores[:5]:
    print("  [%d] %s | %s" % (e["score"], e["title"][:50], ",".join(e["matched"][:3])))

# ── 5. 读取CSV ──
r_csv = subprocess.run([THS_PY, "-c",
    "import pandas as pd; df=pd.read_csv(r'%s', index_col=0); print(repr(df.to_csv()))" % CSV_PATH],
    capture_output=True, text=True, timeout=30)
csv_repr = r_csv.stdout.strip()
print("CSV: %d chars" % len(csv_repr))

# ── 6. 事件数据嵌入 → 策略 ──
# 按日聚合事件分数：如果有事件的日子，event_boost从50分起跳
daily_events = {}
for e in event_scores:
    day = e["ctime"]  # unix timestamp
    day_date = __import__('datetime').datetime.fromtimestamp(int(day)).strftime("%Y-%m-%d") if day != "0" else ""
    if day_date:
        if day_date not in daily_events:
            daily_events[day_date] = 0
        daily_events[day_date] += e["score"]

print("有事件的交易日: %d 天" % len(daily_events))
if daily_events:
    max_score = max(daily_events.values())
    print("事件分范围: [%d, %d]" % (min(daily_events.values()), max_score))

# 小盘窗口自适应
W = 10
ATR_P = 8
if mcap > 500e8:
    W = 20; ATR_P = 14

print("窗口: W=%d, ATR=%d" % (W, ATR_P))

# ── 7. 策略代码 ──
# 把daily_events编码进策略
events_json = json.dumps(daily_events)

# 自适应参数：小盘更灵敏
SENS = 4.0 if mcap > 500e8 else 3.0
GAIN = 40 if mcap > 500e8 else 45
BUY = 60 if mcap > 500e8 else 55
GM = 1.5 if mcap > 500e8 else 1.2

print("参数: sens=%.1f gain=%.0f buy=%.0f gm=%.1f" % (SENS, GAIN, BUY, GM))

CODE = (r'''
import pandas as pd, numpy as np, json
from datetime import datetime, date
from io import StringIO

stock = "@STOCK@"
today = date.today().isoformat().replace("-", "")
FUND_CSV = @CSV_REPR@
fund_df = pd.read_csv(StringIO(FUND_CSV), index_col=0, parse_dates=True)

# 事件数据：{日期: 事件总分数}，按天匹配
EVENTS = @EVENTS_JSON@

print("=== R8 真实事件信号 ===")
print("标的: " + stock + ", 事件天数: " + str(len(EVENTS)))

data = get_price(stock, start_date="20251229", end_date=today,
                 fre_step="1d", fields=["close","high","low","open","volume"], fq="pre")
if len(data) < 20: print("数据不足"); raise SystemExit()

close = data["close"]; high = data["high"]; low = data["low"]; volume = data["volume"]
df = data.join(fund_df, how="left").fillna(method="ffill", limit=3)
df["ret"] = df["close"].pct_change()
df["range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
df["gap_pct"] = (df["open"] / df["close"].shift(1) - 1) * 100

# 窗口
W = @W@
ATR_P = @ATR_P@
SENS = @SENS@
GAIN = @GAIN@
BUY = @BUY@
GM = @GM@

df["ma_s"] = df["close"].rolling(5).mean()
df["ma_m"] = df["close"].rolling(W).mean()
df["ma_l"] = df["close"].rolling(W*2).mean()

gap_std = df["gap_pct"].rolling(W).std()
vol_mean = df["volume"].rolling(W).mean(); vol_std = df["volume"].rolling(W).std()
range_mean = df["range_pct"].rolling(W).mean(); range_std = df["range_pct"].rolling(W).std()
vol_z = (df["volume"] - vol_mean) / (vol_std + 1)
df["atr"] = df["range_pct"].rolling(ATR_P).mean()
mfp = df["main_force_pct"].fillna(0)

# ── 事件信号：真实消息 → 基准event_score ──
# 日期字符串匹配，有事件的日期 event_base 从50起跳
event_base = pd.Series(50.0, index=df.index)
for i, dt in enumerate(df.index):
    dstr = dt.strftime("%Y-%m-%d")
    if dstr in EVENTS:
        es = EVENTS[dstr]
        # 事件分数映射到 [-30, +30] 偏移
        # 正分=利好，负分=利空
        if es > 0:
            event_base.iloc[i] = 50 + min(es, 30)
        else:
            event_base.iloc[i] = 50 + max(es, -30)

# 技术代理事件（保留，但权重降低）
gap_up = (df["gap_pct"] > GM * gap_std).astype(int)
gap_down = (df["gap_pct"] < -GM * gap_std).astype(int)
vol_surge = (vol_z > GM * 0.8).astype(int)
range_surge = (df["range_pct"] > range_mean + GM * range_std).astype(int)

tech_event = 50.0 + gap_up*10 - gap_down*12 + vol_surge*5 + range_surge*4
tech_event = tech_event.clip(0, 100)

# 合并事件分：真实事件占70%，技术代理占30%
event_score = event_base * 0.70 + tech_event * 0.30
event_dir = (event_score - 50) / 50.0

# ── 资金确认 ──
fund_conf = mfp.abs() / SENS; fund_conf = fund_conf.clip(0, 1)
fund_dir = np.sign(mfp)
combined_dir = event_dir * fund_dir * fund_conf

# ── 技术趋势 ──
tech_score = 50.0 + (df["close"]>df["ma_s"]).astype(int)*8 + (df["close"]>df["ma_m"]).astype(int)*10 + (df["close"]>df["ma_l"]).astype(int)*12
tech_score = tech_score.clip(0, 100)
tech_dir = (tech_score - 50) / 50.0

total_score = 50.0 + combined_dir * GAIN + tech_dir * 35.0 * 0.15
total_score = total_score.clip(0, 100)

# 仓位+止损
position = pd.Series(0.0, index=df.index)
position[total_score >= BUY] = 1.0
position[total_score <= 40] = 0.0
position[mfp < -3.0] = 0.0

inp = False; ep = 0.0; pp = 0.0; dh = 0
for i in range(len(df)):
    if position.iloc[i] == 1.0 and not inp:
        inp = True; ep = df["close"].iloc[i]; pp = ep; dh = 0
    elif inp:
        pp = max(pp, df["close"].iloc[i]); dh += 1
        c = df["close"].iloc[i]; le = (c-ep)/ep; lp = (c-pp)/pp
        atr = df["atr"].iloc[i]/100.0
        if atr < 0.01: atr = 0.01
        if le < -2.0*atr or (lp < -1.5*atr and le > 0.03) or (le < 0.005 and dh > 12):
            position.iloc[i] = 0.0; inp = False

sr = df["ret"] * position.shift(1)
sc = (1 + sr).cumprod(); bc = (1 + df["ret"]).cumprod()
ts = float(sc.iloc[-1] - 1); tb = float(bc.iloc[-1] - 1)
dd = float(((sc - sc.cummax()) / sc.cummax()).min())
sharpe = float(np.sqrt(252) * sr.mean() / (sr.std() + 1e-10))
beta = float(sr.cov(df["ret"]) / (df["ret"].var() + 1e-10))
alpha = float(((1+ts)**(252/len(df))-1) - beta * ((1+tb)**(252/len(df))-1))
wr = float((sr > 0).sum() / ((sr != 0).sum() + 1e-10))

event_days_with_news = len([d for d in df.index if d.strftime("%Y-%m-%d") in EVENTS])
result = {"stock": stock, "version": "R8_news_events",
          "event_days": event_days_with_news, "total_news": len(EVENTS)}
result["strategy_pct"] = round(ts*100, 2)
result["benchmark_pct"] = round(tb*100, 2)
result["excess_pct"] = round((ts-tb)*100, 2)
result["max_dd_pct"] = round(dd*100, 2)
result["sharpe"] = round(sharpe, 3)
result["alpha_pct"] = round(alpha*100, 2)
result["beta"] = round(beta, 3)
result["win_rate_pct"] = round(wr*100, 2)
result["trades"] = int((position.diff()!=0).sum())
print("===RESULT===")
print(json.dumps(result))
print("===END===")
''').replace("@STOCK@", STOCK).replace("@CSV_REPR@", csv_repr)\
    .replace("@EVENTS_JSON@", events_json)\
    .replace("@W@", str(W)).replace("@ATR_P@", str(ATR_P))\
    .replace("@SENS@", str(SENS)).replace("@GAIN@", str(GAIN))\
    .replace("@BUY@", str(BUY)).replace("@GM@", str(GM))

if 'f"' in CODE: print("ERROR: f-string"); exit(1)
print("代码: %d chars" % len(CODE))

async def main():
    async with SuperMindSession() as sm:
        r = await sm.execute(CODE, timeout=300)
    print(r["stdout"])
    if r["error"]: print("ERR:\n" + r["error"][:1500])

asyncio.run(main())
