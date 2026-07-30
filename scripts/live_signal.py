"""实盘每日信号 (单脚本)

用途: 每天收盘后跑一次, 输出"下一个交易日尾盘"该卖什么、该买什么、买多少股。
与 scripts/wf_v35_breadth_alpha.py 的回测执行逻辑 1:1 对齐 (t1close 模式):
    T 日收盘出信号  ->  T+1 日尾盘成交

核心设计
────────
1. 有状态。持仓/现金存在 data/live/state.json, 每次运行:
     a) 先"结算"上一次的挂单计划 (用真实的 T+1 行情, 含停牌/涨跌停判定)
     b) 再用最新数据训练模型, 生成新的挂单计划
   这样即使某天漏跑, 下次运行也能自动补上, 不会错位。

2. 无未来函数。
     - 训练截止日 = 信号日往前推 LABEL_HORIZON 天 (标签窗口必须完全过去)
     - 特征筛选只用 --feat-cutoff 之前的数据; 默认直接复用已验证回测的特征列表
     - regime 只用截至信号日的行情

3. 计划是"建议", 结算是"事实"。
   计划里给的股数按 T 日收盘价估算; T+1 真实价格不同时按可用资金取 100 股整数倍。
   结算默认按 T+1 收盘价模拟 (贴近尾盘成交); 若实际成交价差异大, 用 --confirm 手工修正。

用法
────
  # 每日信号 (默认参数 = 当前最优方案 em_t1close_s001)
  python scripts/live_signal.py

  # 首次初始化本金
  python scripts/live_signal.py --capital 20000 --init

  # 只看不写状态
  python scripts/live_signal.py --dry-run

  # 手工修正上一笔成交 (JSON: [{"code":"600519","action":"buy","shares":100,"price":1680.5}])
  python scripts/live_signal.py --confirm fills.json

  # 只看当前持仓/现金 (秒级, 不训练模型) —— 网页"当前状态"面板用
  python scripts/live_signal.py --status

  # 整体对账: 手上实际持仓/现金与服务器不一致时, 以券商App为准强制覆盖
  python scripts/live_signal.py --sync-template          # 导出当前状态当表单
  python scripts/live_signal.py --sync my_sync.json      # 改好数字后回填
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402
from pipeline.config import settings  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # 没装 tqdm 时退化成朴素进度
    def tqdm(it, total=None, desc="", **_):
        it = list(it)
        n = total or len(it)
        t0 = datetime.now()
        for i, x in enumerate(it, 1):
            if i % max(1, n // 50) == 0 or i == n:
                el = (datetime.now() - t0).total_seconds()
                eta = el / i * (n - i)
                print(f"\r  {desc} [{i}/{n}] {i/n*100:.0f}% | 已用{el:.0f}s | 剩余~{eta/60:.1f}m",
                      end="", flush=True)
            yield x
        print(flush=True)


# ═══════════════════════════════════════════════════════════════
# 参数 (默认值 = 已验证的最优方案)
# ═══════════════════════════════════════════════════════════════
ap = argparse.ArgumentParser(description="实盘每日信号")
ap.add_argument("--train-file", default="training_data_pit_v24.parquet")
ap.add_argument("--pit-universe", default="universe_pit.parquet")
ap.add_argument("--label", default="5d", choices=["1d", "5d"])
ap.add_argument("--hold-days", type=int, default=10)
ap.add_argument("--tranche-n", type=int, default=3)
ap.add_argument("--portfolio-mode", default="periodic", choices=["periodic", "staggered"])
ap.add_argument("--exec-mode", default="t1close", choices=["t1close", "t1open"])
ap.add_argument("--slippage", type=float, default=0.001)
ap.add_argument("--regime-filter", default="breadth", choices=["off", "ma", "breadth", "both", "any"])
ap.add_argument("--regime-ma", type=int, default=20)
ap.add_argument("--regime-breadth", type=float, default=0.35)
ap.add_argument("--regime-confirm", type=int, default=1)
ap.add_argument("--reversal-guard", type=float, default=0.0)
ap.add_argument("--n-features", type=int, default=80)
ap.add_argument("--corr-threshold", type=float, default=0.9)
ap.add_argument("--feat-cutoff", default="2023-09-19",
                help="特征筛选只允许用该日期之前的数据 (与回测保持一致)")
ap.add_argument("--features-from",
                default="wf_daily_em_t1close_s001_fundfix_ts2022-09-01_te2026-07-27_cap20000.json",
                help="直接复用回测结果 json 里的 selected_features (data/processed/ 下); "
                     "设为 none 则现场重新筛选")
ap.add_argument("--capital", type=float, default=20000.0, help="初始本金 (仅 --init 时使用)")
ap.add_argument("--init", action="store_true", help="重置状态文件, 用 --capital 作为起始现金")
ap.add_argument("--state", default="state.json", help="状态文件名 (data/live/ 下)")
ap.add_argument("--confirm", default=None, help="手工成交回报 json 路径, 用它替代自动结算")
ap.add_argument("--dry-run", action="store_true", help="只打印, 不写状态/不落盘")
ap.add_argument("--allow-stale", action="store_true", help="训练集比K线旧时仍继续")
ap.add_argument("--as-of", default=None,
                help="假装数据只到该日期 (回放/补跑/自测用), 不使用之后的任何数据")
ap.add_argument("--alternates", type=int, default=8, help="额外输出几只候补股")
ap.add_argument("--status", action="store_true",
                help="只打印当前持仓/现金/待执行计划, 不跑模型 (秒级)")
ap.add_argument("--sync", default=None,
                help="整体对账: 用该 json 里的真实持仓/现金覆盖服务器状态 (以券商App为准)")
ap.add_argument("--sync-template", action="store_true",
                help="导出一份以当前状态预填的对账表单 json, 改完再用 --sync 回填")
args = ap.parse_args()

DATA_DIR = settings.DATA_DIR
TRAIN_PATH = DATA_DIR / "processed" / args.train_file
KLINE_DIR = DATA_DIR / "raw" / "kline"
LIVE_DIR = DATA_DIR / "live"
STATE_PATH = LIVE_DIR / args.state

# 计划文件名跟着状态文件走, 否则多条并行线(不同本金/持仓数)会写同一个
# plan_<日期>.json 互相覆盖。state.json -> plan_<日期>.json (兼容旧数据);
# state_aggr5w.json -> plan_aggr5w_<日期>.json
_state_stem = Path(args.state).stem
if _state_stem == "state":
    PLAN_PREFIX = "plan"
elif _state_stem.startswith("state_"):
    PLAN_PREFIX = "plan_" + _state_stem[len("state_"):]
else:
    PLAN_PREFIX = "plan_" + _state_stem

LABEL_RAW = "fwd_1d_ret" if args.label == "1d" else "fwd_5d_ret"
LABEL = "y_target"
LABEL_HORIZON = 1 if args.label == "1d" else 5

LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {"date", "code", "group", LABEL,
             "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret",
             "fwd_1d_excess", "fwd_5d_excess", "fwd_1d_open_ret", "fwd_1d_exec_ret",
             "fwd_1d_t1_open_ret", "fwd_1d_t1_close_ret", "fwd_1d_exec_excess"}
EXCLUDED_FEATS = {"mf_pct_1d", "mf_pct_1d_ma5", "mf_pct_1d_ma20",
                  "macd_signal", "macd_signal_ma5", "macd_signal_ma20"}
LOCKED_PARAMS = dict(
    n_estimators=151, max_depth=4, learning_rate=0.03,
    num_leaves=15, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=50, random_state=42, n_jobs=10, verbosity=-1,
    boosting_type="dart",
)
TRADE_COST = 0.0006
MIN_FEE = 5.0
SLIPPAGE = args.slippage
HOLD_DAYS, TRANCHE_N = args.hold_days, args.tranche_n
PERIODIC = args.portfolio_mode == "periodic"
TARGET_POSITIONS = TRANCHE_N if PERIODIC else HOLD_DAYS * TRANCHE_N
MIN_TRAIN_DAYS = 250
EXEC_FIELD = "open" if args.exec_mode == "t1open" else "close"

_COL_MAP = {"时间": "date", "收盘价": "close", "开盘价": "open",
            "最高价": "high", "最低价": "low", "成交量": "volume", "总金额": "amount"}


# ═══════════════════════════════════════════════════════════════
# 工具 (与回测同源)
# ═══════════════════════════════════════════════════════════════
def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")


def board_limit(code):
    c = str(code)[:6]
    if c.startswith(("300", "688")):
        return 0.20
    if c.startswith(("43", "83", "87", "92")):
        return 0.30
    return 0.10


def fill_px(px, side):
    return px * (1 + SLIPPAGE) if side == "buy" else px * (1 - SLIPPAGE)


class Klines:
    """按需加载单只K线, 带缓存"""

    def __init__(self):
        self._c = {}

    def get(self, code):
        k = str(code)[:6]
        if k not in self._c:
            p = KLINE_DIR / f"{k}.parquet"
            if not p.exists():
                self._c[k] = None
            else:
                kl = pd.read_parquet(p).rename(columns=_COL_MAP)
                kl["date"] = pd.to_datetime(kl["date"])
                self._c[k] = kl.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        return self._c[k]

    def px(self, code, date, field="close"):
        kl = self.get(code)
        if kl is None:
            return None
        r = kl[kl["date"] == pd.Timestamp(date)]
        if not len(r):
            return None
        v = float(r.iloc[0][field])
        return v if v == v and v > 0 else None

    def limit_state(self, code, date, field="close"):
        kl = self.get(code)
        if kl is None:
            return False, False
        idx = kl.index[kl["date"] == pd.Timestamp(date)]
        if len(idx) == 0 or idx[0] < 1:
            return False, False
        pos = idx[0]
        prev, cur = float(kl.iloc[pos - 1]["close"]), float(kl.iloc[pos][field])
        lim = board_limit(code)
        return cur >= prev * (1 + lim) * 0.999, cur <= prev * (1 - lim) * 1.001


def load_names():
    names = {}
    pm = DATA_DIR / "universe" / "pit_metadata.parquet"
    if pm.exists():
        m = pd.read_parquet(pm)
        names = dict(zip(m["code"].astype(str).str.zfill(6), m["name"]))
    p = DATA_DIR / "raw" / "all_stock_list.parquet"
    if p.exists():
        n = pd.read_parquet(p)
        names.update(dict(zip(n["code"].astype(str).str[:6], n["name"])))
    return names


def apply_pit_universe(df, uni_file):
    u = pd.read_parquet(DATA_DIR / "universe" / uni_file)
    u["effective_date"] = pd.to_datetime(u["effective_date"])
    u["code6"] = u["code"].astype(str).str.zfill(6)
    # 用 DatetimeIndex 而非 np.array(sorted(...)): 后者得到 object 数组,
    # 与 datetime64[ns] 的 date 列做 searchsorted 会退化成 Timestamp<int 报错
    # (训练集 parquet 的时间精度 ms/ns 不固定, 这里必须对精度不敏感)
    eff = pd.DatetimeIndex(sorted(pd.to_datetime(u["effective_date"].unique())))
    members = {d: set(g["code6"]) for d, g in u.groupby("effective_date")}
    code6 = df["code"].astype(str).str[:6]
    period = eff.searchsorted(pd.DatetimeIndex(df["date"]), side="right") - 1
    keep = np.zeros(len(df), dtype=bool)
    for i, d in enumerate(eff):
        m = period == i
        if m.any():
            keep[m] = code6[m].isin(members[pd.Timestamp(d)]).values
    out = df[keep].reset_index(drop=True)
    print(f"  PIT 成分约束: {len(df):,} -> {len(out):,} 行, {out['code'].nunique()} 只")
    return out


def compute_market_features():
    """全市场大盘/广度特征。带按最新日期的缓存, 避免每天重扫 5000+ 文件"""
    files = sorted(KLINE_DIR.glob("*.parquet"))
    stamp = max(int(p.stat().st_mtime) for p in files) if files else 0
    cache = LIVE_DIR / f"cache_market_ma{args.regime_ma}_{stamp}.parquet"
    if cache.exists():
        m = pd.read_parquet(cache)
        m["date"] = pd.to_datetime(m["date"])
        print(f"  大盘特征: 命中缓存 {cache.name}")
    else:
        rows = []
        for p in tqdm(files, total=len(files), desc="大盘特征"):
            try:
                kl = pd.read_parquet(p).rename(columns=_COL_MAP)
                kl["date"] = pd.to_datetime(kl["date"])
                kl = kl.sort_values("date")
                kl["above_ma"] = (kl["close"] > kl["close"].rolling(args.regime_ma).mean()).astype(float)
                kl["up"] = (kl["close"].pct_change() > 0).astype(float)
                rows.append(kl[["date", "close", "open", "above_ma", "up"]])
            except Exception:
                continue
        market = pd.concat(rows, ignore_index=True)
        m = market.groupby("date").agg(mkt_close=("close", "mean"), mkt_open=("open", "mean"),
                                       breadth_above_ma=("above_ma", "mean"),
                                       breadth_up=("up", "mean")).sort_index()
        m["mkt_ret_1d"] = m["mkt_close"].pct_change()
        m["mkt_overnight"] = m["mkt_open"] / m["mkt_close"].shift(1) - 1
        m["mkt_intraday"] = m["mkt_close"] / m["mkt_open"] - 1
        for w in (5, 20, 60):
            m[f"mkt_ma{w}"] = m["mkt_close"].rolling(w).mean()
            m[f"mkt_above_ma{w}"] = (m["mkt_close"] > m[f"mkt_ma{w}"]).astype(int)
        m["mkt_mom_5d"] = m["mkt_close"] / m["mkt_close"].shift(5) - 1
        m["mkt_mom_20d"] = m["mkt_close"] / m["mkt_close"].shift(20) - 1
        m["mkt_vol_20d"] = m["mkt_ret_1d"].rolling(20).std()
        m["mkt_vol_5d"] = m["mkt_ret_1d"].rolling(5).std()
        m["mkt_trend_strength"] = m["mkt_ma5"] / m["mkt_ma20"] - 1
        m = m.reset_index()
        m["date"] = pd.to_datetime(m["date"])
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        for old in LIVE_DIR.glob(f"cache_market_ma{args.regime_ma}_*.parquet"):
            old.unlink()
        m.to_parquet(cache, index=False)
    feats = [c for c in m.columns if c.startswith("mkt_")]
    for c in feats:
        m[c] = pd.to_numeric(m[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    regime_src = m[["date", "mkt_close", f"mkt_ma{args.regime_ma}",
                    "breadth_above_ma", "breadth_up"]].copy()
    return m[["date"] + feats], feats, regime_src


def compute_overnight_features(codes):
    out = []
    for code in tqdm(list(codes), desc="隔夜特征"):
        p = KLINE_DIR / f"{str(code)[:6]}.parquet"
        if not p.exists():
            continue
        try:
            kl = pd.read_parquet(p).rename(columns=_COL_MAP)
            kl["date"] = pd.to_datetime(kl["date"])
            kl = kl.sort_values("date").reset_index(drop=True)
            kl["overnight_ret"] = kl["open"] / kl["close"].shift(1) - 1
            kl["intraday_ret"] = kl["close"] / kl["open"] - 1
            kl["ovn_mean_5d"] = kl["overnight_ret"].rolling(5).mean()
            kl["ovn_mean_20d"] = kl["overnight_ret"].rolling(20).mean()
            kl["ovn_std_20d"] = kl["overnight_ret"].rolling(20).std()
            kl["ovn_sum_5d"] = kl["overnight_ret"].rolling(5).sum()
            kl["ovn_pos_ratio_20d"] = (kl["overnight_ret"] > 0).rolling(20).mean()
            kl["code"] = code
            out.append(kl[["date", "code", "overnight_ret", "intraday_ret", "ovn_mean_5d",
                           "ovn_mean_20d", "ovn_std_20d", "ovn_sum_5d", "ovn_pos_ratio_20d"]])
        except Exception:
            continue
    ovn = pd.concat(out, ignore_index=True)
    feats = ["overnight_ret", "intraday_ret", "ovn_mean_5d", "ovn_mean_20d",
             "ovn_std_20d", "ovn_sum_5d", "ovn_pos_ratio_20d"]
    for c in feats:
        ovn[c] = ovn[c].replace([np.inf, -np.inf], np.nan)
    return ovn, feats


def select_features(df, all_features, cutoff):
    if args.features_from and args.features_from.lower() != "none":
        src = DATA_DIR / "processed" / args.features_from
        if not src.exists():
            raise SystemExit(f"ERROR: 找不到特征来源 {src}; 用 --features-from none 现场筛选")
        sel = json.loads(src.read_text(encoding="utf-8"))["selected_features"]
        miss = [f for f in sel if f not in df.columns]
        if miss:
            raise SystemExit(f"ERROR: 训练集缺少回测特征 {miss[:10]} (共{len(miss)}个)")
        print(f"  特征: 复用回测 {args.features_from} 的 {len(sel)} 个特征 (无泄漏, 与回测完全一致)")
        return sel
    print(f"  特征筛选: {len(all_features)} -> top{args.n_features} | 仅用 < {cutoff} 的数据")
    s = df[(df["date"] < pd.Timestamp(cutoff)) & df[LABEL].notna()]
    if len(s) < 10000:
        raise SystemExit(f"ERROR: {cutoff} 之前只有 {len(s)} 行, 不足以筛特征")
    X = s.groupby("code")[all_features].transform(lambda c: c.ffill().fillna(0))
    p = dict(LOCKED_PARAMS, n_estimators=50, boosting_type="gbdt")
    mdl = lgb.LGBMRegressor(**p).fit(X, s[LABEL])
    imp = (pd.DataFrame({"feature": all_features, "importance": mdl.feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    top = imp.head(args.n_features)["feature"].tolist()
    cm = s[top].corr().abs()
    selected, dropped = [], set()
    for f in top:
        if f in dropped:
            continue
        selected.append(f)
        for c in cm.index[cm[f] > args.corr_threshold]:
            if c != f:
                dropped.add(c)
    print(f"    保留 {len(selected)} 个")
    return selected


def build_regime_series(regime_src):
    """返回每个日期的空仓布尔序列 (只用截至当日信息)"""
    if args.regime_filter == "off":
        return None
    r = regime_src.set_index("date").sort_index()
    trend_bad = r["mkt_close"] < r[f"mkt_ma{args.regime_ma}"]
    breadth_bad = r["breadth_above_ma"] < args.regime_breadth
    raw = {"ma": trend_bad, "breadth": breadth_bad,
           "both": trend_bad & breadth_bad, "any": trend_bad | breadth_bad}[args.regime_filter]
    raw = raw.fillna(False)
    k = max(1, args.regime_confirm)
    turn_off = raw.rolling(k).min().fillna(0) == 1
    turn_on = (~raw).rolling(k).min().fillna(0) == 1
    state, out = False, {}
    for d in r.index:
        if not state and bool(turn_off.loc[d]):
            state = True
        elif state and bool(turn_on.loc[d]):
            state = False
        out[d] = state
    return pd.Series(out)


# ═══════════════════════════════════════════════════════════════
# 状态
# ═══════════════════════════════════════════════════════════════
def fingerprint():
    return {k: getattr(args, k) for k in
            ("train_file", "pit_universe", "label", "hold_days", "tranche_n",
             "portfolio_mode", "exec_mode", "slippage", "regime_filter", "regime_ma",
             "regime_breadth", "regime_confirm", "reversal_guard")}


def load_state():
    if args.init or not STATE_PATH.exists():
        return {"config": fingerprint(), "cash": args.capital, "initial_capital": args.capital,
                "lots": [], "next_lot_id": 1, "last_rebal_signal_date": None,
                "last_signal_date": None, "pending": None, "history": [], "calendar": []}
    st = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    diff = {k: (v, st.get("config", {}).get(k)) for k, v in fingerprint().items()
            if st.get("config", {}).get(k) != v}
    if diff:
        raise SystemExit("ERROR: 参数与状态文件不一致, 会导致持仓错位。\n"
                         + "\n".join(f"  {k}: 状态={o} 当前={n}" for k, (n, o) in diff.items())
                         + "\n  确认要换方案请先备份并加 --init 重置。")
    return st


def save_state(st):
    if args.dry_run:
        print("\n[dry-run] 未写入状态文件")
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    st["config"] = fingerprint()
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")
    print(f"状态已更新: {STATE_PATH}")


def backup_state(tag):
    """改动状态前先备份, 出错可回滚"""
    if not STATE_PATH.exists() or args.dry_run:
        return None
    bak = LIVE_DIR / "backup"
    bak.mkdir(parents=True, exist_ok=True)
    p = bak / f"{STATE_PATH.stem}_{tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
    p.write_text(STATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════
# 交易日历 (持仓到期完全按"交易日日期"算, 不依赖任何行号/序号,
#            这样对账、补跑、换数据源都不会让持仓错位)
# ═══════════════════════════════════════════════════════════════
def cal_from_state(st):
    cal = st.get("calendar")
    if not cal:
        raise SystemExit("ERROR: 状态里还没有交易日历。先跑一次正常信号:\n"
                         "       python scripts/live_signal.py")
    return [pd.Timestamp(x) for x in cal]


def cal_pos(cal, d):
    """该日期在交易日历中的位置; 非交易日则取不晚于它的最后一个交易日"""
    idx = cal if isinstance(cal, pd.DatetimeIndex) else pd.DatetimeIndex(cal)
    i = int(idx.searchsorted(pd.Timestamp(d), side="right")) - 1
    if i < 0:
        raise SystemExit(f"ERROR: {pd.Timestamp(d).date()} 早于数据起始日 "
                         f"{pd.Timestamp(idx[0]).date()}")
    return i


def held_days(cal, lot, signal_date):
    """已持有的换仓周期数 = 信号日 与 开仓信号日 之间的交易日跑动次数"""
    return cal_pos(cal, signal_date) - cal_pos(cal, lot["open_signal_date"])


# ═══════════════════════════════════════════════════════════════
# 结算: 用真实的 T+1 行情执行上一次的计划
# ═══════════════════════════════════════════════════════════════
def settle(st, pending, exec_date, kl, names, cal):
    """完全复刻回测执行层: 先卖到期批次, 再按排名买入。返回成交流水"""
    sig_date, in_cash = pending["signal_date"], pending["in_cash"]
    fills = []
    manual = None
    if args.confirm:
        manual = json.loads(Path(args.confirm).read_text(encoding="utf-8"))

    if manual is not None:
        for f in manual:
            code = str(f["code"])[:6]
            px, sh = float(f["price"]), int(f["shares"])
            gross = px * sh
            fee = max(gross * TRADE_COST, MIN_FEE)
            if f["action"] == "sell":
                lot = next((l for l in st["lots"] if str(l["code"])[:6] == code
                            and l["shares"] == sh), None)
                if lot is None:
                    raise SystemExit(f"ERROR: 回报里的卖出 {code} x{sh} 找不到对应持仓")
                st["lots"].remove(lot)
                st["cash"] += gross - fee
                fills.append({"code": code, "action": "sell", "shares": sh, "price": px,
                              "fee": round(fee, 2), "net": round(gross - fee, 2),
                              "source": "manual"})
            else:
                st["cash"] -= gross + fee
                st["lots"].append({"id": st["next_lot_id"], "code": code, "shares": sh,
                                   "buy_price": px, "open_signal_date": sig_date,
                                   "open_date": str(pd.Timestamp(exec_date).date())})
                st["next_lot_id"] += 1
                fills.append({"code": code, "action": "buy", "shares": sh, "price": px,
                              "fee": round(fee, 2), "net": round(-(gross + fee), 2),
                              "source": "manual"})
        return fills

    # ── 1. 卖出 ──
    keep, rejected = [], []
    for lot in st["lots"]:
        matured = held_days(cal, lot, sig_date) >= HOLD_DAYS
        if not (matured or in_cash):
            keep.append(lot)
            continue
        px = kl.px(lot["code"], exec_date, EXEC_FIELD)
        _, limit_down = kl.limit_state(lot["code"], exec_date, EXEC_FIELD)
        if px is None or limit_down:
            rejected.append({"code": lot["code"], "action": "sell",
                             "reason": "停牌" if px is None else "跌停卖不出"})
            keep.append(lot)
            continue
        px = fill_px(px, "sell")
        gross = lot["shares"] * px
        fee = max(gross * TRADE_COST, MIN_FEE)
        st["cash"] += gross - fee
        fills.append({"code": lot["code"], "name": names.get(str(lot["code"])[:6], ""),
                      "action": "sell", "shares": lot["shares"], "price": round(px, 3),
                      "fee": round(fee, 2), "net": round(gross - fee, 2),
                      "reason": "matured" if matured else "regime_exit", "source": "auto"})
    st["lots"] = keep

    # ── 2. 买入 ──
    if not in_cash and pending["is_rebal"]:
        equity = st["cash"] + sum(
            l["shares"] * (kl.px(l["code"], exec_date, EXEC_FIELD) or l["buy_price"])
            for l in st["lots"])
        denom = 1 if PERIODIC else HOLD_DAYS
        remaining = min(equity / denom, st["cash"])
        held = {str(l["code"])[:6] for l in st["lots"]}
        blocked = set(pending.get("blocked", []))
        bought = 0
        for code in pending["ranked"]:
            if bought >= TRANCHE_N:
                break
            c6 = str(code)[:6]
            if c6 in held or c6 in blocked:
                continue
            px = kl.px(code, exec_date, EXEC_FIELD)
            if px is None:
                rejected.append({"code": code, "action": "buy", "reason": "停牌"})
                continue
            limit_up, _ = kl.limit_state(code, exec_date, EXEC_FIELD)
            if limit_up:
                rejected.append({"code": code, "action": "buy", "reason": "涨停买不进"})
                continue
            px = fill_px(px, "buy")
            alloc = remaining / (TRANCHE_N - bought)
            shares = int(alloc / (px * 100)) * 100
            if shares <= 0:
                rejected.append({"code": code, "action": "buy", "reason": "预算不足一手"})
                continue
            gross = shares * px
            fee = max(gross * TRADE_COST, MIN_FEE)
            if gross + fee > st["cash"]:
                rejected.append({"code": code, "action": "buy", "reason": "现金不足"})
                continue
            st["cash"] -= gross + fee
            remaining -= gross + fee
            st["lots"].append({"id": st["next_lot_id"], "code": c6, "shares": shares,
                               "buy_price": px, "open_signal_date": sig_date,
                               "open_date": str(pd.Timestamp(exec_date).date())})
            st["next_lot_id"] += 1
            held.add(c6)
            bought += 1
            fills.append({"code": c6, "name": names.get(c6, ""), "action": "buy",
                          "shares": shares, "price": round(px, 3), "fee": round(fee, 2),
                          "net": round(-(gross + fee), 2), "reason": "new_tranche",
                          "source": "auto"})
    return fills, rejected


# ═══════════════════════════════════════════════════════════════
# 对账 / 查看状态 (不训练模型, 秒级返回; 供网页按钮调用)
# ═══════════════════════════════════════════════════════════════
def snapshot(st, kl, names):
    """当前持仓 + 最新可得收盘价估值"""
    cal = [pd.Timestamp(x) for x in st.get("calendar", [])]
    ref_date = cal[-1] if cal else None
    rows, mv = [], 0.0
    for lot in st["lots"]:
        ref = (kl.px(lot["code"], ref_date, "close") if ref_date else None) or lot["buy_price"]
        val = lot["shares"] * ref
        mv += val
        rows.append({"code": lot["code"], "name": names.get(str(lot["code"])[:6], ""),
                     "shares": lot["shares"], "buy_price": round(lot["buy_price"], 3),
                     "last_close": round(ref, 3), "market_value": round(val, 2),
                     "pnl_pct": round((ref / lot["buy_price"] - 1) * 100, 2),
                     "open_date": lot.get("open_date"),
                     "open_signal_date": lot.get("open_signal_date"),
                     "held_days": (held_days(cal, lot, ref_date) if cal else None)})
    return {"ref_date": str(ref_date.date()) if ref_date is not None else None,
            "cash": round(st["cash"], 2), "market_value": round(mv, 2),
            "equity": round(st["cash"] + mv, 2),
            "initial_capital": st.get("initial_capital"),
            "total_return_pct": (round((st["cash"] + mv) / st["initial_capital"] * 100 - 100, 2)
                                 if st.get("initial_capital") else None),
            "positions": rows,
            "pending": st.get("pending"),
            "last_signal_date": st.get("last_signal_date"),
            "last_rebal_signal_date": st.get("last_rebal_signal_date"),
            "last_synced_at": st.get("last_synced_at")}


def print_status(snap):
    W = 68
    print(f"\n{'='*W}")
    print(f"  服务器记录的当前状态 (估值基准日 {snap['ref_date']})")
    print(f"{'='*W}")
    print(f"  总资产 ¥{snap['equity']:,.2f} = 现金 ¥{snap['cash']:,.2f} + 持仓 ¥{snap['market_value']:,.2f}")
    if snap["total_return_pct"] is not None:
        print(f"  累计收益 {snap['total_return_pct']:+.2f}% (本金 ¥{snap['initial_capital']:,.0f})")
    print(f"\n  持仓 {len(snap['positions'])} 只:")
    for r in snap["positions"]:
        print(f"    {r['code']} {r['name']:<8} x{r['shares']:>6} 成本 {r['buy_price']:>8.2f} "
              f"现价 {r['last_close']:>8.2f} 浮盈 {r['pnl_pct']:+6.2f}% "
              f"(已持{r['held_days']}日/{HOLD_DAYS}日, 买入 {r['open_date']})")
    if not snap["positions"]:
        print("    无")
    p = snap.get("pending")
    if p:
        print(f"\n  待执行计划: {p['signal_date']} 的信号"
              f"{' [空仓]' if p['in_cash'] else ''}{' [换仓日]' if p['is_rebal'] else ''}")
        print(f"    -> 下一交易日{'开盘' if EXEC_FIELD == 'open' else '尾盘'}执行, "
              f"执行完下次运行会自动按真实行情结算")
    else:
        print("\n  待执行计划: 无")
    if snap.get("last_synced_at"):
        print(f"  上次人工对账: {snap['last_synced_at']}")


def do_sync_template(st, kl, names):
    snap = snapshot(st, kl, names)
    tpl = {
        "_说明": "以券商App为准填写真实数据, 然后 python scripts/live_signal.py --sync 本文件",
        "_cash说明": "可用资金(元)。含未交收资金也算进来, 以App显示的可用+待交收为准",
        "_positions说明": "buy_date = 实际成交日期(该日尾盘/开盘买进的那天)。"
                          "留空则沿用服务器原记录; 全新股票留空会被当成今天刚买",
        "cash": snap["cash"],
        "positions": [{"code": r["code"], "name": r["name"], "shares": r["shares"],
                       "buy_price": r["buy_price"], "buy_date": r["open_date"]}
                      for r in snap["positions"]],
        "note": "",
    }
    out = LIVE_DIR / "sync_template.json"
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tpl, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"对账表单已导出: {out}")
    print("  改完里面的 cash / positions 后执行:")
    print(f"  python scripts/live_signal.py --sync {out}")


def do_sync(st, kl, names):
    """以人工填写的真实持仓/现金为准, 整体覆盖服务器状态

    只信任 4 个字段: code / shares / buy_price / buy_date。
    到期时点由 buy_date 反推的"开仓信号日"决定, 所以哪怕服务器完全记错了,
    对账后持仓节奏依然正确。
    """
    payload = json.loads(Path(args.sync).read_text(encoding="utf-8"))
    if "cash" not in payload or "positions" not in payload:
        raise SystemExit("ERROR: 对账文件必须包含 cash 和 positions 两个字段")
    cal = cal_from_state(st)
    before = snapshot(st, kl, names)

    # ── 防误操作: 空表单会把账户抹平 ──
    # 网页端若现金输入框为空, parseFloat("")=NaN 经 ||0 会变成 0, 提交后
    # 静默把现金清零。这种"0现金+0持仓"对纸面账户永远不是合理输入。
    try:
        cash_in = float(payload["cash"])
    except (TypeError, ValueError):
        raise SystemExit(f"ERROR: cash 必须是数字, 收到 {payload['cash']!r}")
    if cash_in < 0:
        raise SystemExit(f"ERROR: cash 不能为负 (收到 {cash_in})")
    if cash_in == 0 and not payload["positions"] and not payload.get("confirm_empty"):
        raise SystemExit(
            "ERROR: 对账内容为「0 现金 + 0 持仓」, 会清空账户, 已拒绝。\n"
            f"  当前状态: 现金 ¥{before['cash']:,.2f} / 总资产 ¥{before['equity']:,.2f}\n"
            "  如果只是想看看, 直接关闭弹窗即可; 若确实要清空,\n"
            "  请在对账 json 里显式加上 \"confirm_empty\": true")
    old_by_code = {str(l["code"])[:6]: l for l in st["lots"]}

    new_lots, notes = [], []
    for i, p in enumerate(payload["positions"], 1):
        code = str(p["code"]).zfill(6)[:6]
        shares = int(p["shares"])
        if shares <= 0:
            raise SystemExit(f"ERROR: 第{i}条 {code} 股数必须为正 (要清掉就直接从列表里删除)")
        if shares % 100 and not code.startswith("688"):
            notes.append(f"{code} 股数 {shares} 不是100的整数倍 (零股只能卖不能买, 已按原样记录)")
        bp = p.get("buy_price")
        if bp in (None, "", 0):
            if code in old_by_code:
                bp = old_by_code[code]["buy_price"]
                notes.append(f"{code} 未填成本价, 沿用服务器记录 {bp:.3f}")
            else:
                raise SystemExit(f"ERROR: 第{i}条 {code} 是新增持仓, 必须填 buy_price")
        bp = float(bp)

        bd = p.get("buy_date") or None
        if bd:
            j = cal_pos(cal, bd)
            if pd.Timestamp(bd) > cal[-1]:
                notes.append(f"{code} 成交日 {bd} 还没有行情数据, 按最后交易日 "
                             f"{cal[-1].date()} 计算持有期")
            open_sig = cal[max(0, j - 1)]      # 成交日的前一交易日 = 出信号那天
            open_exec = cal[j]
        elif code in old_by_code and old_by_code[code].get("open_signal_date"):
            open_sig = pd.Timestamp(old_by_code[code]["open_signal_date"])
            open_exec = pd.Timestamp(old_by_code[code].get("open_date") or open_sig)
            notes.append(f"{code} 未填成交日, 沿用服务器开仓日 {open_exec.date()}")
        else:
            open_sig, open_exec = cal[-2] if len(cal) > 1 else cal[-1], cal[-1]
            notes.append(f"{code} 未填成交日且服务器无记录, 当成 {open_exec.date()} 刚买入 "
                         f"(将完整持有 {HOLD_DAYS} 个交易日)")
        new_lots.append({"id": st["next_lot_id"] + len(new_lots), "code": code,
                         "shares": shares, "buy_price": bp,
                         "open_signal_date": str(pd.Timestamp(open_sig).date()),
                         "open_date": str(pd.Timestamp(open_exec).date()),
                         "source": "sync"})

    dup = [c for c in {l["code"] for l in new_lots}
           if sum(1 for l in new_lots if l["code"] == c) > 1]
    if dup:
        raise SystemExit(f"ERROR: 对账列表里 {dup} 重复出现。同一只股票请合并成一条")

    bak = backup_state("sync")
    st["cash"] = float(payload["cash"])
    st["lots"] = new_lots
    st["next_lot_id"] += len(new_lots)
    # 对账后原挂单计划作废: 下次运行直接按真实持仓重新出信号, 避免重复结算
    dropped_pending = st.get("pending")
    st["pending"] = None
    st["last_synced_at"] = datetime.now().isoformat(timespec="seconds")
    after = snapshot(st, kl, names)
    st["history"].append({"type": "sync", "at": st["last_synced_at"],
                          "note": payload.get("note", ""),
                          "before": {"cash": before["cash"], "equity": before["equity"],
                                     "positions": [(r["code"], r["shares"]) for r in before["positions"]]},
                          "after": {"cash": after["cash"], "equity": after["equity"],
                                    "positions": [(r["code"], r["shares"]) for r in after["positions"]]},
                          "dropped_pending": bool(dropped_pending)})

    W = 68
    print(f"\n{'='*W}")
    print("  人工对账结果 (以你填写的为准)")
    print(f"{'='*W}")
    print(f"  现金   : ¥{before['cash']:,.2f}  ->  ¥{after['cash']:,.2f}")
    print(f"  总资产 : ¥{before['equity']:,.2f}  ->  ¥{after['equity']:,.2f}")
    b = {r["code"]: r["shares"] for r in before["positions"]}
    a = {r["code"]: r["shares"] for r in after["positions"]}
    print("  持仓变化:")
    for code in sorted(set(b) | set(a)):
        nm = names.get(code, "")
        if code not in b:
            print(f"    + 新增 {code} {nm} x{a[code]}")
        elif code not in a:
            print(f"    - 移除 {code} {nm} x{b[code]}")
        elif b[code] != a[code]:
            print(f"    ~ 调整 {code} {nm} {b[code]} -> {a[code]}")
        else:
            print(f"      不变 {code} {nm} x{a[code]}")
    if not (set(b) | set(a)):
        print("    (空仓)")
    for nt in notes:
        print(f"  注意: {nt}")
    if dropped_pending:
        print(f"  已作废挂单计划 ({dropped_pending['signal_date']} 的信号), "
              f"下次运行按真实持仓重新出信号")
    if bak:
        print(f"  旧状态备份: {bak}")
    save_state(st)
    print("\n  下一步: 重新跑一次信号即可拿到基于真实持仓的操作建议")
    print("  python scripts/live_signal.py")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════
t0 = datetime.now()
state = load_state()
names = load_names()
kl = Klines()

# 轻量入口: 看状态 / 导表单 / 对账 —— 不碰数据管道, 秒级返回
if args.status:
    print_status(snapshot(state, kl, names))
    sys.exit(0)
if args.sync_template:
    do_sync_template(state, kl, names)
    sys.exit(0)
if args.sync:
    do_sync(state, kl, names)
    sys.exit(0)

print(f"加载 {TRAIN_PATH.name} ...")
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
df["code"] = df["code"].astype(str)
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
_lab_ok = df[LABEL_RAW].notna()
_last_lab_date = df.loc[_lab_ok, "date"].max()
df = df[_lab_ok | (df["date"] > _last_lab_date)]
if args.pit_universe:
    df = apply_pit_universe(df, args.pit_universe)

print("构建增强特征...")
mkt_df, mkt_features, regime_src = compute_market_features()
regime_series = build_regime_series(regime_src)
df = df.merge(mkt_df, on="date", how="left")
for c in mkt_features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
ovn_df, ovn_features = compute_overnight_features(df["code"].unique())
df = df.merge(ovn_df[["date", "code"] + ovn_features], on=["date", "code"], how="left")
df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

all_cols = [c for c in df.columns if c not in SKIP_COLS and c not in EXCLUDED_FEATS and is_valid_feat(c)]
all_features = [f for f in all_cols if f not in LEAKAGE_FEATS]

if args.as_of:
    df = df[df["date"] <= pd.Timestamp(args.as_of)]
    if df.empty:
        raise SystemExit(f"ERROR: {args.as_of} 之前没有数据")

all_dates = sorted(df["date"].unique())
date_pos = {d: i for i, d in enumerate(all_dates)}
SIGNAL_DATE = all_dates[-1]
mkt_last = pd.Timestamp(mkt_df["date"].max())
if mkt_last > pd.Timestamp(SIGNAL_DATE) and not args.as_of:
    msg = (f"训练集最新 {pd.Timestamp(SIGNAL_DATE).date()} 落后于 K线最新 {mkt_last.date()}, "
           f"需要先重建特征 (training_data)")
    if not args.allow_stale:
        raise SystemExit(f"ERROR: {msg}\n  确认要用旧特征出信号请加 --allow-stale")
    print(f"WARN: {msg}")

print(f"  {len(df):,} 行, {df['code'].nunique()} 只 | 信号日 {pd.Timestamp(SIGNAL_DATE).date()}")
features = select_features(df, all_features, args.feat_cutoff)

# ── 结算上一次的挂单 ──
pending = state.get("pending")
if pending:
    p_sig = pd.Timestamp(pending["signal_date"])
    if p_sig not in date_pos:
        raise SystemExit(f"ERROR: 挂单信号日 {p_sig.date()} 不在交易日历内")
    gp = date_pos[p_sig]
    exec_date = all_dates[gp + 1] if gp + 1 < len(all_dates) else None
    if exec_date is None:
        print(f"\n[结算] {p_sig.date()} 的挂单尚无 T+1 行情, 计划保持有效, 本次不重新出信号。")
        print(f"耗时 {(datetime.now()-t0).total_seconds():.0f}s")
        sys.exit(0)
    print(f"\n[结算] {p_sig.date()} 信号 -> {pd.Timestamp(exec_date).date()} "
          f"{'开盘' if EXEC_FIELD == 'open' else '尾盘'}成交"
          f"{' (手工回报)' if args.confirm else ' (按行情自动模拟)'}")
    res = settle(state, pending, exec_date, kl, names, all_dates)
    fills, rej = res if isinstance(res, tuple) else (res, [])
    for f in fills:
        act = "卖出" if f["action"] == "sell" else "买入"
        print(f"    {act} {f['code']}{f.get('name','')} x{f['shares']} @ {f['price']} "
              f"手续费 {f['fee']} 现金 {f['net']:+,.0f}")
    for r in rej:
        print(f"    跳过 {'买入' if r['action']=='buy' else '卖出'} {r['code']}: {r['reason']}")
    if not fills:
        print("    无成交")
    state["history"].append({"signal_date": str(p_sig.date()),
                             "exec_date": str(pd.Timestamp(exec_date).date()),
                             "fills": fills, "rejected": rej})
    state["pending"] = None
else:
    print("\n[结算] 无待结算挂单")

# ── 训练 + 预测 ──
seq = date_pos[SIGNAL_DATE]
cutoff = all_dates[seq - LABEL_HORIZON]
train_df = df[(df["date"] < cutoff) & df[LABEL].notna()]
if train_df["date"].nunique() < MIN_TRAIN_DAYS:
    raise SystemExit(f"ERROR: 训练集只有 {train_df['date'].nunique()} 天, 少于 {MIN_TRAIN_DAYS}")
print(f"\n[训练] 样本 < {pd.Timestamp(cutoff).date()} | "
      f"{train_df['date'].nunique()} 天 {len(train_df):,} 行 {len(features)} 特征")
X = train_df.groupby("code")[features].transform(lambda c: c.ffill().fillna(0))
model = lgb.LGBMRegressor(**LOCKED_PARAMS).fit(X, train_df[LABEL])

tm = df["date"] == SIGNAL_DATE
Xt = df.loc[tm, features].fillna(0)
preds = model.predict(Xt)
ranked = (pd.DataFrame({"code": df.loc[tm, "code"].astype(str).str[:6].values, "pred": preds})
          .sort_values("pred", ascending=False).reset_index(drop=True))
pred_map = dict(zip(ranked["code"], ranked["pred"]))

blocked = set()
if args.reversal_guard > 0 and "ret_5d" in df.columns:
    r5 = df.loc[tm, ["code", "ret_5d"]].dropna()
    if len(r5) > 10:
        pct = r5["ret_5d"].rank(pct=True)
        blocked = set(r5.loc[pct >= 1 - args.reversal_guard, "code"].astype(str).str[:6])

# ── regime ──
in_cash = bool(regime_series.get(pd.Timestamp(SIGNAL_DATE), False)) if regime_series is not None else False
was_in_cash = bool(state.get("last_in_cash", False))

_r = regime_src[regime_src["date"] == pd.Timestamp(SIGNAL_DATE)]
breadth = float(_r["breadth_above_ma"].iloc[0]) if len(_r) else float("nan")
mkt_c = float(_r["mkt_close"].iloc[0]) if len(_r) else float("nan")
mkt_ma = float(_r[f"mkt_ma{args.regime_ma}"].iloc[0]) if len(_r) else float("nan")

# ── 生成计划 ──
sell_plan, keep_plan = [], []
for lot in state["lots"]:
    _held = held_days(all_dates, lot, SIGNAL_DATE)
    matured = _held >= HOLD_DAYS
    ref = kl.px(lot["code"], SIGNAL_DATE, "close") or lot["buy_price"]
    row = {"code": lot["code"], "name": names.get(str(lot["code"])[:6], ""),
           "shares": lot["shares"], "buy_price": round(lot["buy_price"], 3),
           "ref_close": round(ref, 3), "open_date": lot.get("open_date"),
           "held_days": _held,
           "pnl_pct": round((ref / lot["buy_price"] - 1) * 100, 2)}
    if matured or in_cash:
        row["reason"] = "持满到期" if matured else "大盘转弱清仓"
        row["est_proceeds"] = round(lot["shares"] * fill_px(ref, "sell") * (1 - TRADE_COST), 2)
        sell_plan.append(row)
    else:
        keep_plan.append(row)

# 换仓日判定 (periodic): 距上次换仓满 HOLD_DAYS 个交易日; 另外两种自愈情形 ——
#   a) 卖完后已空仓 (漏跑/对账导致错过周期点时, 不至于长期空着)
#   b) 大盘刚由弱转强, 当天立即回场 (与回测一致)
_last_rebal = state.get("last_rebal_signal_date")
is_rebal = ((not PERIODIC)
            or _last_rebal is None
            or (seq - cal_pos(all_dates, _last_rebal)) >= HOLD_DAYS
            or (was_in_cash and not in_cash)
            or (not keep_plan and not in_cash))

cash_after_sell = state["cash"] + sum(r["est_proceeds"] for r in sell_plan)
buy_plan, alt_plan = [], []
if not in_cash and is_rebal:
    equity = cash_after_sell + sum(r["shares"] * r["ref_close"] for r in keep_plan)
    denom = 1 if PERIODIC else HOLD_DAYS
    remaining = min(equity / denom, cash_after_sell)
    held = {str(r["code"])[:6] for r in keep_plan}
    bought = 0
    for code in ranked["code"]:
        if bought >= TRANCHE_N and len(alt_plan) >= args.alternates:
            break
        if code in held or code in blocked:
            continue
        ref = kl.px(code, SIGNAL_DATE, "close")
        if ref is None:
            continue
        px = fill_px(ref, "buy")
        if bought < TRANCHE_N:
            alloc = remaining / (TRANCHE_N - bought)
            shares = int(alloc / (px * 100)) * 100
            if shares <= 0:
                alt_plan.append({"code": code, "name": names.get(code, ""),
                                 "ref_close": round(ref, 3), "pred": round(pred_map[code], 6),
                                 "note": "预算不足一手"})
                continue
            gross = shares * px
            fee = max(gross * TRADE_COST, MIN_FEE)
            remaining -= gross + fee
            bought += 1
            buy_plan.append({"code": code, "name": names.get(code, ""),
                             "shares": shares, "ref_close": round(ref, 3),
                             "est_price": round(px, 3), "est_cost": round(gross + fee, 2),
                             "pred": round(pred_map[code], 6),
                             "budget": round(alloc, 2)})
        else:
            alt_plan.append({"code": code, "name": names.get(code, ""),
                             "ref_close": round(ref, 3), "pred": round(pred_map[code], 6),
                             "note": "候补"})

# ── 打印 ──
mv = sum(r["shares"] * r["ref_close"] for r in keep_plan + sell_plan)
equity_now = state["cash"] + mv
W = 68
print(f"\n{'='*W}")
print(f"  操作建议 | 信号日 {pd.Timestamp(SIGNAL_DATE).date()} -> "
      f"下一交易日{'开盘' if EXEC_FIELD == 'open' else '尾盘(14:50后)'}执行")
print(f"{'='*W}")
print(f"  方案     : {args.label}标签 / {args.portfolio_mode} / 持有{HOLD_DAYS}天 / "
      f"{TARGET_POSITIONS}只 / 滑点{SLIPPAGE*100:.2f}%")
print(f"  总资产   : ¥{equity_now:,.0f}  (现金 ¥{state['cash']:,.0f} + 持仓 ¥{mv:,.0f})")
print(f"  大盘     : 均价 {mkt_c:.2f} vs MA{args.regime_ma} {mkt_ma:.2f} | "
      f"广度 {breadth*100:.1f}% (阈值 {args.regime_breadth*100:.0f}%)")
print(f"  择时     : {'空仓 (清掉全部持仓, 不开新仓)' if in_cash else '持仓'}"
      f" | 今日{'是' if is_rebal else '非'}换仓日")

print(f"\n  ── 卖出 ({len(sell_plan)}) ──")
if sell_plan:
    for r in sell_plan:
        print(f"    {r['code']} {r['name']:<8} x{r['shares']:>6} 参考 {r['ref_close']:>8.2f} "
              f"成本 {r['buy_price']:>8.2f} 浮盈 {r['pnl_pct']:+6.2f}% | {r['reason']} "
              f"(已持{r['held_days']}日)")
else:
    print("    无")

print(f"\n  ── 买入 ({len(buy_plan)}) ──")
if buy_plan:
    for i, r in enumerate(buy_plan, 1):
        print(f"    {i}. {r['code']} {r['name']:<8} x{r['shares']:>6} 参考 {r['ref_close']:>8.2f} "
              f"约 ¥{r['est_cost']:>9,.0f}  pred={r['pred']:+.4f}")
    print(f"    合计约 ¥{sum(r['est_cost'] for r in buy_plan):,.0f} / 可用 ¥{cash_after_sell:,.0f}")
elif in_cash:
    print("    无 (大盘弱势, 空仓等待)")
elif not is_rebal:
    print("    无 (非换仓日, 继续持有)")
else:
    print("    无")

if alt_plan:
    print(f"\n  ── 候补 (若上面某只涨停/停牌买不进, 按顺序顶上) ──")
    for r in alt_plan[:args.alternates]:
        print(f"    {r['code']} {r['name']:<8} 参考 {r['ref_close']:>8.2f} "
              f"pred={r['pred']:+.4f} {r['note']}")

print(f"\n  ── 继续持有 ({len(keep_plan)}) ──")
for r in keep_plan:
    print(f"    {r['code']} {r['name']:<8} x{r['shares']:>6} 浮盈 {r['pnl_pct']:+6.2f}% "
          f"(已持{r['held_days']}日, 满{HOLD_DAYS}日卖出)")
if not keep_plan:
    print("    无")
print(f"\n  注: 股数按 {pd.Timestamp(SIGNAL_DATE).date()} 收盘价估算。次日价格变动导致资金不足时, "
      f"按可用资金向下取整到 100 股。")

# ── 每日推荐榜 ──
# 模型当天打分最高的前 20 只, 自带名称/分数/收盘价, 网页不用再查行情。
# 注意它只反映模型排序, 不含"买不买得起一手"的资金约束 —— 每只预算
# = 总资产/持仓数, 买不起一手的会被跳过, 所以实际买入以 buy 为准。
recommend = []
for _i, _c in enumerate(list(ranked["code"].head(20))):
    _px = kl.px(_c, SIGNAL_DATE, "close")
    recommend.append({
        "rank": _i + 1,
        "code": _c,
        "name": names.get(_c, ""),
        "pred": round(float(pred_map[_c]), 6),
        "close": round(_px, 3) if _px else None,
        "blocked": _c in blocked,
    })

# ── 落盘 ──
plan = {
    "signal_date": str(pd.Timestamp(SIGNAL_DATE).date()),
    "seq": int(seq),
    "exec_mode": args.exec_mode,
    "exec_hint": "下一交易日" + ("开盘" if EXEC_FIELD == "open" else "尾盘"),
    "in_cash": in_cash, "was_in_cash": was_in_cash, "is_rebal": bool(is_rebal),
    "breadth": round(breadth, 4), "mkt_close": round(mkt_c, 3), "mkt_ma": round(mkt_ma, 3),
    "equity": round(equity_now, 2), "cash": round(state["cash"], 2),
    "cash_after_sell": round(cash_after_sell, 2),
    "sell": sell_plan, "buy": buy_plan, "hold": keep_plan,
    "alternates": alt_plan[:args.alternates],
    "ranked": list(ranked["code"].head(60)),
    "recommend": recommend,
    "blocked": sorted(blocked),
    "config": fingerprint(),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}
if not args.dry_run:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    pp = LIVE_DIR / f"{PLAN_PREFIX}_{plan['signal_date']}.json"
    pp.write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n计划已保存: {pp}")

state["pending"] = {"signal_date": plan["signal_date"],
                    "in_cash": in_cash, "is_rebal": bool(is_rebal),
                    "ranked": plan["ranked"], "blocked": plan["blocked"]}
state["last_signal_date"] = plan["signal_date"]
state["last_in_cash"] = in_cash
if is_rebal and PERIODIC and not in_cash:
    state["last_rebal_signal_date"] = plan["signal_date"]
# 交易日历随状态一起存盘, 让 --status / --sync 不必重跑数据管道
state["calendar"] = [str(pd.Timestamp(d).date()) for d in all_dates]
save_state(state)
print(f"耗时 {(datetime.now()-t0).total_seconds():.0f}s")
