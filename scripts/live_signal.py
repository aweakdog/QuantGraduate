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
import hashlib
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
ap.add_argument("--lot-flex", type=float, default=0.0,
                help="整手粒度救济: 槽位预算买不起一手(100股)时, 若一手成本 <= 预算"
                     "*(1+flex) 且现金足够, 仍买这一手而不是沿排名换下一只。"
                     "与回测 --lot-flex 同义。0 = 关闭")
ap.add_argument("--n-features", type=int, default=80)
ap.add_argument("--corr-threshold", type=float, default=0.9)
ap.add_argument("--feat-cutoff", default="2023-09-19",
                help="特征筛选只允许用该日期之前的数据 (与回测保持一致)")
ap.add_argument("--features-from",
                default="wf_daily_REGRESS_CHK_ts2022-09-01_te2026-07-27_cap50000.json",
                help="直接复用回测结果 json 里的 selected_features (data/processed/ 下); "
                     "设为 none 则现场重新筛选。默认与 live_config.FEATURES_FROM 一致 "
                     "(修复 drop_market_wide 后的干净 80 特征); 旧的 fundfix 58 特征集"
                     "含 8 个市场级日历假象特征, 不要再用")
ap.add_argument("--capital", type=float, default=20000.0, help="初始本金 (仅 --init 时使用)")
ap.add_argument("--init", action="store_true", help="重置状态文件, 用 --capital 作为起始现金")
ap.add_argument("--state", default="state.json", help="状态文件名 (data/live/ 下)")
ap.add_argument("--confirm", default=None, help="手工成交回报 json 路径, 用它替代自动结算")
ap.add_argument("--require-confirm", action="store_true",
                help="实盘模式: 禁止按行情自动记账。有挂单未确认时原地等待, 既不结算也不出"
                     "新信号, 直到用 --confirm 填入真实成交价。适合真金白银在跑的条线 —— "
                     "自动记账会假设你按参考价成交了, 你若没下单账目就会悄悄偏离现实")
ap.add_argument("--dry-run", action="store_true", help="只打印, 不写状态/不落盘")
ap.add_argument("--allow-stale", action="store_true", help="训练集比K线旧时仍继续")
ap.add_argument("--as-of", default=None,
                help="假装数据只到该日期 (回放/补跑/自测用), 不使用之后的任何数据")
ap.add_argument("--alternates", type=int, default=8, help="额外输出几只候补股")
ap.add_argument("--seed-ensemble", default="42,7,123,2024,31337",
                help="逗号分隔的 LightGBM 种子列表。每个种子训一套模型, 预测取平均。"
                     "同 IC 下前3名选股方差是单种子的主要脆弱点(单种子回测 IR 均值"
                     "0.92±0.26, 5种子集成 1.23~1.30, 见 docs/progress_2026-08-02.md)。"
                     "传单个种子如 '42' 退回旧行为")
ap.add_argument("--preds-cache", default=None,
                help="预测结果缓存路径。四条线的模型完全相同(特征/标签/训练集/超参都不"
                     "依赖 tranche_n 与本金), 各训一遍是 4 倍浪费。指定同一个缓存文件, "
                     "则当天第一条线训练并写入, 其余直接读取。输入指纹不符会自动重训")
ap.add_argument("--status", action="store_true",
                help="只打印当前持仓/现金/待执行计划, 不跑模型 (秒级)")
ap.add_argument("--sync", default=None,
                help="整体对账: 用该 json 里的真实持仓/现金覆盖服务器状态 (以券商App为准)")
ap.add_argument("--sync-template", action="store_true",
                help="导出一份以当前状态预填的对账表单 json, 改完再用 --sync 回填")
ap.add_argument("--set-cash", type=float, default=None,
                help="现金校准: 把记录的现金改成券商App里的真实数字。"
                     "用于修正手续费/成交价的累积误差 —— 这是修账不是盈亏, "
                     "所以本金(initial_capital)不动, 收益率会随之修正")
ap.add_argument("--cash-flow", type=float, default=None,
                help="出入金: 正数存入, 负数取出。这是本金变动不是盈亏, "
                     "所以现金和本金同额增减, 收益率保持不变")
ap.add_argument("--drop-lot", default=None,
                help="删除一笔持仓 (填6位代码)。必须同时说明现金怎么算: "
                     "配 --sold-at <成交价> 表示已在券商卖出(现金增加); "
                     "配 --phantom 表示系统记错了从没持有过(现金不变)")
ap.add_argument("--sold-at", type=float, default=None,
                help="配合 --drop-lot: 真实卖出价, 现金按 股数x价格-手续费 增加")
ap.add_argument("--phantom", action="store_true",
                help="配合 --drop-lot: 这笔持仓本就不存在(记账错误), 只删记录不动现金")
ap.add_argument("--drop-shares", type=int, default=None,
                help="配合 --drop-lot: 只处理部分股数(部分卖出/部分记错), "
                     "不填则整笔删除。剩下的仓位保留原开仓日, 到期节奏不变")
ap.add_argument("--note", default="", help="给现金/持仓类操作附一句备注")
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
ENSEMBLE_SEEDS = [int(s) for s in str(args.seed_ensemble).split(",") if s.strip()]
if not ENSEMBLE_SEEDS:
    raise SystemExit("ERROR: --seed-ensemble 不能为空")
TRADE_COST = 0.0006
MIN_FEE = 5.0
SLIPPAGE = args.slippage
LOT_FLEX = args.lot_flex
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
    """到期时钟: 距下次该评估还有多久, 用来判定是否持满 HOLD_DAYS。

    注意续持会把它归零 (见 settle 里的 roll 分支) —— 这正是它该有的行为,
    续持就是重新起算一个换仓周期。要显示"这笔一共拿了多久"请用 tenure_days。
    """
    return cal_pos(cal, signal_date) - cal_pos(cal, lot["open_signal_date"])


def tenure_days(cal, lot, signal_date):
    """真实持有时长: 从最初开仓那天算起, 续持不清零。

    与 held_days 回答的是不同问题, 两个都要有:
      held_days   -> "什么时候会动它"  (到期时钟, 续持归零)
      tenure_days -> "这笔一共拿了多久" (真实时长, 只增不减)
    只显示前者会让人以为刚买的; 只显示后者则看不出哪天该操作。

    老持仓没有 first_open_signal_date 字段(续持功能上线前建的), 此时
    两者本就相等, 退回 open_signal_date 即为正确答案。
    """
    first = lot.get("first_open_signal_date") or lot["open_signal_date"]
    return cal_pos(cal, signal_date) - cal_pos(cal, first)


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
        # 强制先卖后买, 不管上报顺序如何。
        # 换仓日通常是"卖掉到期的, 用卖出所得买新的" —— A股卖出资金当天可用,
        # 所以这在现实里成立。但若先扣买入款, 中途现金会假性为负;
        # 更要紧的是下面那道现金检查会误判。排序后语义与实际执行顺序一致。
        manual = sorted(manual, key=lambda f: 0 if f.get("action") == "sell" else 1)
        cash_before_manual = float(st["cash"])
        for f in manual:
            code = str(f["code"])[:6]
            px, sh = float(f["price"]), int(f["shares"])
            gross = px * sh
            fee = max(gross * TRADE_COST, MIN_FEE)
            if f["action"] == "sell":
                # 支持部分卖出: 实盘常有只成交一部分的情况(挂单没全成、
                # 自己只卖了一半)。原先要求股数与持仓完全相等, 一旦部分成交
                # 就直接报错退出, 逼人去做整体对账。
                # 多笔同代码时按开仓先后(FIFO)扣减; 剩余仓位保留原
                # open_signal_date, 所以到期节奏不变。
                lots = [l for l in st["lots"] if str(l["code"])[:6] == code]
                if not lots:
                    raise SystemExit(f"ERROR: 回报里卖出了 {code}, 但持仓里没有这只")
                total = sum(l["shares"] for l in lots)
                if sh > total:
                    raise SystemExit(
                        f"ERROR: 回报里卖出 {code} x{sh} 股, 但只持有 {total} 股")
                lots.sort(key=lambda l: (str(l.get("open_date") or ""), l.get("id") or 0))
                left = sh
                for l in lots:
                    if left <= 0:
                        break
                    take = min(l["shares"], left)
                    l["shares"] -= take
                    left -= take
                st["lots"] = [l for l in st["lots"] if l["shares"] > 0]
                st["cash"] += gross - fee
                fills.append({"code": code, "action": "sell", "shares": sh, "price": px,
                              "fee": round(fee, 2), "net": round(gross - fee, 2),
                              "partial": sh < total, "remaining": total - sh,
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

        # 现金变负说明上报的成交自相矛盾 —— 真实账户不可能用没有的钱买入。
        # 最常见的原因是"买入勾了、卖出忘了勾": 那笔卖出的钱没进账,
        # 却把买入的钱扣了。此时必须拦住, 否则账目从此带着一个负现金往下走。
        if st["cash"] < -0.01:
            buys = sum(-f["net"] for f in fills if f["action"] == "buy")
            sells = sum(f["net"] for f in fills if f["action"] == "sell")
            raise SystemExit(
                f"ERROR: 按上报的成交算下来现金会变成 ¥{st['cash']:,.2f} (负数), 已拒绝。\n"
                f"  原有现金 ¥{cash_before_manual:,.2f} + 卖出所得 ¥{sells:,.2f} "
                f"- 买入支出 ¥{buys:,.2f}\n"
                "  常见原因: 买入报了、卖出漏报了。换仓日是靠卖出所得来买的,\n"
                "  所以卖出那几笔也要一起确认。\n"
                "  若确认买卖都没漏报, 说明系统记的现金本身偏低, 请先用「校准现金」\n"
                "  改成券商App里的真实数字, 再来确认成交。")
        return fills

    # ── 0. 目标组合 ──
    # 到期但仍在目标名单里的就续持, 不卖了再买回 —— 一次往返要付
    # (佣金+滑点)*2, 白付这笔钱却回到同样的持仓。与回测 roll_set 一致。
    roll_set = set()
    if PERIODIC and pending["is_rebal"] and not in_cash:
        _blocked = set(pending.get("blocked", []))
        for code in pending["ranked"]:
            if len(roll_set) >= TRANCHE_N:
                break
            c6 = str(code)[:6]
            if c6 in _blocked:
                continue
            roll_set.add(c6)

    # ── 1. 卖出 ──
    keep, rejected = [], []
    n_rolled = 0
    for lot in st["lots"]:
        matured = held_days(cal, lot, sig_date) >= HOLD_DAYS
        if not (matured or in_cash):
            keep.append(lot)
            continue
        if matured and not in_cash and str(lot["code"])[:6] in roll_set:
            # 续持: 重置到期时钟(下个换仓周期再评估), 成本价与开仓日不动。
            # first_open_signal_date 留住真实起始日, 好让页面显示真实持有时长。
            lot.setdefault("first_open_signal_date", lot["open_signal_date"])
            lot["open_signal_date"] = sig_date
            lot["rolled"] = lot.get("rolled", 0) + 1
            keep.append(lot)
            n_rolled += 1
            fills.append({"code": str(lot["code"])[:6],
                          "name": names.get(str(lot["code"])[:6], ""),
                          "action": "roll", "shares": lot["shares"],
                          "price": round(lot["buy_price"], 3), "fee": 0.0, "net": 0.0,
                          "reason": "still_ranked", "source": "auto"})
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
        # 从已持仓数起算, 否则续持的(以及卖单被拒仍持有的)不计数, 会超过 TRANCHE_N
        bought = len(st["lots"])
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
                # 整手粒度救济: 与回测引擎同义 (wf_v35 --lot-flex)
                if LOT_FLEX > 0 and px * 100 <= alloc * (1 + LOT_FLEX):
                    shares = 100
                else:
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


def do_set_cash(st, kl, names):
    """现金校准: 把记录的现金改成券商 App 里的真实数字。

    为什么需要: 自动记账用的是收盘价 + 估算的手续费, 和你真实成交价、真实
    佣金总有零点几个百分点的差。几十次换仓累积下来就是可观的偏差。

    关键: 这是"修账", 不是盈亏, 也不是出入金。所以只动 cash, 不动
    initial_capital —— 收益率的分母保持原样, 分子被修正, 于是收益率跟着
    修正到真实水平。这正是我们想要的: 之前记的收益率是虚的, 现在变实。
    """
    new_cash = float(args.set_cash)
    if new_cash < 0:
        raise SystemExit(f"ERROR: 现金不能为负 (收到 {new_cash})")
    before = snapshot(st, kl, names)
    old_cash = float(st["cash"])
    delta = new_cash - old_cash

    # 防手滑: 一次性改动超过总资产的 50% 极可能是输错(比如少打一位)
    if before["equity"] > 0 and abs(delta) > before["equity"] * 0.5:
        raise SystemExit(
            f"ERROR: 这次校准要把现金从 ¥{old_cash:,.2f} 改成 ¥{new_cash:,.2f} "
            f"(变动 ¥{delta:+,.2f}), 超过总资产 ¥{before['equity']:,.2f} 的一半, 已拒绝。\n"
            "  校准是用来修几百块的累积误差的。如果确实要大改, 说明该用整体对账\n"
            "  (--sync) 把持仓也一起改; 若是存取现金请用 --cash-flow。")

    bak = backup_state("setcash")
    st["cash"] = new_cash
    after = snapshot(st, kl, names)
    st.setdefault("history", []).append({
        "type": "set_cash", "at": datetime.now().isoformat(timespec="seconds"),
        "note": args.note, "delta": round(delta, 2),
        "before": {"cash": before["cash"], "equity": before["equity"]},
        "after": {"cash": after["cash"], "equity": after["equity"]},
    })

    W = 68
    print(f"\n{'='*W}\n  现金校准 (修账, 不计为盈亏)\n{'='*W}")
    print(f"  现金   : ¥{before['cash']:,.2f}  ->  ¥{after['cash']:,.2f}  ({delta:+,.2f})")
    print(f"  总资产 : ¥{before['equity']:,.2f}  ->  ¥{after['equity']:,.2f}")
    print(f"  本金   : ¥{st.get('initial_capital', 0):,.2f}  (不变)")
    if before["total_return_pct"] is not None and after["total_return_pct"] is not None:
        print(f"  收益率 : {before['total_return_pct']:+.2f}%  ->  "
              f"{after['total_return_pct']:+.2f}%  (修正到真实水平)")
    if args.note:
        print(f"  备注   : {args.note}")
    if bak:
        print(f"  已备份 : {bak.name}")
    save_state(st)


def do_cash_flow(st, kl, names):
    """出入金: 存入(正)或取出(负)。

    关键: 这是本金变动, 不是赚了或亏了。所以 cash 和 initial_capital 同额
    增减 —— 否则存进 1 万会被算成"赚了 1 万", 收益率立刻虚高一大截。
    """
    amt = float(args.cash_flow)
    if amt == 0:
        raise SystemExit("ERROR: 金额不能为 0")
    before = snapshot(st, kl, names)
    if amt < 0 and float(st["cash"]) + amt < 0:
        raise SystemExit(
            f"ERROR: 要取出 ¥{-amt:,.2f}, 但可用现金只有 ¥{st['cash']:,.2f}。\n"
            "  持仓市值不能直接取现, 需要先卖出。")

    bak = backup_state("cashflow")
    st["cash"] = float(st["cash"]) + amt
    st["initial_capital"] = float(st.get("initial_capital") or 0) + amt
    after = snapshot(st, kl, names)
    st.setdefault("history", []).append({
        "type": "deposit" if amt > 0 else "withdraw",
        "at": datetime.now().isoformat(timespec="seconds"),
        "note": args.note, "amount": round(amt, 2),
        "before": {"cash": before["cash"], "equity": before["equity"],
                   "initial_capital": before["initial_capital"]},
        "after": {"cash": after["cash"], "equity": after["equity"],
                  "initial_capital": after["initial_capital"]},
    })

    W = 68
    act = "存入" if amt > 0 else "取出"
    print(f"\n{'='*W}\n  {act}现金 ¥{abs(amt):,.2f} (本金变动, 不计为盈亏)\n{'='*W}")
    print(f"  现金   : ¥{before['cash']:,.2f}  ->  ¥{after['cash']:,.2f}")
    print(f"  总资产 : ¥{before['equity']:,.2f}  ->  ¥{after['equity']:,.2f}")
    print(f"  本金   : ¥{before['initial_capital']:,.2f}  ->  "
          f"¥{after['initial_capital']:,.2f}  (同额调整)")
    if before["total_return_pct"] is not None and after["total_return_pct"] is not None:
        print(f"  收益率 : {before['total_return_pct']:+.2f}%  ->  "
              f"{after['total_return_pct']:+.2f}%  (应基本不变)")
    n = TRANCHE_N
    print(f"  每只预算: ¥{before['equity']/n:,.0f}  ->  ¥{after['equity']/n:,.0f} "
          f"(总资产/{n}只, 决定能买多高价的股票)")
    if args.note:
        print(f"  备注   : {args.note}")
    if bak:
        print(f"  已备份 : {bak.name}")
    save_state(st)


def do_drop_lot(st, kl, names):
    """删除一笔持仓。

    有两种完全相反的情形, 现金的处理正好相反, 所以必须由调用方显式指明,
    绝不猜:

      --sold-at <价>  你已经在券商 App 里卖掉了 —— 钱真的回到账户,
                      所以 现金 += 股数x价格 - 手续费, 并记一笔卖出流水。
      --phantom       系统记错了, 你从没持有过这只 —— 纯修账, 现金不动。

    猜错任何一边都会让账目失真: 前者少记钱会让总资产凭空缩水,
    后者多记钱会让总资产凭空膨胀。
    """
    code = str(args.drop_lot).zfill(6)[:6]
    said_sold = args.sold_at is not None
    if said_sold and args.phantom:
        raise SystemExit("ERROR: --sold-at 和 --phantom 不能同时用 —— "
                         "一个现金增加一个现金不变, 只能选一个")
    if not said_sold and not args.phantom:
        raise SystemExit(
            "ERROR: 删除持仓必须说明现金怎么算, 二选一:\n"
            "  --sold-at <成交价>  已在券商卖出 -> 现金增加\n"
            "  --phantom           系统记错了   -> 现金不变")
    if args.sold_at is not None and args.sold_at <= 0:
        raise SystemExit(f"ERROR: 卖出价必须为正 (收到 {args.sold_at})")

    hits = [l for l in st["lots"] if str(l["code"])[:6] == code]
    if not hits:
        held = ", ".join(sorted(str(l["code"])[:6] for l in st["lots"])) or "(空仓)"
        raise SystemExit(f"ERROR: 持仓里没有 {code}。当前持仓: {held}")
    nm = names.get(code, "")
    total = sum(l["shares"] for l in hits)
    avg_cost = (sum(l["shares"] * l["buy_price"] for l in hits) / total) if total else 0

    n_drop = total if args.drop_shares is None else int(args.drop_shares)
    if n_drop <= 0:
        raise SystemExit(f"ERROR: 股数必须为正 (收到 {n_drop})")
    if n_drop > total:
        raise SystemExit(f"ERROR: 要处理 {n_drop} 股, 但 {code} 只持有 {total} 股")
    partial = n_drop < total
    before = snapshot(st, kl, names)

    bak = backup_state("droplot")
    # 多笔同代码时按开仓先后(FIFO)扣减; 剩下的仓位不改
    # open_signal_date, 所以到期节奏不变
    hits.sort(key=lambda l: (str(l.get("open_date") or ""), l.get("id") or 0))
    left = n_drop
    for l in hits:
        if left <= 0:
            break
        take = min(l["shares"], left)
        l["shares"] -= take
        left -= take
    st["lots"] = [l for l in st["lots"] if l["shares"] > 0]

    if args.sold_at is not None:
        px = float(args.sold_at)
        gross = n_drop * px
        fee = max(gross * TRADE_COST, MIN_FEE)
        st["cash"] = float(st["cash"]) + gross - fee
        kind, detail = "drop_sold", (f"按 {px} 卖出 {n_drop} 股, "
                                     f"净入 ¥{gross - fee:,.2f} (手续费 {fee:,.2f})")
    else:
        px, gross, fee = None, 0.0, 0.0
        kind, detail = "drop_phantom", f"记账错误, 删掉 {n_drop} 股记录, 现金不动"

    after = snapshot(st, kl, names)
    st.setdefault("history", []).append({
        "type": kind, "at": datetime.now().isoformat(timespec="seconds"),
        "note": args.note,
        "lot": {"code": code, "name": nm, "shares": n_drop,
                "buy_price": round(avg_cost, 4), "held_before": total},
        "partial": partial, "remaining": total - n_drop,
        "sold_at": px, "fee": round(fee, 2) if fee else 0.0,
        "before": {"cash": before["cash"], "equity": before["equity"]},
        "after": {"cash": after["cash"], "equity": after["equity"]},
    })

    W = 68
    title = "部分删除持仓" if partial else "删除持仓"
    print(f"\n{'='*W}\n  {title} {code} {nm}\n{'='*W}")
    print(f"  原持仓 : {total} 股, 均价成本 {avg_cost:.3f}")
    print(f"  本次   : {n_drop} 股" + (f", 剩下 {total - n_drop} 股继续持有"
                                       f"(到期节奏不变)" if partial else " (全部)"))
    print(f"  处理   : {detail}")
    if args.sold_at is not None:
        pnl = gross - fee - n_drop * avg_cost
        print(f"  这笔盈亏: ¥{pnl:+,.2f}")
    print(f"  现金   : ¥{before['cash']:,.2f}  ->  ¥{after['cash']:,.2f}")
    print(f"  总资产 : ¥{before['equity']:,.2f}  ->  ¥{after['equity']:,.2f}")
    print(f"  持仓数 : {len(before['positions'])}  ->  {len(after['positions'])}")
    if args.phantom:
        print("  注: 总资产下降是因为删掉了一笔本不该存在的持仓, 账目现在更接近真实")
    if st.get("pending"):
        print(f"  注: 挂单计划({st['pending']['signal_date']})里若含这只股, "
              f"结算时会自动跳过(它已不在持仓里), 不会重复卖出")
    if args.note:
        print(f"  备注   : {args.note}")
    if bak:
        print(f"  已备份 : {bak.name}")
    save_state(st)


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
if args.set_cash is not None and args.cash_flow is not None:
    raise SystemExit("ERROR: --set-cash 和 --cash-flow 不能同时用 —— 一个是修账"
                     "(本金不动), 一个是出入金(本金同额变动), 混在一起账就说不清了")
if args.set_cash is not None:
    do_set_cash(state, kl, names)
    sys.exit(0)
if args.cash_flow is not None:
    do_cash_flow(state, kl, names)
    sys.exit(0)
if args.drop_lot:
    do_drop_lot(state, kl, names)
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
        # 带了成交回报却无法结算时必须报错。否则网页那边会以为提交成功
        # (进程退出码 0), 而账目其实一动没动 —— 用户会以为已经记上了。
        if args.confirm:
            raise SystemExit(
                f"ERROR: {p_sig.date()} 的计划还没到执行日的行情 —— 数据最新只到 "
                f"{pd.Timestamp(all_dates[-1]).date()}, 现在无法结算成交。\n"
                "  执行日收盘、当晚数据更新完成后再提交成交回报。")
        print(f"\n[结算] {p_sig.date()} 的挂单尚无 T+1 行情, 计划保持有效, 本次不重新出信号。")
        print(f"耗时 {(datetime.now()-t0).total_seconds():.0f}s")
        sys.exit(0)
    # 实盘模式的闸门: 没有真实成交回报就不许动账。
    # 自动记账会假设"你按参考价买到了", 如果你其实没下单, 状态就会静默偏离
    # 真实账户, 而且越积越歪。所以这里原地停住, 也不出新信号 —— 新信号依赖
    # 当前持仓, 拿一份错的持仓算出来的信号只会误导人。
    if args.require_confirm and not args.confirm:
        print(f"\n[等待确认] {p_sig.date()} 的计划应在 "
              f"{pd.Timestamp(exec_date).date()}"
              f"{'开盘' if EXEC_FIELD == 'open' else '尾盘'}执行。")
        print("  这条线是实盘模式, 不会按行情自动记账。请在网页上填写真实成交价")
        print("  (或用 --confirm 提交成交回报) 后, 才会结算并出下一份信号。")
        print("  若当天实际没有下单, 在网页上选「未成交」即可跳过。")
        state["awaiting_confirm"] = {
            "signal_date": str(p_sig.date()),
            "exec_date": str(pd.Timestamp(exec_date).date()),
            "since": datetime.now().isoformat(timespec="seconds"),
        }
        save_state(state)
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
                             "fills": fills, "rejected": rej,
                             "source": "manual" if args.confirm else "auto"})
    state["pending"] = None
    state["awaiting_confirm"] = None
else:
    print("\n[结算] 无待结算挂单")

# ── 训练 + 预测 ──
seq = date_pos[SIGNAL_DATE]
cutoff = all_dates[seq - LABEL_HORIZON]
train_df = df[(df["date"] < cutoff) & df[LABEL].notna()]
if train_df["date"].nunique() < MIN_TRAIN_DAYS:
    raise SystemExit(f"ERROR: 训练集只有 {train_df['date'].nunique()} 天, 少于 {MIN_TRAIN_DAYS}")
# 四条线的模型完全相同 —— 特征、标签、训练集、超参都不依赖 tranche_n 与本金,
# 只有建仓环节不同。各训一遍就是 4 倍 CPU 白烧。缓存键取"能影响预测的一切",
# 任何一项变了都会重训, 所以不存在读到过期预测的风险。
_cache_key = None
if args.preds_cache:
    _cache_key = hashlib.sha1(json.dumps({
        "signal_date": str(SIGNAL_DATE), "cutoff": str(cutoff),
        "train_file": args.train_file, "pit_universe": args.pit_universe,
        "label": LABEL, "features": list(features),
        "params": {k: str(v) for k, v in sorted(LOCKED_PARAMS.items())},
        "seeds": ENSEMBLE_SEEDS,
        "rows": int(len(train_df)),
    }, sort_keys=True, default=str).encode()).hexdigest()[:16]

# tm 在缓存命中时也要用(下面算 blocked 靠它), 所以放在分支外
tm = df["date"] == SIGNAL_DATE

ranked = None
if _cache_key:
    _cp = Path(args.preds_cache)
    if _cp.exists():
        try:
            _c = json.loads(_cp.read_text(encoding="utf-8"))
            if _c.get("key") == _cache_key:
                ranked = pd.DataFrame({"code": _c["codes"], "pred": _c["preds"]})
                print(f"\n[训练] 命中预测缓存 (同日同输入, 由 {_c.get('by','?')} 生成), "
                      f"跳过训练")
        except Exception as e:
            print(f"[训练] 预测缓存读取失败, 改为重新训练: {e}")

if ranked is None:
    print(f"\n[训练] 样本 < {pd.Timestamp(cutoff).date()} | "
          f"{train_df['date'].nunique()} 天 {len(train_df):,} 行 {len(features)} 特征 | "
          f"{len(ENSEMBLE_SEEDS)} 种子集成")
    X = train_df.groupby("code")[features].transform(lambda c: c.ffill().fillna(0))
    Xt = df.loc[tm, features].fillna(0)
    # 多种子集成: 同数据同超参, 只换 random_state, 预测取普通平均(保留收益率量纲)。
    # 前3名选股对训练噪声极度敏感(两两种子 top3 重合率仅 75%), 平均后方差抵消。
    # 回测验证: plain-mean 与 z-mean top3 重合 99.3%, 集成 IC 与单种子持平。
    preds = None
    for _sd in ENSEMBLE_SEEDS:
        _m = lgb.LGBMRegressor(**dict(LOCKED_PARAMS, random_state=_sd)).fit(X, train_df[LABEL])
        _p = _m.predict(Xt)
        preds = _p if preds is None else preds + _p
    preds = preds / len(ENSEMBLE_SEEDS)
    ranked = (pd.DataFrame({"code": df.loc[tm, "code"].astype(str).str[:6].values, "pred": preds})
              .sort_values("pred", ascending=False).reset_index(drop=True))
    if _cache_key and not args.dry_run:
        try:
            Path(args.preds_cache).write_text(json.dumps({
                "key": _cache_key, "signal_date": str(SIGNAL_DATE),
                "by": STATE_PATH.stem, "at": datetime.now().isoformat(timespec="seconds"),
                "codes": list(ranked["code"]), "preds": [float(x) for x in ranked["pred"]],
            }, ensure_ascii=False), encoding="utf-8")
            print(f"[训练] 预测已缓存, 同日其余条线可直接复用")
        except Exception as e:
            print(f"[训练] 预测缓存写入失败(不影响本次运行): {e}")

ranked = ranked.sort_values("pred", ascending=False).reset_index(drop=True)
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
# 换仓日判定 (periodic): 距上次换仓满 HOLD_DAYS 个交易日; 另外两种自愈情形 ——
#   a) 到期后已无保留仓 (漏跑/对账导致错过周期点时, 不致于长期空着)
#   b) 大盘刚由弱转强, 当天立即回场 (与回测一致)
# 必须先算 is_rebal, 因为"到期的该不该续持"取决于本次是否换仓。
_last_rebal = state.get("last_rebal_signal_date")
_n_matured = sum(1 for l in state["lots"]
                 if held_days(all_dates, l, SIGNAL_DATE) >= HOLD_DAYS)
is_rebal = ((not PERIODIC)
            or _last_rebal is None
            or (seq - cal_pos(all_dates, _last_rebal)) >= HOLD_DAYS
            or (was_in_cash and not in_cash)
            or (_n_matured == len(state["lots"]) and not in_cash))

# 目标组合: ranked 里前 TRANCHE_N 个未被持禁的。已持仓且仍在目标里的,
# 到期也不卖 —— 卖了再买回要付一次往返 (佣金+滑点)*2, 白付这笔钱
# 却回到同样的持仓。与回测 wf_v35 的 roll_set 逻辑一致。
roll_set = set()
if PERIODIC and is_rebal and not in_cash:
    for code in ranked["code"]:
        if len(roll_set) >= TRANCHE_N:
            break
        if code in blocked:
            continue
        roll_set.add(code)

sell_plan, keep_plan = [], []
for lot in state["lots"]:
    _held = held_days(all_dates, lot, SIGNAL_DATE)
    matured = _held >= HOLD_DAYS
    ref = kl.px(lot["code"], SIGNAL_DATE, "close") or lot["buy_price"]
    row = {"code": lot["code"], "name": names.get(str(lot["code"])[:6], ""),
           "shares": lot["shares"], "buy_price": round(lot["buy_price"], 3),
           "ref_close": round(ref, 3), "open_date": lot.get("open_date"),
           "held_days": _held,
           # 真实持有时长与累计续持次数: 续持会把 held_days 归零, 光看它
           # 会以为是刚买的。两个都给前端, 各自回答不同的问题。
           "tenure_days": tenure_days(all_dates, lot, SIGNAL_DATE),
           "n_rolled": int(lot.get("rolled", 0)),
           "first_open_date": lot.get("first_open_signal_date") or lot.get("open_signal_date"),
           "pnl_pct": round((ref / lot["buy_price"] - 1) * 100, 2)}
    rolled = matured and not in_cash and str(lot["code"])[:6] in roll_set
    if rolled:
        # 续持: 不产生交易, 也不让用户做任何操作
        row["rolled"] = True
        row["reason"] = "到期但仍在前列, 续持不动"
        keep_plan.append(row)
    elif matured or in_cash:
        row["reason"] = "持满到期" if matured else "大盘转弱清仓"
        row["est_proceeds"] = round(lot["shares"] * fill_px(ref, "sell") * (1 - TRADE_COST), 2)
        sell_plan.append(row)
    else:
        keep_plan.append(row)

cash_after_sell = state["cash"] + sum(r["est_proceeds"] for r in sell_plan)
buy_plan, alt_plan = [], []
if not in_cash and is_rebal:
    equity = cash_after_sell + sum(r["shares"] * r["ref_close"] for r in keep_plan)
    denom = 1 if PERIODIC else HOLD_DAYS
    remaining = min(equity / denom, cash_after_sell)
    held = {str(r["code"])[:6] for r in keep_plan}
    # 从已持仓数起算: 续持的那几只已经占着仓位, 不从 0 起算会超配
    bought = len(keep_plan)
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
                # 整手粒度救济: 计划与结算必须同一口径, 否则挂单和入账会对不上
                if LOT_FLEX > 0 and px * 100 <= alloc * (1 + LOT_FLEX):
                    shares = 100
                else:
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
