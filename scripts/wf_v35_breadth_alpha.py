"""
Walk-Forward v35: 最小验证版 —— 只改两件事, 检验能否跑赢等权基准

相对 v33 的改动 (对症诊断证据 2 和 3):
  A. 去掉概念中性化 -> 只按日期 demean
     证据: 53.8% 的(日期,概念组)只有1只股票, demean 后标签恒为0;
           corr(excess_group, raw)=0.63 而 corr(excess_date, raw)=1.00
  D. 广度 + 换手
     top_n: 3 -> 10 (5档 x 每档2只), 常年满仓
     持有期: 1天全换 -> 5日分档, 每天只换 1/5 仓位
     证据: 平均部署率仅 28.9%, 成本 14.4%

刻意【不】开启 (保持变量干净, 后续再单独验证):
  - IC 置信度择时 (--ic-timing 开启)
  - 波动率目标仓位 (--vol-target 开启)

保留 v33 有效的部分: 特征筛选(importance+去相关)、大盘特征、隔夜特征

强制输出与【等权买入持有】基准的对比: 超额 / IR / beta / alpha
"""
import pandas as pd, numpy as np, json, warnings, argparse
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
import lightgbm as lgb

parser = argparse.ArgumentParser(description="WF v35 breadth + alpha vs benchmark")
parser.add_argument("--test-start", type=str, default="2022-09-01")
parser.add_argument("--test-end", type=str, default="2026-07-16")
parser.add_argument("--initial-capital", type=float, default=100000.0)
parser.add_argument("--label", type=str, default="5d", choices=["1d", "5d"],
                    help="预测目标: 1d=次日收益, 5d=5日收益(与5日持有对齐)")
parser.add_argument("--hold-days", type=int, default=5, help="持有天数(分档数)")
parser.add_argument("--tranche-n", type=int, default=2, help="每档买入只数; 总仓位=hold_days*tranche_n")
parser.add_argument("--n-features", type=int, default=80)
parser.add_argument("--corr-threshold", type=float, default=0.9)
parser.add_argument("--drop-market-wide", type=float, default=0.0,
                    help="剔除截面变异比(当日跨股票std / 总体std)低于该阈值的特征。"
                         "标签是按日期demean的截面收益, 这类特征当日对所有股票同值, "
                         "单独预测能力恒为0, 在树里只能按'日期'切分=记忆训练区间。"
                         "建议 0.01; 0 = 关闭(保留旧行为)")
parser.add_argument("--select-seeds", type=int, default=5,
                    help="特征筛选时用几个随机种子平均 gain 重要度。"
                         "1=旧行为(单次拟合, 排名不稳定)")
parser.add_argument("--select-pool-mult", type=int, default=3,
                    help="去相关的候选池 = n_features * 该倍数。"
                         "取1则退化为旧行为(先砍到n_features再去相关, 名额会剩不满)")
parser.add_argument("--ic-timing", action="store_true", help="开启IC置信度择时(默认关)")
parser.add_argument("--vol-target", action="store_true", help="开启波动率目标仓位(默认关)")
parser.add_argument("--objective", type=str, default="l2", choices=["l2", "rank"],
                    help="B: l2=回归(优化截面中部) rank=lambdarank(优化头部排序)")
parser.add_argument("--neutralize-style", action="store_true",
                    help="C: 对预测值做风格中性化, 消除高波动/估值/流动性偏好")
parser.add_argument("--portfolio-mode", type=str, default="staggered",
                    choices=["staggered", "periodic"],
                    help="staggered=每日开一档共HOLD_DAYS档(适合大资金); "
                         "periodic=每HOLD_DAYS天整体换仓, 仅持TRANCHE_N只(适合小资金)")
parser.add_argument("--no-roll", action="store_true",
                    help="关掉续持: 到期一律卖掉再买回, 即使它仍在目标名单里。"
                         "默认开启续持(省一次往返成本)。此开关仅用于 A/B 对照")
parser.add_argument("--exec-mode", type=str, default="close",
                    choices=["close", "t1open", "t1close"],
                    help="close=T日收盘信号T日收盘成交(未来函数, 不可实盘); "
                         "t1open=T日收盘信号 T+1日开盘成交(可实盘); "
                         "t1close=T日收盘信号 T+1日尾盘成交(可实盘, 可用尾盘集合竞价)")
parser.add_argument("--slippage", type=float, default=0.0,
                    help="单边滑点(小数), 买入成交价上浮/卖出下浮该比例。"
                         "例: 0.001 = 单边0.1%%(往返0.2%%)。仅影响成交价, 不影响持仓估值")
parser.add_argument("--trade-cost", type=float, default=0.0006,
                    help="单边交易费率(小数), 含佣金+过户费+印花税的综合估值。"
                         "默认 0.0006(万六)。万一口径约 0.0002 (佣金万一+卖出印花税摊平到双边)")
parser.add_argument("--min-fee", type=float, default=5.0,
                    help="单笔最低佣金(元)。小资金下这是硬约束: 设 0 即模拟“免五”")
parser.add_argument("--regime-filter", type=str, default="off",
                    choices=["off", "ma", "breadth", "both", "any"],
                    help="大盘 regime 空仓过滤: ma=大盘跌破MA; breadth=上涨广度低迷; "
                         "both=两者同时成立才空仓(保守); any=任一成立即空仓(激进)")
parser.add_argument("--regime-ma", type=int, default=20, help="regime 趋势均线窗口")
parser.add_argument("--regime-breadth", type=float, default=0.40,
                    help="广度阈值: 全市场收盘价站上MA20的比例低于此值视为弱势")
parser.add_argument("--regime-confirm", type=int, default=2,
                    help="连续N个信号日满足条件才切换状态(双向), 抑制来回打脸")
parser.add_argument("--train-file", type=str, default="training_data_v24.parquet",
                    help="训练集文件名 (data/processed/ 下)")
parser.add_argument("--pit-universe", type=str, default=None,
                    help="PIT 股票池 parquet (data/universe/ 下), 如 universe_pit.parquet; "
                         "开启后每行样本只保留当期生效成分股, 训练/预测/基准均受约束")
parser.add_argument("--reversal-guard", type=float, default=0.0,
                    help="反转护栏: 排除信号日近5日涨幅处于截面前 X 分位的候选股 "
                         "(如 0.10 = 剔除涨幅前10%的股票, 0 = 关闭)。"
                         "针对急涨行情中追高被打脸的问题")
parser.add_argument("--lot-flex", type=float, default=0.0,
                    help="整手粒度救济: 槽位预算买不起一手(100股)时, 若一手成本 "
                         "<= 槽位预算*(1+flex) 且现金足够, 仍买这一手, 而不是沿排名 "
                         "换下一只。用等权纪律换信号保真度。0 = 关闭(旧行为)")
parser.add_argument("--roll-rank", type=int, default=0,
                    help="卖出容忍: 到期持仓只要仍在当日排名前 M 名就继续持有, 而不是"
                         "只有进前 TRANCHE_N 名才续持。买入门槛不变(仍只买最强的)。"
                         "从第2名滑到第7名的股不再值得新买, 但也不值得付往返成本换掉。"
                         "0 = 关闭(等于 TRANCHE_N, 旧行为)")
parser.add_argument("--save-preds", type=str, default=None,
                    help="把逐日模型预测结果缓存到此 pickle (data/processed/ 下)")
parser.add_argument("--load-preds", type=str, default=None,
                    help="从 pickle 加载预测结果, 跳过训练。仅当只改执行层参数"
                         "(regime/护栏/持仓数/成本) 时可用, 模型相关参数必须与缓存一致")
parser.add_argument("--features-from", type=str, default=None,
                    help="直接复用另一份回测 json 里的 selected_features (data/processed/ 下), "
                         "不现场筛选。用于隔离'数据变了'和'特征集变了'两个变量")
parser.add_argument("--skip-boards", type=str, default="",
                    help="不能买的板块代码前缀, 逗号分隔。如 '30,688' = 创业板+科创板"
                    " (没开通权限的账户)。空 = 不限制")
parser.add_argument("--skip-boards-mode", choices=["substitute", "cash"],
                    default="substitute",
                    help="遇到受限板块股怎么办: substitute=顺位递补下一名主板股(默认);"
                    " cash=该槽位留现金到下轮(不递补)")
parser.add_argument("--min-pred", type=float, default=0.0,
                    help="建仓信号强度门槛: 只买 pred >= X 的候选(单位=预期5日超额收益,"
                    " 如 0.005 = 0.5%)。不达标的槽位留现金。需要 v2 缓存(带 pred_vals)。"
                    " 0 = 关闭(无条件买满)")
parser.add_argument("--fill-daily", action="store_true",
                    help="空槽位逐日补买: 非换仓日若持仓 < TRANCHE_N 且当日有过门槛的候选,"
                    " 立即补买(新批次自带 5 日到期时钟)。与 --min-pred 搭配使用")
parser.add_argument("--export-matrix", type=str, default=None,
                    help="把准备好的训练矩阵(含市场/隔夜特征、demean标签、选定特征列)导出到 "
                         "data/processed/ 下的 parquet(+同名.meta.json), 然后退出。"
                         "给外部模型实验(如 GPU MLP)用, 保证数据口径与本脚本完全一致")
parser.add_argument("--lgb-seed", type=int, default=42,
                    help="LightGBM random_state。不是可调超参 —— 仅供多种子集成实验"
                         "(同一模型训多个种子平均排名, 降低前3名选择方差)使用。"
                         "禁止用它挑好看的单种子结果")
parser.add_argument("--tag", type=str, default=None)
args = parser.parse_args()

from pipeline.config import settings
DATA_DIR = settings.DATA_DIR
TRAIN_PATH = DATA_DIR / "processed" / args.train_file
KLINE_DIR = DATA_DIR / "raw" / "kline"

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
LOCKED_PARAMS["random_state"] = args.lgb_seed   # 仅多种子集成用, 见 --lgb-seed 说明

TRADE_COST = args.trade_cost
MIN_FEE = args.min_fee
SLIPPAGE = args.slippage
INIT_CAPITAL = args.initial_capital
TEST_START, TEST_END = args.test_start, args.test_end
HOLD_DAYS, TRANCHE_N = args.hold_days, args.tranche_n
PERIODIC = args.portfolio_mode == "periodic"
NO_ROLL = args.no_roll
LOT_FLEX = args.lot_flex
ROLL_RANK = args.roll_rank
MIN_PRED = args.min_pred
FILL_DAILY = args.fill_daily
SKIP_BOARDS = tuple(s.strip() for s in args.skip_boards.split(",") if s.strip())
SKIP_CASH = args.skip_boards_mode == "cash"
TARGET_POSITIONS = TRANCHE_N if PERIODIC else HOLD_DAYS * TRANCHE_N
MIN_TRAIN_DAYS = 250

_tag = args.tag or f"v35_{args.label}_hold{HOLD_DAYS}_n{TARGET_POSITIONS}"
_out = f"wf_daily_{_tag}_ts{TEST_START}_te{TEST_END}_cap{int(INIT_CAPITAL)}"
OUT_PATH = DATA_DIR / "processed" / f"{_out}.json"

_COL_MAP = {"时间": "date", "收盘价": "close", "开盘价": "open",
            "最高价": "high", "最低价": "low", "成交量": "volume", "总金额": "amount"}


STYLE_CANDIDATES = ["atr_pct", "pb", "vol_ratio", "con_amount"]
RANK_BUCKETS = 10


def apply_pit_universe(df, uni_file):
    """只保留每行日期所属生效期的成分股 (PIT 无偏)

    生效日 T 的成分列表适用于 [T, 下一个生效日) 区间。
    早于首个生效日的行无法判定成分, 直接丢弃(宁可少用数据也不引入前视)。
    """
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
    n0, c0 = len(df), df["code"].nunique()
    out = df[keep].reset_index(drop=True)
    n_early = int((period < 0).sum())
    print(f"  PIT 成分约束({uni_file}): {n0:,} -> {len(out):,} 行, "
          f"{c0} -> {out['code'].nunique()} 只 | 生效期 {len(eff)} 个"
          f"{f', 丢弃首生效日之前 {n_early:,} 行' if n_early else ''}")
    return out


def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")


def to_buckets(s):
    """截面标签 -> 0..9 整数相关度, 供 lambdarank 使用"""
    r = s.rank(method="first", pct=True)
    return np.clip((r * RANK_BUCKETS).astype(int), 0, RANK_BUCKETS - 1)


def neutralize(preds, style_df):
    """截面 OLS 取残差: 去掉预测值中可被风格因子解释的部分"""
    if style_df.shape[1] == 0 or len(preds) < style_df.shape[1] + 2:
        return preds
    S = style_df.astype(float)
    S = S.fillna(S.median()).fillna(0.0)
    S = S.rank(pct=True)                       # 秩归一化, 抑制肥尾
    A = np.column_stack([np.ones(len(S)), S.values])
    try:
        coef, *_ = np.linalg.lstsq(A, preds, rcond=None)
    except np.linalg.LinAlgError:
        return preds
    return preds - A @ coef


# ═══════════════════════════════════════════════════════════════
# 特征构建 (沿用 v33)
# ═══════════════════════════════════════════════════════════════
def compute_market_features():
    print("  大盘特征...")
    all_daily = []
    for p in sorted(KLINE_DIR.glob("*.parquet")):
        try:
            kl = pd.read_parquet(p).rename(columns=_COL_MAP)
            kl["date"] = pd.to_datetime(kl["date"])
            kl = kl.sort_values("date")
            kl["above_ma"] = (kl["close"] > kl["close"].rolling(args.regime_ma).mean()).astype(float)
            kl["up"] = (kl["close"].pct_change() > 0).astype(float)
            all_daily.append(kl[["date", "close", "open", "above_ma", "up"]])
        except Exception:
            continue
    market = pd.concat(all_daily, ignore_index=True)
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
    feats = [c for c in m.columns if c.startswith("mkt_")]
    for c in feats:
        m[c] = pd.to_numeric(m[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    regime_src = m[["date", "mkt_close", f"mkt_ma{args.regime_ma}",
                    "breadth_above_ma", "breadth_up"]].copy()
    return m[["date"] + feats], feats, regime_src


def build_regime_state(regime_src):
    """大盘 regime -> 每个信号日的空仓布尔值 (只用截至当日的信息, PIT 安全)

    弱势判定:
      ma      : 全市场平均收盘价跌破 MA(regime_ma)
      breadth : 站上均线的个股比例 < regime_breadth
    切换需连续 regime_confirm 天确认(进出双向), 避免震荡市反复空/满仓。
    """
    if args.regime_filter == "off":
        return None
    r = regime_src.set_index("date").sort_index()
    trend_bad = r["mkt_close"] < r[f"mkt_ma{args.regime_ma}"]
    breadth_bad = r["breadth_above_ma"] < args.regime_breadth
    raw = {"ma": trend_bad, "breadth": breadth_bad,
           "both": trend_bad & breadth_bad, "any": trend_bad | breadth_bad}[args.regime_filter]
    raw = raw.fillna(False)
    k = max(1, args.regime_confirm)
    turn_off = raw.rolling(k).min().fillna(0) == 1        # 连续k天弱 -> 空仓
    turn_on = (~raw).rolling(k).min().fillna(0) == 1      # 连续k天不弱 -> 回场
    state, out = False, {}
    for d in r.index:
        if not state and bool(turn_off.loc[d]):
            state = True
        elif state and bool(turn_on.loc[d]):
            state = False
        out[d] = state
    return pd.Series(out)


def compute_overnight_features(codes):
    print("  隔夜特征...")
    out = []
    for code in codes:
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


def drop_market_wide(df, feats, label_col, cutoff, thresh):
    """剔除对截面标签零信息的市场级特征。

    标签是 y = fwd_ret - 当日截面均值。若某特征在同一天对所有股票取同一个值
    (商品价格/汇率/国债收益率/PMI/全球指数/全市场事件聚合等), 那么它在截面内
    没有任何变异, 单独的截面 IC 恒为 0。

    树模型仍可能用它做分裂, 但那种分裂会把当天【所有】股票分到同一侧 ——
    等价于按日期切分训练集, 是记忆训练区间而非学习选股规律。筛选样本只有
    约 255 天, 几次这样的分裂就足以过拟合。

    判据 (两条取并集, 任一命中即剔除):
      A. 截面变异比 = (每日跨股票 std 的均值) / (总体 std)
      B. 每日独立取值比 = median(当日 nunique / 当日股票数)

    【为何必须有 B】判据 A 不具备股票池规模不变性。市场级序列按各股自己的
    交易日 as-of 对齐时, 停牌股会拿到上一期的陈旧值, 于是同一天出现少数几个
    不同取值。池子越大停牌/日历异质性越多, 这种假截面变异越强:
        cn_pmi 变异比   497只池 = 0.0052 (正确剔除)
                       2370只池 = 0.0180 (穿破 0.01 阈值, 漏网)
        同日存在多个 PMI 取值的天数  7.5% -> 52.5%
    漏网后它在 2370 只池子里 gain 重要度排第 2, 模型学到的是 PMI 公布前后
    哪些股票停牌, 不是选股 alpha。
    根因是 PMI 跳变使那 2 个陈旧/新值相距很远, 少量样本即可推高 std。
    判据 B 只数取值个数不看取值大小, 对此免疫: cn_pmi 在 2337 只股票里每日
    仅约 2 个不同取值 (独值比 0.00085), 两个池子都远低于阈值。
    实测两池分离度: 市场级 0.0000~0.0020 vs 个股级 0.0390~1.0000。

    只用 cutoff 之前的数据计算, 与 select_features 口径一致。

    【必须按母特征判定】feature_engine 在单只股票内部对几乎所有列做
    rolling(5/20).mean()。市场级序列原始值的截面变异比精确为 0, 但因为每只
    股票停牌日不同, 滚动窗口覆盖的日历不同, 其 _ma5/_ma20 会凭空产生
    0.001~0.010 的截面变异 —— 这部分变异实测与"近20日缺勤天数"显著相关
    (a50_futures_chg_21d_ma20: rho=+0.137, t=3.4), 编码的是停牌模式而非
    个股信息。若只按特征自身的变异比过滤, 会有约 31 个这样的假特征漏网,
    且它们的 gain 重要度常年排在最前。因此: 母特征是市场级的, 其所有 MA
    派生一律剔除。
    """
    s = df[(df["date"] < pd.Timestamp(cutoff)) & df[label_col].notna()]
    # 判据A: 截面变异比
    overall = s[feats].std().replace(0, np.nan)
    cs = s.groupby("date")[feats].std().mean()
    ratio = (cs / overall).fillna(0)
    # 判据B: 每日独立取值比 (对市场级序列的陈旧值跳变免疫)
    n_per_day = s.groupby("date")["code"].nunique()
    uniq = s.groupby("date")[feats].nunique().div(n_per_day, axis=0).median().fillna(0)

    def base_of(f):
        for suf in ("_ma5", "_ma20"):
            if f.endswith(suf):
                return f[: -len(suf)]
        return f

    keep, dropped, by_ma, by_uniq = [], [], 0, 0
    for f in feats:
        b = base_of(f)
        # 母特征在数据里就用母特征判定, 否则退回用自身
        k = b if (b in ratio.index) else f
        hit_var, hit_uniq = ratio[k] < thresh, uniq[k] < thresh
        if hit_var or hit_uniq:
            dropped.append(f)
            if not hit_var:
                by_uniq += 1
            if b != f and ratio[f] >= thresh and uniq[f] >= thresh:
                by_ma += 1
        else:
            keep.append(f)
    print(f"  剔除市场级特征(母特征 变异比 或 独值比 < {thresh}): "
          f"{len(feats)} -> {len(keep)} 个候选, 剔除 {len(dropped)} 个")
    if by_ma:
        print(f"    其中 {by_ma} 个是靠停牌日历伪造出截面变异的 MA 派生"
              f"(自身变异比 >= {thresh}, 但母特征为市场级)")
    if by_uniq:
        print(f"    其中 {by_uniq} 个仅被独值比判据捕获"
              f"(变异比 >= {thresh} 已漏网, 但每日跨股票取值数极少)")
    if dropped:
        print(f"    示例: {dropped[:6]}")
    return keep


def select_features(df, all_features, label_col, n_top, corr_thresh, cutoff,
                    n_seeds=5, pool_mult=3, return_table=False):
    """筛特征只允许用 cutoff 之前的数据。

    cutoff = 首个可出信号日。绝不能退化成全样本: 那会把测试期的 fwd 标签
    喂给筛选器, 造成未来泄漏, 且结果对数据尾部极度敏感。

    2026-07 重写, 修了 4 个问题 (原实现使特征集不可复现, 同一份数据两次
    重建可能得到差异巨大的特征集 -> 回测收益 24.7% vs 171.3%):

    1. 用 gain 而非 split 计数。split 计数是小整数, 排名 70~85 名会全部
       并列在同一个值上, 截断线切在并列区里, 保留谁纯看排序实现细节。
    2. 多种子平均。colsample_bytree=0.8 使单次拟合的重要度带随机性,
       单种子结果不稳定。每个种子内先归一化再平均, 避免量级差异主导。
    3. 稳定且确定的排序。原 sort_values 默认 kind="quicksort" 非稳定,
       并列项顺序不确定; 改为 (重要度降, 特征名升) + mergesort。
    4. 去相关从 n_top*pool_mult 的候选池里填满 n_top 个名额, 而不是先砍到
       n_top 再去相关(那样最终只剩 57/80, 名额被浪费)。
       同时只与【已选中】的特征比相关性, 不再连坐。
    """
    print(f"  特征筛选: {len(all_features)} 个候选 -> 目标 {n_top} 个 "
          f"| gain重要度 x {n_seeds} 种子平均 | 去相关阈值 {corr_thresh} "
          f"| 仅用 < {pd.Timestamp(cutoff).date()} 的数据")
    s = df[(df["date"] < pd.Timestamp(cutoff)) & df[label_col].notna()]
    if len(s) < 10000:
        raise SystemExit(
            f"ERROR: {pd.Timestamp(cutoff).date()} 之前只有 {len(s)} 行, 不足以筛特征。\n"
            f"       请把 --test-start 往后推, 或补更早的历史数据。\n"
            f"       (绝不退化为全样本筛选 —— 那会造成未来数据泄漏)")
    X = s.groupby("code")[all_features].transform(lambda c: c.ffill().fillna(0))
    y = s[label_col]

    # ── 1+2: gain 重要度, 对【日期区块重采样】求平均 ──
    # 关键: 必须令 colsample_bytree=1.0。它按列索引抽样, 会使重要度依赖
    # 候选特征的列顺序 —— 修复基本面数据新增 30 列后列序一变, 特征集就洗牌
    # (实测打乱列序后 Jaccard 仅 68~76%)。关掉后给定数据即完全确定。
    # 稳健性改由重采样数据获得: 按 5 日区块抽样(尊重 5 日重叠标签的相关结构),
    # 每次抽 80% 区块, 这样 imp_std 反映的是"换一段样本排名会不会变"。
    uniq_dates = np.array(sorted(s["date"].unique()))
    blocks = np.array_split(np.arange(len(uniq_dates)),
                            max(1, len(uniq_dates) // LABEL_HORIZON))
    mat = np.zeros((n_seeds, len(all_features)), dtype=float)
    for i in range(n_seeds):
        if n_seeds == 1:
            m = np.ones(len(s), dtype=bool)
        else:
            rng = np.random.default_rng(1000 + i)
            pick = rng.choice(len(blocks), size=max(1, int(len(blocks) * 0.8)),
                              replace=False)
            keep = set(uniq_dates[np.concatenate([blocks[b] for b in sorted(pick)])])
            m = s["date"].isin(keep).values
        p = dict(LOCKED_PARAMS, n_estimators=50, boosting_type="gbdt",
                 colsample_bytree=1.0, random_state=42, importance_type="gain")
        mdl = lgb.LGBMRegressor(**p).fit(X[m], y[m])
        g = np.asarray(mdl.booster_.feature_importance(importance_type="gain"),
                       dtype=float)
        tot = g.sum()
        mat[i] = g / tot if tot > 0 else 0.0
    imp_mean, imp_std = mat.mean(axis=0), mat.std(axis=0)

    # ── 3: 确定性排序, 并列时按特征名升序, 保证可复现 ──
    imp = pd.DataFrame({"feature": all_features,
                        "importance": imp_mean, "imp_std": imp_std})
    imp = (imp.sort_values(["importance", "feature"], ascending=[False, True],
                           kind="mergesort").reset_index(drop=True))

    # 诊断: 0 重要度的特征直接不参选; 并统计截断线附近的并列程度
    nz = imp[imp["importance"] > 0].reset_index(drop=True)
    n_zero = len(imp) - len(nz)
    if len(nz) > n_top:
        cut_val = nz.loc[n_top - 1, "importance"]
        n_tied = int((nz["importance"] == cut_val).sum())
    else:
        cut_val, n_tied = float("nan"), 0
    print(f"    重要度: 非零 {len(nz)} 个, 恒为0 {n_zero} 个(已排除)")
    if n_tied > 1:
        print(f"    !! 截断线({cut_val:.3g})上有 {n_tied} 个并列特征, "
              f"排名仍受实现细节影响")

    # ── 4: 从更大候选池里去相关, 填满 n_top 个名额 ──
    pool = nz["feature"].head(max(n_top * pool_mult, n_top)).tolist()
    cm = s[pool].corr().abs()
    selected, n_dropped = [], 0
    for f in pool:
        if len(selected) >= n_top:
            break
        if any(cm.at[f, g] > corr_thresh for g in selected):
            n_dropped += 1
            continue
        selected.append(f)
    print(f"    候选池 {len(pool)} 个 -> 去相关剔除 {n_dropped} 个 -> "
          f"选中 {len(selected)} 个")
    print(f"    top10: {nz.head(10)['feature'].tolist()}")
    # 重要度排名的可信度: 变异系数越大说明该特征的重要度越不稳定
    sel_imp = imp.set_index("feature").loc[selected]
    cv = (sel_imp["imp_std"] / sel_imp["importance"].replace(0, np.nan)).median()
    print(f"    选中特征重要度的种子间变异系数(中位数): {cv:.2f} "
          f"({'稳定' if cv < 0.3 else '偏噪声, 建议加大 n_seeds 或减少候选'})")
    if return_table:
        return selected, imp
    return selected


# ═══════════════════════════════════════════════════════════════
# K线 / 执行辅助
# ═══════════════════════════════════════════════════════════════
def load_all_klines():
    cache = {}
    for p in sorted(KLINE_DIR.glob("*.parquet")):
        if not p.is_file():
            continue
        kl = pd.read_parquet(p).rename(columns=_COL_MAP)
        kl["date"] = pd.to_datetime(kl["date"])
        cache[p.stem] = kl.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return cache


def get_close(klines, code, date):
    kl = klines.get(str(code)[:6])
    if kl is None:
        return None
    r = kl[kl["date"] == pd.Timestamp(date)]
    return float(r.iloc[0]["close"]) if len(r) else None


def board_limit(code):
    """各板块涨跌幅限制"""
    c = str(code)[:6]
    if c.startswith(("300", "688")):          # 创业板 / 科创板
        return 0.20
    if c.startswith(("43", "83", "87", "92")):  # 北交所
        return 0.30
    return 0.10                               # 主板


def get_px(klines, code, date, field):
    kl = klines.get(str(code)[:6])
    if kl is None:
        return None
    r = kl[kl["date"] == pd.Timestamp(date)]
    if not len(r):
        return None
    v = float(r.iloc[0][field])
    return v if v == v and v > 0 else None


def fill_px(px, side):
    """成交价 = 参考价 +- 滑点。买入吃上浮, 卖出吃下浮。

    只用于成交金额计算; 持仓估值仍用未加滑点的收盘价, 避免虚增/虚减净值。
    """
    return px * (1 + SLIPPAGE) if side == "buy" else px * (1 - SLIPPAGE)


def _limit_state(klines, code, date, field="close"):
    """返回 (是否涨停, 是否跌停), 按所属板块的真实限幅判断"""
    kl = klines.get(str(code)[:6])
    if kl is None:
        return False, False
    idx = kl.index[kl["date"] == pd.Timestamp(date)]
    if len(idx) == 0 or idx[0] < 1:
        return False, False
    pos = idx[0]
    prev, cur = kl.iloc[pos - 1]["close"], float(kl.iloc[pos][field])
    lim = board_limit(code)
    return cur >= prev * (1 + lim) * 0.999, cur <= prev * (1 - lim) * 1.001


# ═══════════════════════════════════════════════════════════════
# 数据准备
# ═══════════════════════════════════════════════════════════════
print(f"加载 {TRAIN_PATH.name} ...")
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
df["code"] = df["code"].astype(str)
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
# fwd 标签在数据末尾天然缺失(未来收益还没发生)。历史段照旧剔除缺失标签,
# 但保留"末尾整段无标签的日期"用于出信号 —— 否则最新 LABEL_HORIZON 天
# 会连同 all_dates 一起被删掉, 最近几天永远无法交易。
_lab_ok = df[LABEL_RAW].notna()
_last_lab_date = df.loc[_lab_ok, "date"].max()
df = df[_lab_ok | (df["date"] > _last_lab_date)]
if args.pit_universe:
    df = apply_pit_universe(df, args.pit_universe)
if SKIP_BOARDS and not args.load_preds:
    # 训练时就把受限板块剪掉: 训练样本/截面 demean/候选池/基准全部只看能买的股。
    # 这是"没权限账户"的诚实世界 —— 比执行层事后跳过更彻底(模型不再把注意力
    # 花在永远买不了的股上)。注意基准也变成主板等权, IR 不能直接与全板块版比。
    _n0 = len(df)
    df = df[~df["code"].astype(str).str.startswith(SKIP_BOARDS)]
    print(f"板块过滤({args.skip_boards}): {_n0} -> {len(df)} 行, "
          f"{df['code'].nunique()} 只 (训练+候选+基准全部只看主板)")

print("构建增强特征...")
mkt_df, mkt_features, regime_src = compute_market_features()
regime_state = build_regime_state(regime_src)
df = df.merge(mkt_df, on="date", how="left")
for c in mkt_features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
ovn_df, ovn_features = compute_overnight_features(df["code"].unique())
df = df.merge(ovn_df[["date", "code"] + ovn_features], on=["date", "code"], how="left")

# ── 改动 A: 只按日期 demean (保序, 与 raw 收益 corr=1.00) ──
df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

all_cols = [c for c in df.columns if c not in SKIP_COLS and c not in EXCLUDED_FEATS and is_valid_feat(c)]
all_features = [f for f in all_cols if f not in LEAKAGE_FEATS]
print(f"  {len(df)} 行, {df['code'].nunique()} 只, {len(all_features)} 个候选特征")

all_dates = sorted(df["date"].unique())
date_pos = {d: i for i, d in enumerate(all_dates)}

mask = df["date"] >= pd.Timestamp(TEST_START)
if TEST_END:
    mask &= df["date"] <= pd.Timestamp(TEST_END)
dates = sorted(df[mask]["date"].unique())
if not dates:
    raise SystemExit(f"ERROR: {TEST_START} 之后没有数据")

# ── 首个真正能出信号的日期 (准入条件与下方训练循环完全一致) ──
# 特征筛选只能用这一天之前的数据, 否则会用到测试期的 fwd 标签 = 未来泄漏
_labeled_dates = np.array(sorted(df.loc[df[LABEL].notna(), "date"].unique()))
FIRST_PRED = None
for _d in dates:
    _gp = date_pos[_d]
    if _gp - LABEL_HORIZON < 0:
        continue
    if (_labeled_dates < all_dates[_gp - LABEL_HORIZON]).sum() >= MIN_TRAIN_DAYS:
        FIRST_PRED = _d
        break
if FIRST_PRED is None:
    raise SystemExit(f"ERROR: 数据不足, 无法在 {TEST_START} 之后凑出 {MIN_TRAIN_DAYS} 天训练集")

FEAT_IMPORTANCE = None
if args.features_from:
    _src = DATA_DIR / "processed" / args.features_from
    if not _src.exists():
        raise SystemExit(f"ERROR: 找不到特征来源 {_src}")
    _sel = json.load(open(_src, encoding="utf-8"))["selected_features"]
    _miss = [f for f in _sel if f not in df.columns]
    features = [f for f in _sel if f in df.columns]
    print(f"  特征: 复用 {args.features_from} 的 {len(_sel)} 个, "
          f"当前数据里可用 {len(features)} 个")
    if _miss:
        print(f"    缺失 {len(_miss)} 个: {_miss}")
else:
    _cand = all_features
    if args.drop_market_wide > 0:
        _cand = drop_market_wide(df, _cand, LABEL, FIRST_PRED,
                                 args.drop_market_wide)
    features, _imp_table = select_features(
        df, _cand, LABEL, args.n_features, args.corr_threshold,
        FIRST_PRED, n_seeds=args.select_seeds,
        pool_mult=args.select_pool_mult, return_table=True)
    FEAT_IMPORTANCE = _imp_table.head(120).to_dict("records")
style_cols = [c for c in STYLE_CANDIDATES if c in df.columns]

if args.export_matrix:
    _mp = DATA_DIR / "processed" / args.export_matrix
    _cols = ["date", "code", LABEL, LABEL_RAW] + features
    df[_cols].to_parquet(_mp, index=False)
    _meta = {"features": features, "label": LABEL, "label_raw": LABEL_RAW,
             "label_horizon": LABEL_HORIZON, "first_pred": str(pd.Timestamp(FIRST_PRED).date()),
             "min_train_days": MIN_TRAIN_DAYS,
             "test_start": TEST_START, "test_end": TEST_END,
             "train_file": args.train_file, "pit_universe": args.pit_universe,
             "features_from": args.features_from,
             "说明": "行 = PIT池内样本; y_target 已按日 demean; 末尾无标签日期仅供出信号; "
                   "训练时 X 按 code 先 ffill 再 fillna(0), 预测日直接 fillna(0) (与 wf_v35 同口径)"}
    with open(str(_mp) + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(_meta, fh, ensure_ascii=False, indent=1)
    print(f"已导出训练矩阵: {_mp} ({len(df)} 行, {len(features)} 特征) + meta。退出。")
    raise SystemExit(0)

mkt_position = None
if args.vol_target:
    va = mkt_df.set_index("date")["mkt_vol_20d"] * np.sqrt(252)
    mkt_position = (0.15 / va).clip(0.3, 1.0)

print(f"\n走进式回测: {len(dates)} 天 ({dates[0].date()} ~ {dates[-1].date()})")
print(f"标签: {LABEL_RAW} (按日期demean) | 持有 {HOLD_DAYS} 天 | 组合模式 {args.portfolio_mode}"
      f" | 目标持仓 {TARGET_POSITIONS} 只 | 本金 ¥{INIT_CAPITAL:,.0f}")
print(f"目标函数: {args.objective} | 风格中性化: {'开 ' + str(style_cols) if args.neutralize_style else '关'}")
print(f"择时: {'开' if args.ic_timing else '关'} | 波动率目标仓位: {'开' if args.vol_target else '关'}")
if regime_state is not None:
    _rs = regime_state.loc[(regime_state.index >= pd.Timestamp(TEST_START))]
    print(f"大盘空仓过滤: {args.regime_filter} (MA{args.regime_ma}, 广度<{args.regime_breadth:.0%}, "
          f"确认{args.regime_confirm}天) | 回测期内弱势日占比 {_rs.mean()*100:.1f}%")
else:
    print("大盘空仓过滤: 关")

# ═══════════════════════════════════════════════════════════════
# 训练 (或从缓存加载预测)
# ═══════════════════════════════════════════════════════════════
daily_preds = []
t0 = datetime.now()

if args.load_preds:
    import pickle
    _cache = DATA_DIR / "processed" / args.load_preds
    with open(_cache, "rb") as fh:
        cached = pickle.load(fh)
    meta, daily_preds = cached["meta"], cached["preds"]
    # 校验模型相关参数一致, 不一致则拒绝复用。
    # 外部模型缓存(meta 带 "model" 字段, 如 wf_mlp_gpu / 种子集成)不比 objective:
    # 它们本来就不是 LightGBM, 但数据口径字段必须全部一致。
    _checks = [("train_file", args.train_file), ("pit_universe", args.pit_universe),
               ("label", args.label),
               ("test_start", TEST_START), ("test_end", TEST_END),
               ("neutralize_style", args.neutralize_style),
               ("n_features", len(features)),
               ("feat_cutoff", f"{pd.Timestamp(FIRST_PRED):%Y-%m-%d}")]
    if "model" not in meta:
        _checks.append(("objective", args.objective))
    else:
        print(f"  外部模型缓存: model={meta['model']} (跳过 objective 校验)")
    for k, v in _checks:
        if meta.get(k) != v:
            raise SystemExit(f"ERROR: 预测缓存不匹配 ({k}: 缓存={meta.get(k)} 当前={v})")
    # 护栏只依赖行情不依赖模型, 每次按当前参数重算
    for dp in daily_preds:
        blocked = set()
        if args.reversal_guard > 0 and "ret_5d" in df.columns:
            r5 = df.loc[df["date"] == dp["date"], ["code", "ret_5d"]].dropna()
            if len(r5) > 10:
                pct = r5["ret_5d"].rank(pct=True)
                blocked = set(r5.loc[pct >= 1 - args.reversal_guard, "code"])
        dp["blocked"] = blocked
    print(f"从缓存加载 {len(daily_preds)} 个预测日 ({args.load_preds}), 跳过训练 "
          f"[{(datetime.now()-t0).total_seconds():.0f}s]")

for i, pred_date in enumerate([] if args.load_preds else dates):
    # 防泄漏: 标签窗口 D..D+H 必须完全早于预测日
    gpos = date_pos[pred_date]
    if gpos - LABEL_HORIZON < 0:
        continue
    cutoff = all_dates[gpos - LABEL_HORIZON]
    train_df = df[(df["date"] < cutoff) & df[LABEL].notna()]
    if train_df["date"].nunique() < MIN_TRAIN_DAYS:
        continue

    if args.objective == "rank":
        tdf = train_df.sort_values("date", kind="mergesort")
        X = tdf.groupby("code")[features].transform(lambda c: c.ffill().fillna(0))
        y = tdf.groupby("date")[LABEL].transform(to_buckets)
        groups = tdf.groupby("date", sort=True).size().values
        model = lgb.LGBMRanker(**LOCKED_PARAMS, label_gain=list(range(RANK_BUCKETS)))
        model.fit(X, y, group=groups)
    else:
        X = train_df.groupby("code")[features].transform(lambda c: c.ffill().fillna(0))
        model = lgb.LGBMRegressor(**LOCKED_PARAMS).fit(X, train_df[LABEL])

    tm = df["date"] == pred_date
    Xt = df.loc[tm, features].fillna(0)
    yt = df.loc[tm, LABEL]
    preds = model.predict(Xt)
    if args.neutralize_style and style_cols:
        preds = neutralize(preds, df.loc[tm, style_cols])
    # 最新几天 fwd 标签还没发生, IC 无法计算 -> NaN, 不影响出信号
    _m = yt.notna().values
    ic = spearmanr(preds[_m], yt[_m])[0] if _m.sum() > 5 else np.nan

    ranked = (pd.DataFrame({"code": df.loc[tm, "code"].values, "pred": preds})
              .sort_values("pred", ascending=False))

    # 反转护栏: 把当日近5日涨幅排在截面顶部的股票拉黑
    # ret_5d = close.pct_change(5), 纯滞后数据, PIT 安全
    blocked = set()
    if args.reversal_guard > 0 and "ret_5d" in df.columns:
        r5 = df.loc[tm, ["code", "ret_5d"]].dropna()
        if len(r5) > 10:
            pct = r5["ret_5d"].rank(pct=True)
            blocked = set(r5.loc[pct >= 1 - args.reversal_guard, "code"])

    daily_preds.append({"date": pred_date, "ranked": list(ranked["code"]),
                        "pred_vals": [round(float(v), 6) for v in ranked["pred"]],
                        "ic": ic, "blocked": blocked})

    if i % 50 == 0 or i == len(dates) - 1:
        print(f"  [{i+1}/{len(dates)}] {pred_date.date()} IC={ic:+.4f} "
              f"({(datetime.now()-t0).total_seconds():.0f}s)")

if args.save_preds:
    import pickle
    cache_path = DATA_DIR / "processed" / args.save_preds
    meta = {"train_file": args.train_file, "pit_universe": args.pit_universe,
            "label": args.label, "objective": args.objective,
            "test_start": TEST_START, "test_end": TEST_END,
            "neutralize_style": args.neutralize_style,
            "n_features": len(features),
            "feat_cutoff": f"{pd.Timestamp(FIRST_PRED):%Y-%m-%d}"}
    # v2 格式: 多存 pred_vals (与 ranked 对齐的预测值)。没有它就无法做
    # "信号强度门槛/离散度择时"一类实验 —— 旧 v1 缓存只存了排名。
    slim = [{"date": dp["date"], "ranked": dp["ranked"], "ic": dp["ic"],
             "pred_vals": dp.get("pred_vals")} for dp in daily_preds]
    with open(cache_path, "wb") as fh:
        pickle.dump({"meta": meta, "preds": slim}, fh)
    print(f"预测缓存已保存: {cache_path}")

print(f"\n训练/加载完成 {(datetime.now()-t0).total_seconds():.0f}s | 加载K线...")
klines = load_all_klines()
print(f"  {len(klines)} 个K线文件")

# ═══════════════════════════════════════════════════════════════
# 执行: 5日分档, 每天只换 1/HOLD_DAYS 仓位
# ═══════════════════════════════════════════════════════════════
# t1open 按次日开盘价成交; close/t1close 按收盘价成交
EXEC_FIELD = "open" if args.exec_mode == "t1open" else "close"
sched = []
for dp in daily_preds:
    if args.exec_mode == "close":          # 同日收盘 (未来函数)
        sched.append((dp, dp["date"]))
    else:                                  # T日收盘出信号 -> T+1 开盘或尾盘
        gp = date_pos[dp["date"]]
        if gp + 1 < len(all_dates):
            sched.append((dp, all_dates[gp + 1]))

mode_desc = {
    "close": "T日收盘信号 -> T日收盘成交 (同时性问题, 不可实盘)",
    "t1open": "T日收盘信号 -> T+1日开盘成交 (可实盘)",
    "t1close": "T日收盘信号 -> T+1日尾盘成交 (可实盘, 持仓比t1open晚一个交易日起算)",
}[args.exec_mode]
pf_desc = (f"每{HOLD_DAYS}天整体换仓, 持仓 {TRANCHE_N} 只" if PERIODIC
           else f"{HOLD_DAYS}日分档轮动, 每档 {TRANCHE_N} 只 (共 {TARGET_POSITIONS} 只)")
print(f"\n[执行] {pf_desc} | {mode_desc}")
print(f"[成本] 佣金 {TRADE_COST*100:.3f}%/边(最低¥{MIN_FEE:.0f}) | "
      f"滑点 {SLIPPAGE*100:.3f}%/边 (往返 {(TRADE_COST+SLIPPAGE)*2*100:.3f}%)")
cash = INIT_CAPITAL
lots = []           # 每个建仓批次: {code, shares, buy_price, open_idx}
daily_records, trade_log = [], []
buy_fee_sum = sell_fee_sum = 0.0
rejected_buy = rejected_sell = 0
# 拒单原因分解: 停牌/涨停/一手都买不起/现金不足/跌停卖不掉
rej = {"buy_halt": 0, "buy_limit_up": 0, "buy_lot_too_big": 0,
       "buy_no_cash": 0, "sell_halt": 0, "sell_limit_down": 0}
n_lot_flex = 0               # 整手粒度救济触发次数 (--lot-flex)
n_below_thresh = 0           # 因信号强度门槛留空的槽位次数 (--min-pred)
n_daily_fill = 0             # 非换仓日补买成交笔数 (--fill-daily)
n_board_skip = 0             # 因板块权限跳过的候选次数 (--skip-boards)
if MIN_PRED > 0 and daily_preds and daily_preds[0].get("pred_vals") is None:
    raise SystemExit("ERROR: --min-pred 需要带 pred_vals 的 v2 预测缓存, "
                     "当前缓存是旧格式(只存排名)")
ic_hist, in_cash, ic_cash = [], False, False
n_cash_days = 0
# 续持次数: 到期但仍在目标名单里, 省下一次往返成本
n_rolled = 0

for i, (dp, exec_date) in enumerate(sched):
    d, dstr = exec_date, str(pd.Timestamp(exec_date).date())
    sig_str = str(dp["date"].date())
    ic = dp["ic"]

    if args.ic_timing:
        if not np.isnan(ic):
            ic_hist.append(ic)
        if len(ic_hist) >= 10:
            if all(x < 0 for x in ic_hist[-3:]):
                ic_cash = True
            if ic_cash and np.mean(ic_hist[-10:]) > 0:
                ic_cash = False
    regime_cash = bool(regime_state.get(dp["date"], False)) if regime_state is not None else False
    was_in_cash = in_cash
    in_cash = ic_cash or regime_cash
    n_cash_days += int(in_cash)

    pos_size = 1.0
    if args.vol_target and mkt_position is not None and dp["date"] in mkt_position.index:
        v = float(mkt_position.loc[dp["date"]])
        pos_size = 1.0 if np.isnan(v) else v

    # is_rebal 要在卖出之前就知道 —— 因为“该不该续持”取决于本次是否换仓
    is_rebal = (not PERIODIC) or (i % HOLD_DAYS == 0) or (was_in_cash and not in_cash)

    # 换仓日的目标组合: ranked 里前 TRANCHE_N 个未被持禁的。
    # 已持仓且仍在目标里的, 到期也不卖 —— 卖了再买回要付一次往返
    # (佣金+滑点)*2, 白付这笔钱却回到同样的持仓。
    # 空仓信号(in_cash)时不适用: 那是要清光。
    # 分档轮动模式(非 PERIODIC)不适用: 各档到期时点不同, 续持会模糊档位。
    roll_set = set()
    if NO_ROLL:
        pass                  # --no-roll: 退回旧行为(到期一律卖掉再买回), 仅供 A/B
    elif PERIODIC and (is_rebal or FILL_DAILY) and not in_cash:
        # FILL_DAILY 时天天建 roll_set: 中途补买的批次会在非换仓日到期,
        # 它们也该享受同样的续持经济(仍在目标名单就不白付往返成本)。
        # 主周期批次只在换仓日到期, 不受影响。
        _blocked = dp.get("blocked") or set()
        # --roll-rank: 卖出门槛比买入宽。新买仍只买前 TRANCHE_N 名,
        # 但已持仓只要没掉出前 M 名就不卖 (避免为微小排名变化付往返成本)
        _lim = max(TRANCHE_N, ROLL_RANK) if ROLL_RANK else TRANCHE_N
        for code in dp["ranked"]:
            if len(roll_set) >= _lim:
                break
            if code in _blocked:
                continue
            if SKIP_BOARDS and str(code).startswith(SKIP_BOARDS):
                continue          # 买不了的板块不可能持有, 也不占目标名单位置
            roll_set.add(code)

    # ── 1. 卖出到期批次 (持满 HOLD_DAYS) ──
    sell_fee = 0.0
    keep = []
    rolled_lots = []          # 本轮续持的, 稍后要把权重配平回等权
    for lot in lots:
        matured = (i - lot["open_idx"]) >= HOLD_DAYS
        if not (matured or in_cash):
            keep.append(lot)
            continue
        if matured and not in_cash and lot["code"] in roll_set:
            # 续持: 重置到期时钟, 成本价不动(没发生交易)
            lot["open_idx"] = i
            lot["rolled"] = lot.get("rolled", 0) + 1
            keep.append(lot)
            rolled_lots.append(lot)
            n_rolled += 1
            continue
        px = get_px(klines, lot["code"], d, EXEC_FIELD)
        _, limit_down = _limit_state(klines, lot["code"], d, EXEC_FIELD)
        if px is None or limit_down:
            rejected_sell += 1
            rej["sell_halt" if px is None else "sell_limit_down"] += 1
            keep.append(lot)          # 卖不掉, 继续持有
            continue
        px = fill_px(px, "sell")
        gross = lot["shares"] * px
        fee = max(gross * TRADE_COST, MIN_FEE)
        cash += gross - fee
        sell_fee += fee
        trade_log.append({"date": dstr, "signal_date": sig_str, "code": lot["code"],
                          "action": "sell", "shares": lot["shares"], "price": px,
                          "gross": gross, "fee": fee, "net": gross - fee,
                          "reason": "matured"})
    lots = keep

    # ── 2. 开新批次: 用 1/HOLD_DAYS 的权益买 TRANCHE_N 只 ──
    # (is_rebal 已在上面算过; 空仓转回场当天允许立即建仓)
    # FILL_DAILY: 非换仓日也进入买入段, 但只在有空槽位时真正买得成
    buy_fee = 0.0
    if not in_cash and (is_rebal or FILL_DAILY):
        equity = cash + sum(l["shares"] * (get_px(klines, l["code"], d, EXEC_FIELD) or l["buy_price"])
                            for l in lots)

        # 续持仓配平: 续持省下了往返成本, 但如果就这么放着, 它的权重会随
        # 涨跌漂移, 等权再平衡(高抛低吸)这个收益来源就没了。所以只交易
        # 差额把它拉回目标市值: 既保住等权纪律, 又避开整笔进出。
        # 不设容忍带 —— 100股的整手粒度已经天然把琐碎调整挡掉了,
        # 再加阀值就多一个可拟合的参数。
        target_val = equity * pos_size / TRANCHE_N
        for lot in rolled_lots:
            rpx = get_px(klines, lot["code"], d, EXEC_FIELD)
            if rpx is None:
                continue
            limit_up_r, limit_down_r = _limit_state(klines, lot["code"], d, EXEC_FIELD)
            delta_val = target_val - lot["shares"] * rpx
            if delta_val < 0:                       # 超配 -> 减仓
                if limit_down_r:
                    continue                        # 跌停卖不掉
                spx = fill_px(rpx, "sell")
                sh = int(-delta_val / (spx * 100)) * 100
                sh = min(sh, lot["shares"] - 100)   # 至少留一手, 否则就不是续持了
                if sh <= 0:
                    continue
                gross = sh * spx
                fee = max(gross * TRADE_COST, MIN_FEE)
                cash += gross - fee
                sell_fee += fee
                lot["shares"] -= sh
                trade_log.append({"date": dstr, "signal_date": sig_str, "code": lot["code"],
                                  "action": "sell", "shares": sh, "price": spx,
                                  "gross": gross, "fee": fee, "net": gross - fee,
                                  "reason": "roll_trim"})
            elif delta_val > 0:                     # 欠配 -> 加仓
                if limit_up_r:
                    continue                        # 涨停买不进
                bpx = fill_px(rpx, "buy")
                sh = int(delta_val / (bpx * 100)) * 100
                if sh <= 0:
                    continue
                gross = sh * bpx
                fee = max(gross * TRADE_COST, MIN_FEE)
                if gross + fee > cash:
                    continue
                cash -= gross + fee
                buy_fee += fee
                # 加仓后成本价按股数加权平均, 否则盈亏会算错
                tot = lot["shares"] + sh
                lot["buy_price"] = (lot["buy_price"] * lot["shares"] + bpx * sh) / tot
                lot["shares"] = tot
                trade_log.append({"date": dstr, "signal_date": sig_str, "code": lot["code"],
                                  "action": "buy", "shares": sh, "price": bpx,
                                  "gross": gross, "fee": fee, "net": -(gross + fee),
                                  "reason": "roll_topup"})

        denom = 1 if PERIODIC else HOLD_DAYS
        tranche_budget = min(equity * pos_size / denom, cash)
        remaining = tranche_budget
        held = {l["code"] for l in lots}
        blocked = dp.get("blocked") or set()
        # 从已持仓数起算, 否则续持的那几只不计数, 总持仓会超过 TRANCHE_N。
        # (旧逻辑下卖单被拒(停牌/跌停)时也有同样的超配问题)
        bought = len(lots)
        _pred_of = None
        if MIN_PRED > 0:
            _pv = dp.get("pred_vals")
            _pred_of = dict(zip(dp["ranked"], _pv)) if _pv else {}
        for code in dp["ranked"]:
            if bought >= TRANCHE_N:
                break
            # 信号强度门槛: ranked 按 pred 降序, 第一个不达标的之后全都不达标。
            # 留空的槽位不沿排名退而求其次 —— 那正是要避免的"买弱信号"。
            if _pred_of is not None and _pred_of.get(code, -1.0) < MIN_PRED:
                n_below_thresh += TRANCHE_N - bought
                break
            if code in held:            # 已持有则跳到下一名, 保证广度
                continue
            if code in blocked:         # 刚急涨过的不追
                continue
            if SKIP_BOARDS and str(code).startswith(SKIP_BOARDS):
                n_board_skip += 1
                if SKIP_CASH:
                    # 留现金模式: 槽位被"消耗"但不买, 预算也预留不给后面的槽位。
                    # (否则就变成变相递补: 钱流给下一名, 只是换个名字)
                    alloc = remaining / (TRANCHE_N - bought)
                    remaining -= alloc
                    bought += 1
                continue                # 递补模式: 直接看下一名
            px = get_px(klines, code, d, EXEC_FIELD)
            if px is None:                          # 停牌/无行情
                rejected_buy += 1
                rej["buy_halt"] += 1
                continue
            limit_up, _ = _limit_state(klines, code, d, EXEC_FIELD)
            if limit_up:
                rejected_buy += 1
                rej["buy_limit_up"] += 1
                continue
            px = fill_px(px, "buy")
            alloc = remaining / (TRANCHE_N - bought)
            shares = int(alloc / (px * 100)) * 100
            if shares <= 0:                         # 预算不足一手(100股)
                # 整手粒度救济: 一手成本在预算容忍带内就买, 保住排名靠前的信号。
                # 超出的部分由 remaining 自然扣减, 后面的槽位预算相应变小。
                if LOT_FLEX > 0 and px * 100 <= alloc * (1 + LOT_FLEX):
                    shares = 100
                    n_lot_flex += 1
                else:
                    rejected_buy += 1
                    rej["buy_lot_too_big"] += 1
                    continue
            gross = shares * px
            fee = max(gross * TRADE_COST, MIN_FEE)
            if gross + fee > cash:
                rejected_buy += 1
                rej["buy_no_cash"] += 1
                continue
            cash -= gross + fee
            remaining -= gross + fee
            buy_fee += fee
            lots.append({"code": code, "shares": shares, "buy_price": px, "open_idx": i})
            held.add(code)
            bought += 1
            if not is_rebal:
                n_daily_fill += 1
            trade_log.append({"date": dstr, "signal_date": sig_str, "code": code,
                              "action": "buy", "shares": shares, "price": px,
                              "gross": gross, "fee": fee, "net": -(gross + fee),
                              "reason": "new_tranche" if is_rebal else "daily_fill"})

    buy_fee_sum += buy_fee
    sell_fee_sum += sell_fee

    mv = sum(l["shares"] * (get_close(klines, l["code"], d) or l["buy_price"]) for l in lots)
    pv = cash + mv
    prev = daily_records[-1]["portfolio_value"] if daily_records else INIT_CAPITAL
    daily_records.append({
        "date": dstr, "portfolio_value": round(pv, 2), "cash": round(cash, 2),
        "daily_ret": round(pv / prev - 1 if prev > 0 else 0.0, 6),
        "n_holdings": len(lots), "holdings": [l["code"] for l in lots],
        "deployed": round(mv / pv, 4) if pv > 0 else 0.0,
        "sell_cost": round(sell_fee, 2), "buy_cost": round(buy_fee, 2),
        "ic": round(ic, 4) if not np.isnan(ic) else None,
        "in_cash": bool(in_cash),
    })

# 期末清仓
last_d = pd.Timestamp(sched[-1][1])
final_value = cash
for l in lots:
    px = get_close(klines, l["code"], last_d)
    if px is None:
        final_value += l["shares"] * l["buy_price"]
        continue
    px = fill_px(px, "sell")
    gross = l["shares"] * px
    fee = max(gross * TRADE_COST, MIN_FEE)
    final_value += gross - fee
    trade_log.append({"date": str(last_d.date()), "signal_date": str(last_d.date()),
                      "code": l["code"], "action": "force_sell",
                      "shares": l["shares"], "price": px, "gross": gross,
                      "fee": fee, "net": gross - fee, "reason": "end"})

# ═══════════════════════════════════════════════════════════════
# 指标 + 基准对比
# ═══════════════════════════════════════════════════════════════
rdf = pd.DataFrame(daily_records)
rdf["date"] = pd.to_datetime(rdf["date"])
n = len(rdf)
r = rdf["daily_ret"]

bench = df.groupby("date")["fwd_1d_ret"].mean().shift(1)   # T日实现的是T-1日标签
bench = bench.reindex(rdf["date"]).fillna(0.0).values
bench_s = pd.Series(bench, index=rdf.index)

cum = (1 + r).cumprod()
max_dd = float((cum / cum.expanding().max() - 1).min())
ann = float((1 + r).prod() ** (252 / n) - 1) if n else 0.0
total_ret = (final_value / INIT_CAPITAL - 1) * 100
sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

bench_cum = float((1 + bench_s).prod() - 1)
bench_ann = float((1 + bench_s).prod() ** (252 / n) - 1) if n else 0.0
excess = r - bench_s
beta = float(np.cov(r, bench_s)[0, 1] / np.var(bench_s)) if np.var(bench_s) > 0 else 0.0
alpha_d = float(r.mean() - beta * bench_s.mean())
ir = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

ics = [x["ic"] for x in daily_records if x["ic"] is not None]
ic_mean, ic_std = (float(np.mean(ics)), float(np.std(ics))) if ics else (0.0, 0.0)
ic_t = ic_mean / ic_std * np.sqrt(len(ics)) if ics and ic_std > 0 else 0.0
total_cost_pct = (buy_fee_sum + sell_fee_sum) / INIT_CAPITAL * 100

print(f"\n{'='*64}")
print(f"  v35 [{args.label}标签/持有{HOLD_DAYS}天/{TARGET_POSITIONS}只] 结果")
print(f"{'='*64}")
print(f"  交易日           : {n}")
print(f"  特征数           : {len(features)}")
print(f"  IC               : {ic_mean:+.4f} (std {ic_std:.4f}, t={ic_t:.2f})")
print(f"  平均部署率       : {rdf['deployed'].mean()*100:.1f}%")
print(f"  平均持仓         : {rdf['n_holdings'].mean():.1f} 只")
print(f"  总费用           : {total_cost_pct:.1f}% of capital")
print(f"  拒单             : buy={rejected_buy} sell={rejected_sell}")
print(f"    买入被拒   : 停牌 {rej['buy_halt']} | 涨停 {rej['buy_limit_up']} | "
      f"买不起一手 {rej['buy_lot_too_big']} | 现金不足 {rej['buy_no_cash']}")
print(f"    卖出被拒   : 停牌 {rej['sell_halt']} | 跌停 {rej['sell_limit_down']}")
print(f"  空仓天数         : {n_cash_days} ({100*n_cash_days/n if n else 0:.1f}%)")
print(f"  交易笔数         : {len(trade_log)}")
print(f"  续持次数         : {n_rolled} (到期但仍在目标里, 省下往返成本)")
print(f"  ─────────────── 收益 ───────────────")
print(f"  期末净值         : ¥{final_value:,.0f} (起始 ¥{INIT_CAPITAL:,.0f})")
print(f"  总收益           : {total_ret:+.1f}%   年化 {ann*100:+.1f}%")
print(f"  夏普             : {sharpe:.2f}   最大回撤 {max_dd*100:.1f}%")
print(f"  ────────── vs 等权买入持有基准 ──────────")
print(f"  基准总收益       : {bench_cum*100:+.1f}%   年化 {bench_ann*100:+.1f}%")
print(f"  超额(日均)       : {excess.mean()*100:+.4f}%/day  年化 {excess.mean()*252*100:+.1f}%")
print(f"  信息比率 IR      : {ir:.2f}")
print(f"  beta             : {beta:.3f}")
print(f"  alpha(年化)      : {alpha_d*252*100:+.1f}%")
verdict = "跑赢基准 ✓" if excess.mean() > 0 else "跑输基准 ✗"
print(f"  结论             : {verdict}")

# ── 分段稳健性: 防止在整段测试集上挑参数造成过拟合 ──
halves = []
for name, sl in (("前半段", slice(0, n // 2)), ("后半段", slice(n // 2, n))):
    rr, bb = r.iloc[sl], bench_s.iloc[sl]
    ex = (rr - bb)
    k = len(rr)
    halves.append({
        "segment": name,
        "period": f"{rdf['date'].iloc[sl].iloc[0]:%Y-%m-%d} ~ {rdf['date'].iloc[sl].iloc[-1]:%Y-%m-%d}",
        "strategy_pct": round(float((1 + rr).prod() - 1) * 100, 1),
        "benchmark_pct": round(float((1 + bb).prod() - 1) * 100, 1),
        "excess_annual_pct": round(float(ex.mean()) * 252 * 100, 1),
        "ir": round(float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0, 2),
    })
print(f"  ─────────── 分段稳健性 ───────────")
for h in halves:
    flag = "✓" if h["excess_annual_pct"] > 0 else "✗"
    print(f"  {h['segment']} {h['period']}: 策略 {h['strategy_pct']:+.1f}% "
          f"vs 基准 {h['benchmark_pct']:+.1f}% | 年化超额 {h['excess_annual_pct']:+.1f}% "
          f"IR {h['ir']:.2f} {flag}")
both = all(h["excess_annual_pct"] > 0 for h in halves)
print(f"  两段都跑赢       : {'是 ✓' if both else '否 ✗'}")
print(f"  耗时             : {(datetime.now()-t0).total_seconds():.0f}s")

json.dump({
    "label": LABEL_RAW,
    "neutralization": "date_demean_only",
    "objective": args.objective,
    "exec_mode": args.exec_mode,
    "slippage": SLIPPAGE, "trade_cost": TRADE_COST, "min_fee": MIN_FEE,
    "portfolio_mode": args.portfolio_mode,
    "neutralize_style": args.neutralize_style,
    "style_cols": style_cols if args.neutralize_style else [],
    "hold_days": HOLD_DAYS, "tranche_n": TRANCHE_N, "target_positions": TARGET_POSITIONS,
    "ic_timing": args.ic_timing, "vol_target": args.vol_target,
    "train_file": args.train_file, "pit_universe": args.pit_universe,
    "regime_filter": args.regime_filter, "regime_ma": args.regime_ma,
    "regime_breadth": args.regime_breadth, "regime_confirm": args.regime_confirm,
    "reversal_guard": args.reversal_guard,
    "lot_flex": LOT_FLEX,
    "roll_rank": ROLL_RANK,
    "min_pred": MIN_PRED, "fill_daily": FILL_DAILY,
    "skip_boards": list(SKIP_BOARDS), "skip_boards_mode": args.skip_boards_mode,
    "features": len(features), "selected_features": features,
    "feat_select_cutoff": f"{pd.Timestamp(FIRST_PRED):%Y-%m-%d}",
    "feat_select_seeds": args.select_seeds,
    "feat_select_pool_mult": args.select_pool_mult,
    "feat_select_n_target": args.n_features,
    "feat_importance_top120": FEAT_IMPORTANCE,
    "period": f"{rdf['date'].iloc[0]:%Y-%m-%d} ~ {rdf['date'].iloc[-1]:%Y-%m-%d}",
    "n_days": n, "initial_capital": INIT_CAPITAL, "top_n": TARGET_POSITIONS,
    "summary": {
        "ic_mean": round(ic_mean, 4), "ic_std": round(ic_std, 4), "ic_tstat": round(ic_t, 2),
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_ret, 1), "annualized_return_pct": round(ann * 100, 1),
        "sharpe": round(sharpe, 2), "max_dd_pct": round(max_dd * 100, 1),
        "benchmark_total_pct": round(bench_cum * 100, 1),
        "benchmark_annual_pct": round(bench_ann * 100, 1),
        "excess_daily_pct": round(excess.mean() * 100, 4),
        "excess_annual_pct": round(excess.mean() * 252 * 100, 1),
        "information_ratio": round(ir, 2), "beta": round(beta, 3),
        "alpha_annual_pct": round(alpha_d * 252 * 100, 1),
        "avg_deployed_pct": round(rdf["deployed"].mean() * 100, 1),
        "avg_holdings": round(rdf["n_holdings"].mean(), 1),
        "total_cost_pct": round(total_cost_pct, 1),
        "rejected_buy": rejected_buy, "rejected_sell": rejected_sell,
        "reject_breakdown": rej,
        "n_trades": len(trade_log),
        "n_rolled": n_rolled,
        "n_lot_flex": n_lot_flex,
        "n_below_thresh": n_below_thresh,
        "n_daily_fill": n_daily_fill,
        "n_board_skip": n_board_skip,
        "cash_days": n_cash_days,
        "cash_days_pct": round(100 * n_cash_days / n, 1) if n else 0.0,
        "beat_benchmark": bool(excess.mean() > 0),
        "beat_both_halves": bool(both),
    },
    "stability": halves,
    "daily": daily_records, "trades": trade_log,
}, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
print(f"\n已保存: {OUT_PATH}")
