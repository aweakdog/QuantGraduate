"""每日收盘后自动重建流水线 (K线 -> 特征 -> 训练集 -> 实盘信号)

设计要点:
  * 幂等: 同一天重复跑不会重复结算、不会污染状态 (特征增量跳过, 信号自带挂单结算逻辑)
  * 原子: 训练集先建到临时文件, 校验通过才替换正式文件, 旧文件留备份
  * 非交易日自动跳过: K线最新日未前进则直接退出, 不碰训练集和状态
  * 全程写 pipeline_status.json, 供网页展示数据新鲜度和上次结果

用法:
  python scripts/daily_rebuild.py                 # 完整流程 (519池)
  python scripts/daily_rebuild.py --full-market    # 同时更新全市场K线 (广度用)
  python scripts/daily_rebuild.py --skip-kline     # 跳过拉数, 只重建+出信号
  python scripts/daily_rebuild.py --dry-run        # 只检查不落盘
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
PY = sys.executable
DATA = ROOT / "data"
KLINE_DIR = DATA / "raw" / "kline"
PROC = DATA / "processed"
LIVE = DATA / "live"

WATCHLIST = "watchlist_pit.json"
TRAIN_FILE = "training_data_pit_v24.parquet"
TMP_FILE = "training_data_pit_v24_new.parquet"
STATUS_PATH = LIVE / "pipeline_status.json"

# live_signal 的参数统一从 live_config 取 —— 禁止在此处硬编码。
# 一旦这里和网页/命令行用的参数不一致, live_signal 的指纹校验会直接报错退出,
# 而它发生在收盘后无人值守的时段, 不容易发现。
import trading_calendar  # noqa: E402
from live_config import FEATURES_FROM, PROFILES, signal_args  # noqa: E402

# 最新日允许有多少比例的"模型实际使用特征"全为空。
# 注意区分两个量: 训练集有 440 列, 而线上模型只用其中 80 列。
# 外部宏观数据(商品/美股/汇率/国债, 约127列)一列都不在这 80 列里,
# 它们断更对预测没有任何影响, 所以不能拿全表 NaN 比例当拦截依据 ——
# 那会让一次无害的构建被拦下, 反而害得当天没有信号。
USED_FEAT_NAN_LIMIT = 0.10

# live_signal 在运行时现算、不落在训练集里的特征。校验时必须跳过,
# 否则会把它们误判为"在用特征缺列"。
# 来源: live_signal.compute_overnight_features() 与 mkt_* 市场缓存。
# 注意: 这份名单与 live_signal 里的实现存在漂移风险, 改那边记得同步这里。
RUNTIME_FEAT_PREFIXES = ("ovn_", "overnight_", "intraday_", "mkt_")

# 已知的"死特征": 在整个训练集上恒为常量, 毫无信息量。
# 成因: dde_net / mtss_balance / fund_flow 三列原本来自 thsdk(同花顺), 该源停后
# feature_engine 的 fillna(0) 把"源列不存在"静默变成了"值为0", 于是整段历史被抹平。
# 后果: 线上模型名义上用 80 特征, 实际只有 69 个。
#
# 2026-08-09 已清空 —— 不是放弃检查, 而是根因都消掉了:
#   * fund_flow 改由 tushare moneyflow 供给, 不再是常量(非空 15.9% -> 99.9%)
#   * fillna(0) 已按经济含义拆开(见 feature_engine._impute): 存量列前值填充,
#     流量列只填首尾之间的散点缺口, 首尾之外留 NaN, 不再把"没数据"伪造成"值为0"
#   * dde_net* / con_* 已从特征集剔除(FEATURES_FROM=F1B), 不在"在用特征"里
# 实测: 在用的 71 个特征里恒常量 0 个 (预检脚本对 f1b 矩阵逐列 nunique 验证)。
#
# 保持空集是有意的: 名单只能缩短不能加长。空集意味着此后任何一个特征退化成常量
# 都会立刻拦住管线, 而不是被一份过期的豁免名单默默放过 —— 上次就是这样带着
# 11 个死特征跑了好几天没人发现。
KNOWN_DEAD_FEATS = set()

status = {"started_at": None, "finished_at": None, "ok": False,
          "skipped_reason": None, "stages": {}}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def save_status():
    LIVE.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


def stage(name, **kv):
    status["stages"][name] = {"at": datetime.now().isoformat(timespec="seconds"), **kv}
    save_status()


def run(cmd, name, timeout):
    """跑子进程, 全量日志进 stdout, 失败抛异常"""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout,
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.returncode and (r.stderr or "") or "")
    tail = "\n".join(l for l in out.splitlines() if l.strip())[-1500:]
    el = time.time() - t0
    if r.returncode != 0:
        stage(name, ok=False, seconds=round(el), log=tail)
        raise RuntimeError(f"{name} 失败 rc={r.returncode}\n{tail}")
    log(f"  {name} 完成 {el:.0f}s")
    stage(name, ok=True, seconds=round(el), log=tail)
    return out


def kline_max_date():
    """PIT 池里 K线的最新交易日 (取众数, 避免个别停牌股拉低)"""
    codes = json.loads((DATA / "universe" / WATCHLIST).read_text())
    items = codes.get("watchlist", codes) if isinstance(codes, dict) else codes
    dates = []
    for it in items:
        c6 = str(it["code"] if isinstance(it, dict) else it)[:6]
        p = KLINE_DIR / f"{c6}.parquet"
        if p.exists():
            try:
                dates.append(pd.read_parquet(p, columns=["date"])["date"].max())
            except Exception:
                pass
    if not dates:
        return None
    return pd.Series(dates).mode().iloc[0]


def train_max_date(path):
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=["date"])["date"].max()


def live_used_features():
    """读线上模型实际使用的特征列表 (FEATURES_FROM 里的 selected_features)。

    取不到就返回 None, 调用方退回到"全表 NaN 比例"的旧口径, 宁可粗糙也不能不校验。
    """
    try:
        d = json.loads((PROC / FEATURES_FROM).read_text(encoding="utf-8"))
        feats = d.get("selected_features")
        return list(feats) if feats else None
    except Exception as e:
        log(f"警告: 读取特征集 {FEATURES_FROM} 失败 ({e}), 退回全表NaN口径")
        return None


def validate_new_train(new_path, old_path, require_advance=True):
    """新训练集必须: 日期前进 + 列集不缺 + 行数不异常缩水 + 最新日在用特征不全空

    require_advance=False 用于 --force 重建(如只为应用代码修复),
    此时日期不前进是预期的, 不应当成错误。

    最新日的空值校验只看模型在用的那 80 列: 未被使用的列(如已断更的宏观数据)
    再空也不影响信号, 只记 warning 不拦截。
    """
    new = pd.read_parquet(new_path)
    n_max = new["date"].max()
    problems = []

    if old_path.exists():
        old = pd.read_parquet(old_path, columns=["date", "code"])
        o_max = old["date"].max()
        if require_advance and n_max <= o_max:
            problems.append(f"日期未前进: 新{pd.Timestamp(n_max).date()} <= 旧{pd.Timestamp(o_max).date()}")
        elif n_max < o_max:
            problems.append(f"日期倒退: 新{pd.Timestamp(n_max).date()} < 旧{pd.Timestamp(o_max).date()}")
        if len(new) < len(old) * 0.95:
            problems.append(f"行数缩水: {len(new):,} < 旧{len(old):,} 的95%")
        if new["code"].nunique() < old["code"].nunique() * 0.95:
            problems.append(f"股票数缩水: {new['code'].nunique()} < 旧{old['code'].nunique()}")
        old_cols = set(pd.read_parquet(old_path).columns)
        missing = old_cols - set(new.columns)
        if missing:
            problems.append(f"缺列 {len(missing)} 个: {sorted(missing)[:8]}")

    last = new[new["date"] == n_max]
    warnings = []
    used_nan = []
    info = {}
    if last.empty:
        problems.append("最新日无数据行")
    else:
        nan_cols = set(last.columns[last.isna().all(axis=0)])
        all_nan = len(nan_cols)
        used = live_used_features()
        # tk_* 逐笔列不在基础 v24 矩阵里 —— 它们由 §4.5 的 build_tick_augmented
        # 基于本矩阵另行加列, 存在性/覆盖率在 §4.5 里检。在这里要求会误拦
        # (2026-08-16 彩排实拍: 切 V24PUT 后本校验报"在用特征缺列 11 个 tk_*")。
        if used is not None:
            used = [c for c in used if not c.startswith("tk_")]
        if used is None:
            if all_nan > len(new.columns) * 0.3:
                problems.append(f"最新日全NaN列过多: {all_nan}/{len(new.columns)}")
        else:
            absent = [c for c in used if c not in new.columns
                      and not c.startswith(RUNTIME_FEAT_PREFIXES)]
            if absent:
                problems.append(f"在用特征缺列 {len(absent)} 个: {sorted(absent)[:8]}")
            used_nan = sorted(c for c in used if c in nan_cols)
            if len(used_nan) > len(used) * USED_FEAT_NAN_LIMIT:
                problems.append(
                    f"最新日在用特征全NaN过多: {len(used_nan)}/{len(used)} "
                    f"(上限{USED_FEAT_NAN_LIMIT:.0%}) {used_nan[:8]}")
            elif used_nan:
                warnings.append(
                    f"最新日有 {len(used_nan)}/{len(used)} 个在用特征全NaN, "
                    f"未超上限但已降级: {used_nan}")
        # 零方差体检: 恒为常量的特征对模型毫无贡献, 而且一旦是“曾经有值、后来被抹平”,
        # 模型会拿当年学到的分裂点去切一个常量, 预测会系统性偏。
        # 这种退化不会报错也不会产生 NaN, 只能靠主动体检发现。
        if used is not None:
            chk = [c for c in used if c in new.columns]
            const = {c for c in chk if new[c].nunique(dropna=True) <= 1}
            new_dead = sorted(const - KNOWN_DEAD_FEATS)
            if new_dead:
                problems.append(
                    f"新出现恒常量特征 {len(new_dead)} 个(数据源可能已静默断更): {new_dead[:8]}")
            still_dead = sorted(const & KNOWN_DEAD_FEATS)
            if still_dead:
                warnings.append(
                    f"已知死特征仍为常量 {len(still_dead)}/{len(used)}, "
                    f"模型实际有效特征只有 {len(used) - len(still_dead)} 个: {still_dead}")
            info["dead_feats"] = still_dead

        unused_nan = all_nan - len(used_nan)
        if unused_nan > len(new.columns) * 0.2:
            warnings.append(
                f"最新日有 {unused_nan} 个未被模型使用的列全NaN "
                f"(占全表{unused_nan / len(new.columns):.0%}), 疑有外部数据源断更, "
                f"不影响当日信号但需排查")

    info.update({"rows": len(new), "codes": int(new["code"].nunique()),
                 "cols": len(new.columns), "max_date": str(pd.Timestamp(n_max).date()),
                 "last_day_rows": len(last), "used_feat_nan": used_nan,
                 "warnings": warnings})
    return problems, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-market", action="store_true",
                    help="同时全量更新市场K线 (广度/择时用, 约25分钟)")
    ap.add_argument("--skip-kline", action="store_true", help="跳过K线更新")
    ap.add_argument("--force", action="store_true",
                    help="即使K线日期没前进也强制重建")
    ap.add_argument("--procs", type=int, default=10, help="K线并发进程数")
    ap.add_argument("--feat-procs", type=int, default=8,
                    help="特征构建并行进程数 (pandas 有 GIL 瓶颈, 必须用进程不能用线程)。"
                         "默认从 24 降为 8: 这是共用服务器, 而整条管线的长杆是网络"
                         "(K线下载 ≈25分钟), 特征构建快上几十秒对完成时间几乎无影响")
    ap.add_argument("--dry-run", action="store_true", help="只检查不落盘")
    a = ap.parse_args()

    t0 = time.time()
    status["started_at"] = datetime.now().isoformat(timespec="seconds")
    save_status()

    train_path = PROC / TRAIN_FILE
    tmp_path = PROC / TMP_FILE
    before_train = train_max_date(train_path)
    log(f"当前训练集最新日: {pd.Timestamp(before_train).date() if before_train is not None else '无'}")

    # ── 1. K线 ──
    if not a.skip_kline:
        if a.full_market:
            # scope=all 已是 519 池的超集, 不必再单独拉一遍
            run([PY, "-u", "scripts/update_kline_akshare.py",
                 "--scope", "all", "--start", "20190101",
                 "--procs", str(a.procs), "--timeout", "25"],
                "kline_full_market", timeout=7200)
        else:
            run([PY, "-u", "scripts/update_kline_akshare.py",
                 "--scope", "universe", "--watchlist", WATCHLIST,
                 "--start", "20190101", "--procs", str(a.procs), "--timeout", "25"],
                "kline_universe", timeout=3600)
    else:
        stage("kline", ok=True, skipped=True)

    k_max = kline_max_date()
    status["kline_max_date"] = str(pd.Timestamp(k_max).date()) if k_max is not None else None
    log(f"K线最新交易日: {status['kline_max_date']}")

    # ── 1.5 交易日历缓存 ──
    # 网页要靠它算"这份计划该今天还是明天执行"。放在非交易日提前退出之前,
    # 这样即使当天不开市也会把日历补上。整块 try 兜住: 日历抓不到只是页面
    # 少显示一个具体日期, 绝不能因此让当晚的信号出不来。
    try:
        ok_cal = trading_calendar.ensure_fresh()
        _cal_days, _cal_meta = trading_calendar.load()
        stage("trading_calendar", ok=bool(ok_cal),
              covers_to=(_cal_meta or {}).get("last"),
              fetched_at=(_cal_meta or {}).get("fetched_at"))
    except Exception as e:
        log(f"WARN 交易日历环节异常(不影响出信号): {e}")
        stage("trading_calendar", ok=False, error=str(e)[:300])

    # ── 2. 非交易日/数据未更新 -> 干净退出 ──
    if (not a.force and before_train is not None and k_max is not None
            and pd.Timestamp(k_max) <= pd.Timestamp(before_train)):
        status["skipped_reason"] = (
            f"K线最新日 {pd.Timestamp(k_max).date()} 未超过训练集 "
            f"{pd.Timestamp(before_train).date()}, 判定非交易日或数据未更新")
        status["ok"] = True
        status["finished_at"] = datetime.now().isoformat(timespec="seconds")
        save_status()
        log(f"跳过: {status['skipped_reason']}")
        return 0

    if a.dry_run:
        log("dry-run: 到此为止, 不重建训练集")
        status["ok"] = True
        status["finished_at"] = datetime.now().isoformat(timespec="seconds")
        save_status()
        return 0

    # ── 2.5 外部数据日更 (资金流 / 两融) ──
    # 2026-08-09 查出的根因: 这条链路里【从来没有】资金流拉取, pull_fundflow_shard.py
    # 只被手工调用过。于是资金流表停在最后一次人工操作的日期(08-04), 而 K 线一直在走,
    # 导致全部资金面特征在最新日逐渐变 NaN —— 不报错、只是慢慢腐烂。
    #
    # 用 tushare 而非旧源: moneyflow 覆盖每日 5100+ 只(旧源的 dde_net 只有 243 只),
    # 且 net_mf_amount 与旧源 fund_flow 对账中位比值 10000.0028 / 相关 0.9996,
    # 是同一口径的万元→元换算。margin_detail.rzrqye 与 mtss_balance 同理(比值 1.000000)。
    #
    # 不能让它拖垮整条管线: 拉取失败只是数据停在昨天(特征退化由 validate_new_train
    # 的在用特征 NaN 上限兜住), 而 raise 会导致当晚彻底没有信号。故与 trading_calendar
    # 同样处理 —— 整块 try 兜住, 失败只记 stage 不中断。
    # k_max 可能为 None(K线日期取不到): 早退分支要求 k_max is not None, 所以那种情况
    # 会一路走到这里。此时退回用当天算回看起点, 不能让 pd.Timestamp(None)=NaT 废掉这步。
    _since = ((pd.Timestamp(k_max) if k_max is not None else pd.Timestamp.now())
              - pd.Timedelta(days=90)).strftime("%Y%m%d")
    for _iface in ("moneyflow", "margin_detail"):
        try:
            # --refresh-days 5: margin_detail 是延迟发布的, 实测当日傍晚只有 1994 只、
            # 次日才补齐到 4422 只。默认的"拉过就不再拉"会把这份残缺永久冻结, 所以
            # 每次重拉尾部 5 天。重拉安全 —— save_by_year 按主键 keep="last" 覆盖。
            run([PY, "-u", "-m", "pipeline.pull_tushare", _iface,
                 "--start", _since, "--refresh-days", "5"],
                f"pull_{_iface}", timeout=1800)
        except Exception as e:
            # 缺口超过 90 天要手动全量补: python -m pipeline.pull_tushare moneyflow
            log(f"WARN {_iface} 日更失败(不中断, 数据将停在上一次成功日): {e}")
            stage(f"pull_{_iface}", ok=False, error=str(e)[:300])

    # ── 2.6 宏观数据日更 (SOX/商品指数/中债/美债/汇率) ──
    # 2026-08-13 上线: 旧 iFinD 系宏观 parquet 从来没进过日更(和资金流同一类根因),
    # SOX/USDCNH 等停在 7 月初, FB 特征集的 13 个宏观特征在最新日全 NaN。
    # 现改由 tushare(汇率/美债) + akshare(SOX/商品指数/中债) 供给,
    # 全部源与旧值 2024 年后逐日对账通过(见 pipeline/pull_macro.py 头注)。
    # 同样 try 兜住: 拉取失败只是宏观停在昨天(calc_commodity_features 对齐交易日时
    # ffill limit=5, 断供 5 个交易日后宏观特征才开始变 NaN, 有缓冲窗口), 不能拖垮管线。
    try:
        run([PY, "-u", "-m", "pipeline.pull_macro"], "pull_macro", timeout=900)
    except Exception as e:
        log(f"WARN 宏观日更失败(不中断, 宏观特征将沿用前值): {e}")
        stage("pull_macro", ok=False, error=str(e)[:300])

    # ── 3. 重建特征 -> 临时训练集 ──
    run([PY, "-u", "-m", "pipeline.feature_engine",
         "--no-incremental" if a.force else "--incremental",
         "--procs", str(a.feat_procs),
         "--watchlist", WATCHLIST, "--out", TMP_FILE],
        "build_features", timeout=14400)

    # ── 4. 校验后原子替换 ──
    problems, info = validate_new_train(tmp_path, train_path,
                                        require_advance=not a.force)
    status["new_train_info"] = info
    for w in info.get("warnings", []):
        log(f"警告: {w}")
    if problems:
        stage("validate", ok=False, problems=problems, info=info)
        raise RuntimeError("新训练集校验不通过:\n  - " + "\n  - ".join(problems))
    stage("validate", ok=True, info=info)
    log(f"校验通过: {info}")

    if train_path.exists():
        bak_dir = PROC / "backup"
        bak_dir.mkdir(exist_ok=True)
        bak = bak_dir / f"{train_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.parquet"
        shutil.copy2(train_path, bak)
        # 只留最近5份备份, 训练集单个几百MB
        olds = sorted(bak_dir.glob(f"{train_path.stem}_*.parquet"))
        for p in olds[:-5]:
            p.unlink()
        log(f"旧训练集已备份: {bak.name}")
    tmp_path.replace(train_path)
    stage("swap_train", ok=True, path=str(train_path))
    log(f"训练集已更新 -> {info['max_date']}")

    # ── 4.5 逐笔增广矩阵 (线上全部条线的输入, 2026-08-16 起) ──
    # 逐笔特征面板由 eez040 抽取后 rsync 到本机 data/processed/tick_micro/
    # (链路见 scripts/tick_daily_extract.py 头注), 这里在干净的 v24 之上加列,
    # 产出 training_data_pit_v24_tick1.parquet。
    # 2026-08-16 全线切 V24PUT(逐笔 lag1 + 剔 qfq 价格列)后这一步是硬失败:
    # tick1 建不出来/末日不对齐 => 当晚宁可没信号, 也不用陈旧矩阵出错信号。
    # --require-fresh 3: 供应商断供 3 个交易日内用 ffill 兜(回测证过 lag5 仍有
    # 正超额), 超限则拒绝更新 —— 停更被下面的末日对齐检查拦成硬失败。
    run([PY, "-u", "scripts/build_tick_augmented.py", "--lag", "1",
         "--require-fresh", "3"], "build_tick1", timeout=1800)
    tick1_path = PROC / "training_data_pit_v24_tick1.parquet"
    t1_max = pd.read_parquet(tick1_path, columns=["date"])["date"].max()
    if pd.Timestamp(t1_max) != pd.Timestamp(info["max_date"]):
        stage("build_tick1", ok=False,
              error=f"tick1 末日 {t1_max} != v24 末日 {info['max_date']}")
        raise RuntimeError(
            f"tick1 矩阵末日 {t1_max} 与 v24 末日 {info['max_date']} 不对齐 ——\n"
            "  多半是逐笔面板断供超过 --require-fresh 限度。去查 eez040 的\n"
            "  ~/logs/tickfeat_*.log 与供应商 123 盘更新; 或临时回滚 live_config\n"
            "  到 V24B(无逐笔)再手动重跑本脚本。")
    used_tk = [c for c in (live_used_features() or []) if c.startswith("tk_")]
    if used_tk:
        t1_cols = set(pq.read_schema(tick1_path).names)
        tk_missing = [c for c in used_tk if c not in t1_cols]
        have = [c for c in used_tk if c in t1_cols]
        t1_last = pd.read_parquet(tick1_path, columns=["date"] + have)
        t1_last = t1_last[t1_last["date"] == t1_max]
        tk_dead = [c for c in have if t1_last[c].isna().all()]
        if tk_missing or tk_dead:
            stage("build_tick1", ok=False,
                  error=f"在用逐笔列 缺{tk_missing[:4]} 末日全空{tk_dead[:4]}")
            raise RuntimeError(
                f"tick1 在用逐笔列异常: 缺列 {tk_missing} / 末日全NaN {tk_dead} ——\n"
                "  多半是面板断供或 build_tick_augmented 回归, 当晚宁可无信号。")
    stage("build_tick1", ok=True, max_date=str(t1_max), tk_used=len(used_tk))
    log(f"tick1 矩阵已对齐 -> {t1_max}, 在用逐笔列 {len(used_tk)} 个末日全有值")

    # ── 5. 逐条线出信号 ──
    # 四条线用同一份行情/训练集, 但各自一份状态与计划文件。
    # 故意不用 fail-fast: 某条线出错(例如指纹不匹配)不应连带拖死其他条线,
    # 否则一条线的配置问题会让当天所有人都没信号可看。
    sig_results = {}
    for pid in PROFILES:
        try:
            out = run([PY, "-u", "scripts/live_signal.py"] + signal_args(pid),
                      f"live_signal_{pid}", timeout=1800)
            line = next((l.strip() for l in out.splitlines()
                         if "信号日" in l and "->" in l), None)
            sig_results[pid] = {"ok": True, "signal_line": line}
            log(f"[{pid}] 信号完成")
        except Exception as e:
            sig_results[pid] = {"ok": False, "error": str(e)[:800]}
            log(f"[{pid}] 信号失败: {e}")
    status["signals"] = sig_results

    if not any(v["ok"] for v in sig_results.values()):
        raise RuntimeError("所有 profile 的信号都失败了")

    status["ok"] = True
    status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    status["total_seconds"] = round(time.time() - t0)
    save_status()
    log(f"全流程完成 {status['total_seconds']}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        status["ok"] = False
        status["error"] = str(e)[:2000]
        status["finished_at"] = datetime.now().isoformat(timespec="seconds")
        save_status()
        log(f"FAILED: {e}")
        sys.exit(1)
