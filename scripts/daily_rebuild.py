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
from live_config import PROFILES, signal_args  # noqa: E402

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


def validate_new_train(new_path, old_path, require_advance=True):
    """新训练集必须: 日期前进 + 列集不缺 + 行数不异常缩水 + 最新日特征不全空

    require_advance=False 用于 --force 重建(如只为应用代码修复),
    此时日期不前进是预期的, 不应当成错误。
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
    if last.empty:
        problems.append("最新日无数据行")
    else:
        all_nan = int(last.isna().all(axis=0).sum())
        if all_nan > len(new.columns) * 0.3:
            problems.append(f"最新日全NaN列过多: {all_nan}/{len(new.columns)}")

    info = {"rows": len(new), "codes": int(new["code"].nunique()),
            "cols": len(new.columns), "max_date": str(pd.Timestamp(n_max).date()),
            "last_day_rows": len(last)}
    return problems, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-market", action="store_true",
                    help="同时全量更新市场K线 (广度/择时用, 约25分钟)")
    ap.add_argument("--skip-kline", action="store_true", help="跳过K线更新")
    ap.add_argument("--force", action="store_true",
                    help="即使K线日期没前进也强制重建")
    ap.add_argument("--procs", type=int, default=10, help="K线并发进程数")
    ap.add_argument("--feat-procs", type=int, default=24,
                    help="特征构建并行进程数 (pandas 有 GIL 瓶颈, 必须用进程不能用线程)")
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
                 "--scope", "all", "--procs", str(a.procs), "--timeout", "25"],
                "kline_full_market", timeout=7200)
        else:
            run([PY, "-u", "scripts/update_kline_akshare.py",
                 "--scope", "universe", "--watchlist", WATCHLIST,
                 "--procs", str(a.procs), "--timeout", "25"],
                "kline_universe", timeout=3600)
    else:
        stage("kline", ok=True, skipped=True)

    k_max = kline_max_date()
    status["kline_max_date"] = str(pd.Timestamp(k_max).date()) if k_max is not None else None
    log(f"K线最新交易日: {status['kline_max_date']}")

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
