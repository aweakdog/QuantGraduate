"""2019 扩容过夜编排 (服务器上 nohup 跑, 不依赖 ssh 存活)

顺序 (每步写状态, 重跑自动跳过已完成步骤):
  0. 等资金流全池重灌收尾 (pull_fundflow_sina --refill-all), 没合并成功则续跑最多 2 次
  1. K线扩到 2019: 先 519 池 (特征用), 再全市场 (PIT排名+广度用)
  2. 幸存者偏差处置: 从 pit_metadata 找 2019 前上市的候选股, 给缺K线的
     (多为已退市) 补拉; 拉不到的计数写报告 —— 缺口超阈值就停下等人裁决
  3. PIT 成分重建 2019-07 起 -> universe_pit_2019.parquet + watchlist_pit_2019.json
  4. 特征全量重建 -> training_data_pit_2019.parquet (全程不碰 v24 与线上任何文件)
  5. 汇总报告 -> data/processed/expand_2019_report.json

明早人工步骤: 审报告 -> 跑 wf 对比 (2019矩阵 vs v24) -> 决定是否切换。
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_config import FEATURES_FROM  # noqa: E402

PY = sys.executable
DATA = ROOT / "data"
KLINE = DATA / "raw" / "kline"
STATUS = DATA / "processed" / "expand_2019_status.json"
REPORT = DATA / "processed" / "expand_2019_report.json"
FF_CONS = DATA / "raw" / "fund_flow_full" / "fundflow_history.parquet"

UNI_OUT = DATA / "universe" / "universe_pit_2019.parquet"
WL_OUT = DATA / "universe" / "watchlist_pit_2019.json"
TRAIN_OUT = "training_data_pit_2019.parquet"
FEATURES_DIR = "features_2019"
FEATURE_CUTOFF = "2019-01-01"
PIT_START = "2019-07-01"
PIT_END = "2026-07-27"
EXCLUDED_PREFIXES = ("200", "900")
WF_TEST_START = "2022-09-01"
WF_TEST_END = PIT_END

state = json.loads(STATUS.read_text()) if STATUS.exists() else {"stages": {}}


def log(msg):
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def mark(stage, **kv):
    state["stages"][stage] = {"at": datetime.now().isoformat(timespec="seconds"), **kv}
    STATUS.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str))


def done(stage):
    return state["stages"].get(stage, {}).get("ok", False)


def usable_kline_codes(codes):
    usable = set()
    for code in codes:
        path = KLINE / f"{code}.parquet"
        if not path.exists():
            continue
        try:
            if len(pd.read_parquet(path, columns=["date"])):
                usable.add(code)
        except Exception:
            continue
    return usable


def run(cmd, name, timeout):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout,
                       capture_output=True, text=True)
    tail = ((r.stdout or "") + (r.stderr or ""))[-2000:]
    if r.returncode != 0:
        mark(name, ok=False, log=tail)
        raise RuntimeError(f"{name} rc={r.returncode}\n{tail}")
    return tail


# ── 0. 资金流重灌收尾 ──────────────────────────────────────────────
def stage_fundflow():
    if done("fundflow"):
        return
    relaunch = 0
    while True:
        alive = subprocess.run(["pgrep", "-f", "pull_fundflow_sina"],
                               capture_output=True).returncode == 0
        if alive:
            time.sleep(120)
            continue
        logtxt = Path("/tmp/sina_ff.log").read_text(errors="ignore")
        if "重灌完成" in logtxt:
            ff = pd.read_parquet(FF_CONS, columns=["date", "code"])
            mark("fundflow", ok=True, rows=len(ff),
                 codes=int(ff["code"].nunique()),
                 span=f"{ff['date'].min():%F} ~ {ff['date'].max():%F}")
            log(f"资金流重灌完成: {len(ff):,} 行 / {ff['code'].nunique()} 只")
            return
        if relaunch >= 2:
            mark("fundflow", ok=False, error="重灌进程结束但未见合并成功, 续跑 2 次仍失败")
            raise RuntimeError("资金流重灌失败, 停止过夜队列")
        relaunch += 1
        log(f"重灌进程死了但没合并成功, 续跑第 {relaunch} 次 (断点续传)")
        with open("/tmp/sina_ff.log", "a") as log_file:
            subprocess.Popen(
                [PY, "-m", "pipeline.pull_fundflow_sina", "--refill-all",
                 "--since", "2019-01-01", "--sleep", "1.2", "--skip-unit-check"],
                cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT)
        time.sleep(300)


# ── 1. K线扩容 ────────────────────────────────────────────────────
def stage_kline():
    if not done("kline_universe"):
        out = run([PY, "-u", "scripts/update_kline_akshare.py",
                   "--scope", "universe", "--watchlist", "watchlist_pit.json",
                   "--start", "20190101", "--procs", "6", "--timeout", "30"],
                  "kline_universe", timeout=7200)
        mark("kline_universe", ok=True, log=out[-400:])
    if not done("kline_all"):
        out = run([PY, "-u", "scripts/update_kline_akshare.py",
                   "--scope", "all", "--start", "20190101",
                   "--procs", "8", "--timeout", "30"],
                  "kline_all", timeout=14400)
        mark("kline_all", ok=True, log=out[-400:])


# ── 2. 幸存者偏差处置 ─────────────────────────────────────────────
def stage_survivorship():
    if done("survivorship"):
        return
    meta = pd.read_parquet(DATA / "universe" / "pit_metadata.parquet")
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    meta["list_date"] = pd.to_datetime(meta["list_date"], errors="coerce")
    meta["delist_date"] = pd.to_datetime(meta.get("delist_date"), errors="coerce")
    eligible = ~meta["code"].str.startswith(EXCLUDED_PREFIXES)
    in_window = meta["delist_date"].isna() | (meta["delist_date"] > pd.Timestamp(PIT_START))
    cand = meta[eligible & in_window & (meta["list_date"] < pd.Timestamp(PIT_END))]
    have = usable_kline_codes(cand["code"])
    target = cand[~cand["code"].isin(have)]
    log(f"A股候选 {len(cand)} 只, 回测窗相关且缺K线 {len(target)} 只")

    pulled = failed = 0
    if len(target):
        from update_kline_akshare import process as kl_process
        today = pd.Timestamp.now().normalize()
        for _, row in target.iterrows():
            end_date = today if pd.isna(row["delist_date"]) else min(today, row["delist_date"])
            _, st, _, _ = kl_process(
                row["code"], end_date.strftime("%Y%m%d"), False, 2, 30, "20190101"
            )
            if st.startswith("ok"):
                pulled += 1
            else:
                failed += 1
            time.sleep(0.3)
    # 拉完仍缺的、且退市时间在训练窗内的 = 真正的幸存者空洞
    have = usable_kline_codes(cand["code"])
    hole = cand[~cand["code"].isin(have)]
    info = {"candidates": len(cand), "missing_before": len(target),
            "pulled": pulled, "failed": failed, "survivor_hole": len(hole),
            "hole_codes_sample": hole["code"].head(20).tolist()}
    log(f"幸存者处置: 补拉成功 {pulled}, 仍缺 {len(hole)} 只在窗内存活/退市的候选")
    if len(hole) > 200:
        mark("survivorship", ok=False, **info)
        raise RuntimeError(f"幸存者空洞过大 ({len(hole)} 只), 停下等人工裁决")
    mark("survivorship", ok=True, **info)


# ── 3. PIT 成分 + watchlist ───────────────────────────────────────
def stage_pit():
    if done("pit"):
        return
    run([PY, "-u", "scripts/build_pit_universe.py",
         "--top-n", "300", "--freq", "semiannual", "--rank-by", "mcap",
         "--start", PIT_START, "--end", PIT_END,
         "--exclude-prefixes", ",".join(EXCLUDED_PREFIXES),
         "--out", str(UNI_OUT), "--jobs", "12"],
        "pit_build", timeout=3600)
    u = pd.read_parquet(UNI_OUT)
    meta = pd.read_parquet(DATA / "universe" / "pit_metadata.parquet")
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    names = dict(zip(meta["code"], meta.get("name", meta["code"]), strict=True))

    def suffix(c):
        return "SH" if c.startswith("6") else (
            "BJ" if c[:2] in ("43", "83", "87", "88", "92") else "SZ")

    codes = sorted(u["code"].astype(str).str.zfill(6).unique())
    wl = {"watchlist": [{"code": f"{c}.{suffix(c)}", "name": names.get(c, c)}
                        for c in codes]}
    WL_OUT.write_text(json.dumps(wl, ensure_ascii=False, indent=1))
    mark("pit", ok=True, effective_dates=int(u["effective_date"].nunique()),
         union_codes=len(codes),
         span=f"{u['effective_date'].min()} ~ {u['effective_date'].max()}")
    log(f"PIT 成分: {u['effective_date'].nunique()} 个生效日, 并集 {len(codes)} 只")


# ── 4. 特征矩阵 ───────────────────────────────────────────────────
def stage_features():
    if done("features"):
        return
    run([PY, "-u", "-m", "pipeline.feature_engine", "--no-incremental",
         "--procs", "8", "--watchlist", WL_OUT.name, "--out", TRAIN_OUT,
         "--cutoff", FEATURE_CUTOFF, "--features-dir", FEATURES_DIR],
        "features", timeout=21600)
    t = pd.read_parquet(DATA / "processed" / TRAIN_OUT, columns=["date", "code"])
    first = pd.Timestamp(t["date"].min())
    if first >= pd.Timestamp("2020-01-01"):
        mark("features", ok=False, rows=len(t), codes=int(t["code"].nunique()),
             span=f"{t['date'].min():%F} ~ {t['date'].max():%F}",
             error="矩阵没有 2019 年训练历史")
        raise RuntimeError(f"特征矩阵最早日期 {first:%F}, 无法执行 2019训/2022测")
    mark("features", ok=True, rows=len(t), codes=int(t["code"].nunique()),
         span=f"{t['date'].min():%F} ~ {t['date'].max():%F}",
         cutoff=FEATURE_CUTOFF, features_dir=FEATURES_DIR)
    log(f"矩阵: {len(t):,} 行, {t['code'].nunique()} 只, "
        f"{t['date'].min():%F} ~ {t['date'].max():%F}")


# ── 5. 汇总 ──────────────────────────────────────────────────────
def stage_report():
    t = pd.read_parquet(DATA / "processed" / TRAIN_OUT)
    feature_spec = DATA / "processed" / FEATURES_FROM
    if not feature_spec.exists():
        raise RuntimeError(f"锁定特征文件不存在: {feature_spec}")
    locked_features = json.loads(feature_spec.read_text())["selected_features"]
    generated = {"overnight_ret", "intraday_ret"}
    missing_features = [
        f for f in locked_features
        if f not in t.columns
        and not f.startswith(("mkt_", "ovn_"))
        and f not in generated
    ]
    if missing_features:
        raise RuntimeError(f"2019 矩阵缺少 {len(missing_features)} 个锁定特征: {missing_features}")
    key_feats = [c for c in ("mf_net_amt_z21", "mtss_z", "turn_z21", "mom_20d")
                 if c in t.columns]
    by_year = {}
    for y, g in t.groupby(t["date"].dt.year):
        by_year[int(y)] = {"rows": len(g),
                           **{f"nan_{c}": round(float(g[c].isna().mean()), 3)
                              for c in key_feats}}
    wf_args = [
        ".venv/bin/python", "scripts/wf_v35_breadth_alpha.py",
        "--train-file", TRAIN_OUT,
        "--pit-universe", UNI_OUT.name,
        "--test-start", WF_TEST_START,
        "--test-end", WF_TEST_END,
        "--features-from", FEATURES_FROM,
        "--label", "5d", "--hold-days", "5",
        "--portfolio-mode", "periodic", "--exec-mode", "t1close",
        "--slippage", "0.002", "--regime-filter", "off",
    ]
    rep = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "stages": state["stages"],
        "by_year": by_year,
        "experiment": {
            "raw_feature_start": FEATURE_CUTOFF,
            "pit_train_start": PIT_START,
            "test_start": WF_TEST_START,
            "test_end": WF_TEST_END,
            "features_from": FEATURES_FROM,
            "features_locked": True,
            "feature_count": len(locked_features),
            "wf_base_command": " ".join(wf_args),
        },
    }
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"报告 -> {REPORT}")


def main():
    t0 = time.time()
    for fn in (stage_fundflow, stage_kline, stage_survivorship,
               stage_pit, stage_features, stage_report):
        fn()
    log(f"过夜队列全部完成, 用时 {(time.time() - t0) / 3600:.1f} 小时")


if __name__ == "__main__":
    main()
