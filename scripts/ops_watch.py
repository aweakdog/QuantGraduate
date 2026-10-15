"""数据日更与系统健康告警 (2026-10-15 用户要求: 日更失败、数据源 token/权限失效等也要发邮件)。

两个入口:
  1. daily_rebuild 每次结束(成功/失败/跳过)时调 report_run(status): 把这次运行摘要追加到
     data/live/pipeline_runs.jsonl, 并当场判 —— 硬失败 / 软失败环节(分 token·权限、超时·网络、其他) /
     有线出信号失败 / 逐笔面板陈旧。pipeline_status.json 会被 19:30、21:30 的补跑覆盖,
     摘要历史才留得住 17:30 主跑的细节。
  2. systemd quant-ops-watch.timer 周一至五 21:15 / 23:45 跑 --deadline: 交易日到点仍没有一次
     成功且数据推进到当天的日更 -> 告警 (日更根本没跑、卡死、或数据源当天没更新)。
  3. systemd quant-tick-watch.timer 周一至五 13:05 / 17:05 跑 --tick: 040 供的上一交易日逐笔/T1A
     面板还没到 041 -> 告警 (赶在 17:30 日更之前, 留出补救时间)。
告警走 qmt_sync.alert_once: 企业微信给详情(长串疑似密钥一律打码), 邮件只给精简说明; 同日同类只发一次。
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
LIVE = ROOT / "data" / "live"
RUNS = "pipeline_runs.jsonl"

TOKEN_PAT = re.compile(r"token|权限|积分|unauthori|forbidden|\b401\b|\b403\b|过期|expired|认证|apikey|api key", re.I)
NET_PAT = re.compile(r"timed out|timeout|超时|connection|网络|reset by peer|temporarily|name resolution|ssl", re.I)
SECRET_PAT = re.compile(r"[A-Za-z0-9_\-]{32,}(?:\.[A-Za-z0-9_\-]{6,})*")
STALE_PAT = re.compile(r"陈旧 (\d+) 个交易日")
STAGE_CN = {"kline_full_market": "全市场K线(akshare)", "kline_universe": "股票池K线(akshare)", "kline": "K线",
            "trading_calendar": "交易日历", "pull_moneyflow": "资金流(Tushare)", "pull_margin_detail": "两融(Tushare)",
            "pull_macro": "宏观数据(Tushare/akshare)", "build_features": "特征构建", "validate": "训练集校验",
            "swap_train": "训练集替换", "build_cgo": "CGO增广", "build_t1": "T1增广", "build_tick1": "逐笔增广(040 供数)"}


def stage_cn(name):
    if name.startswith("live_signal_"):
        return f"{name[len('live_signal_'):]} 出信号"
    return STAGE_CN.get(name, name)


def mask(text):
    return SECRET_PAT.sub("***", str(text or ""))


def classify(err):
    e = str(err or "")
    if TOKEN_PAT.search(e):
        return "token"
    if NET_PAT.search(e):
        return "network"
    return "other"


def summarize_run(status):
    stages = {}
    for name, v in (status.get("stages") or {}).items():
        if not isinstance(v, dict):
            continue
        err = v.get("error") or ("; ".join(v.get("problems") or []) if v.get("problems") else "")
        if v.get("ok") is False and not err:
            err = (v.get("log") or "")[-400:]
        stages[name] = {"ok": v.get("ok"), "error": mask(err)[:400] if err else None}
        m = STALE_PAT.findall(v.get("log") or "")
        if name == "build_tick1" and m:
            stages[name]["stale_days"] = int(m[-1])
    return {"started_at": status.get("started_at"), "finished_at": status.get("finished_at"),
            "ok": bool(status.get("ok")), "skipped_reason": status.get("skipped_reason"),
            "error": mask(status.get("error"))[:600] if status.get("error") else None,
            "kline_max_date": status.get("kline_max_date"), "stages": stages,
            "signals": {pid: bool(v.get("ok")) for pid, v in (status.get("signals") or {}).items()},
            "signal_errors": {pid: mask(v.get("error"))[:300] for pid, v in (status.get("signals") or {}).items()
                              if not v.get("ok")},
            "warnings": [mask(w)[:300] for w in ((status.get("new_train_info") or {}).get("warnings") or [])]}


def evaluate_run(rec):
    """一次运行该发哪些告警。返回 [(类别, 企业微信详情, 邮件精简)]。"""
    day = str(rec.get("started_at") or datetime.now().isoformat())[:10]
    md, hhmm = day[5:], str(rec.get("started_at") or "")[11:16]
    retry = " 19:30 / 21:30 会自动重试。" if hhmm and hhmm < "19:00" else (" 21:30 会自动重试。" if hhmm and hhmm < "21:00" else "")
    out = []
    if not rec.get("ok"):
        err = (rec.get("error") or "未知原因").splitlines()[0][:200]
        failed = [stage_cn(n) for n, s in rec["stages"].items() if s.get("ok") is False]
        where = f"({'、'.join(failed[:3])})" if failed else ""
        kind = classify(rec.get("error"))
        hint = " 看起来是数据源 token/权限问题。" if kind == "token" else (" 看起来是网络/超时。" if kind == "network" else "")
        out.append(("fail", f"【数据】{day} {hhmm} 日更失败{where}: {err}{hint} 今晚可能没有新信号。{retry}",
                    f"【数据】{md} 日更失败{where}{hint} 今晚可能没有新信号。{retry}"))
    soft = {n: s for n, s in rec["stages"].items() if s.get("ok") is False and not n.startswith("live_signal_")}
    if rec.get("ok") and soft:
        groups = {}
        for n, s in soft.items():
            groups.setdefault(classify(s.get("error")), []).append(n)
        for kind, names in groups.items():
            cn = "、".join(stage_cn(n) for n in names)
            detail = "; ".join(f"{stage_cn(n)}: {(soft[n].get('error') or '')[:160]}" for n in names)
            what = {"token": "数据源 token 失效或接口权限不足", "network": "数据源超时/网络错误", "other": "数据环节出错"}[kind]
            out.append((f"soft_{kind}", f"【数据】{day} {what}: {detail}。日更继续跑完, 但这些数据停在上一次成功日。",
                        f"【数据】{md} {what}: {cn}。日更已跑完, 但这些数据停在上一次成功日, 需要处理。"))
    bad = sorted(pid for pid, ok in (rec.get("signals") or {}).items() if not ok)
    if bad:
        detail = "; ".join(f"{p}: {(rec['signal_errors'].get(p) or '')[:120]}" for p in bad)
        out.append(("signal", f"【数据】{day} 有 {len(bad)} 条线出信号失败: {detail}",
                    f"【数据】{md} 有 {len(bad)} 条线出信号失败 ({'、'.join(bad)}), 这些线今晚没有新计划。"))
    # 陈旧度正常就是 1 (lag1: 重建日当天的包次日早上才到); >=2 才是 040 供数/供应商断了
    stale = (rec["stages"].get("build_tick1") or {}).get("stale_days") or 0
    if stale >= 2:
        out.append(("tick_stale", f"【数据】{day} 逐笔面板陈旧 {stale} 个交易日(正常是 1), 已断供 {stale - 1} 天; "
                                  "3 天内用前值兜底, 陈旧超过 3 天当晚无信号。查 040 ~/logs/tickfeat_*.log 与供应商 123 云盘。",
                    f"【数据】{md} 040 逐笔数据已断供 {stale - 1} 个交易日, 陈旧超过 3 天当晚将没有信号。"))
    return out


# ── 040 逐笔供数到货检查 ──────────────────────────────────────────
# 040 cron 7:40 / 12:40 / 16:40 跑 tick_daily_extract: 拉供应商 123 云盘 D-1 的包 -> 抽取
# tick_micro 与 t1a_daily -> rsync 到 041。正常 8:05 前就到; 13:05 没到 = 早班失败(还有 16:40 一班),
# 17:05 没到 = 今晚 17:30 日更只能用前值兜底。在 041 上看「到没到」, 不依赖能不能登 040。
PANELS = (("tick_micro", "逐笔(tk_*)"), ("t1a_daily", "订单结构(T1A)"))


def _prev_trading_day(day, days):
    prior = [d for d in (days or []) if d < day]
    return max(prior) if prior else None


def tick_arrival_check(now=None, proc_dir=None, live_dir=LIVE, days=None, send=None):
    """交易日应已到达上一交易日的逐笔/T1A 面板; 缺就告警。返回缺失的面板名列表。"""
    now = now or datetime.now()
    today = now.date()
    proc_dir = Path(proc_dir) if proc_dir else ROOT / "data" / "processed"
    if days is None:
        import trading_calendar
        days, _ = trading_calendar.load()
    if not days or today not in set(days):
        return []
    prev = _prev_trading_day(today, days)
    if prev is None:
        return []
    stamp = prev.strftime("%Y%m%d")
    missing = [cn for d, cn in PANELS if not (proc_dir / d / f"{stamp}.parquet").exists()]
    if not missing:
        return []
    # 连续缺了几天: 从上一交易日往回数, 到最近一个两个面板都在的交易日为止
    gap, d = 0, prev
    while d is not None and gap < 10 and any(not (proc_dir / p / f"{d:%Y%m%d}.parquet").exists() for p, _ in PANELS):
        gap += 1
        d = _prev_trading_day(d, days)
    final = now.strftime("%H:%M") >= "17:00"
    tail = ("今晚 17:30 日更会用前值兜底; 陈旧超过 3 天当晚将没有信号。" if final else "16:40 还有一班会重试。")
    what = "/".join(missing)
    day = today.isoformat()
    _alert(live_dir, f"{day}:ops:{'tick_missing_final' if final else 'tick_missing'}",
           f"【数据】{day} {now:%H:%M} 040 的{what}面板还没到 041 ({stamp} 的包, 已连续缺 {gap} 个交易日)。{tail}"
           f"查 040 ~/logs/tickfeat_{now:%Y%m}.log、~/logs/sync123_{now:%Y%m}.log 与供应商 123 云盘。",
           f"【数据】{day[5:]} {now:%H:%M} 040 逐笔数据还没到 ({prev:%m-%d} 的包, 已连续缺 {gap} 个交易日)。{tail}", send)
    return missing


def _alert(live_dir, key, text, email_text, send=None):
    from qmt_sync import alert_once
    return alert_once(live_dir, key, text, send=send, email_text=email_text)


def report_run(status, live_dir=LIVE, send=None):
    rec = summarize_run(status)
    p = Path(live_dir) / RUNS
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    day = str(rec.get("started_at") or "")[:10]
    sent = []
    for kind, text, email_text in evaluate_run(rec):
        _alert(live_dir, f"{day}:ops:{kind}", text, email_text, send)
        sent.append(kind)
    return rec, sent


def runs_on(live_dir, day):
    p = Path(live_dir) / RUNS
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if str(r.get("started_at") or "").startswith(day):
            out.append(r)
    return out


def deadline_check(now=None, live_dir=LIVE, is_trading=None, send=None):
    """交易日到点仍没有「成功且 K 线推进到当天」的日更 -> 告警。返回告警类别或 None。"""
    now = now or datetime.now()
    day = now.date().isoformat()
    if is_trading is None:
        import trading_calendar
        is_trading = trading_calendar.is_trading_day(now.date())
        if is_trading is None:
            is_trading = now.weekday() < 5
    if not is_trading:
        return None
    runs = runs_on(live_dir, day)
    if any(r.get("ok") and not r.get("skipped_reason") and str(r.get("kline_max_date")) == day for r in runs):
        return None
    final = now.strftime("%H:%M") >= "23:00"
    last = runs[-1] if runs else None
    if not runs:
        why = "今天还没有任何一次日更运行记录 (定时器没触发或服务挂了)"
    elif last.get("skipped_reason"):
        why = f"K 线没有更新到今天 ({last.get('skipped_reason')})"
    else:
        why = f"最近一次运行 {str(last.get('started_at'))[11:16]} {'失败' if not last.get('ok') else '未推进到今天'}"
    kind = "deadline_final" if final else "deadline"
    tail = "今天彻底没有新数据和新信号, 明早需要人工处理。" if final else "21:30 还会补跑一次。"
    _alert(live_dir, f"{day}:ops:{kind}", f"【数据】{day} {now:%H:%M} 日更仍未完成: {mask(why)}。{tail}",
           f"【数据】{day[5:]} {now:%H:%M} 日更仍未完成。{tail}", send)
    return kind


WATCHED_UNITS = ("quant-web.service", "quant-daily.timer", "quant-qmt-confirm.timer")


def service_check(now=None, live_dir=LIVE, active_of=None, send=None):
    """网页服务和关键定时器必须在运行(网页挂了, 这个独立任务照样能发)。返回挂掉的单元列表。"""
    import subprocess
    now = now or datetime.now()

    def _active(unit):
        try:
            r = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return True             # 查不了就不误报
        return r.stdout.strip() in ("active", "activating", "reloading")

    down = [u for u in WATCHED_UNITS if not (active_of or _active)(u)]
    if down:
        day = now.date().isoformat()
        _alert(live_dir, f"{day}:ops:service", f"【系统】{day} {now:%H:%M} 这些服务没在运行: {', '.join(down)}。",
               f"【系统】{day[5:]} {now:%H:%M} 有 {len(down)} 个服务没在运行 ({', '.join(down)})。", send)
    return down


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deadline", action="store_true", help="交易日截止检查 (systemd 21:15 / 23:45)")
    ap.add_argument("--tick", action="store_true", help="040 逐笔供数到货检查 (systemd 13:05 / 17:05)")
    args = ap.parse_args(argv)
    if args.deadline:
        print(f"[{datetime.now():%F %T}] deadline: {deadline_check()} services_down: {service_check()}")
    if args.tick:
        print(f"[{datetime.now():%F %T}] tick panels missing: {tick_arrival_check()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
