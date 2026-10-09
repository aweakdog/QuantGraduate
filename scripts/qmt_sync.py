"""QMT 快照接收、对账与自动确认的纯逻辑 (041 侧)。

数据流 (阶段一, 2026-10-08):
  Windows 计划任务 (9:35 开盘检查 / 15:10 收盘) 读 miniQMT 只读快照
    -> HMAC 签名 POST /api/qmt/snapshot -> 本模块校验并存到 data/live/qmt/
  日更链之后 qmt_autoconfirm.py 读当天收盘后的最终快照, 把真实成交转成
  live_signal --confirm 的回报格式, 替 QMT 线完成「确认成交」。

原则:
  - 快照只是数据, 不碰 state; 写账仍只走 live_signal (唯一写入者)。
  - 任何对不上的情况 (线外代码、持仓不一致、快照不是收盘后) 都不自动确认,
    原样停在「待确认」并报警, 交给人判断 —— 宁可停, 不可错记。
  - 现金只报告不拦: 账户里有线外现金 (用户会转走), 估算手续费与真实佣金也有分毫差。
"""
import hashlib
import hmac
import json
import os
import time
from datetime import datetime
from pathlib import Path

QMT_LINES = ("qmt10w",)          # 由 QMT 真实成交记账的线
STOCK_BUY, STOCK_SELL = 23, 24   # xtconstant.STOCK_BUY / STOCK_SELL
CLOSE_TIME = "15:00:00"          # 晚于此时的快照才算当日最终成交
SIGN_WINDOW = 300                # 签名时间戳允许的偏差(秒)
MAX_BODY = 2_000_000
KEY_PATH = Path(os.environ.get("QUANT_QMT_KEY_FILE", str(Path.home() / ".config" / "quant" / "qmt_bridge.key")))


# ── 签名 ──────────────────────────────────────────────────────────
def sign(body: bytes, ts: str, key: bytes) -> str:
    return hmac.new(key, str(ts).encode() + b"\n" + body, hashlib.sha256).hexdigest()


def verify(body: bytes, ts: str, sig: str, key: bytes, now=None, window=SIGN_WINDOW):
    """返回 (是否通过, 原因)。时间戳必须在窗口内, 签名用常数时间比较。"""
    if not key:
        return False, "服务器未配置 QMT 桥密钥"
    try:
        t = int(ts)
    except (TypeError, ValueError):
        return False, "缺少或非法时间戳"
    now = time.time() if now is None else now
    if abs(now - t) > window:
        return False, "时间戳超出允许范围"
    if not sig or not hmac.compare_digest(sign(body, str(t), key), str(sig)):
        return False, "签名不符"
    return True, ""


def load_key(path=None):
    p = Path(path) if path else KEY_PATH
    try:
        k = p.read_text(encoding="utf-8").strip()
    except OSError:
        return b""
    return k.encode() if k else b""


def alert_once(live_dir, key, text, send=None):
    """同一 key(日期+类别)只推送一次, 结果记在 data/live/qmt/alerts_sent.json。返回是否发送成功。"""
    sent_p = Path(live_dir) / "qmt" / "alerts_sent.json"
    try:
        sent = json.loads(sent_p.read_text(encoding="utf-8")) if sent_p.exists() else {}
    except (OSError, ValueError):
        sent = {}
    if key in sent:
        return False
    try:
        if send is None:
            from notify_channels import get_channel
            send = get_channel().send
        ok = bool(send(text))
    except Exception as e:  # noqa: BLE001 —— 告警失败不能拖垮调用方
        print(f"[qmt] 告警发送失败: {e}")
        ok = False
    sent[key] = {"at": datetime.now().isoformat(timespec="seconds"), "ok": ok}
    sent_p.parent.mkdir(parents=True, exist_ok=True)
    sent_p.write_text(json.dumps(sent, ensure_ascii=False, indent=1), encoding="utf-8")
    return ok


# ── 快照 ──────────────────────────────────────────────────────────
def validate_snapshot(snap):
    """最小结构校验; 返回问题列表(空 = 通过)。"""
    if not isinstance(snap, dict):
        return ["快照必须是 JSON 对象"]
    probs = []
    try:
        datetime.fromisoformat(str(snap.get("probe_at")))
    except ValueError:
        probs.append("probe_at 缺失或格式不对")
    if snap.get("kind") not in ("morning", "close", "manual"):
        probs.append("kind 必须是 morning/close/manual")
    if snap.get("account_online"):
        for k in ("positions", "orders", "trades"):
            if not isinstance(snap.get(k), list):
                probs.append(f"{k} 必须是数组")
        if not isinstance(snap.get("asset"), dict):
            probs.append("asset 缺失")
    return probs


def store_snapshot(snap, live_dir):
    """存一份带时间戳的快照 + latest.json (原子替换)。返回文件路径。"""
    d = Path(live_dir) / "qmt"
    d.mkdir(parents=True, exist_ok=True)
    t = datetime.fromisoformat(str(snap["probe_at"]))
    path = d / f"snap_{t:%Y%m%d_%H%M%S}_{snap['kind']}.json"
    text = json.dumps(snap, ensure_ascii=False, indent=2, default=str)
    for p in (path, d / "latest.json"):
        tmp = p.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)
    return path


def final_snapshot_for(live_dir, date):
    """某交易日收盘后(>=15:00)最后一份、账户在线的快照; 没有返回 None。"""
    d = Path(live_dir) / "qmt"
    best = None
    for p in sorted(d.glob(f"snap_{str(date).replace('-', '')}_*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        t = datetime.fromisoformat(str(s.get("probe_at")))
        if s.get("account_online") and f"{t:%H:%M:%S}" >= CLOSE_TIME:
            best = s
    return best


# ── 对账 ──────────────────────────────────────────────────────────
def code6(c):
    return str(c)[:6]


def line_positions(state):
    out = {}
    for lot in (state or {}).get("lots") or []:
        out[code6(lot["code"])] = out.get(code6(lot["code"]), 0) + int(lot.get("shares") or 0)
    return {c: s for c, s in out.items() if s > 0}


def qmt_positions(snap):
    out = {}
    for p in (snap or {}).get("positions") or []:
        v = int(p.get("volume") or 0)
        if v > 0:
            out[code6(p["stock_code"])] = out.get(code6(p["stock_code"]), 0) + v
    return out


def reconcile(snap, state):
    """快照 vs 线账本: 持仓逐只比, 现金只报差额(线外现金)。"""
    q, ln = qmt_positions(snap), line_positions(state)
    diffs = [{"code": c, "qmt": q.get(c, 0), "line": ln.get(c, 0)}
             for c in sorted(set(q) | set(ln)) if q.get(c, 0) != ln.get(c, 0)]
    acct_cash = ((snap or {}).get("asset") or {}).get("cash")
    line_cash = (state or {}).get("cash")
    return {"probe_at": (snap or {}).get("probe_at"), "kind": (snap or {}).get("kind"),
            "account_online": bool((snap or {}).get("account_online")),
            "positions_match": not diffs, "diffs": diffs,
            "account_cash": acct_cash, "line_cash": None if line_cash is None else round(float(line_cash), 2),
            "outside_cash": (None if acct_cash is None or line_cash is None
                             else round(float(acct_cash) - float(line_cash), 2))}


# ── 阶段二: 下单单据 / 总开关 / 执行回报 ─────────────────────────
def load_switch(live_dir):
    """网页总开关(双钥匙之一)。默认关; 文件坏了也当关。"""
    p = Path(live_dir) / "qmt" / "trading_switch.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"enabled": False}
    return {"enabled": d.get("enabled") is True, "by": d.get("by"), "at": d.get("at")}


def save_switch(live_dir, enabled, by):
    p = Path(live_dir) / "qmt" / "trading_switch.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    d = {"enabled": bool(enabled), "by": str(by), "at": datetime.now().isoformat(timespec="seconds")}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    log = Path(live_dir) / "qmt" / "switch_log.jsonl"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return d


def build_ticket(pid, state, plan, today, exec_date_of, switch, tranche_n):
    """当天要执行的单据。返回 (ticket, problems); problems 非空 = 今天不得下单。

    exec_date_of(signal_date) -> 该信号的执行日(交易日历)。单据只在以下全部成立时有效:
      计划的信号日 = 挂单信号日, 其执行日 = today; 线不在「待确认」(上一笔已按真实成交入账);
      计划不是空仓却要买入之类的自相矛盾不存在。
    """
    probs = []
    pend = (state or {}).get("pending") or {}
    if (state or {}).get("awaiting_confirm"):
        probs.append(f"线还在待确认 {state['awaiting_confirm'].get('exec_date')}, 账本未更新, 不下单")
    if not plan:
        probs.append("找不到挂单对应的计划文件")
    sig = str((plan or {}).get("signal_date") or "")
    if plan and str(pend.get("signal_date") or "") != sig:
        probs.append(f"计划信号日 {sig} 与挂单 {pend.get('signal_date')} 不一致")
    exec_date = exec_date_of(sig) if sig else None
    if str(exec_date) != str(today):
        probs.append(f"计划执行日 {exec_date} 不是今天 {today}")
    if plan and plan.get("in_cash") and plan.get("buy"):
        probs.append("空仓计划里却有买入, 计划自相矛盾")
    sells = [{"code": code6(s["code"]), "shares": int(s["shares"]), "ref_close": s.get("ref_close")}
             for s in (plan or {}).get("sell") or []]
    buys = [{"code": code6(b["code"]), "shares": int(b["shares"]), "ref_close": b.get("ref_close"),
             "budget": b.get("budget"), "pred": b.get("pred")} for b in (plan or {}).get("buy") or []]
    held = line_positions(state)
    for s in sells:
        if s["shares"] > held.get(s["code"], 0):
            probs.append(f"计划卖出 {s['code']} {s['shares']} 股超过线持仓 {held.get(s['code'], 0)}")
    line_cash = float((state or {}).get("cash") or 0)
    equity = float((plan or {}).get("equity") or line_cash)
    ticket = {"profile": pid, "today": str(today), "signal_date": sig, "exec_date": str(exec_date),
              "trading_enabled": bool(switch.get("enabled")), "line_cash": round(line_cash, 2),
              "slot_cap": round(equity / max(1, tranche_n) * 1.3, 2),
              "lots": [{"code": c, "shares": s} for c, s in sorted(held.items())],
              "sell": sells, "buy": buys, "in_cash": bool((plan or {}).get("in_cash")),
              "plan_generated_at": (plan or {}).get("generated_at")}
    return ticket, probs


def store_exec_report(live_dir, report):
    d = Path(live_dir) / "qmt"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"exec_{str(report['today']).replace('-', '')}_{report.get('mode', 'shadow')}.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


# ── 自动确认 ──────────────────────────────────────────────────────
def _trade_date(t):
    v = t.get("traded_time")
    if isinstance(v, (int, float)) and v > 10**9:
        return datetime.fromtimestamp(v).date().isoformat()
    s = str(v or "")
    if len(s) >= 8 and s[:8].isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return None


def aggregate_trades(trades, on_date=None):
    """同代码同方向的多笔成交合并成一笔(股数相加, 价格取成交额加权)。返回 (fills, problems)。"""
    agg, probs = {}, []
    for t in trades or []:
        side = {STOCK_BUY: "buy", STOCK_SELL: "sell"}.get(t.get("order_type"))
        if side is None:
            probs.append(f"未知成交类型 order_type={t.get('order_type')} ({t.get('stock_code')})")
            continue
        vol = int(t.get("traded_volume") or 0)
        px = float(t.get("traded_price") or 0)
        if vol <= 0 or px <= 0:
            probs.append(f"成交股数或价格非正 ({t.get('stock_code')})")
            continue
        td = _trade_date(t)
        if on_date and td and td != str(on_date):
            probs.append(f"成交日期 {td} 不是执行日 {on_date} ({t.get('stock_code')})")
            continue
        amt = float(t.get("traded_amount") or vol * px)
        k = (code6(t.get("stock_code")), side)
        a = agg.setdefault(k, [0, 0.0])
        a[0] += vol
        a[1] += amt
    fills = [{"code": c, "action": side, "shares": v, "price": round(amt / v, 4)}
             for (c, side), (v, amt) in sorted(agg.items(), key=lambda kv: (kv[0][1] != "sell", kv[0][0]))]
    return fills, probs


def build_autoconfirm(snap, state, plan, exec_date):
    """把执行日收盘后的 QMT 快照转成确认回报。返回 (fills, problems, notes)。

    problems 非空 = 不得自动确认(停下等人工)。规则:
      1. 快照必须是执行日 15:00 之后、账户在线
      2. 卖出只能是线账本里持有的代码, 且不超过持有股数
      3. 买入只能是该计划买入单或候补里的代码
      4. 按回报改完后的线持仓必须与 QMT 持仓逐只完全一致 (含无成交的情形)
    """
    notes, probs = [], []
    if not snap:
        return [], [f"没有 {exec_date} 收盘后的 QMT 快照"], notes
    t = datetime.fromisoformat(str(snap.get("probe_at")))
    if t.date().isoformat() != str(exec_date) or f"{t:%H:%M:%S}" < CLOSE_TIME:
        probs.append(f"快照时间 {snap.get('probe_at')} 不是 {exec_date} 收盘后")
    if not snap.get("account_online"):
        probs.append("快照里资金账号不在线")
    fills, p = aggregate_trades(snap.get("trades"), exec_date)
    probs += p
    held = line_positions(state)
    plan_buys = {code6(b["code"]) for b in (plan or {}).get("buy") or []}
    alts = {code6(a["code"]) for a in (plan or {}).get("alternates") or []}
    for f in fills:
        if f["action"] == "sell":
            if f["code"] not in held:
                probs.append(f"卖出了线外代码 {f['code']}")
            elif f["shares"] > held[f["code"]]:
                probs.append(f"卖出 {f['code']} {f['shares']} 股, 线账本只有 {held[f['code']]} 股")
        elif f["code"] in plan_buys:
            pass
        elif f["code"] in alts:
            notes.append(f"买入 {f['code']} 来自候补名单")
        else:
            probs.append(f"买入了计划外代码 {f['code']}")
    after = dict(held)
    for f in fills:
        after[f["code"]] = after.get(f["code"], 0) + (f["shares"] if f["action"] == "buy" else -f["shares"])
    after = {c: s for c, s in after.items() if s > 0}
    q = qmt_positions(snap)
    if after != q:
        diff = {c: (q.get(c, 0), after.get(c, 0)) for c in sorted(set(q) | set(after)) if q.get(c, 0) != after.get(c, 0)}
        probs.append(f"按回报改完后的持仓与 QMT 不一致 (代码: QMT 股数, 线账本股数): {diff}")
    return fills, probs, notes
