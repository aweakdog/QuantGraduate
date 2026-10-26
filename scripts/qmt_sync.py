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


SITE_URL = "http://eez041.ece.ust.hk:8737"
_KIND_TEXT = {"morning": "开盘检查", "close": "收盘检查", "manual": "手动检查"}
# 读屏快照 (10-26 起) 离线时带 error_kind, 邮件按类别说人话 (不含任何交易细节)
OFFLINE_REASON = {
    "layout": "读屏失败, QMT 界面可能变了 (客户端升级或新弹窗)。今天的对账和自动确认暂停, 需要人工看一下。",
    "read": "读屏结果前后不一致或数字对不上, 已按离线处理, 今天不会用它自动记账。",
    "window": "QMT 窗口状态不对 (分辨率变了/被挡住/没能最大化), 读屏暂停。",
    "login": "QMT 未登录或交易不在线, 自动登录也没成功, 需要人工登录。",
    "exception": "读屏程序出错 (可能是 DeepSeek 接口或网络问题), 今天的对账暂停。",
    "other": "读屏没通过检查, 已按离线处理。",
}


def email_summary(key, text=""):
    """邮件版告警: 只说发生了什么、去网页看, 不含股票代码/股数/价格/金额/线名 (10-15 用户要求)。

    邮件会经过系里的中继(有投递日志), 退信也会落到 EEZ041 地址, 所以细节只留在需登录的网页和企业微信群。"""
    parts = str(key).split(":")
    day, kind = (parts + ["", ""])[:2]
    md = day[5:] if len(day) >= 10 else day
    if kind in _KIND_TEXT and parts[2:3] == ["offline"]:
        why = OFFLINE_REASON.get(parts[3] if len(parts) > 3 else "", "资金账号离线, 请检查 QMT 是否已登录。")
        line = f"【QMT】{md} {_KIND_TEXT[kind]}: {why}"
    elif kind == "close" and parts[2:3] == ["mismatch"]:
        line = f"【QMT】{md} 收盘对账: 持仓与券商不一致, 自动确认会停下等人工核对。"
    elif kind == "autoconfirm_ok":
        line = f"【QMT】{md} 已按券商成交自动确认。"
    elif kind == "autoconfirm_blocked":
        line = f"【QMT】{md} 自动确认被拦下, 需要人工核对后在网页上确认。"
    elif kind == "autoconfirm_failed":
        line = f"【QMT】{md} 自动确认执行失败, 需要人工处理。"
    elif kind == "exec":
        mode = {"shadow": "影子演算", "live": "自动下单", "skipped": "未下单"}.get(parts[2] if len(parts) > 2 else "", "执行")
        line = f"【QMT】{md} {mode}有新结果或问题。"
    else:
        line = "【QMT】有一条新告警。"
    return f"{line}\n详情请登录网页查看: {SITE_URL}"


def _default_senders():
    """主推送通道 + (配置了 qmt_email_to 时) 邮件。返回 [(名字, send)]。"""
    from notify_channels import EmailChannel, get_channel, load_config
    cfg = load_config()
    out = [("main", get_channel(cfg=cfg).send)]
    if cfg.get("qmt_email_to"):
        out.append(("email", EmailChannel(cfg["qmt_email_to"]).send))
    return out


def alert_once(live_dir, key, text, send=None, email_text=None):
    """同一 key(日期+类别)只推送一次, 结果记在 data/live/qmt/alerts_sent.json。返回是否至少一个通道成功。

    send=None 时发到主通道和邮件(若配置), 每个通道单独记成败; 一个通道失败不影响另一个。
    邮件内容: 调用方给了 email_text 就用它(数据告警), 否则按 key 生成精简版(QMT 告警, 不含交易细节)。"""
    sent_p = Path(live_dir) / "qmt" / "alerts_sent.json"
    try:
        sent = json.loads(sent_p.read_text(encoding="utf-8")) if sent_p.exists() else {}
    except (OSError, ValueError):
        sent = {}
    if key in sent:
        return False
    try:
        senders = [("main", send)] if send is not None else _default_senders()
    except Exception as e:  # noqa: BLE001 —— 告警失败不能拖垮调用方
        print(f"[qmt] 告警通道配置有误: {e}")
        senders = []
    results = {}
    for name, fn in senders:
        try:
            if name == "email":
                payload = f"{email_text}\n详情请登录网页查看: {SITE_URL}" if email_text else email_summary(key, text)
            else:
                payload = text
            results[name] = bool(fn(payload))
        except Exception as e:  # noqa: BLE001
            print(f"[qmt] 告警发送失败 ({name}): {e}")
            results[name] = False
    ok = any(results.values())
    sent[key] = {"at": datetime.now().isoformat(timespec="seconds"), "ok": ok, "channels": results}
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
# 现金类品种 (10-22 起): 货币 ETF 与国债逆回购只是「停放现金」, 不是策略持仓 ——
# 不参与持仓对账、不进成交回报, 市值并入账户现金一起报告线外现金。
CASH_EQUIV = {"511880", "511990", "511660", "511690", "511850", "159001", "159003", "159005"}
REPO_PREFIXES = ("204", "1318")          # 上交所 GC 系列 204xxx / 深交所 R 系列 1318xx


def code6(c):
    return str(c)[:6]


def is_cash_equiv(code):
    c = code6(code)
    return c in CASH_EQUIV or c.startswith(REPO_PREFIXES)


def line_positions(state):
    out = {}
    for lot in (state or {}).get("lots") or []:
        out[code6(lot["code"])] = out.get(code6(lot["code"]), 0) + int(lot.get("shares") or 0)
    return {c: s for c, s in out.items() if s > 0}


def qmt_positions(snap):
    """策略口径的券商持仓 (排除现金类品种)。"""
    out = {}
    for p in (snap or {}).get("positions") or []:
        v = int(p.get("volume") or 0)
        if v > 0 and not is_cash_equiv(p["stock_code"]):
            out[code6(p["stock_code"])] = out.get(code6(p["stock_code"]), 0) + v
    return out


def cash_equiv_value(snap):
    return round(sum(float(p.get("market_value") or 0) for p in (snap or {}).get("positions") or []
                     if int(p.get("volume") or 0) > 0 and is_cash_equiv(p["stock_code"])), 2)


def reconcile(snap, state):
    """快照 vs 线账本: 持仓逐只比(不含现金类), 现金只报差额(线外现金 = 账户现金 + 现金类市值 - 线现金)。"""
    q, ln = qmt_positions(snap), line_positions(state)
    diffs = [{"code": c, "qmt": q.get(c, 0), "line": ln.get(c, 0)}
             for c in sorted(set(q) | set(ln)) if q.get(c, 0) != ln.get(c, 0)]
    acct_cash = ((snap or {}).get("asset") or {}).get("cash")
    line_cash = (state or {}).get("cash")
    ceq = cash_equiv_value(snap)
    return {"probe_at": (snap or {}).get("probe_at"), "kind": (snap or {}).get("kind"),
            "account_online": bool((snap or {}).get("account_online")),
            "positions_match": not diffs, "diffs": diffs,
            "account_cash": acct_cash, "cash_equiv_value": ceq,
            "line_cash": None if line_cash is None else round(float(line_cash), 2),
            "outside_cash": (None if acct_cash is None or line_cash is None
                             else round(float(acct_cash) + ceq - float(line_cash), 2))}


# ── 人工确认闸门 ──────────────────────────────────────────────────
MANUAL_AFTER = "21:00:00"   # 最后一次自动确认 20:50, 之后才放人工


def _autoconfirm_records(live_dir):
    p = Path(live_dir) / "qmt" / "autoconfirm.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def manual_confirm_gate(live_dir, pid, state, now=None):
    """QMT 线能不能人工「确认成交」。返回 (允许, 说明)。

    10-09 有人在 18:34 手点了「一笔都没成交」, 抢在 18:50 自动确认之前 —— 那天确实没成交所以无害,
    但一旦有真实成交, 人工抢先报「未成交」会让账本和券商对不上。所以 QMT 线在执行日当晚只认自动确认,
    下面三种情况才放人工: 自动确认被拦下/执行失败; 21:00 之后仍没确认; 执行日已过(错过了当晚)。
    """
    aw = (state or {}).get("awaiting_confirm")
    if pid not in QMT_LINES or not aw:
        return True, ""
    now = now or datetime.now()
    exec_date = str(aw.get("exec_date") or "")
    if exec_date < now.date().isoformat():
        return True, "执行日当晚没有完成自动确认, 请对照券商成交人工确认。"
    for rec in _autoconfirm_records(live_dir):
        if rec.get("pid") == pid and str(rec.get("exec_date")) == exec_date and rec.get("status") in ("blocked", "failed"):
            why = "; ".join(rec.get("problems") or []) or f"returncode={rec.get('returncode')}"
            return True, f"自动确认没有通过 ({why}), 请对照券商成交人工确认。"
    if now.strftime("%H:%M:%S") >= MANUAL_AFTER:
        return True, "今晚自动确认没有完成, 请对照券商成交人工确认。"
    return False, ("这条线由系统在 18:50 按券商真实成交自动确认 (20:50 补跑), 不用手动确认。"
                   "自动确认被拦下、或 21:00 后仍未确认时, 才开放人工确认。")


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
        if is_cash_equiv(t.get("stock_code")):
            continue                      # 货币 ETF / 逆回购是现金管理, 不是策略成交
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


def first_online_snapshot_after(live_dir, date):
    """执行日之后(严格晚于那天)第一份账户在线的快照, 任何 kind; 没有返回 None。"""
    day = str(date).replace("-", "")
    for p in sorted((Path(live_dir) / "qmt").glob("snap_*.json")):
        stamp = p.name[5:13]
        if not stamp.isdigit() or stamp <= day:
            continue
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if s.get("account_online"):
            return s
    return None


def build_fallback_confirm(later_snap, state, exec_date):
    """执行日没有收盘快照(QMT 离线, 如 10-14 Windows 更新重启)时的兜底。返回 (fills, problems, notes)。

    当日成交第二天就查不到了, 但持仓查得到: A 股 T+1, 执行日的买入次日一定体现为持仓, 卖出一定体现为减仓。
    所以之后第一份在线快照的持仓若与线账本逐只一致 -> 执行日什么都没成交, 按「无成交」确认;
    不一致 -> 推不出成交价, 停下等人工。
    """
    if not later_snap:
        return [], [f"没有 {exec_date} 收盘后的 QMT 快照, 之后也还没有在线快照"], []
    q, ln = qmt_positions(later_snap), line_positions(state)
    if q != ln:
        diff = {c: (q.get(c, 0), ln.get(c, 0)) for c in sorted(set(q) | set(ln)) if q.get(c, 0) != ln.get(c, 0)}
        return [], [f"没有 {exec_date} 收盘快照, 且 {later_snap.get('probe_at')} 的持仓与线账本不一致 "
                    f"(代码: QMT 股数, 线账本股数) {diff}, 推不出执行日成交"], []
    return [], [], [f"{exec_date} 没有收盘快照 (QMT 离线); {later_snap.get('probe_at')} 的持仓与线账本一致, 执行日无成交"]
