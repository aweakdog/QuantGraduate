"""账户「历史操作」: 把 state_<pid>.json 的 history 回放成带当时收益率的时间线。

只读。绝不改状态文件 —— 状态只由 live_signal.py 单写。

history 里有两类记录, 字段形状不同:
  * 结算批次  {signal_date, exec_date, fills[], rejected[], source}
        fills 每笔 {code, action: buy/sell/roll, shares, price, fee, net}
        net 是这笔对现金的净影响 (买入为负, 含手续费; 续持为 0)
  * 改账操作  {type: deposit/withdraw/set_cash/set_capital/fix_lot/
                       drop_sold/drop_phantom/sync, at, before, after, ...}
        before/after 自带当时的现金/总资产, 直接采用, 不重算。

结算批次没有记"当时总资产", 这里按执行日收盘价复算:
    总资产 = 回放后的现金 + Σ(股数 x 执行日收盘)
收益率 = 总资产 / 当时本金 - 1。本金随存取现金(cash-flow)和本金重记
(set_capital)变化, 所以要按时间回放, 不能拿今天的本金去除历史总资产。

回放完成后与当前状态对账 (现金 / 各股股数), 不一致就明说
(`reconciled=False` + 差额), 不静默展示一串对不上的数字。
"""
import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

KIND_LABEL = {
    "buy": "买入", "sell": "卖出", "roll": "续持", "rejected": "未成交",
    "deposit": "存入现金", "withdraw": "取出现金", "set_cash": "校准现金",
    "set_capital": "本金重记", "fix_lot": "校准持仓", "drop_sold": "删除持仓(已卖出)",
    "drop_phantom": "删除持仓(记账错误)", "sync": "整体对账", "epoch": "策略配置切换",
    "init": "建立账户",
}
SELL_REASON = {"matured": "持满到期", "regime_exit": "大盘转弱清仓",
               "drain_exit": "切换策略前排空", "still_ranked": "到期仍在前列, 续持"}
COLUMNS = [("date", "日期"), ("time", "时间"), ("kind", "操作"), ("code", "代码"),
           ("name", "名称"), ("shares", "股数"), ("price", "价格"), ("fee", "手续费"),
           ("cash_delta", "现金变动"), ("cash_after", "现金"),
           ("market_value_after", "持仓市值"), ("equity_after", "总资产"),
           ("capital_after", "本金"), ("return_pct_after", "收益率%"),
           ("source", "来源"), ("note", "说明")]


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_names(root: Path):
    names = {}
    for p, ccol, ncol in ((root / "data" / "universe" / "pit_metadata.parquet", "code", "name"),
                          (root / "data" / "raw" / "all_stock_list.parquet", "code", "name")):
        if not p.exists():
            continue
        try:
            m = pd.read_parquet(p, columns=[ccol, ncol])
        except Exception:
            continue
        names.update(dict(zip(m[ccol].astype(str).str[:6].str.zfill(6), m[ncol].astype(str), strict=True)))
    return names


class Closes:
    """执行日收盘价, 按股缓存。返回 (价格, 是否恰在该日) ; 无数据 (None, False)。

    停牌股当天没有K线, 退回不晚于该日的最后一根 —— 与 live_signal 的
    市值口径一致, 但要把"不是当日价"告诉调用方。"""

    def __init__(self, root: Path):
        self.kdir = root / "data" / "raw" / "kline"
        self._c = {}

    def _load(self, code):
        if code in self._c:
            return self._c[code]
        p = self.kdir / f"{code}.parquet"
        kl = None
        if p.exists():
            for cols in (["date", "close"], ["时间", "收盘价"]):
                try:
                    kl = pd.read_parquet(p, columns=cols).set_axis(["date", "close"], axis=1)
                    break
                except Exception:
                    continue
        if kl is not None:
            kl["date"] = pd.to_datetime(kl["date"])
            kl["close"] = pd.to_numeric(kl["close"], errors="coerce")
            kl = kl.dropna().sort_values("date").drop_duplicates("date", keep="last")
            kl = kl[kl["close"] > 0].set_index("date")["close"]
        self._c[code] = kl
        return kl

    def at(self, code, date):
        kl = self._load(code)
        if kl is None or kl.empty:
            return None, False
        d = pd.Timestamp(date)
        i = int(kl.index.searchsorted(d, side="right")) - 1
        if i < 0:
            return None, False
        return float(kl.iloc[i]), bool(kl.index[i] == d)


def _initial_capital(state, history):
    """建户时的本金: 从当前本金倒推出入金与本金重记。"""
    cap = _f(state.get("initial_capital"), 0.0)
    for e in history:
        t = e.get("type")
        if t in ("deposit", "withdraw"):
            cap -= _f(e.get("amount"), 0.0)
        elif t == "set_capital":
            cap -= _f(e.get("delta"), 0.0)
    return cap


def _label(kind, entry=None, fill=None):
    if fill is not None and fill.get("action") == "sell":
        why = SELL_REASON.get(fill.get("reason"))
        if fill.get("partial"):
            why = (why + ", " if why else "") + "部分成交"
        return why or ""
    if fill is not None and fill.get("action") == "roll":
        return SELL_REASON["still_ranked"]
    if fill is not None and fill.get("action") == "buy":
        return "新建仓" if fill.get("reason") == "new_tranche" else ""
    return ""


def build_ledger(root: Path, pid: str, state: dict, names=None):
    """回放一条线的历史, 返回 {"rows": [...], "summary": {...}}。"""
    names = names if names is not None else load_names(root)
    closes = Closes(root)
    history = list(state.get("history") or [])
    capital = _initial_capital(state, history)
    cash = capital
    positions = {}          # code -> {"shares": int, "cost": float}
    rows = []
    approx_dates = set()
    unknown_price = set()
    # 每次现金/持仓/本金变动后的快照, 给逐日收益曲线用: 两次变动之间账户是静止的,
    # 逐日只需换收盘价重新估值
    checkpoints = []

    def checkpoint(date):
        checkpoints.append({"date": date, "cash": cash, "capital": capital,
                            "positions": {c: dict(p) for c, p in positions.items() if p["shares"] > 0}})

    def market_value(date):
        mv = 0.0
        for code, pos in positions.items():
            if pos["shares"] <= 0:
                continue
            px, exact = closes.at(code, date)
            if px is None:
                px = pos.get("cost") or 0.0
                unknown_price.add(code)
            elif not exact:
                approx_dates.add(str(pd.Timestamp(date).date()))
            mv += pos["shares"] * px
        return mv

    def snapshot_row(base, cash_after, equity_after, mv_after=None):
        ret = (equity_after / capital - 1) * 100 if capital and equity_after is not None else None
        base.update({"cash_after": round(cash_after, 2),
                     "market_value_after": None if mv_after is None else round(mv_after, 2),
                     "equity_after": None if equity_after is None else round(equity_after, 2),
                     "capital_after": round(capital, 2),
                     "return_pct_after": None if ret is None else round(ret, 2)})
        return base

    def name_of(code):
        return names.get(code, "")

    def add_shares(code, delta, cost=None):
        pos = positions.setdefault(code, {"shares": 0, "cost": cost})
        if delta > 0 and cost is not None:
            total = pos["shares"] + delta
            prev_cost = pos.get("cost") or cost
            pos["cost"] = (prev_cost * pos["shares"] + cost * delta) / total if total else cost
        pos["shares"] += delta
        if pos["shares"] <= 0:
            positions.pop(code, None)

    first_date = None
    rows.append(snapshot_row({"date": None, "time": None, "kind": KIND_LABEL["init"],
                              "code": "", "name": "", "shares": None, "price": None,
                              "fee": None, "cash_delta": None, "source": "系统",
                              "note": f"起始本金 ¥{capital:,.2f}", "signal_date": None},
                             cash, capital, 0.0))

    for e in history:
        t = e.get("type")
        if not t:                                     # 结算批次
            exec_date = e.get("exec_date")
            first_date = first_date or exec_date
            batch = []
            source = "手工确认" if e.get("source") == "manual" else "自动记账"
            for f in e.get("fills") or []:
                code = str(f.get("code"))[:6]
                act = f.get("action")
                shares = int(f.get("shares") or 0)
                price = _f(f.get("price"))
                net = _f(f.get("net"), 0.0)
                if act == "buy":
                    add_shares(code, shares, price)
                elif act == "sell":
                    add_shares(code, -shares)
                cash += net
                batch.append({"date": exec_date, "time": None, "kind": KIND_LABEL.get(act, act),
                              "code": code, "name": f.get("name") or name_of(code),
                              "shares": shares, "price": price, "fee": _f(f.get("fee")),
                              "cash_delta": round(net, 2), "source": source,
                              "note": _label(act, e, f), "signal_date": e.get("signal_date")})
            for r in e.get("rejected") or []:
                code = str(r.get("code"))[:6]
                batch.append({"date": exec_date, "time": None, "kind": KIND_LABEL["rejected"],
                              "code": code, "name": name_of(code), "shares": None, "price": None,
                              "fee": None, "cash_delta": 0.0, "source": source,
                              "note": f"计划{'买入' if r.get('action') == 'buy' else '卖出'}未成交: {r.get('reason', '')}",
                              "signal_date": e.get("signal_date")})
            if not batch:
                continue
            checkpoint(exec_date)
            mv = market_value(exec_date)
            for b in batch:
                rows.append(snapshot_row(b, cash, cash + mv, mv))
            continue

        at = str(e.get("at") or "")
        date, time = (at[:10] or None), (at[11:19] or None)
        before, after = e.get("before") or {}, e.get("after") or {}
        base = {"date": date, "time": time, "kind": KIND_LABEL.get(t, t), "code": "",
                "name": "", "shares": None, "price": None, "fee": None, "cash_delta": None,
                "source": "网页/人工", "note": e.get("note") or "", "signal_date": None}
        if t in ("deposit", "withdraw"):
            amt = _f(e.get("amount"), 0.0)
            capital += amt
            cash = _f(after.get("cash"), cash + amt)
            base["cash_delta"] = round(amt, 2)
        elif t == "set_cash":
            cash = _f(after.get("cash"), cash + _f(e.get("delta"), 0.0))
            base["cash_delta"] = round(_f(e.get("delta"), 0.0), 2)
            base["note"] = (base["note"] + " · " if base["note"] else "") + "修账, 不计为盈亏"
        elif t == "set_capital":
            capital += _f(e.get("delta"), 0.0)
            base["note"] = (base["note"] + " · " if base["note"] else "") + \
                f"本金 {_f(before.get('initial_capital'), 0):,.2f} → {_f(after.get('initial_capital'), 0):,.2f}"
        elif t == "fix_lot":
            code = str(e.get("code"))[:6]
            d_sh = int(before.get("shares") or 0), int(after.get("shares") or 0)
            if code in positions:
                positions[code]["shares"] += d_sh[1] - d_sh[0]
                if after.get("buy_price") is not None:
                    positions[code]["cost"] = _f(after.get("buy_price"), positions[code].get("cost"))
                if positions[code]["shares"] <= 0:
                    positions.pop(code, None)
            base.update({"code": code, "name": name_of(code), "shares": d_sh[1],
                         "price": _f(after.get("buy_price"))})
            base["note"] = (base["note"] + " · " if base["note"] else "") + \
                f"成本 {_f(before.get('buy_price'), 0):.3f}→{_f(after.get('buy_price'), 0):.3f}, 股数 {d_sh[0]}→{d_sh[1]}"
        elif t in ("drop_sold", "drop_phantom"):
            lot = e.get("lot") or {}
            code = str(lot.get("code"))[:6]
            add_shares(code, -int(lot.get("shares") or 0))
            cash = _f(after.get("cash"), cash)
            base.update({"code": code, "name": lot.get("name") or name_of(code),
                         "shares": int(lot.get("shares") or 0), "price": _f(e.get("sold_at")),
                         "fee": _f(e.get("fee")),
                         "cash_delta": round(_f(after.get("cash"), 0) - _f(before.get("cash"), 0), 2)})
        elif t == "sync":
            positions.clear()
            for code, sh in after.get("positions") or []:
                positions[str(code)[:6]] = {"shares": int(sh), "cost": None}
            cash = _f(after.get("cash"), cash)
            base["cash_delta"] = round(_f(after.get("cash"), 0) - _f(before.get("cash"), 0), 2)
            base["note"] = (base["note"] + " · " if base["note"] else "") + "以券商真实持仓/现金覆盖"
        equity = _f(after.get("equity"))
        mv = None if equity is None else max(0.0, equity - cash)
        rows.append(snapshot_row(base, cash, equity, mv))
        if date:
            checkpoint(date)

    for ep in (state.get("strategy_epochs") or [])[1:]:
        rows.append({"date": ep.get("since"), "time": None, "kind": KIND_LABEL["epoch"],
                     "code": "", "name": "", "shares": None, "price": None, "fee": None,
                     "cash_delta": None, "cash_after": None, "market_value_after": None,
                     "equity_after": None, "capital_after": None, "return_pct_after": None,
                     "source": "系统", "note": ep.get("note") or "策略参数切换, 持仓与账目保留",
                     "signal_date": None})

    rows.sort(key=lambda r: (r["date"] or "", r["time"] or ""))
    state_lots = {}
    for lot in state.get("lots") or []:
        code = str(lot.get("code"))[:6]
        state_lots[code] = state_lots.get(code, 0) + int(lot.get("shares") or 0)
    replay_lots = {c: p["shares"] for c, p in positions.items() if p["shares"] > 0}
    cash_diff = round(_f(state.get("cash"), 0.0) - cash, 2)
    # 每笔 net 是四舍五入到分再记的, 而账本现金按未取整值累加: 几十笔下来会差几分钱。
    # 这属于取整差不是记账缺项, 一角以内视为一致 (真缺一笔成交差的是几百几千)。
    reconciled = abs(cash_diff) <= 0.10 and state_lots == replay_lots
    last = next((r for r in reversed(rows) if r.get("equity_after") is not None), None)
    summary = {
        "profile": pid,
        "initial_capital_at_start": round(_initial_capital(state, history), 2),
        "current_capital": round(_f(state.get("initial_capital"), 0.0), 2),
        "first_exec_date": first_date,
        "n_events": len(history),
        "n_rows": len(rows),
        "n_trades": sum(1 for r in rows if r["kind"] in (KIND_LABEL["buy"], KIND_LABEL["sell"])),
        "last_equity": None if last is None else last["equity_after"],
        "last_return_pct": None if last is None else last["return_pct_after"],
        "reconciled": reconciled,
        "cash_diff_vs_state": cash_diff,
        "positions_match_state": state_lots == replay_lots,
        "approx_quote_dates": sorted(approx_dates),
        "unknown_price_codes": sorted(unknown_price),
        "epochs": len(state.get("strategy_epochs") or []),
        "note": (("回放结果与当前账本一致。" + (f"(现金四舍五入差 {cash_diff:+.2f} 元)" if cash_diff else ""))
                 if reconciled else
                 "回放结果与当前账本不完全一致(历史记录可能缺项), 表中总资产/收益率仅供参考。"),
    }
    curve = build_curve(state, checkpoints, closes)
    summary["curve_stats"] = curve_stats(curve)
    return {"rows": rows, "summary": summary, "curve": curve}


def build_curve(state, checkpoints, closes):
    """逐交易日净值: 两次变动之间账户静止, 只按当日收盘重新估值。

    日历用 state.calendar (与 live_signal 同一份交易日历), 从首次变动日到日历末日。
    同一天多次变动取最后一次快照 (与「历史操作」表末行一致)。
    收益率 = 总资产 / 当时本金 - 1; 回撤 = 相对历史最高收益率指数的跌幅。
    停牌股沿用不晚于当日的最后收盘; 完全无行情的股退回成本价并在 flag 里标出。
    """
    if not checkpoints:
        return []
    cal = [str(pd.Timestamp(d).date()) for d in (state.get("calendar") or [])]
    by_date = {}
    for cp in checkpoints:
        by_date[cp["date"]] = cp                       # 后者覆盖前者 = 当天最后一次
    dated = sorted(by_date)
    start = dated[0]
    days = [d for d in cal if d >= start] or dated
    for d in dated:                                    # 非交易日发生的改账也要有点
        if d not in days:
            days.append(d)
    days.sort()
    out, i, cur = [], 0, None
    peak = None
    for d in days:
        while i < len(dated) and dated[i] <= d:
            cur = by_date[dated[i]]
            i += 1
        if cur is None:
            continue
        mv, approx, unknown = 0.0, False, False
        for code, pos in cur["positions"].items():
            px, exact = closes.at(code, d)
            if px is None:
                px, unknown = (pos.get("cost") or 0.0), True
            elif not exact:
                approx = True
            mv += pos["shares"] * px
        equity = cur["cash"] + mv
        capital = cur["capital"]
        nav = equity / capital if capital else None
        if nav is not None:
            peak = nav if peak is None else max(peak, nav)
        dd = None if nav is None or not peak else (nav / peak - 1) * 100
        out.append({"date": d, "cash": round(cur["cash"], 2), "market_value": round(mv, 2),
                    "equity": round(equity, 2), "capital": round(capital, 2),
                    "return_pct": None if nav is None else round((nav - 1) * 100, 2),
                    "drawdown_pct": None if dd is None else round(dd, 2),
                    "n_positions": len(cur["positions"]),
                    "flag": "unknown_price" if unknown else ("approx" if approx else "")})
    return out


def curve_stats(curve):
    """曲线的几个摘要数字。年化用交易日/244 折算, 样本短时不报年化 (< 60 个交易日)。"""
    pts = [c for c in curve if c["return_pct"] is not None]
    if not pts:
        return None
    rets = [c["return_pct"] for c in pts]
    navs = [1 + r / 100 for r in rets]
    daily = [navs[k] / navs[k - 1] - 1 for k in range(1, len(navs))]
    n = len(pts)
    ann = None
    if n >= 60 and navs[0] > 0:
        ann = round(((navs[-1] / navs[0]) ** (244 / (n - 1)) - 1) * 100, 2)
    up = sum(1 for x in daily if x > 0)
    worst = min(pts, key=lambda c: c["drawdown_pct"] if c["drawdown_pct"] is not None else 0)
    return {"start": pts[0]["date"], "end": pts[-1]["date"], "n_days": n,
            "total_return_pct": rets[-1], "peak_return_pct": max(rets), "trough_return_pct": min(rets),
            "max_drawdown_pct": worst["drawdown_pct"], "max_drawdown_date": worst["date"],
            "annualized_pct": ann, "up_days": up, "down_days": sum(1 for x in daily if x < 0),
            "best_day_pct": round(max(daily) * 100, 2) if daily else None,
            "worst_day_pct": round(min(daily) * 100, 2) if daily else None,
            "invested_days": sum(1 for c in pts if c["n_positions"] > 0)}


def ledger_xlsx(pid: str, display: str, ledger: dict) -> bytes:
    """生成 Excel 字节串 (不落盘: 每个账户的导出彼此独立, 不留缓存文件)。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "操作明细"
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="1F4E78")
    ws.append([label for _, label in COLUMNS])
    for c in ws[1]:
        c.font, c.fill = head_font, head_fill
        c.alignment = Alignment(horizontal="center")
    for r in ledger["rows"]:
        ws.append([r.get(key) for key, _ in COLUMNS])
    widths = {"日期": 12, "时间": 10, "操作": 16, "代码": 9, "名称": 12, "股数": 8, "价格": 10,
              "手续费": 9, "现金变动": 13, "现金": 14, "持仓市值": 14, "总资产": 14, "本金": 14,
              "收益率%": 10, "来源": 10, "说明": 40}
    for i, (_, label) in enumerate(COLUMNS, 1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(label, 12)
    money_cols = [i for i, (k, _) in enumerate(COLUMNS, 1)
                  if k in ("price", "fee", "cash_delta", "cash_after", "market_value_after",
                           "equity_after", "capital_after")]
    for row in ws.iter_rows(min_row=2):
        for i in money_cols:
            row[i - 1].number_format = "#,##0.00"
        row[[k for k, _ in COLUMNS].index("return_pct_after")].number_format = "0.00"
    ws.freeze_panes = "A2"

    curve = wb.create_sheet("净值曲线")
    curve.append(["日期", "现金", "持仓市值", "总资产", "本金", "收益率%", "回撤%", "持仓只数", "备注"])
    for c in curve[1]:
        c.font, c.fill = head_font, head_fill
    flag_txt = {"approx": "含停牌股, 按前收盘估值", "unknown_price": "有股票无行情, 按成本估值", "": ""}
    for r in ledger.get("curve") or []:
        curve.append([r["date"], r["cash"], r["market_value"], r["equity"], r["capital"],
                      r["return_pct"], r["drawdown_pct"], r["n_positions"], flag_txt.get(r["flag"], r["flag"])])
    for col in "BCDE":
        for c in curve[col][1:]:
            c.number_format = "#,##0.00"
    for col in "FG":
        for c in curve[col][1:]:
            c.number_format = "0.00"
    for col, w in zip("ABCDEFGHI", (12, 14, 14, 14, 14, 10, 10, 9, 28), strict=True):
        curve.column_dimensions[col].width = w
    curve.freeze_panes = "A2"

    info = wb.create_sheet("说明")
    s = ledger["summary"]
    lines = [("账户", display), ("条线 id", pid),
             ("导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
             ("起始本金", s["initial_capital_at_start"]), ("当前本金", s["current_capital"]),
             ("首次成交日", s["first_exec_date"] or "--"),
             ("最新总资产", s["last_equity"]), ("最新收益率%", s["last_return_pct"]),
             ("买卖笔数", s["n_trades"]), ("与当前账本一致", "是" if s["reconciled"] else "否"),
             ("备注", s["note"]),
             ("口径", "总资产 = 现金 + 股数×执行日收盘价; 收益率 = 总资产/当时本金 − 1。"
                     "存取现金与本金重记会改变本金, 不计为盈亏; 校准现金是修账, 不是盈亏。"),
             ("停牌日近似", ", ".join(s["approx_quote_dates"]) or "无"),
             ("无行情股票", ", ".join(s["unknown_price_codes"]) or "无")]
    for k, v in lines:
        info.append([k, v])
    info.column_dimensions["A"].width = 16
    info.column_dimensions["B"].width = 90
    for c in info["A"]:
        c.font = Font(bold=True)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def ledger_from_file(root: Path, pid: str):
    p = root / "data" / "live" / f"state_{pid}.json"
    if not p.exists():
        return None
    state = json.loads(p.read_text(encoding="utf-8"))
    return build_ledger(root, pid, state)
