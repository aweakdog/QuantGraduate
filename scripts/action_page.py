"""明日操作页 — 行动优先的极简界面

现有 web_server.py 的 "/" 是数据仪表盘 (账户概览/持仓表/计划表), 信息密度高但
需要读者自己判断该干什么。本模块提供另一种呈现: 把 plan_*.json 归一化成
"明天几点、卖什么、买什么、买多少股" 的清单, 让不懂策略的人也能照着执行。

对外只暴露两个东西:
    build_today(root)  -> dict    归一化后的操作载荷 (GET /api/today)
    ACTION_HTML        -> str     单页 HTML (GET /)

设计约束:
  * 只读。绝不改 state.json —— 状态只能由 live_signal.py 写, 避免双写错位。
  * 数据不新鲜时必须显式报警, 而不是把过期计划当成今天的操作展示。
  * 执行时点由 exec_mode 决定, 不能写死: t1close -> 次日尾盘, t1open -> 次日开盘。
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from live_config import (DEFAULT_PROFILE, PROFILES, capital_of, display_name,
                         is_auto, is_locked, state_file)

# 尾盘集合竞价前的下单窗口; t1open 则是次日开盘
EXEC_WHEN = {
    "t1close": "下一个交易日 14:50–15:00 (尾盘)",
    "t1open":  "下一个交易日 09:30 (开盘)",
}


def _load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _plan_glob(pid):
    """计划文件名与 live_signal.PLAN_PREFIX 的命名规则保持一致"""
    return f"plan_{pid}_*.json"


def _latest_plan(live_dir: Path, pid):
    plans = sorted(live_dir.glob(_plan_glob(pid)))
    return _load_json(plans[-1]) if plans else None


def _next_rebal(state, plan):
    """下次换仓的交易日 + 还差几个交易日。

    换仓规则与 live_signal 一致: 距 last_rebal_signal_date 满 hold_days 个交易日。
    日历存在 state.calendar 里, 直接查表, 不重新推算交易日。
    """
    cal = state.get("calendar") or []
    hold = (state.get("config") or {}).get("hold_days")
    last = state.get("last_rebal_signal_date")
    if not cal or not hold or not last:
        return None, None
    idx = pd.DatetimeIndex([pd.Timestamp(x) for x in cal])
    i_last = int(idx.searchsorted(pd.Timestamp(last), side="right")) - 1
    if i_last < 0:
        return None, None
    i_next = i_last + hold
    sig_date = (plan or {}).get("signal_date") or state.get("last_signal_date")
    i_now = int(idx.searchsorted(pd.Timestamp(sig_date), side="right")) - 1 if sig_date else i_last
    left = max(0, i_next - i_now)
    # 未来日期可能还没进日历(日历只到最新交易日), 此时只报剩余天数
    date = str(idx[i_next].date()) if i_next < len(idx) else None
    return date, left


def _freshness(root: Path, state, plan):
    """判断展示的计划是否对应最新交易日。

    K线最新日 > 计划信号日 说明当日流水线还没跑(或失败), 页面上的操作是旧的。
    """
    pipe = _load_json(root / "data" / "live" / "pipeline_status.json") or {}
    kline_date = pipe.get("kline_max_date")
    train_date = pipe.get("train_max_date") or (pipe.get("new_train_info") or {}).get("max_date")
    sig = (plan or {}).get("signal_date")
    stale, note = False, ""
    if kline_date and sig:
        if pd.Timestamp(sig) < pd.Timestamp(kline_date):
            stale = True
            note = (f"计划信号日 {sig} 落后于最新行情日 {kline_date}, "
                    f"当日流水线可能未跑或失败")
    if not sig:
        stale, note = True, "还没有任何操作计划"
    return {
        "kline_date": kline_date,
        "train_date": train_date,
        "signal_date": sig,
        "pipeline_ok": pipe.get("ok"),
        "pipeline_finished_at": pipe.get("finished_at"),
        "pipeline_skipped": pipe.get("skipped_reason"),
        "stale": stale,
        "note": note,
    }


def list_profiles():
    """给前端做切换用的简表"""
    return [{"id": k, "name": display_name(k), "default_name": v["name"],
             # capital 用生效值而不是代码默认值 —— 网页上重置时可能改过
             "capital": capital_of(k), "default_capital": v["capital"],
             "positions": v["tranche-n"],
             "desc": v["desc"], "auto": is_auto(k), "locked": is_locked(k)}
            for k, v in PROFILES.items()]


def build_recommend(root: Path, pid=None):
    """每日推荐看板: 模型当天打分最高的股票。

    模型排序本身与本金/持仓数无关, 四条线的 recommend 完全一样;
    差异只在于"买不买得起" —— 每只预算 = 总资产/持仓数, 不足一手(100股)
    的会被跳过。所以这里按当前 profile 标出 affordable, 避免照榜买入后
    发现根本买不了。
    """
    pid = pid if pid in PROFILES else DEFAULT_PROFILE
    prof = PROFILES[pid]
    live = root / "data" / "live"
    plan = _latest_plan(live, pid)
    state = _load_json(live / state_file(pid))

    rec = list((plan or {}).get("recommend") or [])
    # 每只预算决定"买不买得起", 必须用当前真实总资产, 否则存取现金后
    # 这里还按旧数字标"买不起", 会误导人。
    st = state or {}
    equity = float(st.get("cash") or 0) + sum(
        (l.get("shares") or 0) * (l.get("buy_price") or 0) for l in (st.get("lots") or []))
    if equity <= 0:
        equity = (plan or {}).get("equity") or capital_of(pid)
    n = prof["tranche-n"]
    budget = equity / n if n else None

    held = {str(h.get("code"))[:6] for h in ((plan or {}).get("hold") or [])}
    buying = {str(b.get("code"))[:6] for b in ((plan or {}).get("buy") or [])}

    rows = []
    for r in rec:
        px = r.get("close")
        lot = px * 100 if px else None
        rows.append({
            "rank": r.get("rank"),
            "code": r.get("code"),
            "name": r.get("name") or "",
            "pred": r.get("pred"),
            "close": px,
            "lot_cost": round(lot, 0) if lot else None,
            "affordable": (lot is not None and budget is not None and lot <= budget),
            "held": r.get("code") in held,
            "buying": r.get("code") in buying,
            "blocked": bool(r.get("blocked")),
        })

    return {
        "profile": pid,
        "profile_name": display_name(pid),
        "auto": is_auto(pid),
        "profiles": list_profiles(),
        "signal_date": (plan or {}).get("signal_date"),
        "exec_when": EXEC_WHEN.get(((plan or {}).get("config") or {}).get("exec_mode", "t1close"),
                                   "下一个交易日尾盘"),
        "per_slot_budget": round(budget, 0) if budget else None,
        "positions": n,
        "items": rows,
        "freshness": _freshness(root, state or {}, plan),
        "note": ("榜单只是模型打分排序, 不等于当天要买的清单。"
                 "实际买入受换仓周期和每只预算限制, 以操作清单为准。"),
    }


def build_today(root: Path, pid=None):
    """把 state + plan + pipeline_status 归一化成"明天该做什么"。"""
    pid = pid if pid in PROFILES else DEFAULT_PROFILE
    prof = PROFILES[pid]
    live = root / "data" / "live"
    state = _load_json(live / state_file(pid))
    plan = _latest_plan(live, pid)
    if state is None:
        return {"profile": pid, "profile_name": display_name(pid),
                "auto": is_auto(pid), "profiles": list_profiles(),
                "action": "init", "headline": "这条线还没建仓",
                "subline": f"本金 {prof['capital']:,.0f} / {prof['tranche-n']} 只, "
                           f"还没初始化。跑 init_profiles.py 建立。",
                "sell": [], "buy": [], "hold": [], "alternates": []}

    cfg = state.get("config") or {}
    fresh = _freshness(root, state, plan)
    sell = list((plan or {}).get("sell") or [])
    buy = list((plan or {}).get("buy") or [])

    # 持仓必须以 state.lots 为准, 不能用 plan.hold ——
    # plan 是出信号那一刻的快照, 而删除持仓/对账只改 state。若用 plan,
    # 状态里存在但计划里没有的持仓就不会显示, 用户也就没法删它。
    # 计划里的同一只股票只用来补展示字段(名称/参考价/盈亏/已持天数)。
    plan_hold = {str(h.get("code"))[:6]: h for h in ((plan or {}).get("hold") or [])}
    # 名称在多处出现, 都拿来当字典用, 尽量别让界面上只剩一串代码
    name_src = {}
    for grp in ("hold", "sell", "buy", "alternates", "recommend"):
        for r in ((plan or {}).get(grp) or []):
            c6 = str(r.get("code"))[:6]
            if r.get("name") and c6 not in name_src:
                name_src[c6] = r["name"]

    cal = [str(pd.Timestamp(d).date()) for d in (state.get("calendar") or [])]

    def _days_since(lot, key):
        """从 lot[key] 那天到信号日走了多少个交易日。

        持有时长一律按交易日而不是自然日 —— 标签 fwd_5d_ret 就是在只含
        交易日的面板上上移 5 行算的, 持有期必须与标签 horizon 同口径。
        日历缺失或对不上时返回 None 而不是编一个数。
        """
        d = lot.get(key)
        sig = (plan or {}).get("signal_date")
        if not d or not sig or d not in cal or sig not in cal:
            return None
        return cal.index(sig) - cal.index(d)

    def _held_days(lot):
        """到期时钟: 还有几天该评估它 (续持会归零)"""
        return _days_since(lot, "open_signal_date")

    def _tenure_days(lot):
        """真实持有时长: 从最初开仓那天算起, 续持不清零。

        没有 first_open_signal_date 的是续持功能上线前建的仓, 此时
        两者本就相等, 退回 open_signal_date 即为正确答案。
        """
        if lot.get("first_open_signal_date"):
            return _days_since(lot, "first_open_signal_date")
        return _days_since(lot, "open_signal_date")

    hold = []
    for lot in (state.get("lots") or []):
        c6 = str(lot.get("code"))[:6]
        ph = plan_hold.get(c6) or {}
        ref = ph.get("ref_close") or lot.get("buy_price")
        bp = lot.get("buy_price") or 0
        # 没有当日行情时不谎报 0%, 显示 "--" 更诚实
        pnl = ph.get("pnl_pct")
        if pnl is None and ph.get("ref_close") and bp:
            pnl = round(ph["ref_close"] / bp * 100 - 100, 2)
        hold.append({
            "code": c6,
            "name": ph.get("name") or name_src.get(c6, ""),
            "shares": lot.get("shares") or 0,
            "ref_close": ref,
            "pnl_pct": pnl,
            # 两个天数回答不同问题, 所以都给:
            #   held_days   -> 什么时候会动它 (到期时钟, 续持归零)
            #   tenure_days -> 这笔一共拿了多久 (只增不减)
            # 只显示前者会让续持过的仓看起来像刚买的。
            "held_days": ph.get("held_days", _held_days(lot)),
            "tenure_days": ph.get("tenure_days", _tenure_days(lot)),
            "n_rolled": ph.get("n_rolled", int(lot.get("rolled") or 0)),
        })
    in_cash = bool((plan or {}).get("in_cash"))
    is_rebal = bool((plan or {}).get("is_rebal"))

    # ── 行动类型: 决定页面主横幅 ──
    if fresh["stale"]:
        action = "stale"
        headline = "数据未更新"
        subline = fresh["note"]
    elif in_cash:
        action = "cash"
        headline = "清仓避险"
        subline = "大盘转弱, 卖出全部持仓且不开新仓"
    elif sell or buy:
        action = "trade"
        n = len(sell) + len(buy)
        headline = f"需要执行 {n} 笔操作"
        subline = f"卖出 {len(sell)} 只, 买入 {len(buy)} 只"
    else:
        action = "none"
        headline = "明天不用操作"
        subline = f"继续持有 {len(hold)} 只, 到期自动提示卖出" if hold else "当前空仓, 等待下个换仓日"

    nxt_date, nxt_left = _next_rebal(state, plan)
    # 现金必须取 state 而不是 plan —— plan 是出信号那一刻的快照, 之后的
    # 现金校准/出入金/删除持仓只写 state, 用 plan 的话页面会一直显示旧数字,
    # 而这正是「防止偏差」这类功能最不能出的错。
    # hold 上面已按 state.lots 构建, 市值直接用它算即可。
    cash = float(state.get("cash") or 0)
    mv = sum((h["shares"] or 0) * (h["ref_close"] or 0) for h in hold)
    equity = cash + mv
    init_cap = state.get("initial_capital") or 0

    def _fmt_row(r, side):
        px = r.get("ref_close") or r.get("ref_price") or 0
        sh = r.get("shares") or 0
        return {
            "code": str(r.get("code", ""))[:6],
            "name": r.get("name") or "",
            "shares": sh,
            "ref_price": round(px, 3) if px else None,
            "amount": round(sh * px, 0) if px else None,
            "pnl_pct": r.get("pnl_pct"),
            "held_days": r.get("held_days"),
            "tenure_days": r.get("tenure_days"),
            "n_rolled": r.get("n_rolled"),
            # 卖出预估到账金额(已扣手续费)。换仓日的买入靠这笔钱,
            # 所以界面上要把这个链条显示出来
            "est_proceeds": r.get("est_proceeds"),
            "est_cost": r.get("est_cost"),
            "side": side,
        }

    return {
        "profile": pid,
        "profile_name": display_name(pid),
        "profile_desc": prof["desc"],
        "auto": is_auto(pid),
        # 实盘模式下的"等你填真实成交价"状态。存在时这条线已停止推进:
        # 不会记账也不会出新信号, 直到提交成交回报。
        "awaiting_confirm": state.get("awaiting_confirm") or None,
        # 现在能不能提交成交回报。只有系统确实在等确认时才能提交 ——
        # 计划刚出、执行日还没到时提交是没法结算的(没有执行日行情),
        # 前端据此把打勾锁住, 免得用户白填一遍。
        "can_confirm": (not is_auto(pid)) and bool(state.get("awaiting_confirm")),
        "profiles": list_profiles(),
        "action": action,
        "headline": headline,
        "subline": subline,
        "signal_date": (plan or {}).get("signal_date"),
        "exec_when": EXEC_WHEN.get(cfg.get("exec_mode", "t1close"), "下一个交易日尾盘"),
        "is_rebal": is_rebal,
        "in_cash": in_cash,
        # 换仓日的资金链: 现有现金 + 卖出所得 = 买入可用。
        # A股卖出资金当天可用, 所以先卖后买在现实里成立; 但卖不掉(停牌/
        # 跌停)或只成交一部分时, 钱就不够买全部, 所以这两个数字必须告知用户。
        "cash_after_sell": (plan or {}).get("cash_after_sell"),
        "est_sell_proceeds": round(sum((r.get("est_proceeds") or 0) for r in sell), 2),
        "est_buy_cost": round(sum((r.get("est_cost") or 0) for r in buy), 2),
        "sell": [_fmt_row(r, "sell") for r in sell],
        "buy": [_fmt_row(r, "buy") for r in buy],
        "hold": [_fmt_row(r, "hold") for r in hold],
        "alternates": [_fmt_row(r, "alt") for r in ((plan or {}).get("alternates") or [])],
        "account": {
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "market_value": round(mv, 2),
            "initial_capital": init_cap,
            "total_return_pct": round(equity / init_cap * 100 - 100, 2) if init_cap else None,
            # 绝对盈亏。出入金后百分比会被稀释(分母变大), 但这个数不会 ——
            # 存款同额加进 cash 和 initial_capital, 相减后盈亏不变。
            # 所以做过出入金的条线应以这个数为准。
            "total_pnl": round(equity - init_cap, 2) if init_cap else None,
        },
        "next_rebal": {"date": nxt_date, "trading_days_left": nxt_left},
        "strategy": {
            "hold_days": cfg.get("hold_days"),
            "positions": cfg.get("tranche_n"),
            "exec_mode": cfg.get("exec_mode"),
            "regime_filter": cfg.get("regime_filter"),
            # 每只预算决定了能买的最高股价(一手=100股), 是低本金的关键约束
            "per_slot_budget": round(equity / cfg["tranche_n"], 0) if cfg.get("tranche_n") else None,
        },
        "freshness": fresh,
        "market": {
            "breadth": (plan or {}).get("breadth"),
            "mkt_close": (plan or {}).get("mkt_close"),
            "mkt_ma": (plan or {}).get("mkt_ma"),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "plan_generated_at": (plan or {}).get("generated_at"),
    }


ACTION_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>明日操作</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
       background:#0b0d12;color:#e8eaed;padding:16px 14px 40px;font-size:15px;line-height:1.5}
  .wrap{max-width:560px;margin:0 auto}

  .top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px}
  .top h1{font-size:17px;font-weight:600}
  .top .date{font-size:13px;color:#7c8598}

  /* 条线切换: 四条并行线各自独立记账 */
  .profs{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:12px}
  .prof{background:#14171e;border:1px solid #232833;border-radius:10px;padding:9px 11px;
        cursor:pointer;transition:.15s}
  .prof.on{border-color:#2563eb;background:#161c2b}
  .prof .pn{font-size:14px;font-weight:600;color:#e8eaed}
  .prof .pm{font-size:11px;color:#6f7889;margin-top:2px}
  .prof .pr{font-size:13px;font-weight:600;margin-top:3px}
  /* 记账方式徽标: 纸面=自动按行情, 实盘=等你确认成交 */
  .mode{display:inline-block;font-size:10px;font-weight:700;padding:1px 5px;
        border-radius:4px;margin-left:4px;vertical-align:middle}
  .mode-auto{background:#1e3a2b;color:#86efac}
  .mode-man{background:#3a2a12;color:#fcd34d}

  /* 操作按钮 */
  .acts{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}
  .btn{flex:1;min-width:96px;text-align:center;padding:9px 8px;border-radius:9px;
       background:#1a1f29;color:#c9cdd6;font-size:13px;font-weight:600;cursor:pointer;
       border:1px solid #262c38;transition:.15s;user-select:none}
  .btn:active{transform:scale(.97)}
  .btn-on{background:#14532d;border-color:#1a7a43;color:#bbf7d0}
  .btn-off{background:#3a2a12;border-color:#7c5310;color:#fcd34d}
  .btn-pri{background:#2563eb;border-color:#2563eb;color:#fff}
  .btn[aria-disabled="true"]{opacity:.45;pointer-events:none}

  /* 待确认成交面板 */
  .cf{background:#14171e;border:1px solid #7c5310;border-radius:12px;
      padding:14px;margin-bottom:14px}
  .cf h3{font-size:15px;font-weight:600;color:#fcd34d;margin-bottom:4px}
  .cf .cfs{font-size:12px;color:#8a93a6;margin-bottom:12px;line-height:1.7}
  /* 纸面模式的操作行是只读的: 打勾不影响账目, 给勾反而误导 */
  .op.ro{cursor:default;display:flex;align-items:center;gap:11px}
  .op.ro .tick{display:none}
  /* 实盘模式: 上半行点击打勾, 勾上后下半行出现真实股数/成交价 */
  .op .rowtop{display:flex;align-items:center;gap:11px;cursor:pointer}
  .fillbox{display:flex;align-items:center;gap:7px;margin-top:9px;padding-top:9px;
           border-top:1px dashed #2a3040}
  .fillbox .fl{font-size:12px;color:#8a93a6;flex:0 0 auto}
  .fillbox input{width:82px;background:#0b0d12;border:1px solid #2a3040;color:#e8eaed;
                 border-radius:6px;padding:6px 7px;font-size:13px;text-align:right;
                 font-family:inherit}
  .fillbox input:focus{outline:none;border-color:#2563eb}
  .fillbox .unit{font-size:11px;color:#6f7889;flex:0 0 auto}
  /* 说明条: 讲清打勾到底算不算数 */
  .tipbox{background:#0b0d12;border-radius:9px;padding:10px 12px;margin-bottom:11px;
          font-size:12px;color:#8a93a6;line-height:1.75}
  .tipbox b{color:#c9cdd6}
  .tipbox.warn-tip{border:1px solid #7c5310}
  .tipbox.warn-tip b{color:#fcd34d}
  /* 换仓日资金链: 买入靠卖出所得, 必须显示够不够 */
  .money{background:#0b0d12;border-radius:9px;padding:10px 12px;margin-top:11px;
         font-size:12px;color:#8a93a6;line-height:1.8}
  .money b{color:#c9cdd6}
  .money .mnote{color:#6f7889}
  .money .mok{color:#86efac;font-weight:600}
  .money .mbad{color:#fca5a5;font-weight:600}

  /* 弹窗 */
  .mask{position:fixed;inset:0;background:rgba(0,0,0,.72);display:flex;
        align-items:center;justify-content:center;padding:20px;z-index:50}
  .modal{background:#14171e;border:1px solid #2a3040;border-radius:14px;
         padding:18px;width:100%;max-width:340px}
  .modal h3{font-size:16px;font-weight:600;margin-bottom:6px}
  .modal p{font-size:13px;color:#8a93a6;line-height:1.7;margin-bottom:12px}
  .modal input[type=text],.modal input[type=password],.modal input[type=number]{
         width:100%;background:#0b0d12;border:1px solid #2a3040;
         color:#e8eaed;border-radius:8px;padding:10px;font-size:15px;font-family:inherit}
  .modal input:focus{outline:none;border-color:#2563eb}
  .modal .err{color:#fca5a5;font-size:12px;margin-top:8px;min-height:16px}
  .modal .hint{font-size:12px;color:#6f7889;margin-top:8px;line-height:1.6}
  .modal .cur{background:#0b0d12;border-radius:8px;padding:9px 11px;margin-bottom:10px;
              font-size:12px;color:#8a93a6;line-height:1.7}
  /* 二选一的选项卡: 两种删除语义现金处理相反, 必须让用户显式选 */
  .opt{background:#0b0d12;border:1px solid #262c38;border-radius:9px;
       padding:10px 11px;margin-bottom:8px;cursor:pointer;transition:.15s}
  .opt.on{border-color:#2563eb;background:#161c2b}
  .opt .ot{font-size:14px;font-weight:600;color:#e8eaed}
  .opt .od{font-size:12px;color:#8a93a6;line-height:1.6;margin-top:3px}
  /* 改账里最重的操作(清零重建)用警示色, 与其他按钮拉开距离 */
  .btn-danger{background:#3b1518;border-color:#5b1f24;color:#fca5a5}
  .btn-danger:active{background:#5b1f24}
  .modal label.fl{display:block;font-size:12px;color:#8a93a6;margin:10px 0 5px}
  /* 基准线的说明框 (占住原本按钮组的位置) */
  .lockbox{background:#0b0d12;border:1px solid #262c38;border-radius:9px;
           padding:10px 12px;font-size:12px;color:#8a93a6;line-height:1.7}
  .lockbox b{color:#c9cdd6}
  .mode-lock{background:#3b2f14;color:#e0a83a}
  /* 持仓行上的删除入口 */
  .del{flex:0 0 auto;width:28px;height:28px;border-radius:7px;background:#1e222b;
       color:#8a93a6;font-size:13px;display:flex;align-items:center;
       justify-content:center;cursor:pointer;margin-left:8px;transition:.15s}
  .del:active{background:#4a1d1d;color:#fca5a5}
  .mbtns{display:flex;gap:8px;margin-top:14px}
  .toast{position:fixed;left:50%;bottom:28px;transform:translateX(-50%);
         background:#1e222b;border:1px solid #2a3040;color:#e8eaed;font-size:13px;
         padding:10px 16px;border-radius:10px;z-index:60;max-width:88%;
         box-shadow:0 8px 24px rgba(0,0,0,.5)}

  /* 主视图切换 */
  .tabs{display:flex;gap:6px;margin-bottom:12px}
  .tab{flex:1;text-align:center;padding:9px 0;border-radius:9px;background:#14171e;
       color:#8a93a6;font-size:14px;font-weight:600;cursor:pointer;border:1px solid #1e222b}
  .tab.on{background:#2563eb;color:#fff;border-color:#2563eb}

  /* 推荐看板 */
  .rec{display:flex;align-items:center;gap:10px;padding:10px 2px;border-bottom:1px solid #1e222b}
  .rec:last-child{border-bottom:none}
  .rec .rk{width:24px;height:24px;border-radius:6px;background:#1e222b;color:#8a93a6;
           font-size:12px;font-weight:700;display:flex;align-items:center;
           justify-content:center;flex:0 0 auto}
  .rec.top3 .rk{background:#1e3a8a;color:#bfdbfe}
  .rec .rb{flex:1;min-width:0}
  .rec .rn{font-size:15px;font-weight:600}
  .rec .rm{font-size:11px;color:#6f7889;margin-top:2px}
  .rec .rv{text-align:right;flex:0 0 auto;font-size:13px}
  .chip{display:inline-block;font-size:10px;font-weight:700;padding:1px 5px;
        border-radius:4px;margin-left:5px;vertical-align:middle}
  .chip-hold{background:#1e3a8a;color:#bfdbfe}
  .chip-buy{background:#14532d;color:#bbf7d0}
  .chip-no{background:#3a2a12;color:#fcd34d}
  .chip-blk{background:#4a1d1d;color:#fca5a5}

  /* 主横幅: 一眼看出要不要动手 */
  .banner{border-radius:16px;padding:22px 20px;margin-bottom:14px;position:relative;overflow:hidden}
  .banner .hl{font-size:26px;font-weight:700;letter-spacing:-.3px}
  .banner .sl{font-size:14px;margin-top:6px;opacity:.85}
  .banner .when{font-size:13px;margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,.18)}
  .banner .when b{font-weight:600}
  .b-none{background:linear-gradient(135deg,#14532d,#1a7a43);color:#eafff2}
  .b-trade{background:linear-gradient(135deg,#1e3a8a,#2563eb);color:#eef4ff}
  .b-cash{background:linear-gradient(135deg,#7c2d12,#ea580c);color:#fff5ec}
  .b-stale{background:linear-gradient(135deg,#7f1d1d,#dc2626);color:#fff0f0}
  .b-init{background:#26292f;color:#c9cdd6}

  .card{background:#14171e;border-radius:14px;padding:16px;margin-bottom:12px}
  .card h2{font-size:13px;color:#8a93a6;font-weight:600;letter-spacing:.4px;margin-bottom:12px}

  /* 操作行。外层用 block: 实盘模式下勾上后要在下方再放一行成交输入框,
     横向排布交给内部的 .rowtop (实盘) 或 .op.ro 自身 (纸面只读)。 */
  .op{display:block;padding:13px 12px;border-radius:11px;
      margin-bottom:8px;background:#1b1f28;border-left:4px solid #444}
  .op:last-child{margin-bottom:0}
  .op.sell{border-left-color:#ef4444}
  .op.buy{border-left-color:#22c55e}
  .op.hold{border-left-color:#3f4658}
  /* 打勾 = 这笔会记入系统, 所以是"选中"而不是"划掉", 用高亮不用变灰 */
  .op.done{background:#182a1e;box-shadow:inset 0 0 0 1px #1a7a43}
  .op .tick{width:26px;height:26px;border-radius:7px;border:2px solid #4b5567;flex:0 0 auto;
            display:flex;align-items:center;justify-content:center;font-size:15px;color:transparent;
            cursor:pointer;transition:.15s}
  .op.done .tick{background:#22c55e;border-color:#22c55e;color:#fff}
  .op .body{flex:1;min-width:0}
  .op .line1{font-size:17px;font-weight:600;display:flex;align-items:center;gap:7px;flex-wrap:wrap}
  .op .line1 .tag{font-size:12px;font-weight:700;padding:2px 7px;border-radius:5px}
  .op.sell .line1 .tag{background:#7f1d1d;color:#fecaca}
  .op.buy .line1 .tag{background:#14532d;color:#bbf7d0}
  .op .line2{font-size:13px;color:#8a93a6;margin-top:3px}
  .op .amt{font-size:15px;font-weight:600;color:#c9cdd6;flex:0 0 auto;text-align:right}

  .hold-row{display:flex;justify-content:space-between;align-items:center;
            padding:10px 2px;border-bottom:1px solid #1e222b;font-size:14px}
  .hold-row:last-child{border-bottom:none}
  .hold-row .nm{color:#c9cdd6}
  .hold-row .meta{font-size:12px;color:#6f7889;margin-top:2px}
  /* A股习惯: 红涨绿跌 (与欧美相反)。只用于盈亏/收益率这类涨跌数字;
     "成功/失败"之类的状态色仍按通用语义走绿=好红=坏, 别混为一谈。 */
  .pos{color:#f6465d}.neg{color:#2ebd85}

  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .kv .k{font-size:12px;color:#6f7889}
  .kv .v{font-size:19px;font-weight:700;margin-top:2px}

  .foot{font-size:12px;color:#5e6675;margin-top:16px;text-align:center;line-height:1.8}
  .foot a{color:#7c8598}
  .warn{background:#2a1d1d;border:1px solid #7f1d1d;color:#fca5a5;border-radius:10px;
        padding:11px 13px;font-size:13px;margin-bottom:12px}
  .empty{color:#5e6675;font-size:14px;padding:6px 0}
  .steps{counter-reset:s;margin-top:10px}
  .steps li{list-style:none;counter-increment:s;font-size:14px;color:#a8b0c0;
            padding:7px 0 7px 30px;position:relative}
  .steps li:before{content:counter(s);position:absolute;left:0;top:7px;width:20px;height:20px;
                   border-radius:50%;background:#2563eb;color:#fff;font-size:12px;font-weight:700;
                   display:flex;align-items:center;justify-content:center}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <h1 id="title">明日操作</h1>
    <div class="date" id="sigdate">加载中…</div>
  </div>
  <div class="profs" id="profs"></div>
  <div class="acts" id="acts"></div>
  <div class="tabs">
    <div class="tab on" id="tab-act" onclick="setView('act')">明日操作</div>
    <div class="tab" id="tab-rec" onclick="setView('rec')">每日推荐</div>
  </div>
  <div id="app"></div>
  <div class="foot">
    数据每天收盘后自动更新 · <a href="/pro">运维仪表盘</a><br>
    <span id="gen"></span>
  </div>
</div>
<div id="modal"></div>
<div id="toast"></div>

<script>
const $ = s => document.querySelector(s);

// 当前看哪条线 / 哪个视图, 存起来刷新后不丢
let PID  = localStorage.getItem('pid')  || '';
let VIEW = localStorage.getItem('view') || 'act';
let PROFS = [];

// 勾选状态按 条线+信号日 隔离: 换了一天或换了条线都不会串
let DAY = '';
const doneKey = () => 'done_' + PID + '_' + DAY;
const getDone = () => { try { return JSON.parse(localStorage.getItem(doneKey())||'[]'); } catch(e){ return []; } };
const setDone = a => localStorage.setItem(doneKey(), JSON.stringify(a));

// 手填的实际股数/成交价也要持久化。它们存在 DOM 里, 而定时刷新会
// 重建 DOM —— 不存的话你填到一半被刷新一下就回到计划默认值了。
const fillKey = () => 'fills_' + PID + '_' + DAY;
const getFills = () => { try { return JSON.parse(localStorage.getItem(fillKey())||'{}'); } catch(e){ return {}; } };
const setFills = o => localStorage.setItem(fillKey(), JSON.stringify(o));
function saveFill(id){
  const s = $('#fs_' + id), p = $('#fp_' + id);
  if (!s || !p) return;
  const o = getFills();
  o[id] = {shares: s.value, price: p.value};
  setFills(o);
  refreshMoney();
}

const money = v => v == null ? '--' : '¥' + Number(v).toLocaleString('zh-CN',{maximumFractionDigits:0});

// 操作行有三种形态, 取决于记账方式:
//   纸面模式        -> 只读, 不给打勾。打勾不影响账目, 给了勾反而误导
//   实盘 + 等确认   -> 打勾 = 记入系统, 勾上后可改真实股数与成交价
//   实盘 + 还没到执行日 -> 只读, 提示明天收盘后回来确认
function opRow(r, side, mode){
  const id = side + '_' + r.code;
  const tag = side === 'sell' ? '卖出' : '买入';
  const px = r.ref_price == null ? '' : ' · 参考价 ' + r.ref_price;
  const head = `<div class="body">
      <div class="line1"><span class="tag">${tag}</span>${esc(r.name||r.code)}
        <span style="font-size:13px;color:#6f7889;font-weight:400">${r.code}</span></div>
      <div class="line2">${r.shares} 股${px}</div>
    </div>
    <div class="amt">${money(r.amount)}</div>`;

  if (mode !== 'confirm'){
    return `<div class="op ${side} ro">${head}</div>`;
  }

  const on = getDone().includes(id);
  // 优先用你上次填的值, 没填过才用计划默认值
  const sv = getFills()[id] || {};
  const vS = sv.shares != null ? sv.shares : r.shares;
  const vP = sv.price != null ? sv.price : (r.ref_price == null ? '' : r.ref_price);
  return `<div class="op ${side} ${on?'done':''}" id="row_${id}">
    <div class="rowtop" onclick="tickRow('${id}')">
      <div class="tick">✓</div>${head}
    </div>
    <div class="fillbox" id="fb_${id}" style="display:${on?'flex':'none'}">
      <span class="fl">实际</span>
      <input type="number" id="fs_${id}" value="${vS}" step="100" min="0"
             inputmode="numeric" oninput="saveFill('${id}')">
      <span class="unit">股</span>
      <input type="number" id="fp_${id}" value="${vP}"
             step="0.001" min="0" inputmode="decimal" oninput="saveFill('${id}')">
      <span class="unit">元</span>
    </div>
  </div>`;
}

// 换仓日的资金链: 买入靠的是卖出所得。
// 只读模式按计划的预估值显示; 实盘确认模式按"打勾的行 + 你填的数字"实时算,
// 这样卖不掉或只成交一部分时, 立刻能看出钱够不够买。
let LASTD = null;

function moneyChain(d){
  const cash = (d.account || {}).cash || 0;
  if (!d.can_confirm){
    const got = d.est_sell_proceeds || 0, need = d.est_buy_cost || 0;
    const avail = cash + got;
    return `按计划预估：现金 <b>${money(cash)}</b> + 卖出所得 <b>${money(got)}</b>
      = 可用 <b>${money(avail)}</b>，买入需要 <b>${money(need)}</b>。<br>
      <span class="mnote">卖出资金当天可用，所以先卖后买没问题。
      但若有股票<b>停牌或跌停卖不掉</b>，钱就不够买全部 —— 那时少买一只即可，
      系统会按实际成交记账。</span>`;
  }
  // 实盘确认: 只算打勾的
  const done = getDone();
  let got = 0, need = 0, nS = 0, nB = 0;
  for (const r of (LASTD ? [].concat(LASTD.sell, LASTD.buy) : [])){
    const id = r.side + '_' + r.code;
    if (!done.includes(id)) continue;
    const sh = parseFloat(($('#fs_' + id) || {}).value) || 0;
    const px = parseFloat(($('#fp_' + id) || {}).value) || 0;
    const amt = sh * px;
    if (r.side === 'sell'){ got += amt; nS++; } else { need += amt; nB++; }
  }
  const avail = cash + got;
  const short = need - avail;
  return `已打勾：卖出 ${nS} 笔得 <b>${money(got)}</b>，买入 ${nB} 笔需 <b>${money(need)}</b><br>
    现金 <b>${money(cash)}</b> + 卖出所得 = 可用 <b>${money(avail)}</b>
    ${short > 0.5
      ? `<br><span class="mbad">还差 ${money(short)} —— 买入报多了或卖出漏勾了，
         提交会被拒绝</span>`
      : `<br><span class="mok">资金够，可以提交</span>`}`;
}

function refreshMoney(){
  const box = $('#moneybox');
  if (box && LASTD) box.innerHTML = moneyChain(LASTD);
}

// 打勾 = 这笔真的成交了, 会记入系统; 不打勾 = 没成交, 不动账
function tickRow(id){
  const d = getDone(), i = d.indexOf(id);
  if (i >= 0) d.splice(i, 1); else d.push(id);
  setDone(d);
  const row = $('#row_' + id), fb = $('#fb_' + id);
  if (row) row.classList.toggle('done', i < 0);
  if (fb) fb.style.display = i < 0 ? 'flex' : 'none';
  refreshMoney();
}

// 持仓行的天数: 主显示真实持有时长(只增不减), 另外给出到期倒计时。
// 只显示到期时钟会让续持过的仓看起来像刚买的(续持会把它归零);
// 只显示真实时长则看不出哪天该操作。所以两个都要。天数一律是交易日。
function holdMeta(r){
  const parts = [r.shares + ' 股'];
  const t = r.tenure_days != null ? r.tenure_days : r.held_days;
  parts.push('已持 ' + (t == null ? '--' : t) + ' 日');
  if (r.n_rolled) parts.push('续持 ' + r.n_rolled + ' 次');
  // 到期倒计时: hold_days 未知或天数缺失时干脆不显示, 不编数字
  const H = ((LASTD || {}).strategy || {}).hold_days;
  if (H && r.held_days != null){
    const left = H - r.held_days;
    parts.push(left <= 0 ? '已到期' : '还剩 ' + left + ' 日到期');
  }
  return parts.join(' · ');
}

function holdRow(r){
  const p = r.pnl_pct;
  const cls = p == null ? '' : (p >= 0 ? 'pos' : 'neg');
  const sign = p == null ? '' : (p >= 0 ? '+' : '');
  return `<div class="hold-row">
    <div style="flex:1;min-width:0">
      <div class="nm">${esc(r.name||r.code)} <span style="color:#6f7889;font-size:12px">${r.code}</span></div>
      <div class="meta">${holdMeta(r)}</div></div>
    <div class="${cls}" style="text-align:right;font-weight:600">${sign}${p==null?'--':p+'%'}
         <div class="meta">${money(r.amount)}</div></div>
    ${curProf().locked ? '' : `<div class="del" title="删除这笔持仓"
         onclick="askDrop('${r.code}','${esc(r.name||'')}',${r.shares},${r.ref_price||0})">✕</div>`}
  </div>`;
}

function recRow(r){
  const chips = []
  if (r.held)   chips.push('<span class="chip chip-hold">持有中</span>');
  if (r.buying) chips.push('<span class="chip chip-buy">明天买</span>');
  if (!r.affordable) chips.push('<span class="chip chip-no">买不起一手</span>');
  if (r.blocked) chips.push('<span class="chip chip-blk">急涨回避</span>');
  return `<div class="rec ${r.rank<=3?'top3':''}">
    <div class="rk">${r.rank}</div>
    <div class="rb">
      <div class="rn">${r.name||r.code}
        <span style="font-size:12px;color:#6f7889;font-weight:400">${r.code}</span>${chips.join('')}</div>
      <div class="rm">收盘 ${r.close==null?'--':r.close} · 一手 ${money(r.lot_cost)}</div>
    </div>
    <div class="rv"><span style="color:#8a93a6">得分</span><br>
      <b style="color:#c9cdd6">${r.pred==null?'--':(r.pred*100).toFixed(2)}</b></div>
  </div>`;
}

function setView(v){
  VIEW = v; localStorage.setItem('view', v);
  $('#tab-act').classList.toggle('on', v==='act');
  $('#tab-rec').classList.toggle('on', v==='rec');
  $('#title').textContent = v==='act' ? '明日操作' : '每日推荐';
  load();
}

function setPid(p){
  PID = p; localStorage.setItem('pid', p);
  load();
}

const esc = s => String(s==null?'':s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

let toastT = null;
function toast(msg, ms){
  $('#toast').innerHTML = `<div class="toast">${esc(msg)}</div>`;
  clearTimeout(toastT);
  toastT = setTimeout(() => { $('#toast').innerHTML = ''; }, ms || 3200);
}

function closeModal(){ $('#modal').innerHTML = ''; }

// 所有改账操作都要口令。401 时弹密码框, 输对了自动重试原操作,
// 这样用户不会因为"密码过期"丢掉刚填的表单内容。
async function api(path, body){
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                              body: JSON.stringify(body)});
  const d = await r.json().catch(() => ({}));
  if (r.status === 401 && d.need_password){
    await askPassword();                 // 用户取消会 reject, 直接冒泡出去
    return api(path, body);
  }
  if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
  return d;
}

// 返回一个 Promise: 登录成功 resolve, 用户取消则 reject
function askPassword(){
  return new Promise((resolve, reject) => {
    $('#modal').innerHTML = `
      <div class="mask">
        <div class="modal">
          <h3>需要密码</h3>
          <p>改名、切换记账方式、校准现金、存取现金都会改动账目，
             需要输入密码。一次输入 12 小时内有效。</p>
          <input type="password" id="pw" inputmode="numeric" placeholder="密码"
                 onkeydown="if(event.key==='Enter')window.__pwOk()">
          <div class="err" id="pwerr"></div>
          <div class="mbtns">
            <div class="btn" onclick="window.__pwCancel()">取消</div>
            <div class="btn btn-pri" onclick="window.__pwOk()">确定</div>
          </div>
        </div>
      </div>`;
    setTimeout(() => { const i = $('#pw'); if (i) i.focus(); }, 50);

    window.__pwCancel = () => { closeModal(); reject(new Error('已取消')); };
    window.__pwOk = async () => {
      const pw = ($('#pw') || {}).value || '';
      const r = await fetch('/api/ops/login', {method:'POST',
        headers:{'Content-Type':'application/json'}, body: JSON.stringify({password: pw})});
      if (r.ok){ closeModal(); resolve(); return; }
      const d = await r.json().catch(() => ({}));
      const e = $('#pwerr');
      if (e) e.textContent = d.error || '密码错误';
      const i = $('#pw'); if (i){ i.value = ''; i.focus(); }
    };
  });
}

// 当前这条线的信息 (从 PROFS 里取, 避免和后端载荷字段重复)
const curProf = () => PROFS.find(p => p.id === PID) || {};

function renderProfs(active){
  $('#profs').innerHTML = PROFS.map(p => `
    <div class="prof ${p.id===active?'on':''}" onclick="setPid('${p.id}')">
      <div class="pn">${esc(p.name)}<span class="mode ${p.locked?'mode-lock':(p.auto?'mode-auto':'mode-man')}">${
        p.locked?'基准':(p.auto?'纸面':'实盘')}</span></div>
      <div class="pm">${p.positions} 只 · 每只 ${money(
        (p.equity != null ? p.equity : p.capital) / p.positions)}</div>
    </div>`).join('');
  renderActs();
}

// 最近一次 /api/today 的账户数字, 给现金弹窗做预填和对照
let ACCT = {};

function renderActs(){
  const p = curProf();
  if (!p.id){ $('#acts').innerHTML = ''; return; }
  // 基准线: 一个写操作按钮都不给。后端同样会拒, 这里只是别让人白点。
  if (p.locked){
    $('#acts').innerHTML = `
      <div class="lockbox">
        <b>基准线 · 不可更改</b><br>
        永久纸面自动记账。不能改名、不能切记账方式、不能校准现金、
        不能存取现金、不能删持仓。<br>
        它的用处是给你一条<b>没人动过</b>的参照 ——
        真实账户减去它，差额就是人为干预的代价。
      </div>`;
    return;
  }
  $('#acts').innerHTML = `
    <div class="btn" onclick="askRename()">重命名</div>
    ${p.auto
      ? `<div class="btn btn-off" onclick="askAuto(false)">取消自动操作</div>`
      : `<div class="btn btn-on"  onclick="askAuto(true)">开启自动操作</div>`}
    <div class="btn" onclick="askSetCash()">校准现金</div>
    <div class="btn" onclick="askCashFlow()">存取现金</div>
    <div class="btn btn-danger" onclick="askReset()">从头再来</div>`;
}

function acctBox(){
  return `<div class="cur">
    记录的现金 <b style="color:#c9cdd6">${money(ACCT.cash)}</b><br>
    持仓市值 <b style="color:#c9cdd6">${money(ACCT.market_value)}</b> ·
    总资产 <b style="color:#c9cdd6">${money(ACCT.equity)}</b><br>
    本金 <b style="color:#c9cdd6">${money(ACCT.initial_capital)}</b>
  </div>`;
}

// ── 现金校准 ──
// 自动记账用收盘价 + 估算手续费, 和真实成交总有零点几个百分点的差,
// 几十次换仓累积下来就可观。这里只改现金、不改本金, 所以收益率被修正
// 到真实水平 (而不是被"洗掉")。
function askSetCash(){
  const p = curProf();
  $('#modal').innerHTML = `
    <div class="mask" onclick="if(event.target===this)closeModal()">
      <div class="modal">
        <h3>校准现金 · ${esc(p.name)}</h3>
        ${acctBox()}
        <p>填券商 App 里的<b>真实可用现金</b>。用来消除自动记账
           (按收盘价 + 估算手续费) 累积下来的偏差。<br>
           这是<b>修账不是盈亏</b>：本金不动，所以收益率会被修正到真实水平。</p>
        <input type="number" id="sc" step="0.01" min="0" inputmode="decimal"
               value="${ACCT.cash==null?'':ACCT.cash}">
        <div class="hint">只改现金。如果持仓股数也不对，请去
          <a href="/pro" style="color:#60a5fa">运维页</a>做整体对账。</div>
        <div class="err" id="scerr"></div>
        <div class="mbtns">
          <div class="btn" onclick="closeModal()">取消</div>
          <div class="btn btn-pri" onclick="doSetCash()">保存</div>
        </div>
      </div>
    </div>`;
  setTimeout(() => { const i = $('#sc'); if (i){ i.focus(); i.select(); } }, 50);
}

async function doSetCash(){
  const v = parseFloat(($('#sc') || {}).value);
  const e = $('#scerr');
  if (!(v >= 0)){ if (e) e.textContent = '请填一个不小于 0 的数字'; return; }
  try {
    await api('/api/profile/set-cash', {profile: PID, cash: v, note: '网页校准'});
    closeModal(); toast('现金已校准为 ' + money(v), 4000);
    load();
  } catch(err){
    if (e) e.textContent = err.message; else toast(err.message, 6000);
  }
}

// ── 存取现金 ──
// 本金变动而非盈亏, 所以现金和本金同额增减, 收益率保持不变。
// 若只加现金不加本金, 存进 1 万会被算成"赚了 1 万"。
function askCashFlow(){
  const p = curProf();
  $('#modal').innerHTML = `
    <div class="mask" onclick="if(event.target===this)closeModal()">
      <div class="modal">
        <h3>存取现金 · ${esc(p.name)}</h3>
        ${acctBox()}
        <p>往这条线里<b>加钱或抽钱</b>。填正数是存入，负数是取出。</p>
        <input type="number" id="cfv" step="0.01" inputmode="decimal" placeholder="例如 10000 或 -5000">
        <div class="hint">这是<b>本金变动，不算盈亏</b>：现金和本金同额增减，
          所以<b>累计盈亏的金额不会变</b>。<br>
          但注意：收益<b>百分比会被稀释</b>，因为分母(本金)变大了。
          例如亏 150 元时存入 1 万，-0.75% 会变成 -0.5%，亏的钱其实一样多。
          做过出入金后请看「累计盈亏」的金额，别看百分比。<br>
          存入会提高每只预算 (总资产 ÷ ${p.positions} 只)，也就能买更高价的股票。</div>
        <div class="err" id="cferr"></div>
        <div class="mbtns">
          <div class="btn" onclick="closeModal()">取消</div>
          <div class="btn btn-pri" onclick="doCashFlow()">确认</div>
        </div>
      </div>
    </div>`;
  setTimeout(() => { const i = $('#cfv'); if (i) i.focus(); }, 50);
}

async function doCashFlow(){
  const v = parseFloat(($('#cfv') || {}).value);
  const e = $('#cferr');
  if (!v){ if (e) e.textContent = '请填一个非 0 的数字'; return; }
  try {
    await api('/api/profile/cash-flow', {profile: PID, amount: v, note: '网页出入金'});
    closeModal();
    toast((v > 0 ? '已存入 ' : '已取出 ') + money(Math.abs(v)), 4000);
    load();
  } catch(err){
    if (e) e.textContent = err.message; else toast(err.message, 6000);
  }
}

// ── 删除持仓 ──
// 两种情形现金处理完全相反, 所以让用户显式选, 不替他猜:
//   已卖出   -> 钱回到账户了, 现金增加
//   记错了   -> 本就没这笔, 现金不动
let DROP = {};

function askDrop(code, name, shares, refPx){
  DROP = {code, name, shares, refPx};
  $('#modal').innerHTML = `
    <div class="mask" onclick="if(event.target===this)closeModal()">
      <div class="modal">
        <h3>删除持仓 · ${esc(name||code)}</h3>
        <div class="cur">${esc(name||'')} ${code}<br>
          ${shares} 股 · 最近参考价 ${refPx || '--'}</div>
        <p>为什么要删？这决定<b>现金怎么算</b>，两种情况正好相反：</p>
        <div class="opt" id="opt-sold" onclick="pickDrop('sold')">
          <div class="ot">我已经卖了</div>
          <div class="od">在券商 App 里手动卖掉了。钱回到账户，所以
            <b style="color:#86efac">现金增加</b>，并记一笔卖出流水。</div>
        </div>
        <div class="opt" id="opt-phantom" onclick="pickDrop('phantom')">
          <div class="ot">系统记错了</div>
          <div class="od">我从没持有过这只。纯修账，
            <b style="color:#fcd34d">现金不变</b>，总资产会相应下降。</div>
        </div>
        <div id="droppx" style="display:none;margin-top:10px">
          <input type="number" id="dpx" step="0.001" min="0" inputmode="decimal"
                 value="${refPx || ''}" placeholder="真实卖出价">
          <div class="hint">填券商成交价。手续费按 万6、最低 5 元 估算。</div>
        </div>
        <div class="err" id="dperr"></div>
        <div class="mbtns">
          <div class="btn" onclick="closeModal()">取消</div>
          <div class="btn btn-pri" id="dpok" aria-disabled="true"
               onclick="doDrop()">确认删除</div>
        </div>
      </div>
    </div>`;
}

function pickDrop(mode){
  DROP.mode = mode;
  $('#opt-sold').classList.toggle('on', mode === 'sold');
  $('#opt-phantom').classList.toggle('on', mode === 'phantom');
  $('#droppx').style.display = mode === 'sold' ? 'block' : 'none';
  $('#dpok').setAttribute('aria-disabled', 'false');
  if (mode === 'sold') setTimeout(() => { const i = $('#dpx'); if (i){ i.focus(); i.select(); } }, 50);
}

async function doDrop(){
  const e = $('#dperr');
  if (!DROP.mode){ if (e) e.textContent = '请先选一种情况'; return; }
  const body = {profile: PID, code: DROP.code, mode: DROP.mode, note: '网页删除持仓'};
  if (DROP.mode === 'sold'){
    const px = parseFloat(($('#dpx') || {}).value);
    if (!(px > 0)){ if (e) e.textContent = '请填真实卖出价'; return; }
    body.price = px;
  }
  try {
    await api('/api/profile/drop-lot', body);
    closeModal();
    toast(DROP.mode === 'sold'
      ? `已按卖出记账，${DROP.name||DROP.code} 移出持仓`
      : `已删除记错的持仓 ${DROP.name||DROP.code}（现金未变）`, 5000);
    load();
  } catch(err){
    if (e) e.textContent = err.message; else toast(err.message, 6000);
  }
}

// ── 从头再来 ──
// 改账操作里最重的一个: 等于把这条线的历史业绩清零。所以
//   1. 先把"将要失去什么"摊开给人看 (总资产/盈亏/持仓/成交笔数)
//   2. 必须勾选确认才能点确定, 防误触
//   3. 本金和重置放在同一个动作里 —— 本金只在重置时作为起始现金生效,
//      单独改它不会有任何效果, 分成两个按钮只会让人以为改了其实没改
function askReset(){
  const p = curProf();
  const pnl = ACCT.total_pnl, pct = ACCT.total_return_pct;
  const cls = pnl == null ? '' : (pnl >= 0 ? 'pos' : 'neg');
  $('#modal').innerHTML = `
    <div class="mask" onclick="if(event.target===this)closeModal()">
      <div class="modal">
        <h3>从头再来 · ${esc(p.name)}</h3>
        <div class="cur">
          即将<b style="color:#fca5a5">清零</b>以下记录：<br>
          持仓 <b style="color:#c9cdd6">${ACCT.positions ? ACCT.positions.length : 0} 只</b> ·
          总资产 <b style="color:#c9cdd6">${money(ACCT.equity)}</b><br>
          累计盈亏 <b class="${cls}">${pnl == null ? '--' : (pnl >= 0 ? '+' : '') + money(pnl)}</b>
          ${pct == null ? '' : `(${pct >= 0 ? '+' : ''}${pct}%)`}
        </div>
        <p>持仓与历史成交全部清空，现金回到本金，然后<b>立刻重新建仓</b>。<br>
           旧账会归档到 <code>data/live/archive/</code>，不是直接删掉。</p>
        <label class="fl">本金</label>
        <input type="number" id="rscap" step="1000" min="5000" inputmode="numeric"
               value="${p.capital}">
        <div class="hint">不改就保持 ${money(p.capital)}。每只预算 = 本金 ÷ ${p.positions} 只。
          <span id="rsbud"></span></div>
        <div class="opt" id="rsack" onclick="ackReset()" style="margin-top:12px">
          <div class="ot">我知道这会清零历史业绩</div>
          <div class="od">这条线过去的成交记录与收益都不再计入。基准线不受影响，
            仍按原本金继续跑。</div>
        </div>
        <div class="err" id="rserr"></div>
        <div class="mbtns">
          <div class="btn" onclick="closeModal()">取消</div>
          <div class="btn btn-danger" id="rsok" aria-disabled="true"
               onclick="doReset()">清零并重新建仓</div>
        </div>
      </div>
    </div>`;
  const upd = () => {
    const v = parseFloat(($('#rscap') || {}).value);
    const b = $('#rsbud');
    if (b) b.textContent = (v > 0 && p.positions)
      ? `按现在填的值是每只 ${money(v / p.positions)}。` : '';
  };
  const i = $('#rscap');
  if (i){ i.oninput = upd; }
  upd();
}

function ackReset(){
  const box = $('#rsack');
  if (box) box.classList.add('on');
  const ok = $('#rsok');
  if (ok) ok.setAttribute('aria-disabled', 'false');
}

async function doReset(){
  const e = $('#rserr');
  if ($('#rsok').getAttribute('aria-disabled') !== 'false'){
    if (e) e.textContent = '请先勾选上面那行确认'; return;
  }
  const v = parseFloat(($('#rscap') || {}).value);
  if (!(v >= 5000)){ if (e) e.textContent = '本金至少 5,000'; return; }
  try {
    const d = await api('/api/profile/reset', {profile: PID, capital: v});
    closeModal();
    toast(`已清零，正在按本金 ${money(d.capital)} 重新建仓…`, 6000);
    pollRun('已重新建仓，新计划已生成');
  } catch(err){
    if (e) e.textContent = err.message; else toast(err.message, 6000);
  }
}

function askRename(){
  const p = curProf();
  $('#modal').innerHTML = `
    <div class="mask" onclick="if(event.target===this)closeModal()">
      <div class="modal">
        <h3>重命名</h3>
        <p>只改显示名字，不影响这条线的本金、持仓数和已有持仓。<br>
           留空则恢复默认名「${esc(p.default_name)}」。</p>
        <input type="text" id="rn" maxlength="16" value="${esc(p.name)}"
               placeholder="${esc(p.default_name)}">
        <div class="mbtns">
          <div class="btn" onclick="closeModal()">取消</div>
          <div class="btn btn-pri" onclick="doRename()">保存</div>
        </div>
      </div>
    </div>`;
  setTimeout(() => { const i = $('#rn'); if (i){ i.focus(); i.select(); } }, 50);
}

async function doRename(){
  const name = ($('#rn') || {}).value || '';
  try {
    const d = await api('/api/profile/rename', {profile: PID, name});
    closeModal(); toast('已改名为「' + d.name + '」');
    load();
  } catch(e){ toast('改名失败: ' + e.message); }
}

function askAuto(on){
  const p = curProf();
  const body = on
    ? `<p>切回<b style="color:#86efac">纸面模式</b>：系统每天按第二天的真实收盘价
         自动记账，不需要你做任何确认。<br><br>
         注意这是<b>模拟跟踪</b> —— 它假设你按计划成交了。如果你其实没下单，
         账面就会和你的真实账户脱节。</p>`
    : `<p>切到<b style="color:#fcd34d">实盘模式</b>：系统<b>不再自动记账</b>。
         每次换仓后要你填真实成交价，填了才入账、才会出下一份计划。<br><br>
         适合你真金白银在跑的那条线。代价是<b>你不确认它就会一直停在那</b>，
         不会自己往前走。</p>`;
  $('#modal').innerHTML = `
    <div class="mask" onclick="if(event.target===this)closeModal()">
      <div class="modal">
        <h3>${on?'开启自动操作':'取消自动操作'} · ${esc(p.name)}</h3>
        ${body}
        <div class="mbtns">
          <div class="btn" onclick="closeModal()">取消</div>
          <div class="btn ${on?'btn-on':'btn-off'}" onclick="doAuto(${on})">确认切换</div>
        </div>
      </div>
    </div>`;
}

async function doAuto(on){
  try {
    const d = await api('/api/profile/auto', {profile: PID, auto: on});
    closeModal(); toast(d.note, 5000);
    load();
  } catch(e){ toast('切换失败: ' + e.message); }
}

// ── 实盘模式: 打勾即记账 ──
// 打了勾的行才会记入系统, 每行可改真实股数与成交价(股数改小 = 部分成交)。
// 没打勾的一律当作没成交, 不动账。
let OPROWS = [];

async function submitTicked(none){
  let fills = [];
  if (!none){
    const done = getDone();
    for (const r of OPROWS){
      const id = r.side + '_' + r.code;
      if (!done.includes(id)) continue;          // 没打勾 = 没成交
      const sh = parseInt(($('#fs_' + id) || {}).value, 10);
      const px = parseFloat(($('#fp_' + id) || {}).value);
      if (!(sh > 0)){ toast(`${r.name||r.code}: 股数要大于 0，没成交就取消打勾`, 5000); return; }
      if (sh % 100 && r.side === 'buy'){
        toast(`${r.name||r.code}: 买入股数要是 100 的整数倍`, 5000); return;
      }
      if (!(px > 0)){ toast(`${r.name||r.code}: 请填真实成交价`, 5000); return; }
      if (r.side === 'sell' && sh > r.shares){
        toast(`${r.name||r.code}: 卖出 ${sh} 股超过持有的 ${r.shares} 股`, 5000); return;
      }
      fills.push({code: r.code, action: r.side, shares: sh, price: px});
    }
    if (!fills.length){
      toast('一笔都没打勾。如果确实都没成交, 请点「一笔都没成交」', 5000); return;
    }
  }
  try {
    const d = await api('/api/profile/confirm', {profile: PID, fills});
    setDone([]); setFills({});                   // 已入账, 清掉免得下轮串
    toast(d.note, 6000);
    pollRun();
  } catch(e){ toast('提交失败: ' + e.message, 8000); }
}

// 结算/重置都要跑一遍模型, 轮询到跑完再刷新页面
async function pollRun(okMsg){
  for (let i = 0; i < 90; i++){
    await new Promise(r => setTimeout(r, 2000));
    let s;
    try { s = await (await fetch('/api/signal-status')).json(); } catch(e){ continue; }
    if (!s.active){
      const log = s.log || '';
      if (/ERROR|Traceback/.test(log)) toast('后台任务报错, 详情见运维仪表盘', 6000);
      else toast(okMsg || '已结算并生成新计划', 4000);
      load();
      return;
    }
  }
  toast('耗时偏长, 请稍后刷新');
}

async function loadRec(){
  const q = PID ? ('?profile=' + PID) : '';
  let d;
  try { d = await (await fetch('/api/recommend' + q)).json(); }
  catch(e){ $('#app').innerHTML = '<div class="warn">无法连接服务器</div>'; return; }

  PID = d.profile; PROFS = d.profiles || PROFS; renderProfs(PID);
  $('#sigdate').textContent = d.signal_date ? ('信号日 ' + d.signal_date) : '';
  $('#gen').textContent = '';

  let h = '';
  const f = d.freshness || {};
  if (f.stale) h += `<div class="warn">${f.note||'数据未更新'}</div>`;

  h += `<div class="card"><h2>模型评分前 ${(d.items||[]).length} 名</h2>`;
  h += (d.items||[]).length ? d.items.map(recRow).join('') : '<div class="empty">暂无数据</div>';
  h += `</div>`;

  h += `<div class="card"><h2>怎么看这个榜</h2>
    <div style="font-size:13px;color:#8a93a6;line-height:1.9">
      这是模型对未来 5 日涨幅的预测排序，<b style="color:#c9cdd6">不等于明天要买的清单</b>。<br>
      实际买入只发生在换仓日，且只买前 ${d.positions} 名里买得起的。<br>
      当前每只预算 <b style="color:#c9cdd6">${money(d.per_slot_budget)}</b>，
      标了“买不起一手”的股票 100 股就超过这个预算，会被自动跳过。
    </div></div>`;

  $('#app').innerHTML = h;
}

async function loadAct(){
  const q = PID ? ('?profile=' + PID) : '';
  let d;
  try { d = await (await fetch('/api/today' + q)).json(); }
  catch(e){ $('#app').innerHTML = '<div class="warn">无法连接服务器</div>'; return; }

  PID = d.profile; PROFS = d.profiles || PROFS; renderProfs(PID);
  LASTD = d;                     // 资金链重算要用到当前计划与账户
  DAY = d.signal_date || 'na';
  $('#sigdate').textContent = d.signal_date ? ('信号日 ' + d.signal_date) : '';
  $('#gen').textContent = d.plan_generated_at ? ('计划生成于 ' + d.plan_generated_at.replace('T',' ')) : '';

  const bcls = {none:'b-none', trade:'b-trade', cash:'b-cash', stale:'b-stale', init:'b-init'}[d.action] || 'b-init';
  let h = '';

  // 实盘模式在等你确认时, 整条线都停着, 所以置顶提示。
  // 有买卖单时确认入口在下面的操作清单里(打勾); 没有单时这里直接给个按钮。
  if (d.awaiting_confirm){
    const ac = d.awaiting_confirm;
    const hasOrders = d.sell.length || d.buy.length;
    h += `<div class="cf">
      <h3>等你确认 ${ac.exec_date} 的成交</h3>
      <div class="cfs">这条线是<b style="color:#fcd34d">实盘模式</b>，
        在你确认之前<b>不会记账、也不会出新计划</b>。` +
      (hasOrders ? `<br>请在下面的操作清单里给成交了的打勾。</div>`
                 : `<br>当天没有需要买卖的单子。</div>
        <div class="btn btn-pri" onclick="submitTicked(true)">确认无操作, 继续</div>`) +
      `</div>`;
  }

  // 主横幅
  h += `<div class="banner ${bcls}">
      <div class="hl">${d.headline}</div>
      <div class="sl">${d.subline||''}</div>`;
  if (d.action === 'trade' || d.action === 'cash')
    h += `<div class="when">执行时间 <b>${d.exec_when}</b></div>`;
  else if (d.action === 'none' && d.next_rebal && d.next_rebal.trading_days_left != null)
    h += `<div class="when">下次换仓 <b>${d.next_rebal.date || ('还有 ' + d.next_rebal.trading_days_left + ' 个交易日')}</b></div>`;
  h += `</div>`;

  if (d.freshness && d.freshness.stale && d.action !== 'stale')
    h += `<div class="warn">${d.freshness.note}</div>`;

  // 操作清单。打勾的含义完全取决于记账方式, 所以标题和说明也跟着变
  if (d.sell.length || d.buy.length){
    const mode = d.can_confirm ? 'confirm' : 'ro';
    OPROWS = [].concat(d.sell, d.buy);
    h += `<div class="card"><h2>操作清单${
      mode === 'confirm' ? ' · 打勾即记入系统' : ''}</h2>`;

    if (mode === 'confirm'){
      h += `<div class="tipbox warn-tip">
        这条线是<b>实盘模式</b>：<b>打勾的才会记入系统</b>，没打勾的就当没成交。<br>
        勾上后可以改成真实股数和成交价 —— <b>只成交了一部分就改股数</b>。</div>`;
    } else if (!d.auto){
      h += `<div class="tipbox">实盘模式：${d.exec_when}下单后，
        <b>执行日当晚数据更新完</b>再回来打勾确认。现在还没到时候，先照着做即可。</div>`;
    } else {
      h += `<div class="tipbox">纸面模式：系统会<b>按行情自动记账</b>，不用打勾。<br>
        想按你的真实成交来记账，就点上面的「取消自动操作」。</div>`;
    }

    h += d.sell.map(r => opRow(r,'sell',mode)).join('');
    h += d.buy.map(r => opRow(r,'buy',mode)).join('');

    // 换仓日的买入是靠卖出所得来的, 把这个资金链摆明。
    // A股卖出资金当天可用, 所以现实里成立; 但卖不掉或只成交一部分时钱就不够。
    if (d.sell.length && d.buy.length){
      h += `<div class="money" id="moneybox">${moneyChain(d)}</div>`;
    }

    if (mode === 'confirm'){
      h += `<div class="mbtns">
          <div class="btn" onclick="submitTicked(true)">一笔都没成交</div>
          <div class="btn btn-pri" onclick="submitTicked(false)">提交打勾的成交</div>
        </div>`;
    } else {
      h += `<ol class="steps">
          <li>${d.exec_when} 打开券商 App</li>
          <li>先卖后买, 按上面的股数下单</li>
          ${d.auto ? '<li>系统次日按行情自动结算, 你不用回来操作</li>'
                   : '<li>执行日当晚回来打勾, 填真实成交价</li>'}
        </ol>`;
    }
    h += `</div>`;
  }

  // 持仓
  h += `<div class="card"><h2>当前持仓 · ${d.hold.length} 只</h2>`;
  h += d.hold.length ? d.hold.map(holdRow).join('') : '<div class="empty">空仓</div>';
  h += `</div>`;

  // 账户
  const a = d.account || {};
  ACCT = a;                      // 给现金弹窗做预填与对照
  const rp = a.total_return_pct;
  const pl = a.total_pnl;
  h += `<div class="card"><h2>账户</h2><div class="grid">
      <div class="kv"><div class="k">总资产</div><div class="v">${money(a.equity)}</div></div>
      <div class="kv"><div class="k">累计盈亏</div>
        <div class="v ${pl==null?'':(pl>=0?'pos':'neg')}">${
          pl==null?'--':(pl>=0?'+':'-')+money(Math.abs(pl)).slice(1)}</div></div>
      <div class="kv"><div class="k">收益率</div>
        <div class="v ${rp==null?'':(rp>=0?'pos':'neg')}">${rp==null?'--':(rp>=0?'+':'')+rp+'%'}</div></div>
      <div class="kv"><div class="k">本金</div><div class="v">${money(a.initial_capital)}</div></div>
      <div class="kv"><div class="k">现金</div><div class="v">${money(a.cash)}</div></div>
      <div class="kv"><div class="k">持仓市值</div><div class="v">${money(a.market_value)}</div></div>
    </div></div>`;

  // 该条线的方案与数据状态
  const f = d.freshness || {};
  const s = d.strategy || {};
  h += `<div class="card"><h2>这条线的方案</h2>
      <div style="font-size:13px;color:#8a93a6;line-height:1.9">
        ${d.profile_desc||''}<br>
        持仓 <b style="color:#c9cdd6">${s.positions||'--'} 只</b> ·
        每 <b style="color:#c9cdd6">${s.hold_days||'--'} 个交易日</b>整体换仓 ·
        每只预算 <b style="color:#c9cdd6">${money(s.per_slot_budget)}</b><br>
        记账方式 <b style="color:${d.auto?'#86efac':'#fcd34d'}">${
          d.auto ? '纸面 · 每天按行情自动记账' : '实盘 · 等你确认真实成交'}</b>
      </div>
      <div style="font-size:13px;color:#8a93a6;line-height:2;margin-top:10px;
                  border-top:1px solid #1e222b;padding-top:10px">
        最新行情日 <b style="color:#c9cdd6">${f.kline_date||'--'}</b><br>
        计划信号日 <b style="color:#c9cdd6">${f.signal_date||'--'}</b><br>
        上次自动更新 <b style="color:#c9cdd6">${f.pipeline_finished_at? f.pipeline_finished_at.replace('T',' '):'--'}</b>
      </div></div>`;

  $('#app').innerHTML = h;
}

function load(){ return VIEW === 'rec' ? loadRec() : loadAct(); }

setView(VIEW);
// 定时刷新会重建 DOM。填的值虽然已经落到 localStorage 不会丢, 但刷新会
// 把光标和未提交的编辑现场打断, 所以正在确认成交时干脆不刷。
setInterval(() => {
  if ($('#modal').innerHTML) return;
  if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
  // 实盘确认中且已经动过手(有打勾或填过值): 等你提交完再说
  if (LASTD && LASTD.can_confirm &&
      (getDone().length || Object.keys(getFills()).length)) return;
  load();
}, 60000);
</script>
</body>
</html>
"""
