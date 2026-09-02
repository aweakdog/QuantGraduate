"""宏观事件日历: 非交易日盲区的人肉补丁。

为什么需要它 (2026-08-31 立项)
────────
信号在交易日收盘生成, 执行在下一交易日尾盘, 中间隔着最短一晚、最长十来天
(国庆)的窗口 —— 宏观事件照发不误。2026-08-28(周五)收盘后 7 小时, 美联储
主席沃什在杰克逊霍尔放鹰, 金价单日 -3%, 周一开盘计划里的黄金集群 -7~9%。
模型特征只有 A 股量价, 对这些事件结构性全盲; 往特征里塞宏观序列已被
F 系列实验反复否决(加特征全败)。

所以这里不做预测, 只做最便宜的事: 把"接下来几天有什么大事"摆到操作页上,
人工确认成交前扫一眼, 自己决定这单要不要跳过。判断留给人, 不留给模型。

两类事件, 口径必须分开
────────
  规则生成  非农/LPR/中国PMI 这类节奏固定的, 按规则推算。发布日会漂
            1~2 天的(美国CPI)必须标 approx=True, 页面显示"(约)" ——
            宁可标不准, 不能装准。
  静态登记  FOMC 这类只有官方日历才知道的, 手工维护 STATIC_EVENTS,
            只登已确认的日期, 不猜。日期过了自然不再显示, 不用清理。

after_close 字段 (盲区判定的关键)
────────
A 股 15:00 收盘。事件在收盘前还是收盘后发生, 决定了它到底落在盲区
里还是已经被价格消化:
  after_close=False  中国PMI(9:30)/LPR(9:15) 这类上午发布 —— 当日收盘价
                     已含这个信息, 信号日当天的不算盲区
  after_close=True   非农(20:30)/CPI/FOMC(凌晨) 这类盘后发布 —— 信号日
                     当天发生也看不见(沃什就是这种), 算盲区
不分这一层会把每月底的中国 PMI 都误报成盲区, 警示条天天亮 → 失信。

另外从交易日历现算 A 股长假休市(连续 >=3 个自然日不开市), 提示持仓跨节。

对外只暴露 upcoming(start, horizon_days)。纯读、无网络、异常一律返回
空列表 —— 日历是锦上添花, 绝不能拖垮操作页。
"""
from datetime import date, timedelta

import trading_calendar

WEEKDAY_CN = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

# ── 静态登记: 只登官方已公布的日期, 格式 (日期, 名称, 备注) ──
# 维护约定: 新日期公布后在这里追加一行即可, 过期条目不用删(自然过滤)。
# 第四列 after_close: 事件是否在 A 股 15:00 收盘后发生。
STATIC_EVENTS = [
    ("2026-09-16", "FOMC 利率决议(北京时间17日凌晨)",
     "沃什 8/28 杰克逊霍尔放鹰后的首次决议, 市场已定价约六成加息概率", True),
]


def _first_friday(y, m):
    d0 = date(y, m, 1)
    return d0 + timedelta(days=(4 - d0.weekday()) % 7)


def _month_end(y, m):
    nm = date(y + (m == 12), m % 12 + 1, 1)
    return nm - timedelta(days=1)


def _iter_months(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = y + (m == 12), m % 12 + 1


def _rule_events(start, end):
    """节奏固定的例行发布。只列对 A 股当周情绪有实际影响力的, 别堆噪音。"""
    out = []
    for y, m in _iter_months(start, end):
        out.append({"date": _first_friday(y, m), "approx": False, "after_close": True,
                    "name": "美国非农(20:30 北京时间, 周五晚)",
                    "note": "盘后公布 —— 下周一开盘才反映, 典型信号盲区"})
        out.append({"date": date(y, m, 12), "approx": True, "after_close": True,
                    "name": "美国CPI",
                    "note": "发布日 ±2 天漂移, 以官方日历为准"})
        out.append({"date": date(y, m, 20), "approx": False, "after_close": False,
                    "name": "中国LPR报价(9:15)", "note": ""})
        out.append({"date": _month_end(y, m), "approx": False, "after_close": False,
                    "name": "中国官方PMI(9:30)", "note": ""})
    return out


def _closure_events(start, end):
    """A 股长假: 相邻交易日隔 >=3 个自然日不开市。周末(2天)不算。"""
    days, _ = trading_calendar.load()
    if not days:
        return []
    out = []
    for a, b in zip(days, days[1:]):
        closed = (b - a).days - 1
        if closed >= 3 and start <= a <= end:
            # 休市发生在 a 日收盘之后 → after_close=True
            out.append({"date": a, "approx": False, "after_close": True,
                        "name": f"A股休市 {closed} 天({a + timedelta(days=1)} 起, {b} 复市)",
                        "note": "持仓要不要跨节, 计划不会替你判断 —— 长假里宏观照常发生"})
    return out


def upcoming(start=None, horizon_days=14):
    """[start, start+horizon] 内的事件, 按日期排序。

    返回 [{date:'YYYY-MM-DD', weekday:'周X', name, note, approx, after_close}],
    出错给 []。start 接受 str/date/None(=今天) —— 调用方通常传信号日,
    这样"信号生成之后"发生的事件(哪怕已经过去了)也会列出, 供复盘对照。
    """
    try:
        start = trading_calendar._to_date(start) or date.today()
        end = start + timedelta(days=horizon_days)
        evs = []
        for d, name, note, after_close in STATIC_EVENTS:
            d = trading_calendar._to_date(d)
            if d and start <= d <= end:
                evs.append({"date": d, "name": name, "note": note,
                            "approx": False, "after_close": after_close})
        evs += [e for e in _rule_events(start, end) if start <= e["date"] <= end]
        evs += _closure_events(start, end)
        evs.sort(key=lambda e: e["date"])
        return [{"date": str(e["date"]),
                 "weekday": WEEKDAY_CN[e["date"].weekday()],
                 "name": e["name"], "note": e.get("note") or "",
                 "approx": bool(e.get("approx")),
                 "after_close": bool(e.get("after_close"))} for e in evs]
    except Exception:
        return []          # 日历挂了不能连累操作页


if __name__ == "__main__":
    for e in upcoming():
        flag = "约" if e["approx"] else "  "
        print(f"{e['date']} {e['weekday']} [{flag}] {e['name']}  {e['note']}")
