"""交易日历缓存: 让页面能算出"计划到底哪天执行"。

为什么要单独一份缓存
────────
state.calendar 只到最新交易日为止(它是从训练集的日期列来的), 查不到未来。
但页面要回答的恰恰是未来: "这份计划的执行日是今天还是明天?"

而未来交易日无法靠"周一到周五"推出来 —— 2026 国庆 10/01~10/07 里有 5 个工作日
不开市, 只按工作日算会把执行日说早 5 天。所以必须拿官方日历。

职责划分 (与项目既有约定一致: 流水线写, 网页只读)
────────
  refresh()  只由 daily_rebuild 调用, 联网抓取并落盘。
  load()/next_trading_day()  网页调用, 纯读文件, 绝不联网 ——
  网页请求里做网络调用会让页面卡住甚至超时, 而且抓失败不该影响看盘。

拿不到日历时一律返回 None, 由调用方降级成"下一个交易日"这种不写死日期的说法。
宁可说得含糊, 也不能把日期说错。
"""
import json
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "live" / "trading_calendar.json"

# 只留近几年 + 全部未来, 免得缓存里塞进 1990 年以来的八千多行
KEEP_FROM_YEARS = 2


def _to_date(d):
    """接受 str / date / datetime / pd.Timestamp, 统一成 date"""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    s = str(d)[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def load():
    """读缓存, 返回 (days:list[date], meta:dict); 不可用则 (None, {})。

    "不可用"包含两种: 文件缺失/损坏, 以及日历里没有任何未来日期 ——
    后者说明缓存太旧(比如跨年了没更新), 已经答不了"下一个交易日是哪天"。
    """
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        days = sorted({d for d in (_to_date(x) for x in raw.get("days") or []) if d})
    except (OSError, ValueError, AttributeError):
        return None, {}
    if not days:
        return None, {}
    meta = {"fetched_at": raw.get("fetched_at"), "source": raw.get("source"),
            "first": str(days[0]), "last": str(days[-1]), "n": len(days)}
    return days, meta


def is_trading_day(d, days=None):
    """d 是否交易日; 日历不可用或 d 超出覆盖范围则返回 None (不猜)"""
    d = _to_date(d)
    if d is None:
        return None
    if days is None:
        days, _ = load()
    if not days or not (days[0] <= d <= days[-1]):
        return None
    return d in set(days)


def next_trading_day(d, days=None):
    """严格晚于 d 的第一个交易日; 日历不可用或已到覆盖末尾则 None"""
    d = _to_date(d)
    if d is None:
        return None
    if days is None:
        days, _ = load()
    if not days:
        return None
    for x in days:
        if x > d:
            return x
    return None          # d 已在覆盖范围末尾之后, 老实返回 None


def refresh(verbose=True):
    """联网抓官方交易日历并落盘。只该由 daily_rebuild 调用。

    抓到的日历必须含未来日期才写盘 —— 否则等于用一份答不了问题的东西
    覆盖掉原本可能还有效的缓存。
    """
    import warnings

    warnings.filterwarnings("ignore")
    import akshare as ak
    import pandas as pd

    df = ak.tool_trade_date_hist_sina()
    if df is None or not len(df):
        raise RuntimeError("交易日历返回空")
    col = df.columns[0]
    ser = pd.to_datetime(df[col], errors="coerce").dropna()
    today = date.today()
    cutoff = date(today.year - KEEP_FROM_YEARS, 1, 1)
    days = sorted({d.date() for d in ser if d.date() >= cutoff})
    if not days:
        raise RuntimeError("交易日历里没有近几年的数据")
    n_future = sum(1 for d in days if d > today)
    if n_future == 0:
        raise RuntimeError(
            f"交易日历只到 {days[-1]}, 没有未来交易日, 拒绝用它覆盖缓存")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "source": "akshare tool_trade_date_hist_sina",
        "days": [str(d) for d in days],
    }
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE_PATH)      # 原子替换, 免得网页读到写一半的文件
    if verbose:
        print(f"  交易日历已更新: {len(days)} 个交易日 ({days[0]} ~ {days[-1]}), "
              f"其中未来 {n_future} 个")
    return days


def ensure_fresh(max_age_days=7, min_future_days=30, verbose=True):
    """需要时才联网。给 daily_rebuild 用。

    交易日历一年才变一次(节假日安排公布时), 没必要每天抓三遍。只在这三种
    情况下抓: 缓存不存在 / 抓取时间超过 max_age_days / 未来余量不足
    min_future_days 个自然日(说明快到覆盖末尾了, 比如年底)。

    抓失败只告警不抛异常 —— 日历是锦上添花, 不能因为它拖垮整条出信号的管线。
    返回 True 表示缓存现在可用。
    """
    days, meta = load()
    why = None
    if not days:
        why = "缓存不存在或不可用"
    else:
        last = days[-1]
        if (last - date.today()).days < min_future_days:
            why = f"未来余量只到 {last}, 不足 {min_future_days} 天"
        else:
            try:
                age = (datetime.now() - datetime.fromisoformat(meta["fetched_at"])).days
                if age > max_age_days:
                    why = f"缓存已 {age} 天未更新"
            except (TypeError, ValueError, KeyError):
                why = "缓存缺少抓取时间"
    if why is None:
        if verbose:
            print(f"  交易日历无需更新 (覆盖至 {days[-1]})")
        return True
    if verbose:
        print(f"  交易日历需要更新: {why}")
    try:
        refresh(verbose=verbose)
        return True
    except Exception as e:
        # 有旧缓存就继续用旧的; 完全没有则页面退回"下一个交易日"的含糊说法
        if verbose:
            print(f"  WARN 交易日历更新失败({type(e).__name__}: {e}), "
                  f"{'继续用旧缓存' if days else '暂无缓存, 页面将不显示具体执行日'}")
        return bool(days)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="交易日历缓存")
    ap.add_argument("--refresh", action="store_true", help="联网重抓")
    a = ap.parse_args()
    if a.refresh:
        refresh()
    days, meta = load()
    if not days:
        print("缓存不可用 (缺失/损坏/没有未来日期)。跑 --refresh 抓一份。")
        raise SystemExit(1)
    today = date.today()
    print(f"缓存: {meta['n']} 个交易日 {meta['first']} ~ {meta['last']}")
    print(f"抓取于 {meta['fetched_at']}  来源 {meta['source']}")
    print(f"今天 {today} 是交易日: {is_trading_day(today, days)}")
    nxt = next_trading_day(today, days)
    print(f"下一个交易日: {nxt}" + (f" (还有 {(nxt - today).days} 个自然日)" if nxt else ""))
    upcoming = [d for d in days if d > today][:10]
    print("往后 10 个:", ", ".join(str(d) for d in upcoming))
