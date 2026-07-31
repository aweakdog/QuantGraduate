"""访问记录: 知道有谁在什么时候从哪里访问过这个站。

用途是判断网址有没有泄漏, 所以关心的是"来过哪些陌生 IP", 而不是精确的
流量分析。

为什么分成两份存储
────────
  access_stats.json  每天/每 IP 的精确计数。用来回答"总共多少次访问"。
  access_log.jsonl   事件流, 同一 IP 同一路径 5 分钟内只记一条。

只有事件流会去重, 计数永远精确。这样做是因为页面自身每 60 秒轮询一次
/api/today, 若逐条记录, 你自己的浏览器一天就能刷出上千行, 真正需要注意的
陌生访问会被彻底淹没 —— 而那恰恰是这个功能唯一的目的。

不记什么
────────
请求体一律不记 (登录接口的密码就在里面)。查询串也不记。

关于 X-Forwarded-For
────────
本服务直接 uvicorn 裸跑, 前面没有可信代理, 所以 XFF 是访客自己就能随便
伪造的。真实来源只认 TCP 对端地址 (request.client.host)。XFF 仅另存一份并
在界面上标注"不可信", 便于识别有人在尝试伪造。
"""
import json
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
LOG_PATH = LIVE / "access_log.jsonl"
STATS_PATH = LIVE / "access_stats.json"
GEO_CACHE_PATH = LIVE / "ip_geo_cache.json"

DEDUPE_SECONDS = 300         # 同 IP 同路径这么久内只记一条

# 这些路径永不去重: 它们就是密码尝试本身。若合并, 一个 IP 连猜上百次
# 在日志里只会显示成一条, 等于看不见爆破 —— 而那是最需要看见的事。
LOGIN_PATHS = ("/api/view/login", "/api/ops/login", "/api/bt/login")

MAX_LOG_BYTES = 4 * 1024 * 1024
KEEP_DAYS = 90               # 计数只留最近这些天
UA_MAX = 200

_lock = threading.Lock()
_last_seen = {}              # (ip, path) -> 上次记录事件的时间戳
_stats_cache = None          # 内存里的计数, 落盘做节流


def _now():
    return datetime.now()


def _read_json(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json_atomic(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _load_stats():
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = _read_json(STATS_PATH, {"days": {}})
        _stats_cache.setdefault("days", {})
    return _stats_cache


_last_flush = 0.0


def _flush_stats(force=False):
    """计数落盘节流: 每 20 秒最多写一次, 免得每个请求都碰磁盘"""
    global _last_flush
    if not force and time.time() - _last_flush < 20:
        return
    st = _load_stats()
    cutoff = str(date.today() - timedelta(days=KEEP_DAYS))
    for d in [d for d in st["days"] if d < cutoff]:
        del st["days"][d]
    _write_json_atomic(STATS_PATH, st)
    _last_flush = time.time()


def _rotate_if_big():
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > MAX_LOG_BYTES:
            LOG_PATH.replace(LOG_PATH.with_suffix(".jsonl.1"))
    except OSError:
        pass


def is_private(ip):
    """内网/本机地址。这些不是"外部访问", 也没有归属地可查"""
    if not ip:
        return True
    if ip in ("127.0.0.1", "::1", "localhost"):
        return True
    if ip.startswith(("10.", "192.168.", "172.17.", "172.18.", "169.254.", "fd", "fe80")):
        return True
    if ip.startswith("172."):
        try:
            second = int(ip.split(".")[1])
            return 16 <= second <= 31
        except (IndexError, ValueError):
            return False
    return False


def record(ip, method, path, status, ua="", authed=False, xff=""):
    """记一次访问。必须永不抛异常 —— 记日志失败绝不能影响页面能不能打开。"""
    try:
        with _lock:
            today = str(date.today())
            st = _load_stats()
            day = st["days"].setdefault(today, {"total": 0, "ips": {}})
            day["total"] += 1
            day["ips"][ip] = day["ips"].get(ip, 0) + 1
            _flush_stats()

            if path not in LOGIN_PATHS:
                key = (ip, path)
                now = time.time()
                if now - _last_seen.get(key, 0) < DEDUPE_SECONDS:
                    return                   # 计数已加, 事件流跳过
                _last_seen[key] = now

            _rotate_if_big()
            rec = {"t": _now().isoformat(timespec="seconds"), "ip": ip,
                   "m": method, "p": path, "s": status,
                   "ua": (ua or "")[:UA_MAX], "au": 1 if authed else 0}
            if xff:
                rec["xff"] = xff[:120]       # 仅供识别伪造, 界面标注不可信
            LIVE.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def flush():
    """进程退出前把计数写下来"""
    try:
        with _lock:
            _flush_stats(force=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
# IP 归属地
# ══════════════════════════════════════════════════════════
def _load_geo_cache():
    return _read_json(GEO_CACHE_PATH, {})


def resolve_geo(ips, timeout=6):
    """批量查归属地, 结果缓存到本地。

    只在你打开访问日志页时才调用, 不在请求路径上 —— 记日志时联网会把每个
    页面请求都拖慢, 而且外部服务挂了会连带影响整个站。

    查不到就标"未知", 绝不因此报错: 归属地只是辅助判断, 没有它 IP 和时间
    依然可用。
    """
    cache = _load_geo_cache()
    todo = [ip for ip in {i for i in ips if i}
            if ip not in cache and not is_private(ip)]
    if todo:
        try:
            import urllib.request

            # ip-api.com 免费接口, 无需 key, 批量上限 100 个
            for i in range(0, len(todo), 100):
                chunk = todo[i:i + 100]
                body = json.dumps(
                    [{"query": ip, "fields": "query,status,country,regionName,city,isp"}
                     for ip in chunk]).encode()
                rq = urllib.request.Request(
                    "http://ip-api.com/batch", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(rq, timeout=timeout) as resp:
                    for item in json.loads(resp.read().decode()):
                        ip = item.get("query")
                        if not ip:
                            continue
                        if item.get("status") == "success":
                            where = " ".join(x for x in [item.get("country"),
                                                         item.get("regionName"),
                                                         item.get("city")] if x)
                            cache[ip] = {"where": where or "未知",
                                         "isp": item.get("isp") or "",
                                         "at": _now().isoformat(timespec="seconds")}
                        else:
                            cache[ip] = {"where": "查不到", "isp": "",
                                         "at": _now().isoformat(timespec="seconds")}
            _write_json_atomic(GEO_CACHE_PATH, cache)
        except Exception as e:
            # 离线或接口挂了: 已缓存的照样返回, 未知的留给下次
            cache.setdefault("_error", {"msg": f"{type(e).__name__}: {e}"[:200]})

    out = {}
    for ip in {i for i in ips if i}:
        if is_private(ip):
            out[ip] = {"where": "内网/本机", "isp": ""}
        else:
            g = cache.get(ip)
            out[ip] = {"where": (g or {}).get("where", "未解析"),
                       "isp": (g or {}).get("isp", "")}
    return out


# ══════════════════════════════════════════════════════════
# 查询
# ══════════════════════════════════════════════════════════
def read_events(limit=300, only_pages=False, exclude_ips=()):
    """倒序读事件流。文件按行读, 只保留尾部 limit 条。"""
    rows = []
    for p in [LOG_PATH, LOG_PATH.with_suffix(".jsonl.1")]:
        if not p.exists():
            continue
        try:
            with p.open(encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        for ln in reversed(lines):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if only_pages and str(r.get("p", "")).startswith("/api/"):
                continue
            if r.get("ip") in exclude_ips:
                continue
            rows.append(r)
            if len(rows) >= limit:
                return rows
    return rows


def summary(days=30):
    """总量/独立 IP/今日, 以及按 IP 汇总的排行"""
    st = _load_stats()
    all_days = sorted(st["days"].keys())
    cutoff = str(date.today() - timedelta(days=days))
    recent = {d: v for d, v in st["days"].items() if d >= cutoff}

    per_ip, per_day = {}, {}
    for d, v in sorted(recent.items()):
        per_day[d] = v.get("total", 0)
        for ip, n in (v.get("ips") or {}).items():
            per_ip[ip] = per_ip.get(ip, 0) + n

    today = str(date.today())
    ext = {ip: n for ip, n in per_ip.items() if not is_private(ip)}

    # 密码尝试单独统计: 口令弱时, 这个数字是唯一能发现"有人在猜"的东西。
    # 走事件流而不是计数表, 因为需要区分成功与失败的状态码。
    fails, fail_ips = 0, {}
    for r in read_events(limit=4000):
        if r.get("p") in LOGIN_PATHS and int(r.get("s") or 0) in (401, 403, 429):
            fails += 1
            fail_ips[r.get("ip")] = fail_ips.get(r.get("ip"), 0) + 1

    return {
        "failed_logins": fails,
        "failed_login_ips": sorted(fail_ips.items(), key=lambda kv: -kv[1])[:10],
        "window_days": days,
        "total": sum(per_day.values()),
        "today": recent.get(today, {}).get("total", 0),
        "unique_ips": len(per_ip),
        "unique_external_ips": len(ext),
        "first_day": all_days[0] if all_days else None,
        "per_day": per_day,
        "top_ips": sorted(per_ip.items(), key=lambda kv: -kv[1])[:50],
    }


def first_seen_map():
    """每个 IP 第一次出现的日期 —— 判断"这是个新面孔吗"用"""
    st = _load_stats()
    out = {}
    for d in sorted(st["days"].keys()):
        for ip in (st["days"][d].get("ips") or {}):
            out.setdefault(ip, d)
    return out


if __name__ == "__main__":
    s = summary()
    print(f"最近 {s['window_days']} 天: 共 {s['total']} 次访问, "
          f"{s['unique_ips']} 个 IP (其中外部 {s['unique_external_ips']} 个), "
          f"今日 {s['today']} 次")
    ips = [ip for ip, _ in s["top_ips"][:20]]
    geo = resolve_geo(ips)
    fs = first_seen_map()
    print(f"\n{'IP':<18}{'次数':>6}  {'首次':<12}{'归属地'}")
    for ip, n in s["top_ips"][:20]:
        print(f"{ip:<18}{n:>6}  {fs.get(ip, '?'):<12}{geo.get(ip, {}).get('where', '')}")
    print("\n最近事件:")
    for r in read_events(limit=15):
        print(f"  {r['t']}  {r['ip']:<16}{r['m']:<5}{r['p'][:40]:<42}{r['s']}")
