"""实盘信号 Web 服务器 — 轻量 FastAPI

功能:
  GET  /           → 仪表盘 (持仓 + 最新计划 + 操作按钮)
  GET  /api/status → 当前持仓/现金 JSON (秒级, 不跑模型)
  POST /api/signal→ 跑一次信号生成 (后台, ~30s)
  GET  /api/plan   → 最新计划 JSON
  POST /api/sync-template → 导出对账表单 JSON
  POST /api/sync   → 用提交的 JSON 对账覆盖状态
  GET  /api/history→ 历史成交记录

用法:
  python scripts/web_server.py --host 0.0.0.0 --port 8080
"""
import argparse
import asyncio
import base64
import glob
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent))
import access_log  # noqa: E402
from action_page import (ACTION_HTML, _exec_window, build_recommend,  # noqa: E402
                         build_today, list_profiles)
from live_config import (DEFAULT_PROFILE, PROFILES, capital_of,  # noqa: E402
                         display_name, init_args, is_auto, is_locked,
                         set_auto, set_capital, set_name, signal_args,
                         state_file)

# ── 路径 ──
ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "data" / "live"
# /pro 这个运维页只看默认条线 (四条线的对比看首页)
STATE_PATH = LIVE_DIR / state_file(DEFAULT_PROFILE)
KLINE_DIR = ROOT / "data" / "raw" / "kline"
PROC_DIR = ROOT / "data" / "processed"

app = FastAPI(title="实盘信号")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
# 全站只读鉴权
# ══════════════════════════════════════════════════════════════
# 起因: 这个站 uvicorn 直接监听 0.0.0.0:8080, 实测从非校园网 IP 能直连,
# 而在此之前只有"改账"要密码, "看"是完全开放的 —— 网址一旦传出去,
# 任何人都能看到全部持仓、资金和收益率。
#
# 三道口令各管一段, 互不通用 (签名密钥掺入不同的用途字符串):
#   QUANT_VIEW_PASSWORD  能不能打开这个站            30 天
#   QUANT_OPS_PASSWORD   能不能改账                  12 小时
#   QUANT_BT_PASSWORD    能不能看回测页              7 天
# 能看不等于能改, 所以即使登录了, 改账仍要单独输 ops 口令。
VIEW_COOKIE = "view_token"
VIEW_TTL = 30 * 24 * 3600

# 这些路径不需要登录。必须严格控制在"登录本身所需"和"不含任何数据"的范围内。
VIEW_PUBLIC = ("/login", "/api/view/login", "/api/view/status", "/favicon.ico")

# 防爆破。当前口令是 6 位纯数字(共 100 万种), 字典跑得动, 所以限得比
# 一般情况紧: 5 次/30 分钟使单 IP 每天最多试 240 次, 穷举完要十年量级。
# 计数只在内存里, 重启即清 —— 对付脚本足够, 也不致于把自己永久锁死。
LOGIN_MAX_FAILS = 5
LOGIN_WINDOW = 30 * 60
LOGIN_FAIL_DELAY = 0.7       # 每次失败故意拖一下, 拖死高频脚本
LOGIN_HINT_AFTER = 3         # 错这么多次后告知还剩几次, 免得自己被锁了还不知道为何
_login_fails = {}


def _view_password():
    return os.environ.get("QUANT_VIEW_PASSWORD") or ""


def _view_sign(exp: int) -> str:
    key = ("view:" + _view_password()).encode()
    return hmac.new(key, str(exp).encode(), hashlib.sha256).hexdigest()[:32]


def _view_ok(req: Request) -> bool:
    if not _view_password():
        return False
    tok = req.cookies.get(VIEW_COOKIE, "")
    if "." not in tok:
        return False
    exp_s, sig = tok.rsplit(".", 1)
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < time.time():
        return False
    return hmac.compare_digest(sig, _view_sign(exp))


def _client_ip(req: Request) -> str:
    """真实来源只认 TCP 对端。

    前面没有可信代理, 所以 X-Forwarded-For 是访客自己就能伪造的, 绝不能
    用它当来源 —— 否则日志里的 IP 全都可以被随意编造, 这个功能也就没用了。
    """
    return req.client.host if req.client else ""


def _login_blocked(ip):
    rec = _login_fails.get(ip)
    if not rec:
        return False
    n, first = rec
    if time.time() - first > LOGIN_WINDOW:
        _login_fails.pop(ip, None)
        return False
    return n >= LOGIN_MAX_FAILS


def _login_fail(ip):
    n, first = _login_fails.get(ip, (0, time.time()))
    if time.time() - first > LOGIN_WINDOW:
        n, first = 0, time.time()
    _login_fails[ip] = (n + 1, first)


@app.middleware("http")
async def _view_gate(request: Request, call_next):
    """未登录一律拦住, 并记录每次访问。

    未登录时直接返回登录页(带 401)而不是 302 跳转 —— 跳转在 fetch 里会变成
    "拿到一坨 HTML"这种难查的现象, 也容易和前端路由绕成循环。
    """
    path = request.url.path
    authed = _view_ok(request)
    public = path in VIEW_PUBLIC
    try:
        if public or authed:
            resp = await call_next(request)
        elif not _view_password():
            # 未配置口令 -> 关站, 而不是放行。与改账口令同一个取舍:
            # 宁可用不了, 不可裸奔。
            resp = HTMLResponse(NO_PASSWORD_HTML, status_code=503)
        elif path.startswith("/api/"):
            resp = JSONResponse({"error": "需要登录", "need_login": True},
                                status_code=401)
        else:
            resp = HTMLResponse(LOGIN_HTML, status_code=401)
    finally:
        pass
    access_log.record(_client_ip(request), request.method, path,
                      getattr(resp, "status_code", 0),
                      request.headers.get("user-agent", ""), authed,
                      request.headers.get("x-forwarded-for", ""))
    return resp


@app.post("/api/view/login")
async def api_view_login(req: Request):
    pw = _view_password()
    ip = _client_ip(req)
    if not pw:
        return JSONResponse({"error": "服务器未设置 QUANT_VIEW_PASSWORD"},
                            status_code=503)
    if _login_blocked(ip):
        return JSONResponse(
            {"error": f"尝试过多, 请 {LOGIN_WINDOW // 60} 分钟后再试"},
            status_code=429)
    body = await req.json()
    got = str(body.get("password", ""))
    if not hmac.compare_digest(got, pw):
        _login_fail(ip)
        await asyncio.sleep(LOGIN_FAIL_DELAY)
        n = _login_fails.get(ip, (0, 0))[0]
        left = LOGIN_MAX_FAILS - n
        msg = "口令错误"
        if 0 < left <= LOGIN_MAX_FAILS - LOGIN_HINT_AFTER:
            msg += f", 还可试 {left} 次"
        return JSONResponse({"error": msg}, status_code=403)
    _login_fails.pop(ip, None)
    exp = int(time.time()) + VIEW_TTL
    r = JSONResponse({"ok": True})
    r.set_cookie(VIEW_COOKIE, f"{exp}.{_view_sign(exp)}", max_age=VIEW_TTL,
                 httponly=True, samesite="lax", path="/")
    return r


@app.post("/api/view/logout")
async def api_view_logout():
    r = JSONResponse({"ok": True})
    r.delete_cookie(VIEW_COOKIE, path="/")
    return r


@app.get("/api/view/status")
async def api_view_status(req: Request):
    return {"authed": _view_ok(req), "configured": bool(_view_password())}


LOGIN_HTML = """<!DOCTYPE html><html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>登录</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0b0d10;color:#e6e8eb;min-height:100vh;display:flex;
     align-items:center;justify-content:center;padding:20px;
     font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
.box{background:#14171e;border-radius:16px;padding:30px 24px;width:100%;max-width:360px}
h1{font-size:19px;margin-bottom:6px}
.sub{font-size:13px;color:#8a93a6;line-height:1.7;margin-bottom:20px}
input{width:100%;background:#0f1216;border:1px solid #2a2f3a;color:#e6e8eb;
      border-radius:10px;padding:13px 14px;font-size:17px;outline:none}
input:focus{border-color:#2563eb}
.btn{width:100%;margin-top:12px;background:#2563eb;color:#fff;border:none;
     border-radius:10px;padding:13px;font-size:16px;font-weight:600;cursor:pointer}
.btn:disabled{opacity:.5}
.err{color:#fca5a5;font-size:13px;margin-top:12px;min-height:18px}
.disc{color:#6b7280;font-size:12px;line-height:1.7;margin-top:16px;text-align:center}
</style></head><body>
<div class="box">
  <h1>实盘看板</h1>
  <div class="sub">这个站会显示真实持仓与资金，需要口令才能查看。</div>
  <input id="pw" type="password" placeholder="查看口令" autocomplete="current-password">
  <button class="btn" id="go" onclick="go()">进入</button>
  <div class="err" id="err"></div>
  <div class="disc">本站信息仅供参考，不构成任何投资建议</div>
</div>
<script>
const $ = s => document.querySelector(s);
$('#pw').addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
$('#pw').focus();
async function go(){
  const pw = $('#pw').value;
  if (!pw) return;
  $('#go').disabled = true; $('#err').textContent = '';
  try {
    const r = await fetch('/api/view/login', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({password: pw})});
    const d = await r.json().catch(()=>({}));
    if (r.ok) { location.replace('/'); return; }
    $('#err').textContent = d.error || ('登录失败 (' + r.status + ')');
  } catch(e) {
    $('#err').textContent = '无法连接服务器';
  }
  $('#go').disabled = false;
}
</script></body></html>"""


# 未配置口令时显示这个, 而不是放行。说清怎么修, 免得只看到一个 503 干瞪眼。
NO_PASSWORD_HTML = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>未配置口令</title>
<style>body{background:#0b0d10;color:#e6e8eb;font-family:-apple-system,"PingFang SC",sans-serif;
padding:30px;line-height:1.9;font-size:14px}code{background:#1c2029;padding:2px 6px;
border-radius:4px;color:#fcd34d}h1{font-size:18px;margin-bottom:14px}
.w{max-width:620px;margin:0 auto}pre{background:#14171e;padding:14px;border-radius:10px;
overflow-x:auto;font-size:13px;color:#c9cdd6}</style></head><body><div class="w">
<h1>服务器未设置 QUANT_VIEW_PASSWORD</h1>
<p>这个站会显示真实持仓与资金，所以没有口令时<b>整站关闭</b>，而不是放开访问。</p>
<pre>ssh eez041.ece.ust.hk
echo 'QUANT_VIEW_PASSWORD=想用的口令' &gt;&gt; ~/.config/quant-web.env
systemctl --user restart quant-web.service</pre>
</div></body></html>"""


@app.get("/login", response_class=HTMLResponse)
async def page_login():
    return LOGIN_HTML


# ══════════════════════════════════════════════════════════════
# 访问日志查询 (用改账口令保护)
# ══════════════════════════════════════════════════════════════
# 放在 ops 口令后面而不是仅靠"已登录": 访问日志里有 IP 和归属地,
# 属于比持仓更该少露的东西。


@app.get("/api/access/summary")
async def api_access_summary(req: Request, days: int = 30, geo: int = 1):
    if not _pro_ops_ok(req):
        return _pro_ops_deny()
    days = max(1, min(int(days or 30), 90))
    s = access_log.summary(days)
    first = access_log.first_seen_map()
    ips = [ip for ip, _ in s["top_ips"]]
    # geo=0 供离线/不想外发 IP 时使用
    g = access_log.resolve_geo(ips) if geo else {}
    s["ips"] = [{"ip": ip, "hits": n, "first_seen": first.get(ip),
                 "where": (g.get(ip) or {}).get("where", "未解析"),
                 "isp": (g.get(ip) or {}).get("isp", ""),
                 "private": access_log.is_private(ip)}
                for ip, n in s["top_ips"]]
    s.pop("top_ips", None)
    return s


@app.get("/api/access/events")
async def api_access_events(req: Request, limit: int = 200, pages: int = 0,
                            geo: int = 1):
    if not _pro_ops_ok(req):
        return _pro_ops_deny()
    limit = max(1, min(int(limit or 200), 1000))
    rows = access_log.read_events(limit=limit, only_pages=bool(pages))
    g = access_log.resolve_geo([r.get("ip") for r in rows]) if geo else {}
    for r in rows:
        r["where"] = (g.get(r.get("ip")) or {}).get("where", "未解析")
    return {"events": rows, "dedupe_seconds": access_log.DEDUPE_SECONDS}

# ══════════════════════════════════════════════════════════════
# 写操作鉴权
# ══════════════════════════════════════════════════════════════
# 改名/切记账方式/校准现金/出入金 都会改动账目, 需要口令。
# 与回测页 (QUANT_BT_PASSWORD) 用不同的口令和不同的 cookie —— 能看报表
# 不等于能改账。口令只从环境变量读, 绝不写进仓库。
OPS_COOKIE = "ops_token"
OPS_TTL = 12 * 3600          # 12 小时后要重新输, 比回测页的 7 天短得多


def _ops_password():
    return os.environ.get("QUANT_OPS_PASSWORD") or ""


def _ops_sign(exp: int) -> str:
    # 签名密钥掺入用途字符串, 使回测页的 token 无法当作写权限使用
    key = ("ops:" + _ops_password()).encode()
    return hmac.new(key, str(exp).encode(), hashlib.sha256).hexdigest()[:32]


def _ops_ok(req: Request) -> bool:
    if not _ops_password():
        return False
    tok = req.cookies.get(OPS_COOKIE, "")
    if "." not in tok:
        return False
    exp_s, sig = tok.split(".", 1)
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < time.time():
        return False
    return hmac.compare_digest(sig, _ops_sign(exp))


def _ops_deny():
    """401 让前端弹出输密码框"""
    if not _ops_password():
        return JSONResponse(
            {"error": "服务器未设置 QUANT_OPS_PASSWORD, 所有改账操作已禁用",
             "need_password": False}, status_code=503)
    return JSONResponse({"error": "需要密码", "need_password": True}, status_code=401)


def _archive_profile(pid):
    """重置前把旧状态和旧计划挪进 archive/, 而不是直接删。

    重置等于把这条线的历史业绩清零, 是所有改账操作里最重的一个。留一份
    归档, 出问题时还能回溯"重置前到底是什么样"。

    状态用复制(留原件给 --init 覆盖), 计划用移动(旧计划必须从 live/ 消失,
    否则页面会把上一轮的挂单当成这一轮的)。
    """
    arc = LIVE_DIR / "archive"
    arc.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"state": None, "plans": 0}
    src = LIVE_DIR / state_file(pid)
    if src.exists():
        dst = arc / f"{src.stem}_{ts}.json"
        shutil.copy2(src, dst)
        out["state"] = dst.name
    for pp in LIVE_DIR.glob(f"plan_{pid}_*.json"):
        shutil.move(str(pp), str(arc / f"{pp.stem}_{ts}.json"))
        out["plans"] += 1
    return out


def _check_profile(pid, write=True):
    """校验条线 id; write=True 时还要拦住基准线。

    返回 None 表示通过, 否则返回该直接回给前端的错误响应。
    基准线的意义就是"没人动过", 所以拦在后端而不是只靠前端隐藏按钮 ——
    前端藏起来的按钮, 用 curl 一样能打到接口。
    """
    if pid not in PROFILES:
        return JSONResponse({"error": f"未知条线 {pid}"}, status_code=400)
    if write and is_locked(pid):
        return JSONResponse(
            {"error": f"{display_name(pid)} 是基准线, 不接受任何修改。"
                      f"它的作用是提供一条没人动过的参照, 用来衡量人为干预的代价。",
             "locked": True}, status_code=403)
    return None


@app.post("/api/ops/login")
async def api_ops_login(req: Request):
    pw = _ops_password()
    if not pw:
        return JSONResponse({"error": "服务器未设置 QUANT_OPS_PASSWORD"}, status_code=503)
    body = await req.json()
    got = str(body.get("password", ""))
    if not hmac.compare_digest(got, pw):
        return JSONResponse({"error": "密码错误"}, status_code=403)
    exp = int(time.time()) + OPS_TTL
    r = JSONResponse({"ok": True, "expires_in": OPS_TTL})
    r.set_cookie(OPS_COOKIE, f"{exp}.{_ops_sign(exp)}", max_age=OPS_TTL,
                 httponly=True, samesite="lax", path="/")
    return r


@app.post("/api/ops/logout")
async def api_ops_logout():
    r = JSONResponse({"ok": True})
    r.delete_cookie(OPS_COOKIE, path="/")
    return r


@app.get("/api/ops/status")
async def api_ops_status(req: Request):
    return {"authed": _ops_ok(req), "configured": bool(_ops_password())}


# ── 运维面板专用口令 (QUANT_PRO_OPS_PASSWORD) ──
# /pro 页的写操作(生成信号/同步对账/访问日志)用独立口令,
# 与首页改账口令(QUANT_OPS_PASSWORD)互不通用。
PRO_OPS_COOKIE = "pro_ops_token"
PRO_OPS_TTL = 12 * 3600


def _pro_ops_password():
    return os.environ.get("QUANT_PRO_OPS_PASSWORD") or ""


def _pro_ops_sign(exp: int) -> str:
    key = ("pro-ops:" + _pro_ops_password()).encode()
    return hmac.new(key, str(exp).encode(), hashlib.sha256).hexdigest()[:32]


def _pro_ops_ok(req: Request) -> bool:
    if not _pro_ops_password():
        return False
    tok = req.cookies.get(PRO_OPS_COOKIE, "")
    if "." not in tok:
        return False
    exp_s, sig = tok.split(".", 1)
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < time.time():
        return False
    return hmac.compare_digest(sig, _pro_ops_sign(exp))


def _pro_ops_deny():
    if not _pro_ops_password():
        return JSONResponse(
            {"error": "服务器未设置 QUANT_PRO_OPS_PASSWORD, 运维操作已禁用",
             "need_password": False}, status_code=503)
    return JSONResponse({"error": "需要密码", "need_password": True}, status_code=401)


@app.post("/api/pro-ops/login")
async def api_pro_ops_login(req: Request):
    pw = _pro_ops_password()
    if not pw:
        return JSONResponse({"error": "服务器未设置 QUANT_PRO_OPS_PASSWORD"}, status_code=503)
    body = await req.json()
    got = str(body.get("password", ""))
    if not hmac.compare_digest(got, pw):
        return JSONResponse({"error": "密码错误"}, status_code=403)
    exp = int(time.time()) + PRO_OPS_TTL
    r = JSONResponse({"ok": True, "expires_in": PRO_OPS_TTL})
    r.set_cookie(PRO_OPS_COOKIE, f"{exp}.{_pro_ops_sign(exp)}", max_age=PRO_OPS_TTL,
                 httponly=True, samesite="lax", path="/")
    return r


# ── 全局 ──
PY = sys.executable
SIGNAL_SCRIPT = str(ROOT / "scripts" / "live_signal.py")
_running = {"active": False, "log": "", "started_at": None, "done_at": None,
            "task": None}


def _run_bg(cmd, task, timeout=900):
    """后台跑一个子进程, 进度结果写 _running 供 /api/signal-status 轮询。

    共用同一个 _running 是故意的: live_signal 同时读写状态文件,
    两个实例并行会互相覆盖。单一门闩简单且安全。
    """
    _running.update(active=True, log="", task=task,
                    started_at=datetime.now().isoformat(timespec="seconds"), done_at=None)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(ROOT), timeout=timeout)
        _running["log"] = (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        _running["log"] = f"{task} 失败: {e}"
    finally:
        _running["active"] = False
        _running["done_at"] = datetime.now().isoformat(timespec="seconds")


# 网页上的手动触发/对账默认作用于默认条线; 参数从 live_config 取,
# 必须与 daily_rebuild 完全一致, 否则会写出不同指纹的状态
SIGNAL_ARGS = signal_args(DEFAULT_PROFILE)


def _state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return None


def _state_of(pid):
    """按 profile 读状态; 未建立返回 None"""
    p = LIVE_DIR / state_file(pid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_plan():
    plans = sorted(LIVE_DIR.glob(f"plan_{DEFAULT_PROFILE}_*.json"))
    if not plans:
        return None
    return json.loads(plans[-1].read_text(encoding="utf-8"))


def _names():
    """从 all_stock_list 或 watchlist 加载股票名称"""
    names = {}
    for p in [ROOT / "data" / "raw" / "all_stock_list.parquet",
              ROOT / "data" / "universe" / "watchlist_216.json"]:
        if not p.exists():
            continue
        try:
            if p.suffix == ".parquet":
                import pandas as pd
                df = pd.read_parquet(p)
                for _, r in df.iterrows():
                    code = str(r.get("code", r.get("symbol", "")))[:6]
                    nm = str(r.get("name", r.get("stock_name", "")))
                    if code and nm:
                        names[code] = nm
            else:
                data = json.loads(p.read_text(encoding="utf-8"))
                items = data.get("watchlist", data) if isinstance(data, dict) else data
                for item in items:
                    if isinstance(item, dict):
                        names[str(item.get("code", ""))[:6]] = item.get("name", "")
        except Exception:
            pass
    return names


# ── API ──
@app.get("/api/status")
async def api_status():
    st = _state()
    if not st:
        return JSONResponse({"error": "无状态文件, 请先跑一次信号"}, status_code=404)
    names = _names()
    import pandas as pd
    cal = [pd.Timestamp(x) for x in st.get("calendar", [])]
    ref_date = cal[-1] if cal else None
    positions = []
    mv = 0.0
    for lot in st.get("lots", []):
        code6 = str(lot["code"])[:6]
        ref = lot["buy_price"]
        if ref_date and (KLINE_DIR / f"{code6}.parquet").exists():
            try:
                import pandas as pd
                kl = pd.read_parquet(KLINE_DIR / f"{code6}.parquet")
                kl["date"] = pd.to_datetime(kl["date"])
                row = kl[kl["date"] <= ref_date]
                if len(row):
                    ref = float(row.iloc[-1]["close"])
            except Exception:
                pass
        val = lot["shares"] * ref
        mv += val
        # 两个天数都按交易日算, 但回答不同问题:
        #   held_days   -> 到期时钟, 续持会归零 (决定什么时候动它)
        #   tenure_days -> 真实持有时长, 只增不减 (这笔一共拿了多久)
        held = tenure = None
        if cal and lot.get("open_signal_date"):
            try:
                idx = pd.DatetimeIndex(cal)
                i2 = int(idx.searchsorted(ref_date, side="right")) - 1

                def _pos(d):
                    return int(idx.searchsorted(pd.Timestamp(d), side="right")) - 1

                held = max(0, i2 - _pos(lot["open_signal_date"]))
                first = lot.get("first_open_signal_date") or lot["open_signal_date"]
                tenure = max(0, i2 - _pos(first))
            except Exception:
                pass
        positions.append({
            "code": code6, "name": names.get(code6, ""),
            "shares": lot["shares"], "buy_price": round(lot["buy_price"], 3),
            "last_close": round(ref, 3), "market_value": round(val, 2),
            "pnl_pct": round((ref / lot["buy_price"] - 1) * 100, 2),
            "open_date": lot.get("open_date"), "held_days": held,
            "tenure_days": tenure, "n_rolled": int(lot.get("rolled") or 0),
        })
    cash = st.get("cash", 0)
    equity = cash + mv
    init_cap = st.get("initial_capital", cash)
    return {
        # /pro 只看默认那一条线, 把它回给前端明写在页上 ——
        # 否则容易把这一条的数字当成四条线的全部。
        "profile": DEFAULT_PROFILE,
        "profile_name": display_name(DEFAULT_PROFILE),
        "hold_days": (st.get("config") or {}).get("hold_days"),
        "ref_date": str(ref_date.date()) if ref_date else None,
        "cash": round(cash, 2), "market_value": round(mv, 2),
        "equity": round(equity, 2),
        "initial_capital": init_cap,
        "total_return_pct": round(equity / init_cap * 100 - 100, 2) if init_cap else None,
        "positions": positions,
        "pending": st.get("pending"),
        "last_signal_date": st.get("last_signal_date"),
        "last_rebal_signal_date": st.get("last_rebal_signal_date"),
        "last_synced_at": st.get("last_synced_at"),
        "running": _running["active"],
    }


@app.get("/api/plan")
async def api_plan():
    p = _latest_plan()
    if not p:
        return JSONResponse({"error": "尚无计划文件"}, status_code=404)
    # 计划里的 exec_hint 是生成那一刻写下的"下一交易日尾盘", 第二天看就变成
    # 误导。这里按"现在"重算一份, 与首页同一个函数, 两个页面不会口径不一。
    return {**p, "exec_window": _exec_window(p.get("config") or {}, p)}


@app.post("/api/signal")
async def api_signal(req: Request):
    # 跑信号会写 state (结算挂单、生成新计划), 属于改账 —— 与首页那批
    # /api/profile/* 一样得过 ops 口令。之前这里没校验, 等于"能看就能改"。
    if not _pro_ops_ok(req):
        return _pro_ops_deny()
    if _running["active"]:
        return JSONResponse({"error": "信号生成正在运行中, 请等待"}, status_code=409)
    _running.update(active=True, log="", started_at=datetime.now().isoformat(),
                    done_at=None)
    import threading
    def _run():
        try:
            r = subprocess.run(
                [PY, SIGNAL_SCRIPT] + SIGNAL_ARGS,
                capture_output=True, text=True, cwd=str(ROOT), timeout=120,
            )
            _running["log"] = r.stdout + r.stderr
        except Exception as e:
            _running["log"] = str(e)
        finally:
            _running["active"] = False
            _running["done_at"] = datetime.now().isoformat()
    threading.Thread(target=_run, daemon=True).start()
    return {"message": "信号生成已启动, 约30秒完成", "started_at": _running["started_at"]}


@app.get("/api/signal-status")
async def api_signal_status():
    return {
        "active": _running["active"],
        "started_at": _running["started_at"],
        "done_at": _running["done_at"],
        "log": _running["log"][-3000:] if _running["log"] else "",
    }


# 人工对账: 网页入口已下掉 —— 它一把覆盖整个账户状态, 而首页的
# 确认成交/现金校准/删持仓/重置 四个操作语义各自明确且都有备份与
# history 记录, 已完全取代它。接口保留作为应急通道(一次性批量改写),
# 但必须要 ops 口令 —— 它是所有接口里破坏力最大的一个。
@app.post("/api/sync-template")
async def api_sync_template(req: Request):
    if not _pro_ops_ok(req):
        return _pro_ops_deny()
    r = subprocess.run(
        [PY, SIGNAL_SCRIPT] + SIGNAL_ARGS + ["--sync-template"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=30,
    )
    tpl_path = LIVE_DIR / "sync_template.json"
    if tpl_path.exists():
        return json.loads(tpl_path.read_text(encoding="utf-8"))
    return JSONResponse({"error": r.stderr or "导出失败"}, status_code=500)


@app.post("/api/sync")
async def api_sync(req: Request):
    if not _pro_ops_ok(req):
        return _pro_ops_deny()
    body = await req.json()
    tmp = LIVE_DIR / "sync_input.json"
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    r = subprocess.run(
        [PY, SIGNAL_SCRIPT] + SIGNAL_ARGS + ["--sync", str(tmp)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=30,
    )
    if r.returncode != 0:
        return JSONResponse({"error": r.stderr or r.stdout}, status_code=500)
    return {"message": "对账完成", "output": r.stdout}


@app.get("/api/pipeline")
async def api_pipeline():
    """每日重建流水线状态 + 数据新鲜度"""
    p = LIVE_DIR / "pipeline_status.json"
    out = {}
    if p.exists():
        try:
            out = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            out = {"error": str(e)}
    # 训练集实际最新日 (流水线状态之外再独立核对一次)
    try:
        import pandas as pd
        tp = ROOT / "data" / "processed" / "training_data_pit_v24.parquet"
        if tp.exists():
            d = pd.read_parquet(tp, columns=["date"])["date"].max()
            out["train_max_date"] = str(pd.Timestamp(d).date())
    except Exception:
        pass
    return out


# ═══════════════════════════════════════════════════════════════
# 回测页 (隐藏入口 /backtest, 需密码)
# ═══════════════════════════════════════════════════════════════
BT_COOKIE = "bt_token"
BT_TTL = 7 * 24 * 3600     # 登录有效期 7 天


def _bt_password():
    """口令只从环境变量读, 绝不写进仓库; 未设置则整个回测页关闭"""
    return os.environ.get("QUANT_BT_PASSWORD") or ""


def _bt_sign(exp: int) -> str:
    key = _bt_password().encode()
    return hmac.new(key, str(exp).encode(), hashlib.sha256).hexdigest()[:32]


def _bt_make_token() -> str:
    exp = int(time.time()) + BT_TTL
    return f"{exp}.{_bt_sign(exp)}"


def _bt_ok(req: Request) -> bool:
    pw = _bt_password()
    if not pw:
        return False
    tok = req.cookies.get(BT_COOKIE, "")
    if "." not in tok:
        return False
    exp_s, sig = tok.rsplit(".", 1)
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < time.time():
        return False
    return hmac.compare_digest(sig, _bt_sign(exp))


def _bt_deny():
    return JSONResponse({"error": "未授权"}, status_code=401)


@app.post("/api/bt/login")
async def api_bt_login(req: Request):
    pw = _bt_password()
    if not pw:
        return JSONResponse(
            {"error": "服务器未设置 QUANT_BT_PASSWORD, 回测页已禁用"}, status_code=503)
    body = await req.json()
    got = str(body.get("password", ""))
    # compare_digest 防时序侧信道
    if not hmac.compare_digest(got, pw):
        return JSONResponse({"error": "密码错误"}, status_code=403)
    r = JSONResponse({"message": "ok"})
    r.set_cookie(BT_COOKIE, _bt_make_token(), max_age=BT_TTL,
                 httponly=True, samesite="lax", path="/")
    return r


@app.post("/api/bt/logout")
async def api_bt_logout():
    r = JSONResponse({"message": "ok"})
    r.delete_cookie(BT_COOKIE, path="/")
    return r


def _bt_runs():
    """扫描所有回测结果 json, 按修改时间倒序"""
    out = []
    for f in glob.glob(str(PROC_DIR / "wf_daily_*.json")):
        p = Path(f)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        s = d.get("summary", {})
        out.append({
            "name": p.stem,
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            "period": d.get("period"),
            "train_file": d.get("train_file"),
            "regime_filter": d.get("regime_filter"),
            "hold_days": d.get("hold_days"),
            "target_positions": d.get("target_positions"),
            "features": d.get("features"),
            "initial_capital": d.get("initial_capital"),
            "total_return_pct": s.get("total_return_pct"),
            "annualized_return_pct": s.get("annualized_return_pct"),
            "sharpe": s.get("sharpe"),
            "max_dd_pct": s.get("max_dd_pct"),
            "benchmark_total_pct": s.get("benchmark_total_pct"),
            "excess_annual_pct": s.get("excess_annual_pct"),
            "information_ratio": s.get("information_ratio"),
            "alpha_annual_pct": s.get("alpha_annual_pct"),
            "ic_mean": s.get("ic_mean"),
            "ic_tstat": s.get("ic_tstat"),
            "n_trades": s.get("n_trades"),
            "cash_days_pct": s.get("cash_days_pct"),
            "beat_benchmark": s.get("beat_benchmark"),
            "beat_both_halves": s.get("beat_both_halves"),
        })
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


@app.get("/api/bt/list")
async def api_bt_list(req: Request):
    if not _bt_ok(req):
        return _bt_deny()
    return _bt_runs()


@app.get("/api/bt/detail")
async def api_bt_detail(req: Request, name: str):
    if not _bt_ok(req):
        return _bt_deny()
    p = PROC_DIR / f"{Path(name).name}.json"       # 防目录穿越
    if not p.exists():
        return JSONResponse({"error": "找不到该回测"}, status_code=404)
    d = json.loads(p.read_text(encoding="utf-8"))
    daily = d.get("daily", [])
    cap = d.get("initial_capital") or 1.0
    curve = [{"d": x.get("date"),
              "v": round((x.get("portfolio_value") or 0) / cap, 4),
              "c": int(bool(x.get("in_cash")))} for x in daily]

    # 分年度: 策略 vs 基准(用 daily 里的 bench_ret 若有, 否则跳过)
    yearly = {}
    for x in daily:
        y = str(x.get("date", ""))[:4]
        if not y:
            continue
        a = yearly.setdefault(y, {"year": y, "days": 0, "s": 1.0, "cash": 0})
        a["days"] += 1
        a["s"] *= 1 + (x.get("daily_ret") or 0)
        a["cash"] += int(bool(x.get("in_cash")))
    for a in yearly.values():
        a["strategy_pct"] = round((a.pop("s") - 1) * 100, 2)

    return {
        "name": p.stem,
        "config": {k: d.get(k) for k in
                   ("label", "exec_mode", "slippage", "trade_cost", "portfolio_mode",
                    "hold_days", "tranche_n", "target_positions", "train_file",
                    "pit_universe", "regime_filter", "regime_ma", "regime_breadth",
                    "regime_confirm", "feat_select_cutoff", "period", "n_days",
                    "initial_capital", "features")},
        "summary": d.get("summary", {}),
        "stability": d.get("stability", []),
        "curve": curve,
        "yearly": sorted(yearly.values(), key=lambda z: z["year"]),
        "selected_features": d.get("selected_features", []),
        "trades": d.get("trades", [])[-300:],
        "n_trades_total": len(d.get("trades", [])),
    }


@app.get("/api/bt/xlsx")
async def api_bt_xlsx(req: Request, name: str):
    """按需生成该回测的 Excel 并下载"""
    if not _bt_ok(req):
        return _bt_deny()
    stem = Path(name).name
    src = PROC_DIR / f"{stem}.json"
    if not src.exists():
        return JSONResponse({"error": "找不到该回测"}, status_code=404)
    out = PROC_DIR / f"bt_{stem}.xlsx"
    # json 比 xlsx 新时才重新生成, 避免每次点击都重算基准
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        r = subprocess.run(
            [PY, str(ROOT / "scripts" / "export_backtest_excel.py"),
             "--pattern", str(src), "--out", str(out)],
            capture_output=True, text=True, cwd=str(ROOT), timeout=600, env=env)
        if r.returncode != 0 or not out.exists():
            return JSONResponse({"error": (r.stderr or r.stdout)[-1500:]}, status_code=500)
    return FileResponse(str(out), filename=out.name,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


BACKTEST_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>回测</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;padding:12px;font-size:14px}
h1{font-size:19px;margin-bottom:12px}
.card{background:#1c1c1c;border-radius:10px;padding:14px;margin-bottom:12px}
.card-title{font-size:13px;color:#888;margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px}
select,input{width:100%;padding:10px;background:#252525;border:1px solid #333;border-radius:6px;color:#e0e0e0;font-size:15px}
.btn{padding:10px 16px;border:none;border-radius:6px;font-size:14px;cursor:pointer;margin-right:8px;margin-top:8px}
.btn-primary{background:#1976d2;color:#fff}
.btn-plain{background:#333;color:#ccc}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 6px;text-align:right;border-bottom:1px solid #262626;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:#888;font-weight:500}
.green{color:#4caf50}.red{color:#f44336}.gray{color:#888}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:10px}
.kv{background:#232323;border-radius:8px;padding:10px}
.kv .k{font-size:11px;color:#888}
.kv .v{font-size:17px;font-weight:600;margin-top:3px}
.pill{display:inline-block;background:#252525;border-radius:11px;padding:3px 9px;font-size:12px;color:#bbb;margin:2px 4px 2px 0}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px}
.badge-green{background:#1b5e20;color:#a5d6a7}
.badge-red{background:#b71c1c;color:#ffcdd2}
.warn{background:#3a2a00;border-left:3px solid #ff9800;padding:10px;border-radius:6px;font-size:13px;color:#ffcc80;margin-bottom:12px}
.table-wrap{overflow-x:auto}
.empty{color:#666;text-align:center;padding:14px}
#login{max-width:330px;margin:56px auto}
.err{color:#f44336;font-size:13px;margin-top:8px}
.feat{font-size:12px;color:#aaa;line-height:1.9}
</style></head><body>

<div id="login" style="display:none">
  <div class="card">
    <div class="card-title">回测报表 · 需要密码</div>
    <input type="password" id="pw" placeholder="密码" onkeydown="if(event.key==='Enter')doLogin()">
    <button class="btn btn-primary" onclick="doLogin()">进入</button>
    <div class="err" id="login-err"></div>
  </div>
</div>

<div id="main" style="display:none">
  <h1>回测报表</h1>

  <div class="card">
    <div class="card-title">选择回测</div>
    <select id="sel" onchange="loadDetail()"></select>
    <button class="btn btn-primary" onclick="dl()">下载 Excel</button>
    <button class="btn btn-plain" onclick="doLogout()">退出</button>
  </div>

  <div id="body"></div>
</div>

<script>
const $ = id => document.getElementById(id);

async function jget(u){ const r = await fetch(u); if(r.status===401) throw 'unauth'; return r.json(); }

async function boot(){
  try{
    const runs = await jget('/api/bt/list');
    $('login').style.display='none'; $('main').style.display='';
    if(!runs.length){ $('body').innerHTML='<div class="card empty">没有回测结果文件</div>'; return; }
    $('sel').innerHTML = runs.map(r=>{
      const t = (r.total_return_pct>=0?'+':'')+r.total_return_pct+'%';
      return `<option value="${r.name}">${r.name}  [${t}, 夏普${r.sharpe}]</option>`;
    }).join('');
    loadDetail();
  }catch(e){
    $('main').style.display='none'; $('login').style.display='';
  }
}

async function doLogin(){
  $('login-err').textContent='';
  const r = await fetch('/api/bt/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({password:$('pw').value})});
  if(r.ok){ $('pw').value=''; boot(); }
  else { const d = await r.json(); $('login-err').textContent = d.error||'登录失败'; }
}

async function doLogout(){
  await fetch('/api/bt/logout',{method:'POST'});
  location.reload();
}

function dl(){ window.location = '/api/bt/xlsx?name=' + encodeURIComponent($('sel').value); }

function num(v,d=2,suf=''){ return (v==null||isNaN(v)) ? '--' : (+v).toFixed(d)+suf; }
// 收益类数字按 A股习惯: 红涨绿跌 (与欧美相反)。
// 下面的 badge-green/red 是"跑赢/跑输""成功/失败"这类状态语义, 不跟着变。
function sgn(v,d=1,suf='%'){ if(v==null||isNaN(v))return '<span class="gray">--</span>';
  const c = v>=0?'red':'green'; return `<span class="${c}">${v>=0?'+':''}${(+v).toFixed(d)}${suf}</span>`; }

// 纯 SVG 折线, 不引外部图表库
function curveSVG(curve){
  if(!curve||curve.length<2) return '<div class="empty">无曲线数据</div>';
  const W=680,H=210,PL=44,PR=10,PT=10,PB=22;
  const vs=curve.map(p=>p.v), lo=Math.min(...vs,1), hi=Math.max(...vs,1);
  const pad=(hi-lo)*0.08||0.05, y0=lo-pad, y1=hi+pad;
  const X=i=>PL+(W-PL-PR)*i/(curve.length-1);
  const Y=v=>PT+(H-PT-PB)*(1-(v-y0)/(y1-y0));
  const pts=curve.map((p,i)=>`${X(i).toFixed(1)},${Y(p.v).toFixed(1)}`).join(' ');
  // 空仓区间画灰底
  let bands='',st=null;
  curve.forEach((p,i)=>{
    if(p.c&&st===null) st=i;
    if((!p.c||i===curve.length-1)&&st!==null){
      bands+=`<rect x="${X(st).toFixed(1)}" y="${PT}" width="${Math.max(1,X(i)-X(st)).toFixed(1)}" height="${H-PT-PB}" fill="#ffffff" opacity="0.05"/>`;
      st=null;
    }
  });
  let grid='';
  for(let k=0;k<=4;k++){
    const v=y0+(y1-y0)*k/4, y=Y(v);
    grid+=`<line x1="${PL}" y1="${y.toFixed(1)}" x2="${W-PR}" y2="${y.toFixed(1)}" stroke="#2a2a2a"/>`
        +`<text x="${PL-6}" y="${(y+3).toFixed(1)}" fill="#777" font-size="10" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  const one=(y0<=1&&1<=y1)?`<line x1="${PL}" y1="${Y(1).toFixed(1)}" x2="${W-PR}" y2="${Y(1).toFixed(1)}" stroke="#666" stroke-dasharray="3,3"/>`:'';
  const lab=[0,Math.floor(curve.length/2),curve.length-1].map(i=>
    `<text x="${X(i).toFixed(1)}" y="${H-6}" fill="#777" font-size="10" text-anchor="middle">${curve[i].d}</text>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">${grid}${bands}${one}
    <polyline points="${pts}" fill="none" stroke="#42a5f5" stroke-width="1.6"/>${lab}</svg>
    <div style="font-size:11px;color:#777;margin-top:4px">灰底=空仓区间 · 虚线=本金线(净值1.0)</div>`;
}

async function loadDetail(){
  const name=$('sel').value;
  $('body').innerHTML='<div class="card empty">加载中...</div>';
  let d;
  try{ d = await jget('/api/bt/detail?name='+encodeURIComponent(name)); }
  catch(e){ location.reload(); return; }
  const s=d.summary||{}, c=d.config||{};

  const suspicious = (s.excess_annual_pct>30) || (s.sharpe>1.5&&s.information_ratio>1.2);
  let html='';
  if(suspicious) html+=`<div class="warn">这份回测超额年化 ${num(s.excess_annual_pct,1)}% / IR ${num(s.information_ratio,2)}，
    高得不正常。历史上此类结果曾由基本面数据缺陷造成，请先确认数据口径再采信。</div>`;

  html+=`<div class="card"><div class="card-title">核心指标</div><div class="grid">
    <div class="kv"><div class="k">总收益</div><div class="v">${sgn(s.total_return_pct)}</div></div>
    <div class="kv"><div class="k">年化</div><div class="v">${sgn(s.annualized_return_pct)}</div></div>
    <div class="kv"><div class="k">夏普</div><div class="v">${num(s.sharpe)}</div></div>
    <div class="kv"><div class="k">最大回撤</div><div class="v">${sgn(s.max_dd_pct)}</div></div>
    <div class="kv"><div class="k">基准总收益</div><div class="v">${sgn(s.benchmark_total_pct)}</div></div>
    <div class="kv"><div class="k">超额年化</div><div class="v">${sgn(s.excess_annual_pct)}</div></div>
    <div class="kv"><div class="k">信息比率</div><div class="v">${num(s.information_ratio)}</div></div>
    <div class="kv"><div class="k">年化alpha</div><div class="v">${sgn(s.alpha_annual_pct)}</div></div>
    <div class="kv"><div class="k">IC均值</div><div class="v">${num(s.ic_mean,4)}</div></div>
    <div class="kv"><div class="k">IC t值</div><div class="v">${num(s.ic_tstat)}</div></div>
    <div class="kv"><div class="k">交易笔数</div><div class="v">${s.n_trades??'--'}</div></div>
    <div class="kv"><div class="k">空仓占比</div><div class="v">${num(s.cash_days_pct,1,'%')}</div></div>
  </div>
  <div style="margin-top:10px">
    ${s.beat_benchmark?'<span class="badge badge-green">跑赢基准</span>':'<span class="badge badge-red">跑输基准</span>'}
    ${s.beat_both_halves?'<span class="badge badge-green">两段都跑赢</span>':'<span class="badge badge-red">两段未都跑赢</span>'}
  </div></div>`;

  html+=`<div class="card"><div class="card-title">净值曲线 (起点 1.0)</div>${curveSVG(d.curve)}</div>`;

  html+=`<div class="card"><div class="card-title">分年度</div><div class="table-wrap"><table>
    <thead><tr><th>年份</th><th>交易日</th><th>策略</th><th>空仓天数</th></tr></thead><tbody>`
    + (d.yearly||[]).map(y=>`<tr><td>${y.year}</td><td>${y.days}</td>
        <td>${sgn(y.strategy_pct)}</td><td>${y.cash}</td></tr>`).join('')
    + `</tbody></table></div></div>`;

  if(d.stability&&d.stability.length){
    html+=`<div class="card"><div class="card-title">分段稳健性</div><div class="table-wrap"><table>
      <thead><tr><th>区间</th><th>策略</th><th>基准</th><th>超额年化</th><th>IR</th></tr></thead><tbody>`
      + d.stability.map(h=>`<tr><td>${h.period||h.half||''}</td>
          <td>${sgn(h.strategy_pct??h.strategy)}</td><td>${sgn(h.benchmark_pct??h.benchmark)}</td>
          <td>${sgn(h.excess_annual_pct)}</td><td>${num(h.information_ratio)}</td></tr>`).join('')
      + `</tbody></table></div></div>`;
  }

  html+=`<div class="card"><div class="card-title">配置</div><div>`
    + Object.entries(c).map(([k,v])=>`<span class="pill">${k}: ${v}</span>`).join('')
    + `</div></div>`;

  html+=`<div class="card"><div class="card-title">入选特征 (${(d.selected_features||[]).length})</div>
    <div class="feat">` + (d.selected_features||[]).map(f=>`<span class="pill">${f}</span>`).join('') + `</div></div>`;

  const tr=d.trades||[];
  html+=`<div class="card"><div class="card-title">交易明细 (共 ${d.n_trades_total} 笔, 显示最近 ${tr.length})</div>
    <div class="table-wrap"><table><thead><tr>
    <th>信号日</th><th>成交日</th><th>代码</th><th>方向</th><th>股数</th><th>价格</th><th>金额</th><th>费用</th>
    </tr></thead><tbody>`
    + tr.slice().reverse().map(t=>`<tr>
        <td>${t.signal_date??''}</td><td>${t.date??''}</td><td>${t.code??''}</td>
        <td>${t.action??''}</td><td>${t.shares??''}</td><td>${num(t.price)}</td>
        <td>${num(t.gross)}</td><td>${num(t.fee)}</td></tr>`).join('')
    + `</tbody></table></div></div>`;

  $('body').innerHTML=html;
}

boot();
</script></body></html>"""


@app.get("/backtest", response_class=HTMLResponse)
async def page_backtest():
    """隐藏入口: 首页不链接到这里, 需知道地址且有密码"""
    return BACKTEST_HTML


@app.get("/api/history")
async def api_history():
    st = _state()
    if not st:
        return JSONResponse({"error": "无状态文件"}, status_code=404)
    return st.get("history", [])


# ── 首页仪表盘 ──
@app.get("/api/profiles")
async def api_profiles():
    """四条并行线的定义 + 各自当前概况"""
    out = []
    for p in list_profiles():
        st = _state_of(p["id"])
        out.append({**p,
                    "initialized": st is not None,
                    "equity": None if st is None else round(
                        st.get("cash", 0) + sum(l["shares"] * l["buy_price"]
                                                for l in (st.get("lots") or [])), 2),
                    "n_positions": None if st is None else len(st.get("lots") or [])})
    return {"profiles": out, "default": DEFAULT_PROFILE}


@app.post("/api/profile/rename")
async def api_rename(req: Request):
    """改显示名。传空串 = 恢复代码里的默认名"""
    if not _ops_ok(req):
        return _ops_deny()
    body = await req.json()
    pid = body.get("profile")
    if (bad := _check_profile(pid)) is not None:
        return bad
    try:
        name = set_name(pid, body.get("name"))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "profile": pid, "name": name}


@app.post("/api/profile/auto")
async def api_auto(req: Request):
    """切换记账方式。

    auto=true  纸面模式: 每天按 T+1 真实行情自动记账
    auto=false 实盘模式: 不自动记账, 每次换仓都要你确认真实成交价

    切到实盘模式时, 若已有一份挂单计划, 那份计划就会转为"待确认"状态 ——
    因为我们无法知道你到底有没有按它下单。
    """
    if not _ops_ok(req):
        return _ops_deny()
    body = await req.json()
    pid = body.get("profile")
    if (bad := _check_profile(pid)) is not None:
        return bad
    auto = bool(body.get("auto"))
    set_auto(pid, auto)
    st = _state_of(pid) or {}
    return {"ok": True, "profile": pid, "auto": auto,
            "has_pending": bool(st.get("pending")),
            "note": ("已切回纸面模式, 今后按行情自动记账。"
                     if auto else
                     "已切到实盘模式。下次换仓需要你填真实成交价, 否则不会记账、也不会出新信号。")}


@app.post("/api/profile/confirm")
async def api_confirm(req: Request):
    """提交真实成交回报, 结算这条线的挂单并出下一份信号。

    fills 为空数组 = 「当天没下单」, 直接作废该计划且不动账。
    这一步会跑完整模型, 耗时约 30~60 秒, 所以走后台线程 + 轮询 /api/run-status。
    """
    if not _ops_ok(req):
        return _ops_deny()
    body = await req.json()
    pid = body.get("profile")
    if (bad := _check_profile(pid)) is not None:
        return bad
    if _running["active"]:
        return JSONResponse({"error": "已有任务在跑, 请稍候"}, status_code=409)

    fills = body.get("fills")
    if not isinstance(fills, list):
        return JSONResponse({"error": "fills 必须是数组"}, status_code=400)
    clean = []
    for i, f in enumerate(fills, 1):
        try:
            code = str(f["code"]).zfill(6)[:6]
            act = f["action"]
            shares = int(f["shares"])
            price = float(f["price"])
        except (KeyError, TypeError, ValueError):
            return JSONResponse({"error": f"第{i}条成交记录字段不全或格式错误"}, status_code=400)
        if act not in ("buy", "sell"):
            return JSONResponse({"error": f"第{i}条 action 必须是 buy/sell"}, status_code=400)
        if shares <= 0 or price <= 0:
            return JSONResponse({"error": f"第{i}条 {code} 股数和成交价必须为正"}, status_code=400)
        clean.append({"code": code, "action": act, "shares": shares, "price": price})

    cf = LIVE_DIR / f"confirm_{pid}_{datetime.now():%Y%m%d_%H%M%S}.json"
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [PY, "-u", SIGNAL_SCRIPT] + signal_args(pid) + ["--confirm", str(cf)]
    threading.Thread(target=_run_bg, args=(cmd, f"确认成交 {pid}"), daemon=True).start()
    return {"ok": True, "profile": pid, "n_fills": len(clean),
            "confirm_file": cf.name,
            "note": "已提交, 正在结算并生成下一份计划 (约 30~60 秒)"}


@app.post("/api/profile/reset")
async def api_reset(req: Request):
    """从头再来: 清空这条线的持仓与历史, 现金回到本金, 立刻重新建仓。

    可顺便改本金 —— 本金不在指纹里, 但它只在 --init 时作为起始现金生效,
    所以"改本金"和"重置"必须是同一个动作, 分开做没有意义。

    旧状态与旧计划先归档到 data/live/archive/, 不是直接删。改账操作里这是
    最重的一个 (等于把这条线的历史业绩清零), 必须留得下回溯的余地。

    要跑完整模型出建仓计划, 约 30~60 秒, 所以走后台线程 + 轮询 /api/run-status。
    """
    if not _ops_ok(req):
        return _ops_deny()
    body = await req.json()
    pid = body.get("profile")
    if (bad := _check_profile(pid)) is not None:
        return bad
    if _running["active"]:
        return JSONResponse({"error": "已有任务在跑, 请稍候"}, status_code=409)

    # 本金: 不传 = 沿用当前值; 传了就先落盘, 让 init_args 取到新值
    if body.get("capital") is not None:
        try:
            set_capital(pid, body["capital"])
        except (TypeError, ValueError) as e:
            return JSONResponse({"error": str(e) or "本金填得不对"}, status_code=400)
        except PermissionError as e:
            return JSONResponse({"error": str(e)}, status_code=403)

    st = _state_of(pid) or {}
    before = {"cash": st.get("cash"), "n_lots": len(st.get("lots") or []),
              "initial_capital": st.get("initial_capital"),
              "n_history": len(st.get("history") or [])}
    archived = _archive_profile(pid)
    cmd = [PY, "-u", SIGNAL_SCRIPT] + init_args(pid)
    threading.Thread(target=_run_bg, args=(cmd, f"重置 {pid}"), daemon=True).start()
    return {"ok": True, "profile": pid, "capital": capital_of(pid),
            "archived": archived, "before": before,
            "note": "已归档旧账, 正在按新本金重新建仓 (约 30~60 秒)"}


def _cash_op(pid, extra, task):
    """现金类操作统一走 live_signal 的快通道 (秒级, 不跑模型)。

    必须经由 live_signal 而不是网页直接改 state —— 状态文件规定单写,
    而且 live_signal 里有备份、校验和 history 记录这些不能绕过的东西。
    """
    cmd = [PY, "-u", SIGNAL_SCRIPT] + signal_args(pid) + extra
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        # live_signal 的拒绝理由都是 ERROR: 开头的人话, 直接回给前端
        msg = next((ln.strip() for ln in out.splitlines() if "ERROR" in ln), "")
        detail = out[out.find("ERROR"):].strip() if "ERROR" in out else out[-800:]
        return JSONResponse({"error": msg or f"{task}失败", "detail": detail},
                            status_code=400)
    return {"ok": True, "profile": pid, "log": out[-2000:]}


@app.post("/api/profile/set-cash")
async def api_set_cash(req: Request):
    """现金校准: 把记录的现金改成券商 App 里的真实数字。

    修账, 不是盈亏也不是出入金: 只动现金, 本金不动, 所以收益率会被修正
    到真实水平。用来消除自动记账(收盘价+估算佣金)的累积偏差。
    """
    if not _ops_ok(req):
        return _ops_deny()
    body = await req.json()
    pid = body.get("profile")
    if (bad := _check_profile(pid)) is not None:
        return bad
    try:
        cash = float(body.get("cash"))
    except (TypeError, ValueError):
        return JSONResponse({"error": "请填一个数字"}, status_code=400)
    if cash < 0:
        return JSONResponse({"error": "现金不能为负"}, status_code=400)
    extra = ["--set-cash", repr(cash)]
    if body.get("note"):
        extra += ["--note", str(body["note"])[:100]]
    return _cash_op(pid, extra, "现金校准")


@app.post("/api/profile/cash-flow")
async def api_cash_flow(req: Request):
    """存入(正)/取出(负)现金。

    本金变动, 不是盈亏: 现金和本金同额增减, 收益率保持不变。
    否则存进 1 万会被算成"赚了 1 万"。
    """
    if not _ops_ok(req):
        return _ops_deny()
    body = await req.json()
    pid = body.get("profile")
    if (bad := _check_profile(pid)) is not None:
        return bad
    try:
        amt = float(body.get("amount"))
    except (TypeError, ValueError):
        return JSONResponse({"error": "请填一个数字"}, status_code=400)
    if amt == 0:
        return JSONResponse({"error": "金额不能为 0"}, status_code=400)
    extra = ["--cash-flow", repr(amt)]
    if body.get("note"):
        extra += ["--note", str(body["note"])[:100]]
    return _cash_op(pid, extra, "出入金")


@app.post("/api/profile/drop-lot")
async def api_drop_lot(req: Request):
    """删除一笔持仓。

    mode="sold"   : 已在券商卖出, 需带 price -> 现金增加 股数x价格-手续费
    mode="phantom": 系统记错了从没持有 -> 只删记录, 现金不动

    两种情形现金处理正好相反, 所以必须由前端明确传 mode, 后端不猜。
    """
    if not _ops_ok(req):
        return _ops_deny()
    body = await req.json()
    pid = body.get("profile")
    if (bad := _check_profile(pid)) is not None:
        return bad
    code = str(body.get("code") or "").strip()
    if not code.isdigit() or len(code) > 6:
        return JSONResponse({"error": "股票代码不对"}, status_code=400)
    code = code.zfill(6)

    mode = body.get("mode")
    if mode == "sold":
        try:
            price = float(body.get("price"))
        except (TypeError, ValueError):
            return JSONResponse({"error": "请填真实卖出价"}, status_code=400)
        if price <= 0:
            return JSONResponse({"error": "卖出价必须为正"}, status_code=400)
        extra = ["--drop-lot", code, "--sold-at", repr(price)]
    elif mode == "phantom":
        extra = ["--drop-lot", code, "--phantom"]
    else:
        return JSONResponse(
            {"error": "必须说明是「已卖出」还是「记错了」"}, status_code=400)

    if body.get("note"):
        extra += ["--note", str(body["note"])[:100]]
    return _cash_op(pid, extra, "删除持仓")


@app.get("/api/today")
async def api_today(profile: str = None):
    """归一化的"明天该做什么" —— 首页用。只读, 不触发模型。"""
    return build_today(ROOT, profile)


@app.get("/api/recommend")
async def api_recommend(profile: str = None):
    """每日推荐看板: 模型打分最高的股票 + 该线能否买得起"""
    return build_recommend(ROOT, profile)


@app.get("/api/kline")
async def api_kline(code: str, period: str = "day", bars: int = 120):
    """单只股票的 K 线 (日/周/月), 推荐看板点开看图用。只读。

    周/月线由日线现场重采样 (周=自然周五收, 月=自然月末), 和行情软件口径
    一致。价格是入库的前复权价, 与模型看到的一致。
    """
    import pandas as pd  # 懒加载: 只有这个接口用 pandas, 不拖慢服务启动
    c = "".join(ch for ch in str(code) if ch.isdigit())[:6]
    if len(c) != 6:
        return JSONResponse({"error": f"非法代码 {code!r}"}, status_code=400)
    if period not in ("day", "week", "month"):
        return JSONResponse({"error": f"period 只能是 day/week/month"}, status_code=400)
    p = KLINE_DIR / f"{c}.parquet"
    if not p.exists():
        return JSONResponse({"error": f"{c} 没有K线数据"}, status_code=404)
    kl = pd.read_parquet(p).rename(columns={
        "时间": "date", "收盘价": "close", "开盘价": "open",
        "最高价": "high", "最低价": "low", "成交量": "volume"})
    kl["date"] = pd.to_datetime(kl["date"])
    kl = kl.sort_values("date").set_index("date")[["open", "high", "low", "close", "volume"]]
    if period != "day":
        rule = "W-FRI" if period == "week" else "ME"
        kl = kl.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna(subset=["close"])
    bars = max(20, min(int(bars), 500))
    kl = kl.tail(bars)
    return {
        "code": c, "period": period, "n": len(kl),
        "k": [{"d": d.strftime("%Y-%m-%d"),
               "o": round(float(r["open"]), 3), "h": round(float(r["high"]), 3),
               "l": round(float(r["low"]), 3), "c": round(float(r["close"]), 3),
               "v": float(r["volume"] or 0)}
              for d, r in kl.iterrows()],
    }


@app.get("/", response_class=HTMLResponse)
async def action_page():
    """默认页 = 行动清单 (给看的人照着执行)"""
    return ACTION_HTML


@app.get("/pro", response_class=HTMLResponse)
async def dashboard():
    """完整仪表盘 (含对账/手工触发信号等运维操作)"""
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>实盘信号仪表盘</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
         background: #0f1117; color: #e0e0e0; padding: 20px; }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { font-size: 22px; margin-bottom: 16px; color: #fff; }
  .card { background: #1a1d29; border-radius: 12px; padding: 20px; margin-bottom: 16px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
  .card-title { font-size: 14px; color: #888; margin-bottom: 12px; text-transform: uppercase; }
  .stat-row { display: flex; gap: 24px; flex-wrap: wrap; }
  .stat { text-align: left; }
  .stat .label { font-size: 12px; color: #888; }
  .stat .value { font-size: 24px; font-weight: 700; color: #fff; margin-top: 2px; }
  .stat .value.green { color: #4caf50; }
  .stat .value.red { color: #f44336; }
  .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #2a2d39;
           white-space: nowrap; }
  th { color: #888; font-weight: 600; font-size: 12px; text-transform: uppercase; }
  td { color: #ccc; }
  /* A股习惯: 红涨绿跌 (与欧美相反), 与首页保持一致 */
  .pnl-pos { color: #f6465d; }
  .pnl-neg { color: #2ebd85; }
  .btn { display: inline-block; padding: 10px 20px; border: none; border-radius: 8px;
         font-size: 14px; cursor: pointer; margin-right: 8px; margin-bottom: 8px;
         transition: opacity 0.2s; }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: #2962ff; color: #fff; }
  .btn-warn { background: #ff9800; color: #000; }
  .btn-danger { background: #f44336; color: #fff; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
  .badge-blue { background: #1a237e; color: #82b1ff; }
  .badge-green { background: #1b5e20; color: #69f0ae; }
  .badge-red { background: #b71c1c; color: #ff8a80; }
  .badge-gray { background: #333; color: #aaa; }
  #log { font-family: monospace; font-size: 12px; white-space: pre-wrap;
         max-height: 300px; overflow-y: auto; background: #0a0b10; padding: 12px;
         border-radius: 8px; display: none; margin-top: 12px; }
  .section { margin-bottom: 12px; }
  .pill { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px;
          background: #2a2d39; margin-right: 6px; }
  .empty { color: #666; font-style: italic; padding: 12px 0; }
  .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                   background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center;
                   align-items: flex-start; padding: 40px 20px; overflow-y: auto; }
  .modal { background: #1a1d29; border-radius: 12px; padding: 24px; max-width: 600px; width: 100%; }
  .modal h2 { font-size: 18px; margin-bottom: 16px; color: #fff; }
  .modal label { display: block; font-size: 13px; color: #888; margin-top: 12px; margin-bottom: 4px; }
  .modal input { width: 100%; padding: 8px 12px; border: 1px solid #333; border-radius: 6px;
                 background: #0f1117; color: #fff; font-size: 14px; }

  /* ── 手机端 ── */
  @media (max-width: 640px) {
    body { padding: 12px; }
    h1 { font-size: 18px; }
    .card { padding: 14px; border-radius: 10px; }
    .stat-row { gap: 14px; }
    .stat { flex: 1 1 45%; }
    .stat .value { font-size: 19px; }
    table { font-size: 13px; }
    th, td { padding: 7px 9px; }
    .btn { width: 100%; margin-right: 0; padding: 12px 16px; font-size: 15px; }
    .modal { padding: 16px; }
    #log { font-size: 11px; max-height: 220px; }
  }
</style>
</head>
<body>
<div class="container">
  <h1>📊 运维面板</h1>
  <!-- 这个页只读默认那一条线的状态文件, 必须说清楚: 否则容易
       把这一条的数字当成四条线的全部。 -->
  <div style="background:#1f2430;border-left:3px solid #2962ff;border-radius:8px;
              padding:11px 13px;margin-bottom:16px;font-size:13px;line-height:1.8;color:#9aa3b4">
    本页只显示 <b id="which-profile" style="color:#dbe3f4">默认条线</b> 一条，
    专供看流水线状态与访问日志。<br>
    四条线的持仓、操作清单与所有改账功能在
    <a href="/" style="color:#82b1ff">首页</a>。
  </div>

  <div class="card">
    <div class="card-title">账户概览</div>
    <div class="stat-row" id="overview">
      <div class="stat"><div class="label">总资产</div><div class="value" id="equity">--</div></div>
      <div class="stat"><div class="label">现金</div><div class="value" id="cash">--</div></div>
      <div class="stat"><div class="label">持仓市值</div><div class="value" id="mv">--</div></div>
      <div class="stat"><div class="label">累计收益</div><div class="value" id="ret">--</div></div>
    </div>
    <div style="margin-top:8px;font-size:12px;color:#666" id="meta"></div>
  </div>

  <div class="card">
    <div class="card-title">当前持仓</div>
    <div class="table-wrap">
    <table id="pos-table">
      <thead><tr><th>代码</th><th>名称</th><th>股数</th><th>成本</th><th>现价</th>
                 <th>市值</th><th>浮盈</th><th>持有天数</th></tr></thead>
      <tbody id="pos-body"></tbody>
    </table>
    </div>
    <div id="pending-info" style="margin-top:12px;font-size:13px;color:#888"></div>
  </div>

  <div class="card">
    <div class="card-title">最新操作计划</div>
    <div id="plan-info"></div>
    <div class="table-wrap" style="margin-top:12px">
    <table id="plan-table">
      <thead><tr><th>方向</th><th>代码</th><th>名称</th><th>股数</th><th>参考价</th>
                 <th>预估金额</th><th>pred</th><th>备注</th></tr></thead>
      <tbody id="plan-body"></tbody>
    </table>
    </div>
  </div>

  <div class="card">
    <div class="card-title">数据流水线</div>
    <div id="pipe-info"><span class="empty">加载中...</span></div>
  </div>

  <div class="card">
    <div class="card-title">操作</div>
    <button class="btn btn-primary" id="btn-signal" onclick="runSignal()">🔄 生成今日信号</button>
    <button class="btn" style="background:#37415a;color:#dbe3f4"
            onclick="toggleAccess()">👁 访问日志</button>
    <button class="btn btn-danger" onclick="refresh()">刷新</button>
    <div style="font-size:12px;color:#666;margin-top:6px;line-height:1.8">
      改账操作(确认成交 / 现金校准 / 存取现金 / 删持仓 / 重置)在
      <a href="/" style="color:#82b1ff">首页</a>，那里四条线都能操作且语义更明确。
    </div>
    <div id="log"></div>
  </div>

  <div class="card" id="access-card" style="display:none">
    <div class="card-title">访问日志</div>
    <div id="access-body"><span class="empty">加载中...</span></div>
  </div>

  <div style="font-size:12px;color:#666;text-align:center;margin:18px 0 6px">
    本站信息仅供参考，不构成任何投资建议
  </div>
</div>

<!-- 改账口令弹窗 -->
<div class="modal-overlay" id="ops-modal">
  <div class="modal" style="max-width:340px">
    <h2>需要改账口令</h2>
    <div style="font-size:13px;color:#888;line-height:1.7">
      生成信号会结算挂单并改写账目，所以要口令。一次输入 12 小时内有效。
    </div>
    <label>口令</label>
    <input type="password" id="ops-pw" onkeydown="if(event.key==='Enter')opsSubmit()">
    <div id="ops-err" style="color:#f44336;font-size:12px;margin-top:8px;min-height:16px"></div>
    <div style="margin-top:14px">
      <button class="btn btn-primary" onclick="opsSubmit()">确定</button>
      <button class="btn" style="background:#333;color:#ccc" onclick="opsCancel()">取消</button>
    </div>
  </div>
</div>

<script>
const API = '';

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  return r.json();
}

// ── 改账口令 ──
// 这个页上的写操作(生成信号)之前没校验口令, 等于"能看就能改"。
// 现在与首页一致: 401 弹口令框, 输对后自动重试原操作。
let _opsResolve = null, _opsReject = null;

function askOps(){
  return new Promise((res, rej) => {
    _opsResolve = res; _opsReject = rej;
    document.getElementById('ops-err').textContent = '';
    document.getElementById('ops-pw').value = '';
    document.getElementById('ops-modal').style.display = 'flex';
    setTimeout(() => document.getElementById('ops-pw').focus(), 50);
  });
}

function opsCancel(){
  document.getElementById('ops-modal').style.display = 'none';
  if (_opsReject) _opsReject(new Error('已取消'));
  _opsResolve = _opsReject = null;
}

async function opsSubmit(){
  const pw = document.getElementById('ops-pw').value || '';
  const r = await fetch('/api/pro-ops/login', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({password: pw})});
  if (r.ok){
    document.getElementById('ops-modal').style.display = 'none';
    if (_opsResolve) _opsResolve();
    _opsResolve = _opsReject = null;
    return;
  }
  const d = await r.json().catch(()=>({}));
  document.getElementById('ops-err').textContent = d.error || '口令错误';
  document.getElementById('ops-pw').value = '';
  document.getElementById('ops-pw').focus();
}

// 需要改账权限的请求都走这里
async function opsFetch(url, opts){
  let r = await fetch(url, opts);
  if (r.status === 401){
    const d = await r.json().catch(()=>({}));
    if (d.need_password){
      await askOps();                    // 取消会 reject, 直接冒泡出去
      r = await fetch(url, opts);
    }
  }
  return r;
}

async function refresh() {
  try {
    const s = await fetchJSON('/api/status');
    document.getElementById('equity').textContent = '¥' + (s.equity||0).toLocaleString();
    document.getElementById('cash').textContent = '¥' + (s.cash||0).toLocaleString();
    document.getElementById('mv').textContent = '¥' + (s.market_value||0).toLocaleString();
    const ret = s.total_return_pct;
    const retEl = document.getElementById('ret');
    retEl.textContent = (ret>=0?'+':'') + ret + '%';
    retEl.className = 'value ' + (ret>=0?'red':'green');   // A股: 红涨绿跌
    // 把条线名字填进页顶提示, 避免把这一条的数字当成四条线的全部
    if (s.profile_name)
      document.getElementById('which-profile').textContent =
        s.profile_name + ' (' + s.profile + ')';
    let meta = '估值基准日: ' + (s.ref_date||'--');
    if (s.last_signal_date) meta += ' | 上次信号: ' + s.last_signal_date;
    if (s.last_synced_at) meta += ' | 上次对账: ' + s.last_synced_at;
    if (s.running) meta += ' | <span class="badge badge-blue">信号生成中...</span>';
    document.getElementById('meta').innerHTML = meta;

    const tbody = document.getElementById('pos-body');
    tbody.innerHTML = '';
    if (!s.positions || !s.positions.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty">空仓</td></tr>';
    } else {
      for (const p of s.positions) {
        const cls = p.pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg';
        tbody.innerHTML += `<tr>
          <td>${p.code}</td><td>${p.name}</td><td>${p.shares}</td>
          <td>${p.buy_price}</td><td>${p.last_close}</td>
          <td>¥${p.market_value.toLocaleString()}</td>
          <td class="${cls}">${p.pnl_pct>=0?'+':''}${p.pnl_pct}%</td>
          <td>${daysCell(p, s.hold_days)}</td></tr>`;
      }
    }

    let pendHtml = '';
    if (s.pending) {
      pendHtml = `<span class="badge badge-blue">待执行: ${s.pending.signal_date}</span>`;
      if (s.pending.is_rebal) pendHtml += ' <span class="badge badge-green">换仓日</span>';
      if (s.pending.in_cash) pendHtml += ' <span class="badge badge-red">空仓信号</span>';
    } else {
      pendHtml = '<span class="badge badge-gray">无待执行计划</span>';
    }
    document.getElementById('pending-info').innerHTML = pendHtml;

    document.getElementById('btn-signal').disabled = s.running;
  } catch(e) {
    console.error(e);
  }
  try {
    const p = await fetchJSON('/api/plan');
    renderPlan(p);
  } catch(e) {
    document.getElementById('plan-info').innerHTML = '<span class="empty">尚无计划</span>';
  }
  try {
    renderPipeline(await fetchJSON('/api/pipeline'));
  } catch(e) {
    document.getElementById('pipe-info').innerHTML = '<span class="empty">无流水线记录</span>';
  }
}

function renderPipeline(p) {
  const el = document.getElementById('pipe-info');
  if (!p || (!p.started_at && !p.train_max_date)) {
    el.innerHTML = '<span class="empty">尚未运行过自动重建</span>';
    return;
  }
  let badge;
  if (p.ok === false) badge = '<span class="badge badge-red">失败</span>';
  else if (p.skipped_reason) badge = '<span class="badge badge-gray">本次跳过</span>';
  else if (p.ok) badge = '<span class="badge badge-green">成功</span>';
  else badge = '<span class="badge badge-blue">运行中</span>';

  let html = `<div style="font-size:14px;margin-bottom:8px">${badge}
    <span class="pill">训练集数据至: ${p.train_max_date || '--'}</span>
    <span class="pill">K线至: ${p.kline_max_date || '--'}</span>
  </div>
  <div style="font-size:12px;color:#888">上次运行: ${p.started_at || '--'}`;
  if (p.total_seconds) html += ` · 耗时 ${p.total_seconds}s`;
  html += '</div>';
  if (p.skipped_reason)
    html += `<div style="font-size:12px;color:#888;margin-top:6px">${p.skipped_reason}</div>`;
  if (p.error)
    html += `<div style="font-size:12px;color:#f44336;margin-top:6px">错误: ${p.error}</div>`;
  if (p.stages && Object.keys(p.stages).length) {
    html += '<div style="margin-top:8px">';
    for (const [k, v] of Object.entries(p.stages)) {
      const c = v.ok ? 'badge-green' : 'badge-red';
      const t = v.skipped ? '跳过' : (v.seconds != null ? v.seconds + 's' : '');
      html += `<span class="badge ${c}" style="margin-right:6px">${k} ${t}</span>`;
    }
    html += '</div>';
  }
  el.innerHTML = html;
}

function renderPlan(p) {
  let html = `<div style="font-size:14px;margin-bottom:8px">
    <span class="pill">信号日: ${p.signal_date}</span>
    <span class="pill">执行: ${(p.exec_window && p.exec_window.when_text) || p.exec_hint}</span>
    <span class="pill">总资产: ¥${(p.equity||0).toLocaleString()}</span>
    <span class="pill">大盘广度: ${((p.breadth||0)*100).toFixed(1)}%</span>
    <span class="pill">择时: ${p.in_cash?'空仓':'持仓'}</span>
    <span class="pill">${p.is_rebal?'换仓日':'非换仓日'}</span>
  </div>`;
  document.getElementById('plan-info').innerHTML = html;
  const tbody = document.getElementById('plan-body');
  tbody.innerHTML = '';
  for (const s of (p.sell||[])) {
    tbody.innerHTML += `<tr><td><span class="badge badge-red">卖出</span></td>
      <td>${s.code}</td><td>${s.name}</td><td>${s.shares}</td>
      <td>${s.ref_close}</td><td>¥${(s.est_proceeds||0).toLocaleString()}</td>
      <td>-</td><td>${s.reason||''}</td></tr>`;
  }
  for (const b of (p.buy||[])) {
    tbody.innerHTML += `<tr><td><span class="badge badge-green">买入</span></td>
      <td>${b.code}</td><td>${b.name}</td><td>${b.shares}</td>
      <td>${b.ref_close}</td><td>¥${(b.est_cost||0).toLocaleString()}</td>
      <td>${b.pred?b.pred.toFixed(4):'-'}</td><td>-</td></tr>`;
  }
  for (const a of (p.alternates||[])) {
    tbody.innerHTML += `<tr><td><span class="badge badge-gray">候补</span></td>
      <td>${a.code}</td><td>${a.name}</td><td>-</td>
      <td>${a.ref_close}</td><td>-</td>
      <td>${a.pred?a.pred.toFixed(4):'-'}</td><td>${a.note||''}</td></tr>`;
  }
  if (!tbody.innerHTML) tbody.innerHTML = '<tr><td colspan="8" class="empty">无操作</td></tr>';
}

// 持有天数单元格。与首页同口径: 两个数字回答不同问题, 所以都要给。
//   tenure_days -> 这笔一共拿了多久 (真实时长, 只增不减)
//   held_days   -> 什么时候会动它 (到期时钟, 续持归零)
// 之前这里只显示时钟, 于是续持过的仓看起来像刚买的(显示 0 日)。
function daysCell(p, holdDays){
  const t = p.tenure_days, c = p.held_days;
  if (t == null && c == null) return '--';
  const shown = (t != null ? t : c);
  let s = shown + '日';
  if (p.n_rolled) s += ` <span class="badge badge-blue">续持${p.n_rolled}次</span>`;
  if (holdDays != null && c != null){
    const left = holdDays - c;
    s += left <= 0 ? ' <span class="badge badge-red">已到期</span>'
                   : ` <span style="color:#666;font-size:12px">还剩${left}日</span>`;
  }
  return s;
}

async function runSignal() {
  const btn = document.getElementById('btn-signal');
  const logDiv = document.getElementById('log');
  logDiv.style.display = 'block';
  logDiv.textContent = '信号生成中...';
  btn.disabled = true;
  try {
    const r = await opsFetch('/api/signal', {method: 'POST'});
    const d = await r.json().catch(()=>({}));
    if (!r.ok || d.error) {
      logDiv.textContent = d.error || ('失败 HTTP ' + r.status);
      btn.disabled = false; return;
    }
    pollSignal();
  } catch(e) {
    logDiv.textContent = e.message === '已取消' ? '已取消' : ('请求失败: ' + e);
    btn.disabled = false;
  }
}

async function pollSignal() {
  const logDiv = document.getElementById('log');
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 2000));
    const s = await fetchJSON('/api/signal-status');
    if (s.log) logDiv.textContent = s.log.replace(/\\r/g, '\\n').replace(/\\n{3,}/g, '\\n\\n');
    if (!s.active) {
      logDiv.textContent += '\\n\\n✅ 完成于: ' + s.done_at;
      document.getElementById('btn-signal').disabled = false;
      refresh();
      return;
    }
  }
}

// 此处原有约 60 行对账表单代码, 已随入口一并下掉。缘由见 web_server.py 里
// api_sync 上方的注释。

// ── 访问日志 ──
// 用改账口令保护: 日志里有 IP 和归属地, 比持仓更该少露。
let ACC = {open:false, days:30, pages:1, geo:1};

function toggleAccess(){
  ACC.open = !ACC.open;
  document.getElementById('access-card').style.display = ACC.open ? 'block' : 'none';
  if (ACC.open) loadAccess();
}

const aesc = s => String(s==null?'':s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function loadAccess(){
  const el = document.getElementById('access-body');
  el.innerHTML = '<span class="empty">加载中...</span>';
  try {
    const q = '?days=' + ACC.days + '&geo=' + ACC.geo;
    const [rs, re] = await Promise.all([
      fetch('/api/access/summary' + q),
      fetch('/api/access/events?limit=200&pages=' + ACC.pages + '&geo=' + ACC.geo),
    ]);
    if (rs.status === 401 || re.status === 401){ accessPasswordBox(); return; }
    const s = await rs.json(), e = await re.json();
    if (s.error){ el.innerHTML = '<span class="empty">' + aesc(s.error) + '</span>'; return; }
    renderAccess(s, e);
  } catch(err){
    el.innerHTML = '<span class="empty">加载失败: ' + aesc(err.message) + '</span>';
  }
}

function accessPasswordBox(){
  document.getElementById('access-body').innerHTML =
    '<div style="font-size:13px;color:#888;margin-bottom:8px">' +
    '访问日志含 IP 与归属地, 需要改账口令。</div>' +
    '<input type="password" id="acc-pw" placeholder="改账口令" ' +
    'style="width:200px;padding:8px 12px;border:1px solid #333;border-radius:6px;' +
    'background:#0f1117;color:#fff;font-size:14px" ' +
    'onkeydown="if(event.key===&quot;Enter&quot;)accessLogin()">' +
    '<button class="btn btn-primary" style="margin-left:8px" onclick="accessLogin()">确定</button>' +
    '<div id="acc-err" style="color:#f44336;font-size:12px;margin-top:8px"></div>';
  setTimeout(() => { const i = document.getElementById('acc-pw'); if (i) i.focus(); }, 50);
}

async function accessLogin(){
  const pw = (document.getElementById('acc-pw')||{}).value || '';
  const r = await fetch('/api/pro-ops/login', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({password: pw})});
  if (r.ok){ loadAccess(); return; }
  const d = await r.json().catch(()=>({}));
  const e = document.getElementById('acc-err');
  if (e) e.textContent = d.error || '口令错误';
}

function renderAccess(s, ev){
  let h = '<div class="stat-row" style="margin-bottom:14px">' +
    stat('总访问', s.total) + stat('今日', s.today) +
    stat('独立 IP', s.unique_ips) + stat('其中外部', s.unique_external_ips) + '</div>';

  // 口令是 6 位数字, 这个数字是发现"有人在猜"的唯一途径, 所以非 0 就报警
  if (s.failed_logins){
    h += '<div style="background:#3b1518;border:1px solid #5b1f24;border-radius:8px;' +
      'padding:11px 13px;margin-bottom:12px;font-size:13px;color:#fca5a5">' +
      '口令尝试失败 <b>' + s.failed_logins + '</b> 次';
    const fi = (s.failed_login_ips||[]).filter(x => x[1] >= 3);
    if (fi.length) h += ' · 可疑 IP: ' +
      fi.map(x => aesc(x[0]) + '(' + x[1] + '次)').join(', ');
    h += '<div style="color:#8a93a6;margin-top:5px;font-size:12px">' +
      '当前口令为 6 位纯数字。若这个数字持续上涨且不是你自己输错, ' +
      '建议换长口令。</div></div>';
  }

  h += '<div style="font-size:12px;color:#666;margin-bottom:12px">' +
    '统计窗口 ' + s.window_days + ' 天' +
    (s.first_day ? ' · 最早记录 ' + s.first_day : '') +
    ' · 计数精确, 下方事件流同 IP 同路径 ' +
    Math.round((ev.dedupe_seconds||300)/60) + ' 分钟内合并为一条</div>';

  h += '<div style="margin-bottom:10px">' +
    btnSm('近30天', ACC.days===30, 'ACC.days=30;loadAccess()') +
    btnSm('近7天', ACC.days===7, 'ACC.days=7;loadAccess()') +
    btnSm('只看页面', ACC.pages===1, 'ACC.pages=1;loadAccess()') +
    btnSm('含接口', ACC.pages===0, 'ACC.pages=0;loadAccess()') +
    btnSm(ACC.geo ? '归属地:开' : '归属地:关', false,
          'ACC.geo=' + (ACC.geo?0:1) + ';loadAccess()') +
    '</div>';

  h += '<div style="font-size:13px;color:#888;margin:14px 0 6px">按 IP 汇总</div>' +
       '<div class="table-wrap"><table><thead><tr><th>IP</th><th>次数</th>' +
       '<th>首次出现</th><th>归属地</th><th>网络</th></tr></thead><tbody>';
  for (const r of (s.ips||[])){
    const tag = r.private ? ' <span class="badge badge-gray">内网</span>' : '';
    h += '<tr><td>' + aesc(r.ip) + tag + '</td><td>' + r.hits + '</td><td>' +
         aesc(r.first_seen||'--') + '</td><td>' + aesc(r.where) + '</td><td>' +
         aesc(r.isp||'') + '</td></tr>';
  }
  if (!(s.ips||[]).length) h += '<tr><td colspan="5" class="empty">暂无记录</td></tr>';
  h += '</tbody></table></div>';

  h += '<div style="font-size:13px;color:#888;margin:16px 0 6px">最近访问</div>' +
       '<div class="table-wrap"><table><thead><tr><th>时间</th><th>IP</th>' +
       '<th>归属地</th><th>路径</th><th>状态</th><th>已登录</th>' +
       '<th>设备</th></tr></thead><tbody>';
  for (const r of (ev.events||[])){
    const sc = r.s >= 400 ? ' class="pnl-neg"' : '';
    const xf = r.xff ? ' <span class="badge badge-red" title="无可信代理, 此头可伪造">XFF</span>' : '';
    h += '<tr><td>' + aesc((r.t||'').replace('T',' ')) + '</td><td>' + aesc(r.ip) + xf +
         '</td><td>' + aesc(r.where||'') + '</td><td>' + aesc(r.m + ' ' + r.p) +
         '</td><td' + sc + '>' + r.s + '</td><td>' + (r.au ? '是' : '<b>否</b>') +
         '</td><td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">' +
         aesc(shortUA(r.ua)) + '</td></tr>';
  }
  if (!(ev.events||[]).length) h += '<tr><td colspan="7" class="empty">暂无记录</td></tr>';
  h += '</tbody></table></div>';

  document.getElementById('access-body').innerHTML = h;
}

function stat(label, v){
  return '<div class="stat"><div class="label">' + label + '</div>' +
         '<div class="value">' + (v==null?'--':v) + '</div></div>';
}

function btnSm(label, on, action){
  return '<span onclick="' + action + '" style="display:inline-block;cursor:pointer;' +
    'padding:4px 11px;border-radius:12px;font-size:12px;margin-right:6px;' +
    'background:' + (on ? '#2962ff' : '#2a2d39') + ';color:' + (on ? '#fff' : '#aaa') +
    '">' + label + '</span>';
}

// UA 完整字串太长, 表格里只需认出是什么东西在访问
function shortUA(ua){
  if (!ua) return '';
  const pairs = [['iPhone','iPhone'],['iPad','iPad'],['Android','Android'],
                 ['Macintosh','Mac'],['Windows','Windows'],['Linux','Linux'],
                 ['curl','curl'],['python','脚本'],['bot','爬虫'],['Bot','爬虫']];
  for (const [k, v] of pairs) if (ua.indexOf(k) >= 0) return v;
  return ua.slice(0, 30);
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser(description="实盘信号 Web 服务器")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    print(f"启动: http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port)
