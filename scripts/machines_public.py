"""实验机资源监控 —— 公开只读版, 可分享给实验室同学。

与主站完全隔离: 无登录、无业务数据, 只报三台机的 CPU/内存/GPU/磁盘占用。
独立进程 + 独立端口 (默认 8222, 校防火墙只放行 8xxx 段), 纯标准库无依赖。

用法: python scripts/machines_public.py --port 8222
systemd: ~/.config/systemd/user/resmon.service (eez041)
"""
import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # htop 低调化 (见 scripts/proctitle.py)
    from proctitle import lowkey
    lowkey("mltask/mon")
except Exception:
    pass

HOSTS = ["eez040.ece.ust.hk", "eez041.ece.ust.hk", "eez042.ece.ust.hk"]
PROBE = ("cat /proc/loadavg; echo __SEP__; nproc; echo __SEP__; "
         "free -g | sed -n 2p; echo __SEP__; "
         "ps -eo user:14,pcpu --sort=-pcpu --no-headers | head -12; echo __SEP__; "
         "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
         "--format=csv,noheader,nounits 2>/dev/null || echo NOGPU; "
         "echo __SEP__; df -PB1G / /home 2>/dev/null | tail -n +2")
TTL = 30
_cache = {"at": 0.0, "data": None}
_lock = threading.Lock()


def probe(host):
    local = os.uname().nodename.split(".")[0] == host.split(".")[0]
    cmd = ["bash", "-c", PROBE] if local else \
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         "-o", "StrictHostKeyChecking=accept-new", host, PROBE]
    name = host.split(".")[0]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        if not r.stdout.strip():
            return {"host": name, "ok": False}
        seg = [s.strip() for s in r.stdout.split("__SEP__")]
        load1, load5, load15 = (float(x) for x in seg[0].split()[:3])
        ncpu = int(seg[1].split()[0])
        mem = seg[2].split()
        mem_total, mem_used = int(mem[1]), int(mem[2])
        by_user = {}
        for line in seg[3].splitlines():
            p = line.split()
            if len(p) >= 2:
                by_user[p[0]] = by_user.get(p[0], 0.0) + float(p[1])
        users = [{"user": u, "cores": round(c / 100, 1)}
                 for u, c in sorted(by_user.items(), key=lambda kv: -kv[1])
                 if c >= 50][:4]
        gpus = None
        if seg[4] and "NOGPU" not in seg[4]:
            rows = []
            for line in seg[4].splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 3:
                    try:
                        rows.append((int(p[0]), int(p[1]), int(p[2])))
                    except ValueError:
                        pass
            if rows:
                gpus = {"n": len(rows),
                        "idle": sum(1 for u, m, _ in rows if u < 10 and m < 1024),
                        "max_util": max(u for u, _, _ in rows)}
        disks = []
        seen = set()
        for line in seg[5].splitlines():
            p = line.split()
            if len(p) >= 6 and p[0] not in seen:
                try:
                    disks.append({"mount": p[5], "size_g": int(p[1]),
                                  "used_g": int(p[2]),
                                  "pct": int(p[4].rstrip("%"))})
                    seen.add(p[0])
                except ValueError:
                    pass
        ratio = load1 / ncpu if ncpu else 1.0
        mem_pct = round(mem_used / mem_total * 100) if mem_total else 0
        status = ("idle" if ratio < 0.15 else
                  "busy" if ratio > 0.55 else "partial")
        if mem_pct >= 92:
            status = "busy"
        elif mem_pct >= 80 and status == "idle":
            status = "partial"
        return {"host": name, "ok": True, "ncpu": ncpu,
                "load1": round(load1, 1), "load5": round(load5, 1),
                "load15": round(load15, 1), "ratio": round(ratio, 3),
                "free_cores": max(0, round(ncpu - load1)),
                "status": status,
                "mem_total_g": mem_total, "mem_used_g": mem_used,
                "top_users": users, "gpus": gpus, "disks": disks}
    except Exception:
        return {"host": name, "ok": False}


def collect():
    with _lock:
        if time.time() - _cache["at"] < TTL and _cache["data"]:
            return _cache["data"]
    with ThreadPoolExecutor(max_workers=len(HOSTS)) as ex:
        cards = list(ex.map(probe, HOSTS))
    data = {"at": time.strftime("%H:%M:%S"), "machines": cards,
            "idle": [c["host"] for c in cards
                     if c.get("ok") and c["status"] == "idle"]}
    with _lock:
        _cache.update(at=time.time(), data=data)
    return data


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>实验机资源监控</title><style>
body{background:#111418;color:#dde3ea;font:15px -apple-system,"PingFang SC",sans-serif;margin:0}
.wrap{max-width:760px;margin:0 auto;padding:24px 16px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#8a94a3;font-size:13px;margin-bottom:16px}
.card{background:#1a1f26;border:1px solid #2a313b;border-radius:10px;padding:14px 16px;margin-bottom:12px}
.head{display:flex;align-items:center;gap:10px}
.name{font-size:17px;font-weight:600}
.badge{font-size:12px;padding:2px 8px;border-radius:10px}
.b-idle{background:#173626;color:#4caf7d}.b-partial{background:#3a2f16;color:#e0a03c}
.b-busy{background:#3a1a1a;color:#e05c5c}.b-err{background:#333;color:#999}
.bar{height:8px;background:#242b34;border-radius:4px;margin:10px 0 8px;overflow:hidden}
.fill{height:100%;border-radius:4px}
.row{font-size:13.5px;color:#aeb7c2;margin:3px 0}
.mono{font-family:ui-monospace,Menlo,monospace}
.muted{color:#68717d;font-size:12px}
</style></head><body><div class="wrap">
<h1>实验机资源监控</h1>
<div class="sub">eez040 / eez041 / eez042 · 30 秒自动刷新 · 只读</div>
<div id="cards">加载中…</div>
<div class="muted" id="at"></div>
</div><script>
const fmtG = g => g >= 1024 ? (g/1024).toFixed(1)+'T' : g+'G';
const pcol = p => p >= 90 ? '#e05c5c' : p >= 75 ? '#e0a03c' : '#4caf7d';
async function refresh(){
  try{
    const d = await (await fetch('/api')).json();
    document.getElementById('at').textContent = d.at + ' 采样';
    document.getElementById('cards').innerHTML = d.machines.map(m => {
      if(!m.ok) return '<div class="card"><div class="head"><span class="name">'+m.host+
        '</span><span class="badge b-err">失联</span></div></div>';
      const pct = Math.min(100, Math.round(m.ratio*100));
      const cls = m.status==='idle'?'b-idle':(m.status==='busy'?'b-busy':'b-partial');
      const col = m.status==='idle'?'#4caf7d':(m.status==='busy'?'#e05c5c':'#e0a03c');
      const zh = m.status==='idle'?'空闲':(m.status==='busy'?'忙':'部分占用');
      const mp = Math.round(m.mem_used_g/m.mem_total_g*100);
      const users = (m.top_users||[]).map(u=>u.user+' ≈'+u.cores+'核').join('，')||'无大户';
      return '<div class="card"><div class="head"><span class="name">'+m.host+'</span>'+
        '<span class="badge '+cls+'">'+zh+'</span>'+
        '<span class="muted" style="margin-left:auto">'+m.ncpu+' 核 · 余 ≈'+m.free_cores+' 核</span></div>'+
        '<div class="bar"><div class="fill" style="width:'+pct+'%;background:'+col+'"></div></div>'+
        '<div class="row">负载 <span class="mono">'+m.load1+' / '+m.load5+' / '+m.load15+'</span>（1/5/15分）</div>'+
        '<div class="row">内存 <span class="mono">'+m.mem_used_g+'/'+m.mem_total_g+'G</span> <b style="color:'+pcol(mp)+'">'+mp+'%</b></div>'+
        (m.gpus?'<div class="row">显卡 '+m.gpus.n+' 张 · <b style="color:'+(m.gpus.idle?'#4caf7d':'#e05c5c')+
          '">空闲 '+m.gpus.idle+'</b>（峰值 '+m.gpus.max_util+'%）</div>':'')+
        ((m.disks&&m.disks.length)?'<div class="row">存储 '+m.disks.map(dk=>dk.mount+
          ' <span class="mono">'+fmtG(dk.used_g)+'/'+fmtG(dk.size_g)+'</span> <b style="color:'+
          pcol(dk.pct)+'">'+dk.pct+'%</b>').join(' · ')+'</div>':'')+
        '<div class="row">在用 '+users+'</div></div>';
    }).join('');
  }catch(e){ document.getElementById('cards').textContent = '加载失败'; }
}
refresh(); setInterval(refresh, 30000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api"):
            body = json.dumps(collect(), ensure_ascii=False).encode()
            ctype = "application/json; charset=utf-8"
        elif self.path == "/" or self.path.startswith("/index"):
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):  # 不刷访问日志
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="res monitor")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8222)
    a = ap.parse_args()
    print(f"resmon on http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
