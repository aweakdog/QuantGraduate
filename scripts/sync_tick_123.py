#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""123 云盘逐笔数据日同步: 发现新 .7z -> 自动转存 -> 官方 OpenAPI 下载

把人从回路里拿掉的那一步。整条链路 2026-08-16 凌晨逐环节实证过:

  1. 匿名列供应商分享      share_fs_list @ {uid}.share.123pan.cn/b (无需凭证)
  2. 对比自己账号缺哪些 .7z  官方 OpenAPI /api/v2/file/list
  3. 转存缺的               web 登录态 POST /api/file/copy/async ("保存至云盘")
  4. 下载到本地             官方 OpenAPI /api/v1/file/download_info (实测 ~17MB/s)

已知的坑 (都在本脚本里处理):
  - web 密码登录会作废 open 平台 token -> 任何 web 操作后强制重换 open token
  - open token 缓存 401 时也要重换 (服务端可能提前作废)
  - web JWT 缓存复用, 失效才密码重登, 避免每天一次密码登录画像
  - 下载先写 .part 再原子改名, 半截文件不会被当成完成
  - 本地存在性同时查 123 树和百度树 (0803~0807 由百度渠道先到)
  - 11-04 起供应商把大包切成分卷 (20261104.7z.001/.002, 分享里留了个目录叫
    "带001，002 全下，解压 001 即可"): 分卷照常转存下载, 齐了按序拼回 <日>.7z
    (7-Zip 分卷就是整包按字节切开), 下游只认 <日>.7z。只认 .7z 时这些天被静默漏掉
  - 供应商同名重发 (20261102.7z 11-04 换成新格式, 5.56GB -> 4.04GB): 分享与账号同名但大小
    不同 -> 账号旧版移入回收站后重新转存、下载

凭证文件 (全部 600, 不进仓库):
  ~/.config/123open/credentials.json      {"client_id","client_secret"}
  ~/.config/123open/web_credentials.json  {"passport","password"}
  ~/.config/123open/token.json            open token 缓存 (自动生成)
  ~/.config/123open/web_token.json        web JWT 缓存 (自动生成)

用法:
  sync_tick_123.py --check      只打印三方差异 (分享/账号/本地), 无任何变更
  sync_tick_123.py              完整同步 (转存缺的 + 下载缺的)
  sync_tick_123.py --no-copy    只下载账号里已有而本地缺的

运行环境: ~/venv123/bin/python (3.12, 装了 p123client; open 部分纯标准库)
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- 常量
OPEN_BASE = "https://open-api.123pan.com"
SHARE_BASE = "https://1819606015.share.123pan.cn/b"   # 供应商分享域
YUN_BASE = "https://yun.123pan.cn/b"                   # web 登录态接口域
SHARE_KEY = "98zQjv-9V8rv"                             # 供应商分享 (逐笔日更)
SHARE_PWD = "7wzR"
TOP_FOLDER = "L2增量更新"                               # 分享内与账号内同名顶层目录

CFG = os.path.expanduser("~/.config/123open")
CRED_OPEN = os.path.join(CFG, "credentials.json")
CRED_WEB = os.path.join(CFG, "web_credentials.json")
TOK_OPEN = os.path.join(CFG, "token.json")
TOK_WEB = os.path.join(CFG, "web_token.json")

DEST = "/home/yliog/tickdata123"                       # 本渠道下载目录 (按月分子目录)
BAIDU_TREE = "/home/yliog/tickdata/----逐笔委托成交行情-明细---"  # 百度渠道已有的树

PAUSE = 1.2          # open 平台 QPS 很低, 每次调用间隔
LOCK_FILE = "/tmp/sync_tick_123.lock"
PACK_RE = re.compile(r"^(\d{8})\.7z(?:\.(\d{3}))?$")  # 整包 20261103.7z 或分卷 20261104.7z.001
DL_CONN = 8          # 并行分段下载连接数
DL_TRIES = 30        # 每段最多续传次数


def is_pack(name):
    return bool(PACK_RE.match(name))


def volumes(files):
    """{名字: 大小} -> {日期: {分卷名: 大小}}, 只收 .7z.NNN 分卷"""
    out = {}
    for n, size in files.items():
        m = PACK_RE.match(n)
        if m and m.group(2):
            out.setdefault(m.group(1), {})[n] = size
    return out


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def _save_600(path, obj):
    os.makedirs(CFG, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)


# ---------------------------------------------------------------- open 平台
def _http_json(url, payload=None, headers=None, method=None, timeout=60):
    body = json.dumps(payload).encode() if payload is not None else None
    h = {"Content-Type": "application/json", "Platform": "open_platform",
         "User-Agent": "quant-strategy-feed/1.0"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h,
                                 method=method or ("POST" if body else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8", "replace"))
        except Exception:
            return {"code": e.code, "message": "HTTP %d" % e.code}


def open_token(force=False):
    """open 平台 token; force=True 或缓存过期/被作废时重换"""
    if not force and os.path.exists(TOK_OPEN):
        t = json.load(open(TOK_OPEN))
        margin = time.strftime("%Y-%m-%dT%H:%M:%S",
                               time.localtime(time.time() + 86400))
        if t.get("expiredAt", "") > margin:
            return t["accessToken"]
    c = json.load(open(CRED_OPEN))
    j = _http_json(OPEN_BASE + "/api/v1/access_token",
                   {"clientID": c["client_id"], "clientSecret": c["client_secret"]})
    if j.get("code") != 0:
        raise RuntimeError("换 open token 失败: %s" % json.dumps(j, ensure_ascii=False)[:200])
    _save_600(TOK_OPEN, j["data"])
    log("open token 已刷新, 有效期至", j["data"].get("expiredAt"))
    return j["data"]["accessToken"]


def open_api(path, params=None, payload=None, _retried=False):
    """open 平台调用, 401 自动重换 token 重试一次 (web 登录会作废旧 token)"""
    time.sleep(PAUSE)
    url = OPEN_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    j = _http_json(url, payload=payload,
                   headers={"Authorization": "Bearer " + open_token()})
    if j.get("code") == 401 and not _retried:
        log("open token 被服务端作废, 重换后重试:", path)
        open_token(force=True)
        return open_api(path, params, payload, _retried=True)
    return j


def open_list(parent):
    """列自己账号目录 (翻页), 排除回收站"""
    out, last = [], None
    while True:
        p = {"parentFileId": parent, "limit": 100}
        if last:
            p["lastFileId"] = last
        j = open_api("/api/v2/file/list", p)
        if j.get("code") != 0:
            raise RuntimeError("open list 失败 parent=%s: %s"
                               % (parent, json.dumps(j, ensure_ascii=False)[:200]))
        d = j["data"]
        out += [f for f in d.get("fileList", []) if f.get("trashed") != 1]
        last = d.get("lastFileId")
        if last in (-1, None):
            return out


def open_mkdir(parent, name):
    """在自己账号建目录, 返回目录 id (响应结构不稳的话回退到重列查找)"""
    j = open_api("/upload/v1/file/mkdir", payload={"name": name, "parentID": parent})
    d = j.get("data") or {}
    did = d.get("dirID") or d.get("dirId") or d.get("fileId")
    if j.get("code") == 0 and did:
        return did
    for f in open_list(parent):
        if f.get("filename") == name and f.get("type") == 1:
            return f["fileId"]
    raise RuntimeError("mkdir %r 失败: %s" % (name, json.dumps(j, ensure_ascii=False)[:200]))


# ---------------------------------------------------------------- 分享侧 (匿名)
def share_ls(parent=0):
    from p123client import P123Client
    out, page = [], 1
    while page < 8:
        r = P123Client.share_fs_list(
            None, {"ShareKey": SHARE_KEY, "SharePwd": SHARE_PWD,
                   "ParentFileId": parent, "limit": 100, "Page": page},
            base_url=SHARE_BASE)
        if r.get("code") != 0:
            raise RuntimeError("share list 失败: %s"
                               % json.dumps(r, ensure_ascii=False)[:200])
        d = r.get("data") or {}
        lst = d.get("InfoList") or []
        out += lst
        if str(d.get("Next", "-1")) == "-1" or len(lst) < 100:
            return out
        page += 1
    return out


# ---------------------------------------------------------------- web 登录态
def web_client():
    """优先用缓存 JWT; 失效才密码重登 (减少登录画像 + 少作废 open token)"""
    from p123client import P123Client
    if os.path.exists(TOK_WEB):
        tok = json.load(open(TOK_WEB)).get("token")
        if tok:
            try:
                c = P123Client(token=tok)
                if c.user_info(base_url=YUN_BASE).get("code") == 0:
                    log("web token 缓存可用")
                    return c
            except Exception:
                pass
            log("web token 缓存失效, 走密码重登")
    cred = json.load(open(CRED_WEB))
    c = P123Client(passport=cred["passport"], password=cred["password"])
    _save_600(TOK_WEB, {"token": c.token, "at": time.strftime("%F %T")})
    log("web 密码登录成功, JWT 已缓存 (注意: 这会作废 open token)")
    return c


def share_copy(client, entries, target_parent):
    """转存分享里的文件到自己账号 target_parent 目录"""
    payload = {
        "share_key": SHARE_KEY,
        "share_pwd": SHARE_PWD,
        "file_list": [{
            "file_id": e["FileId"],
            "file_name": e["FileName"],
            "etag": e.get("Etag", ""),
            "size": e.get("Size"),
        } for e in entries],
    }
    r = client.share_fs_copy(payload, parent_id=target_parent, base_url=YUN_BASE)
    if r.get("code") != 0:
        raise RuntimeError("转存失败: %s" % json.dumps(r, ensure_ascii=False)[:300])
    log("转存请求已受理:", ", ".join(e["FileName"] for e in entries))


# ---------------------------------------------------------------- 本地
def local_paths(month, name):
    """一个文件在本地可能存在的所有位置 (123 树 + 百度树)"""
    year = month[:4]
    return [
        os.path.join(DEST, month, name),
        os.path.join(BAIDU_TREE, year, month, name),
    ]


def local_ok(month, name, size):
    p123, pbd = local_paths(month, name)
    if os.path.exists(p123) and os.path.getsize(p123) == size:
        return p123
    # 百度树是另一渠道打的包, 压缩参数不同导致大小天然不等 (34/86 天不一致,
    # 见 findings 2026-08-16 §3)。按精确大小判会把这些天误判为缺失, 每次
    # 同步重复拉 ~100GB。百度版是完整口径, 存在且非空即视为已有。
    # 但符号链接是 tick_daily_extract 从 123 树挂过去的, 不是独立副本 —— 不算,
    # 否则供应商同名重发后永远不会重新下载。
    if os.path.exists(pbd) and not os.path.islink(pbd) and os.path.getsize(pbd) > 0:
        return pbd
    return None


def assemble(month, day, vols):
    """分卷齐了 (编号 001..N 连续、本地大小都对) 就按序拼成 <日>.7z; 拼过且分卷没变则跳过。
    分卷保留不删: 供应商若还在补传后续分卷, 下次能直接重拼。拼残的包打不开, 下游质检会挡。"""
    nums = sorted(int(PACK_RE.match(n).group(2)) for n in vols)
    paths = [os.path.join(DEST, month, n) for n in sorted(vols)]
    final, meta = os.path.join(DEST, month, day + ".7z"), os.path.join(DEST, month, day + ".7z.vols.json")
    if nums != list(range(1, len(nums) + 1)) or not all(
            os.path.exists(p) and os.path.getsize(p) == vols[os.path.basename(p)] for p in paths):
        return None
    if os.path.exists(final) and os.path.exists(meta):
        with open(meta) as f:
            if json.load(f) == vols:
                return None
    with open(final + ".part", "wb") as out:
        for p in paths:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 24)
    os.replace(final + ".part", final)
    with open(meta, "w") as f:
        json.dump(vols, f)
    log(f"分卷合并: {day}.7z <- {' + '.join(sorted(vols))} ({os.path.getsize(final) / 1e9:.2f} GB)")
    return final


def _fetch_range(url, part, start, end, got, tries=DL_TRIES):
    """下载 [start, end] 写进 part 的对应位置; got[start] 记已写字节 (给进度用)。
    读超时/断流就从已写到的位置续传 (11-07~09 每班都因单段超时整班作废、从头再来)。"""
    for i in range(tries):
        pos = start + got[start]
        if pos > end:
            return
        req = urllib.request.Request(url, headers={"User-Agent": "quant-strategy-feed/1.0",
                                                   "Range": f"bytes={pos}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "r+b") as f:
                if r.status != 206:
                    raise RuntimeError(f"服务端不支持分段下载 (HTTP {r.status})")
                f.seek(pos)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    got[start] += len(chunk)
        except RuntimeError:
            raise
        except Exception as e:        # 超时、断流、IncompleteRead: 续传
            if i == tries - 1:
                raise
            log(f"  分段 {start}-{end} 第 {i + 1} 次中断 ({type(e).__name__}), 从 {got[start]} 字节处续传")
            time.sleep(min(60, 5 * (i + 1)))
    if start + got[start] != end + 1:
        raise RuntimeError(f"分段 {start}-{end} 不完整: {got[start]} 字节")


def download(file_id, month, name, size, etag=""):
    """open 平台下载: DL_CONN 段并行写 .part, 校验大小和 MD5 (123 的 etag 就是 MD5) 后原子改名。
    跨境链路白天单连接常只有 0.5~1 MB/s (11-06 15:40 实测), 8 段并行约 3 倍。"""
    j = open_api("/api/v1/file/download_info", {"fileId": file_id})
    if j.get("code") != 0:
        raise RuntimeError("download_info 失败 %s: %s"
                           % (name, json.dumps(j, ensure_ascii=False)[:200]))
    url = (j.get("data") or {}).get("downloadUrl")
    if not url:
        raise RuntimeError("无 downloadUrl: %s" % name)
    dirp = os.path.join(DEST, month)
    os.makedirs(dirp, exist_ok=True)
    part = os.path.join(dirp, name + ".part")
    final = os.path.join(dirp, name)
    with open(part, "wb") as f:
        f.truncate(size)
    step = -(-size // DL_CONN)
    ranges = [(a, min(a + step, size) - 1) for a in range(0, size, step)]
    got = dict.fromkeys((a for a, _ in ranges), 0)
    t0 = time.time()
    with ThreadPoolExecutor(len(ranges)) as ex:
        futs = [ex.submit(_fetch_range, url, part, a, b, got) for a, b in ranges]
        while not all(f.done() for f in futs):
            time.sleep(10)
            total = sum(got.values())
            log("  %s %.2f/%.2f GB  平均 %.1f MB/s"
                % (name, total / 1e9, size / 1e9, total / (time.time() - t0) / 1e6))
        for f in futs:
            f.result()       # 任一段出错就抛, 保留 .part 以便排查
    total = sum(got.values())
    if total != size:
        raise RuntimeError("大小不符 %s: 得到 %d 期望 %d (保留 .part 以便排查)"
                           % (name, total, size))
    if re.fullmatch(r"[0-9a-f]{32}", etag or ""):
        h = hashlib.md5()
        with open(part, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        if h.hexdigest() != etag:
            raise RuntimeError(f"MD5 不符 {name}: {h.hexdigest()} != {etag} (保留 .part 以便排查)")
    os.replace(part, final)
    log("  %s 完成: %.2f GB, %.0fs, 平均 %.1f MB/s"
        % (name, total / 1e9, time.time() - t0, total / (time.time() - t0) / 1e6))
    return final


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="123 云盘逐笔数据日同步")
    ap.add_argument("--check", action="store_true", help="只打印差异, 不做任何变更")
    ap.add_argument("--no-copy", action="store_true", help="跳过转存, 只下载")
    ap.add_argument("--only", default="",
                    help="只处理文件名以此开头的 (如 202608 或 20260413)")
    args = ap.parse_args()

    lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("已有一个同步在跑, 退出")
        return 0

    # 1. 分享侧清单
    log("== 1/4 列供应商分享 (匿名) ==")
    share_root = share_ls(0)
    top = next((x for x in share_root if x.get("FileName") == TOP_FOLDER), None)
    if not top:
        raise RuntimeError("分享里没找到 %r, 结构变了?" % TOP_FOLDER)
    share_files = {}       # month -> {name: entry}
    for m in share_ls(top["FileId"]):
        if m.get("Type") != 1:
            continue
        mon = m["FileName"]
        items = share_ls(m["FileId"])
        share_files[mon] = {f["FileName"]: f for f in items
                            if f.get("Type") != 1 and is_pack(f["FileName"])}
        other = [f["FileName"] for f in items if f["FileName"] not in share_files[mon]]
        if other:       # 供应商换打包方式时先在这里露头 (11-04 的分卷就是这么漏掉的)
            log(f"  {mon} 里不认识、不下载的条目: {', '.join(other)}")
    n_share = sum(len(v) for v in share_files.values())
    log("分享共 %d 个月目录, %d 个包文件 (含分卷), 最新: %s"
        % (len(share_files), n_share,
           max((n for v in share_files.values() for n in v), default="(无)")))

    # 2. 账号侧清单
    log("== 2/4 列自己账号 (openAPI) ==")
    mine_root = open_list(0)
    mine_top = next((x for x in mine_root
                     if x.get("filename") == TOP_FOLDER and x.get("type") == 1), None)
    mine_files, mine_month_ids = {}, {}
    if mine_top:
        for m in open_list(mine_top["fileId"]):
            if m.get("type") != 1:
                continue
            mine_month_ids[m["filename"]] = m["fileId"]
            mine_files[m["filename"]] = {
                f["filename"]: f for f in open_list(m["fileId"])
                if f.get("type") != 1 and is_pack(f["filename"])}
    n_mine = sum(len(v) for v in mine_files.values())
    log("账号共 %d 个月目录, %d 个包文件" % (len(mine_files), n_mine))

    # 3. 差异 (账号里同名但大小不同 = 供应商重发, 旧版先进回收站再转存)
    to_copy, to_replace = {}, []      # month -> [share entry]; [(month, 账号旧文件)]
    for mon, files in share_files.items():
        for n, e in sorted(files.items()):
            old = mine_files.get(mon, {}).get(n)
            if not n.startswith(args.only) or (old and old.get("size") == e.get("Size")):
                continue
            to_copy.setdefault(mon, []).append(e)
            if old:
                to_replace.append((mon, old))
    to_download = []       # (fileId, month, name, size, etag)
    for mon, files in mine_files.items():
        for n, f in sorted(files.items()):
            if n.startswith(args.only) and not local_ok(mon, n, f.get("size", -1)):
                to_download.append((f["fileId"], mon, n, f["size"], f.get("etag", "")))

    log("待转存 %d 个: %s"
        % (sum(len(v) for v in to_copy.values()),
           ", ".join(e["FileName"] for v in to_copy.values() for e in v) or "(无)"))
    if to_replace:
        log(f"其中供应商同名重发 {len(to_replace)} 个: " + ", ".join(
            f"{f['filename']} (账号 {f.get('size', 0) / 1e9:.2f}GB -> "
            f"分享 {share_files[m][f['filename']].get('Size', 0) / 1e9:.2f}GB)" for m, f in to_replace))
    log("待下载 %d 个: %s"
        % (len(to_download), ", ".join(x[2] for x in to_download) or "(无)"))
    if args.check:
        return 0

    # 4. 转存 + 确认 + 下载
    if to_copy and not args.no_copy:
        log("== 3/4 转存 ==")
        if to_replace:     # open 平台操作放在 web 登录之前 (web 登录会作废 open token)
            j = open_api("/api/v1/file/trash", payload={"fileIDs": [f["fileId"] for _, f in to_replace]})
            if j.get("code") != 0:
                raise RuntimeError(f"旧版移入回收站失败: {json.dumps(j, ensure_ascii=False)[:200]}")
            log("账号旧版已移入回收站:", ", ".join(f["filename"] for _, f in to_replace))
            gone = {(m, f["filename"]) for m, f in to_replace}
            to_download = [x for x in to_download if (x[1], x[2]) not in gone]
        c = web_client()
        for mon, entries in sorted(to_copy.items()):
            if mon not in mine_month_ids:
                if not mine_top:
                    raise RuntimeError("账号里没有 %r 顶层目录, 请先手工转存一次建立结构"
                                       % TOP_FOLDER)
                open_token(force=True)   # web 登录可能刚作废过 token
                mine_month_ids[mon] = open_mkdir(mine_top["fileId"], mon)
                log("账号新建月目录 %s (id=%s)" % (mon, mine_month_ids[mon]))
            share_copy(c, entries, mine_month_ids[mon])
        open_token(force=True)           # web 操作之后必须重换
        log("等转存落账...")
        want = {(mon, e["FileName"]) for mon, v in to_copy.items() for e in v}
        for i in range(30):
            time.sleep(10)
            got = set()
            for mon in {m for m, _ in want}:
                names = {f["filename"] for f in open_list(mine_month_ids[mon])}
                got |= {(mon, n) for _, n in want if n in names and _ == mon}
            if got >= want:
                log("全部落账 (第 %d 次确认)" % (i + 1))
                break
            log("已落账 %d/%d ..." % (len(got), len(want)))
        else:
            raise RuntimeError("300s 内转存未全部落账, 中止 (下次运行会自动续)")
        # 落账的文件加入下载队列
        for mon in to_copy:
            for f in open_list(mine_month_ids[mon]):
                n = f.get("filename", "")
                if is_pack(n) and not local_ok(mon, n, f.get("size", -1)) \
                        and all(x[2] != n for x in to_download):
                    to_download.append((f["fileId"], mon, n, f["size"], f.get("etag", "")))

    def assemble_all():
        for mon, files in share_files.items():
            for day, vols in volumes({n: e.get("Size") for n, e in files.items()}).items():
                if day.startswith(args.only):
                    assemble(mon, day, vols)

    assemble_all()                     # 上一班已下齐、但没来得及拼的先拼 (11-05 就卡在这)
    failed = []
    if to_download:
        log("== 4/4 下载 %d 个 (最新优先) ==" % len(to_download))
        for fid, mon, name, size, etag in sorted(to_download, key=lambda x: x[2], reverse=True):
            try:                       # 一个文件失败不拖累其余文件, 也不挡住拼包
                download(fid, mon, name, size, etag)
                assemble_all()
            except Exception as e:
                failed.append(name)
                log(f"  ✗ {name} 下载失败 ({type(e).__name__}: {e}), 下一班重试")
    log("同步完成: 转存 %d, 下载 %d"
        % (sum(len(v) for v in to_copy.values()) if not args.no_copy else 0,
           len(to_download) - len(failed)))
    if failed:
        raise RuntimeError(f"{len(failed)} 个文件没下成: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
