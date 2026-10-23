"""QMT 界面感知原型 (2026-10-23): 截屏 -> DeepSeek deepseek-flash 视觉 -> 结构化的屏幕描述 / 目标坐标。

这一版只读屏幕、只点白名单里没有交易后果的按钮 (报错弹窗的「确定」等), 不碰下单面板。
  python -m qmt_bridge.ui_agent describe shot.png
  python -m qmt_bridge.ui_agent locate shot.png "登录失败弹窗里的确定按钮"
  python -m qmt_bridge.ui_agent dismiss-dialog --title 程序登录失败 --button 确定   (Windows 桌面会话里跑, 要提权)
10-23 实测: 弹窗标题/正文逐字正确, 按钮中心误差 2px; 桌面图标密集时会认错相邻图标 (170px) ——
所以点击前必须过 safe_target(): 模型给出的标签和预期一致、置信度够、坐标落在屏幕内, 否则不点。
密钥: 环境变量 DEEPSEEK_API_KEY, 或 D:\\qmtcode\\deepseek.key, 或 ~/.quant/deepseek.key (都不打印)。
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
KEY_FILES = (r"D:\qmtcode\deepseek.key", str(Path.home() / ".quant" / "deepseek.key"))

DESCRIBE_PROMPT = (
    "这是一张 {w}x{h} 的 Windows 屏幕截图。只返回 JSON, 键为: "
    "windows: 列表, 每项 {{title, kind (dialog|main|other), text (窗口内主要文字, 逐字), "
    "buttons: [{{label, x, y}}]}}。x,y 是元素中心在原图 {w}x{h} 像素坐标系里的位置 (不是归一化坐标)。"
    "看不清的不要猜, 不要编造不存在的窗口。")
LOCATE_PROMPT = (
    "这是一张 {w}x{h} 的 Windows 屏幕截图。找到: {target}。只返回 JSON: "
    "{{found (bool), label (你在图上看到的该元素的文字, 逐字), x, y (元素中心, 原图像素坐标), "
    "confidence (0~1), ambiguous (是否有多个相似元素容易混淆)}}。找不到就 found=false, 不要猜。")


def load_key():
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"].strip()
    for f in KEY_FILES:
        if os.path.exists(f):
            return Path(f).read_text(encoding="utf-8").strip()
    raise SystemExit("找不到 DeepSeek 密钥 (DEEPSEEK_API_KEY 或 " + " / ".join(KEY_FILES) + ")")


def png_size(png):
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("不是 PNG")
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def build_request(png, prompt, model=MODEL, detail="high", max_tokens=1500):
    return {"model": model, "temperature": 0, "max_tokens": max_tokens, "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode(),
                                                    "detail": detail}}]}]}


def parse_reply(resp):
    content = resp["choices"][0]["message"]["content"]
    data = json.loads(content)
    data.pop("type", None)                      # 模型偶尔把 response_format 的 type 也写进来
    return data


def ask(png, prompt, key=None, timeout=60, opener=urllib.request.urlopen):
    body = json.dumps(build_request(png, prompt)).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, headers={"Authorization": "Bearer " + (key or load_key()),
                                                              "Content-Type": "application/json"})
    t0 = time.time()
    with opener(req, timeout=timeout) as r:
        resp = json.load(r)
    return parse_reply(resp), {"latency_s": round(time.time() - t0, 2), "usage": resp.get("usage", {})}


def describe(png, **kw):
    w, h = png_size(png)
    return ask(png, DESCRIBE_PROMPT.format(w=w, h=h), **kw)


def locate(png, target, **kw):
    w, h = png_size(png)
    return ask(png, LOCATE_PROMPT.format(w=w, h=h, target=target), **kw)


VERIFY_PROMPT = (
    "这是屏幕上一小块区域的放大图, 红色十字的交点就是准备点击的位置。只返回 JSON: "
    "{{under_cross (十字交点正下方那个元素上的文字, 逐字; 交点落在空白处就返回空字符串), "
    "is_target (bool: 交点是否正好落在「{target}」上, 而不是旁边的别的元素)}}。")
REFINE_PROMPT = (
    "这是屏幕上一块区域的放大图 ({w}x{h} 像素)。在图里找到: {target}。只返回 JSON: "
    "{{found (bool), label (逐字), x, y (元素中心, 本图像素坐标), confidence (0~1)}}。")


def crop(png, cx, cy, w=360, h=220, scale=2, mark=True):
    """截取以 (cx,cy) 为中心的区域并放大; mark=True 时在中心画红十字。返回 (png, (x0, y0, scale))。"""
    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(png)).convert("RGB")
    x0, y0 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    x1, y1 = min(im.width, x0 + w), min(im.height, y0 + h)
    part = im.crop((x0, y0, x1, y1)).resize(((x1 - x0) * scale, (y1 - y0) * scale))
    if mark:
        d = ImageDraw.Draw(part)
        px, py = (cx - x0) * scale, (cy - y0) * scale
        d.line((px - 18, py, px + 18, py), fill=(255, 0, 0), width=3)
        d.line((px, py - 18, px, py + 18), fill=(255, 0, 0), width=3)
    buf = io.BytesIO()
    part.save(buf, format="PNG")
    return buf.getvalue(), (x0, y0, scale)


def verify_point(png, x, y, target, expect_label, **kw):
    """独立复核: 另起一次请求, 只看点击点周围的放大图, 问十字下面到底是什么。"""
    part, _ = crop(png, x, y)
    v, meta = ask(part, VERIFY_PROMPT.format(target=target), **kw)
    ok = bool(v.get("is_target")) and expect_label in str(v.get("under_cross", ""))
    return ok, {**v, "latency_s": meta["latency_s"]}


def locate_verified(png, target, expect_label, refine=1, **kw):
    """全图定位 -> 复核; 复核不过就在猜测点附近放大重找一次再复核。返回 (loc, 可以点, 记录)。"""
    size = png_size(png)
    loc, _ = locate(png, target, **kw)
    ok, why = safe_target(loc, expect_label, size)
    trail = [("locate", loc, why)]
    if not ok:
        return loc, False, trail
    for attempt in range(refine + 1):
        good, v = verify_point(png, loc["x"], loc["y"], target, expect_label, **kw)
        trail.append(("verify", (loc["x"], loc["y"]), v))
        if good:
            return loc, True, trail
        if attempt == refine:
            break
        part, (x0, y0, s) = crop(png, loc["x"], loc["y"], w=640, h=400, mark=False)
        pw, ph = png_size(part)
        r, _ = ask(part, REFINE_PROMPT.format(w=pw, h=ph, target=target), **kw)
        trail.append(("refine", r))
        if not r.get("found"):
            break
        loc = {**loc, "x": round(x0 + r["x"] / s), "y": round(y0 + r["y"] / s)}
    return loc, False, trail


def safe_target(loc, expect_label, size, min_conf=0.8):
    """点击前的硬检查: 找到了、标签与预期一致、置信度够、不含糊、坐标在屏幕内。返回 (可以点, 原因)。"""
    w, h = size
    if not loc.get("found"):
        return False, "没找到"
    if expect_label and expect_label not in str(loc.get("label", "")):
        return False, f"标签不符: 看到「{loc.get('label')}」, 预期含「{expect_label}」"
    if float(loc.get("confidence") or 0) < min_conf:
        return False, f"置信度 {loc.get('confidence')} < {min_conf}"
    if loc.get("ambiguous"):
        return False, "有相似元素, 容易点错"
    x, y = loc.get("x"), loc.get("y")
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and 0 <= x < w and 0 <= y < h):
        return False, f"坐标越界 ({x}, {y})"
    return True, "ok"


# ── Windows 桌面会话里的截屏与点击 (计划任务, 提权运行: QMT 是提权进程, 低权限注入的点击会被系统丢弃) ──
def screenshot_windows():
    import ctypes
    ctypes.windll.user32.SetProcessDPIAware()
    ps = ("Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
          "$b=[System.Windows.Forms.SystemInformation]::VirtualScreen;"
          "$m=New-Object System.Drawing.Bitmap $b.Width,$b.Height;"
          "$g=[System.Drawing.Graphics]::FromImage($m);$g.CopyFromScreen($b.Left,$b.Top,0,0,$m.Size);"
          "$s=New-Object System.IO.MemoryStream;$m.Save($s,[System.Drawing.Imaging.ImageFormat]::Png);"
          "[Convert]::ToBase64String($s.ToArray())")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
    return base64.b64decode(out.stdout.strip())


def click_windows(x, y):
    import ctypes
    u = ctypes.windll.user32
    u.SetProcessDPIAware()
    u.SetCursorPos(int(x), int(y))
    time.sleep(0.15)
    u.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.06)
    u.mouse_event(0x0004, 0, 0, 0, 0)


DISMISS_WHITELIST = {("程序登录失败", "确定")}         # 只允许点这些没有交易后果的按钮
PRESETS = {"login-failed": ("程序登录失败", "确定")}     # 批处理里只传 ASCII 名字 (cmd.exe 对中文参数编码不可靠)


def dismiss_dialog(title, button, log=print):
    if (title, button) not in DISMISS_WHITELIST:
        raise SystemExit(f"「{title}」/「{button}」不在白名单里, 不点")
    png = screenshot_windows()
    loc, ok, trail = locate_verified(png, f"标题为「{title}」的弹窗里的「{button}」按钮", button)
    log(f"locate+verify: ok={ok} trail={trail}")
    if not ok:
        return False
    click_windows(loc["x"], loc["y"])
    time.sleep(1.5)
    after, _ = describe(screenshot_windows())
    still = [w for w in after.get("windows", []) if title in str(w.get("title", ""))]
    log(f"clicked ({loc['x']},{loc['y']}); dialog still present = {bool(still)}; windows now = "
        f"{[w.get('title') for w in after.get('windows', [])]}")
    return not still


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe")
    d.add_argument("png")
    lo = sub.add_parser("locate")
    lo.add_argument("png")
    lo.add_argument("target")
    di = sub.add_parser("dismiss-dialog")
    di.add_argument("--preset", choices=sorted(PRESETS))
    di.add_argument("--title")
    di.add_argument("--button")
    a = ap.parse_args(argv)
    if a.cmd == "dismiss-dialog":
        title, button = PRESETS[a.preset] if a.preset else (a.title, a.button)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] dismiss-dialog {title}/{button}", flush=True)
        return 0 if dismiss_dialog(title, button) else 1
    png = Path(a.png).read_bytes()
    res, meta = describe(png) if a.cmd == "describe" else locate(png, a.target)
    print(json.dumps({"result": res, "meta": meta}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
