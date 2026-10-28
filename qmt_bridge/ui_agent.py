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
import urllib.error
import urllib.request
from pathlib import Path

# 视觉模型可切换 (10-27 用户要求试 MiniMax): 环境变量 VISION_PROVIDER 或 D:\qmtcode\vision_provider.txt, 默认 deepseek
PROVIDERS = {
    "deepseek": {"url": "https://api.deepseek.com/chat/completions", "model": "deepseek-flash", "key": "deepseek.key",
                 "extra": {"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}}, "image": {}},
    "minimax": {"url": "https://api.minimaxi.com/v1/chat/completions", "model": "MiniMax-M3.1-Flash", "key": "minimax.key",
                "extra": {"reasoning_effort": "low"}, "image": {"max_long_side_pixel": 3840}},   # 原图不缩小, 小字才读得清
    # 10-28 加: 阿里云国际站 (新加坡) 百炼; 用户密钥只认 dashscope-intl
    "qwen": {"url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", "model": "qwen3-vl-plus",
             "key": "qwen.key", "extra": {"vl_high_resolution_images": True}, "image": {}, "coords": "norm1000"},
}
API_URL = PROVIDERS["deepseek"]["url"]
MODEL = PROVIDERS["deepseek"]["model"]
KEY_FILES = (r"D:\qmtcode\deepseek.key", str(Path.home() / ".quant" / "deepseek.key"))


def provider_name():
    if os.environ.get("VISION_PROVIDER"):
        return os.environ["VISION_PROVIDER"].strip()
    f = Path(r"D:\qmtcode\vision_provider.txt")
    return f.read_text(encoding="utf-8").strip() if f.exists() else "deepseek"


def provider_key(name):
    env = os.environ.get(f"{name.upper()}_API_KEY")
    if env:
        return env.strip()
    for f in (Path(r"D:\qmtcode") / PROVIDERS[name]["key"], Path.home() / ".quant" / PROVIDERS[name]["key"]):
        if f.exists():
            return f.read_text(encoding="utf-8").strip()
    raise SystemExit(f"找不到 {name} 的 API 密钥")

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


def build_request(png, prompt, model=None, detail="high", max_tokens=4000, provider=None):
    pv = PROVIDERS[provider or "deepseek"]
    image = {"url": "data:image/png;base64," + base64.b64encode(png).decode(), "detail": detail, **pv["image"]}
    return {"model": model or pv["model"], "temperature": 0, "max_tokens": max_tokens, **pv["extra"],
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": image}]}]}


def parse_reply(resp):
    """取回复里的 JSON。不支持 json 模式的模型可能包代码块或前后带字, 取第一个 { 到最后一个 }。"""
    content = resp["choices"][0]["message"]["content"] or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        i, j = content.find("{"), content.rfind("}")
        if i < 0 or j <= i:
            raise
        data = json.loads(content[i:j + 1])
    data.pop("type", None)                      # 模型偶尔把 response_format 的 type 也写进来
    return data


TRANSIENT = {429, 500, 502, 503, 504, 529}


def _post(req, opener, timeout):
    """发请求; 限流/过载/超时这类临时错误等 2 秒、5 秒各重试一次 (10-28: MiniMax 返回 529 过载, 上午逆回购因此中断)。"""
    for wait in (2, 5, None):
        try:
            with opener(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in TRANSIENT or wait is None:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if wait is None:
                raise
        time.sleep(wait)


# 一家重试后仍不行就按顺序换下一家 (只换有密钥的); MiniMax 挂了先换千问, 交叉核对仍是两家不同的模型 (10-28)
FALLBACK = {"minimax": ("qwen", "deepseek"), "qwen": ("deepseek",), "deepseek": ("qwen",)}


def ask(png, prompt, key=None, timeout=90, opener=urllib.request.urlopen, retries=1, provider=None, fallbacks=None):
    """一次视觉问答。临时错误先重试; 仍不行按 FALLBACK (或传入的 fallbacks) 换下一家顶上 (meta 里记 fallback)。"""
    name = provider or provider_name()
    nxt = FALLBACK.get(name, ()) if fallbacks is None else fallbacks
    chain = [name] + ([] if key is not None else [p for p in nxt if _has_key(p)])
    for i, prov in enumerate(chain):
        try:
            data, meta = _ask_once(png, prompt, key if i == 0 else None, timeout, opener, retries, prov)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == len(chain) - 1:
                raise
            continue
        if i:
            meta["fallback"] = prov
        return data, meta


def _ask_once(png, prompt, key, timeout, opener, retries, name):
    """回复不是合法 JSON (多半是输出被截断) 时提醒精简后重试一次。"""
    key = key or (load_key() if name == "deepseek" else provider_key(name))
    t0 = time.time()
    for attempt in range(retries + 1):
        p = prompt if attempt == 0 else prompt + "\n上次回复不是合法 JSON。请精简内容, 只返回一个完整、合法的 JSON 对象。"
        body = json.dumps(build_request(png, p, provider=name)).encode("utf-8")
        req = urllib.request.Request(PROVIDERS[name]["url"], data=body, headers={"Authorization": "Bearer " + key,
                                                                  "Content-Type": "application/json"})
        resp = _post(req, opener, timeout)
        try:
            return parse_reply(resp), {"latency_s": round(time.time() - t0, 2), "usage": resp.get("usage", {}),
                                       "attempts": attempt + 1}
        except json.JSONDecodeError:
            if attempt == retries:
                raise


def describe(png, **kw):
    w, h = png_size(png)
    return ask(png, DESCRIBE_PROMPT.format(w=w, h=h), **kw)


# 定位不用 DeepSeek (10-28 评测: 它的坐标时而是缩小图上的像素 (约 0.93 倍), 时而是 0-1000 相对值, 无法稳定换算)
LOCATE_FALLBACKS = ("qwen",)


def to_pixels(provider, x, y, w, h):
    """Qwen3-VL 的坐标是 0-1000 相对坐标 (10-28 评测发现), 换算成原图像素; 其他家原样返回。"""
    if PROVIDERS.get(provider or "", {}).get("coords") == "norm1000":
        return round(float(x) / 1000 * w), round(float(y) / 1000 * h)
    return x, y


def locate(png, target, **kw):
    """定位交给 MiniMax (10-27 评测: 密集按钮偏 10px, DeepSeek 偏 500px); 没有 MiniMax 密钥时退回默认。"""
    w, h = png_size(png)
    if "provider" not in kw and _has_key("minimax"):
        kw["provider"] = "minimax"
        kw.setdefault("fallbacks", LOCATE_FALLBACKS)
    loc, meta = ask(png, LOCATE_PROMPT.format(w=w, h=h, target=target), **kw)
    used = meta.get("fallback") or kw.get("provider")
    if loc.get("found") and loc.get("x") is not None and loc.get("y") is not None:
        loc["x"], loc["y"] = to_pixels(used, loc["x"], loc["y"], w, h)
    return loc, meta


def _has_key(name):
    try:
        load_key() if name == "deepseek" else provider_key(name)
        return True
    except SystemExit:
        return False


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
    part = im.crop((x0, y0, x1, y1)).resize((round((x1 - x0) * scale), round((y1 - y0) * scale)))   # scale 可为小数 (10-27: 2.5 倍时崩过)
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


# ── 普通方式登录 (取消「独立交易」) ──────────────────────────────
# 登录窗口出现约 5 秒后自动登录, 比一次视觉调用 (3~5 秒) 还快, 所以第一下用相对窗口的固定偏移
# 先取消「自动登录」让倒计时停下; 之后的勾选状态核对和点击都走 DeepSeek 定位+复核。
# 偏移量 10-23 从 2.1.31.3 的 624x443 登录窗量出: 记住密码 (198,315) 自动登录 (278,315) 独立交易 (367,315) 登录 (240,346)
LOGIN_OFFSETS = {"auto_login": (278, 315), "independent": (367, 315)}
LOGIN_STATE_PROMPT = (
    "这是 Windows 屏幕截图, 上面应有一个「迅投极速交易终端」登录窗口。只返回 JSON: "
    "{{login_window (bool), tabs_selected (顶部 行情+交易/交易/行情 哪个页签是选中状态), "
    "remember_password (bool, 「记住密码」是否勾选), auto_login (bool, 「自动登录」是否勾选), "
    "independent_trading (bool, 「独立交易」是否勾选), other_dialogs (其他弹窗的标题和文字列表)}}")


def find_window(title_part, timeout=15.0, size=None):
    """按标题找可见窗口, 返回 (hwnd, (left, top, w, h)); size=(w,h) 时要求尺寸大致吻合 (±40px)。"""
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    u.SetProcessDPIAware()
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    t0 = time.time()
    while time.time() - t0 < timeout:
        found = []

        def cb(h, _, found=found):
            if u.IsWindowVisible(h):
                buf = ctypes.create_unicode_buffer(256)
                u.GetWindowTextW(h, buf, 256)
                if title_part in buf.value:
                    r = wintypes.RECT()
                    u.GetWindowRect(h, ctypes.byref(r))
                    rect = (r.left, r.top, r.right - r.left, r.bottom - r.top)
                    if size is None or (abs(rect[2] - size[0]) < 40 and abs(rect[3] - size[1]) < 40):
                        found.append((h, rect))
            return True
        u.EnumWindows(enum_proc(cb), 0)
        if found:
            return found[0]
        time.sleep(0.2)
    return None, None


def checkbox_state(x, y):
    """勾选框状态: 取中心 12x12 像素平均色。蓝底=checked, 近白=unchecked, 其他 (灰/黑, 窗口还没画完)=unknown。"""
    from PIL import ImageGrab
    im = ImageGrab.grab(bbox=(x - 6, y - 6, x + 6, y + 6), all_screens=True).convert("RGB")
    px = [im.getpixel((i, j)) for i in range(im.width) for j in range(im.height)]
    r, g, b = (sum(p[i] for p in px) / len(px) for i in range(3))
    rgb = (round(r), round(g), round(b))
    if b > 170 and r < 120:
        return "checked", rgb
    if min(rgb) > 200:
        return "unchecked", rgb
    return "unknown", rgb


def toggle_off(hwnd, x, y, log, tries=4, render_wait=4.0):
    import ctypes
    u = ctypes.windll.user32
    t0, clicks = time.time(), 0
    while True:
        st, rgb = checkbox_state(x, y)
        if st == "unchecked":
            log(f"  checkbox ({x},{y}): unchecked rgb={rgb} after {clicks} click(s)")
            return True
        if st == "unknown":
            if time.time() - t0 > render_wait:
                log(f"  checkbox ({x},{y}): still not rendered rgb={rgb}")
                return False
            time.sleep(0.15)
            continue
        if clicks >= tries:
            log(f"  checkbox ({x},{y}): still checked after {clicks} clicks rgb={rgb}")
            return False
        u.ShowWindow(hwnd, 5)
        u.SetForegroundWindow(hwnd)
        time.sleep(0.1)
        click_windows(x, y)
        clicks += 1
        time.sleep(0.15)
        u.SetCursorPos(x - 150, y - 120)          # 移开鼠标再读颜色, 避免悬停高亮干扰
        time.sleep(0.2)


EXPECTED_SCREEN = (1920, 1080)        # 读屏坐标都是按这个分辨率量的


def ensure_qmt_foreground(log=print):
    """找 QMT 主窗口 (不是 624x443 的登录框): 最小化就恢复, 没最大化就最大化, 切到最前,
    再核对它铺满屏幕、分辨率仍是 1920x1080。返回问题列表 (空 = 可以读屏/点击)。"""
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    u.SetProcessDPIAware()
    screen = (u.GetSystemMetrics(0), u.GetSystemMetrics(1))
    if screen != EXPECTED_SCREEN:
        return [f"屏幕分辨率是 {screen[0]}x{screen[1]}, 读屏坐标按 {EXPECTED_SCREEN[0]}x{EXPECTED_SCREEN[1]} 量的"]
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    found = []

    def cb(h, _):
        buf = ctypes.create_unicode_buffer(256)
        u.GetWindowTextW(h, buf, 256)
        if "QMT实盘" in buf.value and u.IsWindowVisible(h):
            r = wintypes.RECT()
            u.GetWindowRect(h, ctypes.byref(r))
            if u.IsIconic(h) or (r.right - r.left) > 900:           # 主窗口 (登录框只有 624 宽)
                found.append(h)
        return True
    u.EnumWindows(enum_proc(cb), 0)
    if not found:
        return ["找不到 QMT 主窗口 (未登录或已退出)"]
    h = found[0]
    if u.IsIconic(h):
        u.ShowWindow(h, 9)                                       # SW_RESTORE
        time.sleep(0.5)
    if not u.IsZoomed(h):
        u.ShowWindow(h, 3)                                       # SW_MAXIMIZE
        time.sleep(0.8)
    u.keybd_event(0x12, 0, 0, 0)                                 # 按一下 Alt, 解除前台锁再切前台
    u.keybd_event(0x12, 0, 2, 0)
    u.SetForegroundWindow(h)
    time.sleep(0.6)
    r = wintypes.RECT()
    u.GetWindowRect(h, ctypes.byref(r))
    probs = []
    if not u.IsZoomed(h):
        probs.append("QMT 主窗口最大化失败")
    if r.left > 0 or r.top > 0 or r.right < screen[0] or r.bottom < screen[1] - 80:   # 任务栏约占 40~60px
        probs.append(f"QMT 主窗口没铺满屏幕: {(r.left, r.top, r.right, r.bottom)}")
    if u.GetForegroundWindow() != h:
        probs.append("QMT 主窗口没能切到最前 (可能被别的窗口挡住)")
    log(f"QMT 主窗口: rect={(r.left, r.top, r.right, r.bottom)} maximized={bool(u.IsZoomed(h))} "
        f"foreground={u.GetForegroundWindow() == h} problems={probs}")
    return probs


def login_state(log=print):
    state, meta = ask(screenshot_windows(), LOGIN_STATE_PROMPT)
    log(f"login state ({meta['latency_s']}s): {state}")
    return state


def login_normal_mode(log=print):
    """重启 QMT, 在自动登录倒计时期间取消「自动登录」「独立交易」, 再用普通模式点「登录」。不涉及任何交易。"""
    from .ensure_qmt import kill_qmt, launch_qmt
    kill_qmt()
    time.sleep(3)
    t0 = time.time()
    launch_qmt()
    hwnd, rect = find_window("QMT实盘", timeout=20, size=(624, 443))
    if not hwnd:
        log("登录窗口 20 秒内没出现")
        return False
    # 10-23 连拍: 窗口出现后约 2.4 秒才载入保存的勾选并开始倒计时 (登录(5)...(1)), 约 9 秒后自动登录。
    # 倒计时开始前点的勾选会被载入的设置覆盖, 所以等倒计时开始后再取消「自动登录」「独立交易」。
    ax, ay = rect[0] + LOGIN_OFFSETS["auto_login"][0], rect[1] + LOGIN_OFFSETS["auto_login"][1]
    ix, iy = rect[0] + LOGIN_OFFSETS["independent"][0], rect[1] + LOGIN_OFFSETS["independent"][1]
    shown = time.time()
    while time.time() - shown < 6 and checkbox_state(ax, ay)[0] != "checked":
        time.sleep(0.1)
    time.sleep(max(0.0, 3.2 - (time.time() - shown)))
    log(f"登录窗口 {rect} 出现于 {shown - t0:.1f}s; {time.time() - shown:.1f}s 后开始取消勾选")
    off_auto = toggle_off(hwnd, ax, ay, log)
    off_ind = toggle_off(hwnd, ix, iy, log)
    log(f"自动登录已取消={off_auto} 独立交易已取消={off_ind} (窗口出现后 {time.time() - shown:.1f}s)")
    time.sleep(2.5)
    st = login_state(log)
    if not st.get("login_window") or st.get("auto_login") or st.get("independent_trading"):
        log("勾选状态不对或窗口已不在, 停止")
        return False
    loc, ok, trail = locate_verified(screenshot_windows(), "登录窗口里蓝色的「登录」按钮 (不是「脱机」)", "登录")
    log(f"locate 登录: ok={ok} trail={trail}")
    if not ok:
        return False
    click_windows(loc["x"], loc["y"])
    log("已点「登录」(普通模式), 等待 30 秒")
    time.sleep(30)
    after, meta = describe(screenshot_windows())
    log(f"登录后屏幕 ({meta['latency_s']}s): {[(w.get('title'), w.get('kind'), str(w.get('text'))[:80]) for w in after.get('windows', [])]}")
    return True


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
    sub.add_parser("login-normal", help="重启 QMT 并用普通方式登录 (取消自动登录/独立交易)")
    a = ap.parse_args(argv)
    if a.cmd == "login-normal":
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] login-normal", flush=True)
        return 0 if login_normal_mode(log=lambda m: print(m, flush=True)) else 1
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
