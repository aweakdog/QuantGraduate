"""QMT 登录前把交易电脑上别的软件退掉 (用户 11-12 要求)。

11-12 早上有人开着 QQ、WeGame、KOOK, 它们的登录框叠在 QMT 登录框上, 模型找到两个「登录」按钮,
判定容易点错没敢点, 读屏离线、逆回购差点做不成。所以:
  登录前  quit_other_apps(force=False): 给别家窗口发关闭, 再 taskkill (不加 /F) 退出这些程序和常驻托盘的聊天/游戏软件
  登录失败 quit_other_apps(force=True):  taskkill /F 强杀所有带窗口的非白名单程序 + 常驻托盘软件, 然后再登一次
白名单: 系统进程、QMT 自身、我们的 Python/SSH、远程连接 (Tailscale/NetBird/向日葵)、驱动托盘。
"""
import subprocess
import time

KEEP = {
    "system", "explorer.exe", "dwm.exe", "csrss.exe", "winlogon.exe", "svchost.exe", "services.exe", "lsass.exe",
    "smss.exe", "wininit.exe", "fontdrvhost.exe", "sihost.exe", "taskhostw.exe", "ctfmon.exe", "runtimebroker.exe",
    "shellexperiencehost.exe", "startmenuexperiencehost.exe", "searchhost.exe", "searchapp.exe", "textinputhost.exe",
    "applicationframehost.exe", "systemsettings.exe", "lockapp.exe", "securityhealthsystray.exe", "msmpeng.exe",
    "conhost.exe", "openconsole.exe", "windowsterminal.exe", "cmd.exe", "powershell.exe", "pwsh.exe", "taskmgr.exe",
    "python.exe", "pythonw.exe", "sshd.exe", "ssh.exe", "schtasks.exe", "taskkill.exe",
    "xtitclient.exe", "xtminiqmt.exe", "miniquote.exe", "xtupdate.exe",                       # QMT 自身
    "tailscale.exe", "tailscaled.exe", "tailscale-ipn.exe", "netbird.exe", "netbird-ui.exe",  # 我们连这台机器靠它们
    "sunloginclient.exe", "slservice.exe",                                                    # 向日葵: 人工远程兜底
}
KEEP_PREFIX = ("logi", "rtk", "nvidia", "igfx", "amd", "realtek",                            # 驱动托盘
               "awesun", "sunlogin", "todesk")                                                # 远程工具: 别把人工远程断了
TRAY_APPS = {   # 关了窗口也常驻托盘、还会弹登录框/通知的, 不管有没有窗口都退
    "qq.exe", "wegame.exe", "kook.exe", "steam.exe", "steamwebhelper.exe", "dingtalk.exe", "wechat.exe",
    "weixin.exe", "wxwork.exe", "quark.exe", "wps.exe", "et.exe", "wpp.exe", "wpscloudsvr.exe", "baidunetdisk.exe",
    "360tray.exe", "lenovoappstore.exe", "lenovocloud.exe", "qqmusic.exe", "cloudmusic.exe",
}


def keep(name: str) -> bool:
    n = name.lower()
    return n in KEEP or n.startswith(KEEP_PREFIX)


def pick_targets(windowed, running):
    """windowed = 带可见标题窗口的进程名列表; running = 全部进程名。返回要退出的进程名 (小写, 排序)。"""
    names = {n.lower() for n in windowed} | ({n.lower() for n in running} & TRAY_APPS)
    return sorted(n for n in names if not keep(n))


def _windows():
    """[(hwnd, pid, 进程名, 标题)]: 可见、有标题的顶层窗口"""
    import ctypes
    from ctypes import wintypes
    u, k = ctypes.windll.user32, ctypes.windll.kernel32
    out = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def name_of(pid):
        h = k.OpenProcess(0x1000, False, pid)                     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ""
        buf, size = ctypes.create_unicode_buffer(520), wintypes.DWORD(520)
        ok = k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
        k.CloseHandle(h)
        return buf.value.rsplit("\\", 1)[-1] if ok else ""

    def cb(h, _):
        n = u.GetWindowTextLengthW(h)
        if u.IsWindowVisible(h) and n:
            t = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(h, t, n + 1)
            p = wintypes.DWORD()
            u.GetWindowThreadProcessId(h, ctypes.byref(p))
            out.append((h, p.value, name_of(p.value), t.value))
        return True
    u.EnumWindows(enum_proc(cb), 0)
    return out


def _running():
    r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace")
    return [line.split('","')[0].strip('"') for line in (r.stdout or "").splitlines() if line.startswith('"')]


def quit_other_apps(force=False, log=print):
    """退出 QMT 以外的软件; 返回退掉的进程名。force=False 先礼后兵 (发关闭 + taskkill), True 直接强杀。"""
    import ctypes
    wins = [w for w in _windows() if w[2] and not keep(w[2])]
    targets = pick_targets([w[2] for w in wins], _running())
    if not targets:
        return []
    if not force:
        for h, _pid, _name, _title in wins:
            ctypes.windll.user32.PostMessageW(h, 0x0010, 0, 0)    # WM_CLOSE, 等于点右上角 ×
        time.sleep(3)
    for n in targets:
        subprocess.run(["taskkill", *(["/F"] if force else []), "/T", "/IM", n], capture_output=True)
    if not force:
        time.sleep(2)
        left = set(pick_targets([w[2] for w in _windows() if w[2]], _running())) & set(targets)
        for n in sorted(left):                                    # 不理关闭请求的再强杀
            subprocess.run(["taskkill", "/F", "/T", "/IM", n], capture_output=True)
    log(f"登录前退出其他软件 ({'强杀' if force else '正常退出'}): {targets} "
        f"(窗口: {sorted({w[3][:20] for w in wins})[:10]})")
    return targets
