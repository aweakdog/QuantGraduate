"""QMT 普通客户端界面下单流程 (2026-10-23 起)。分两层, 默认只填单不提交:

  dry-fill: 关通知弹窗 -> 选股票 -> 填委托价/数量 -> 读回表单核对。绝不点「买入」、不在价格/数量框里按回车
            (很多交易终端里回车 = 提交)。收盘后也照此执行 (夜市委托会排到下一交易日, 一样不能误交)。
  channel-test (尚未实现): 实盘跌停价挂 100 股再撤单, 需用户对具体委托逐笔确认后才做。

所有点击走 ui_agent.locate_verified (模型定位 + 独立复核); 键盘只输入数字和小数点。
"""
import argparse
import ctypes
import json
import sys
import time

from .ui_agent import ask, click_windows, locate_verified, png_size, screenshot_windows

VK = {**{str(d): 0x30 + d for d in range(10)}, ".": 0xBE}
FORM_PROMPT = (
    "这是 {w}x{h} 的 QMT 交易终端截图。只读左下角「下单面板」, 只返回 JSON: "
    "{{side_tab (当前选中的是 买入 还是 卖出), code (证券代码), name (证券名称), price (委托价输入框里的数), "
    "volume (总量输入框里的数), est_amount (预估金额), max_buy (最大可买), limit_up (涨停价, 看不到填 null), "
    "limit_down (跌停价, 看不到填 null), popup (屏幕上有无弹窗, 有则给标题)}}。数字按原样转成数值, 看不清填 null。")


def _key(vk, up=False):
    ctypes.windll.user32.keybd_event(vk, 0, 2 if up else 0, 0)


def type_digits(text):
    bad = [ch for ch in text if ch not in VK]
    if bad or not text:
        raise ValueError(f"只允许输入数字和小数点, 收到 {text!r}")       # 先整串校验, 不按半截
    for ch in text:
        _key(VK[ch])
        _key(VK[ch], up=True)
        time.sleep(0.03)


def select_all():
    _key(0x11)
    _key(0x41)
    _key(0x41, up=True)
    _key(0x11, up=True)


def backspace(n=12):
    for _ in range(n):
        _key(0x08)
        _key(0x08, up=True)


def click_target(target, expect, log):
    loc, ok, trail = locate_verified(screenshot_windows(), target, expect)
    log(f"  {target}: ok={ok} at ({loc.get('x')},{loc.get('y')}) trail={trail[-1]}")
    if ok:
        click_windows(loc["x"], loc["y"])
    return ok


def dismiss_notice(log):
    """「系统预留信息」通知: 勾「已知悉,下次不再提示」再关闭。没有弹窗就什么都不做。"""
    form = read_form()
    if not form.get("popup"):
        return True
    log(f"popup present: {form.get('popup')}")
    click_target("「系统预留信息」弹窗底部「已知悉，下次不再提示」左边的复选框", "已知悉", log)
    time.sleep(0.5)
    ok = click_target("「系统预留信息」弹窗右上角的关闭按钮 ×", "", log)
    time.sleep(1.0)
    return ok


def read_form():
    png = screenshot_windows()
    w, h = png_size(png)
    form, _ = ask(png, FORM_PROMPT.format(w=w, h=h))
    return form


def set_field(target, expect, value, log):
    """点输入框 -> 全选 -> 退格清空 -> 输入数字。不按回车。"""
    if not click_target(target, expect, log):
        return False
    time.sleep(0.3)
    select_all()
    backspace()
    type_digits(value)
    time.sleep(0.4)
    return True


def dry_fill(code, price, volume, log=print):
    """填单演练: 选股、填价、填量、读回核对。任何一步不对就停, 绝不提交。"""
    dismiss_notice(log)
    form = read_form()
    log(f"form before: {form}")
    if str(form.get("code")) != code:
        log(f"当前证券 {form.get('code')} 不是 {code}: 本演练要求先在界面上选中 {code}, 停止")
        return False, form
    if form.get("side_tab") != "买入":
        click_target("下单面板左侧竖排的「买入」页签", "买", log)
        time.sleep(0.8)
    set_field("下单面板里「委托价」右边的输入框 (数字输入框本身, 不是 +/- 按钮)", "", price, log)
    set_field("下单面板里「总量」右边的输入框 (数字输入框本身, 不是 +/- 按钮)", "", str(volume), log)
    time.sleep(0.8)
    form = read_form()
    log(f"form after: {form}")
    ok = (str(form.get("code")) == code and form.get("side_tab") == "买入"
          and abs(float(form.get("price") or 0) - float(price)) < 1e-6 and int(form.get("volume") or 0) == volume)
    log(f"核对 {'通过' if ok else '不通过'}: 期望 {code} 买入 {price} x {volume}")
    return ok, form


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-fill", help="只填单、读回核对, 不提交")
    d.add_argument("--code", default="600000")
    d.add_argument("--price", required=True)
    d.add_argument("--volume", type=int, default=100)
    sub.add_parser("read-form")
    a = ap.parse_args(argv)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {a.cmd}", flush=True)
    if a.cmd == "read-form":
        print(json.dumps(read_form(), ensure_ascii=False), flush=True)
        return 0
    ok, form = dry_fill(a.code, a.price, a.volume, log=lambda m: print(m, flush=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
