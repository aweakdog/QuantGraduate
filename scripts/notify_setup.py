# -*- coding: utf-8 -*-
"""配置推送通道, 并当场验证它真能发出去。

【为什么要单独一个脚本】
webhook 地址等于"往这个群发消息"的权限, 拿到的人就能冒充系统发假信号 ——
对一个发买卖指令的系统, 这不是小事。所以它:
  · 只落在服务器的 data/live/ 下(该目录已被 .gitignore 排除), 不进版本库
  · 文件权限设成 0600, 同机器上的其他用户读不到
  · 不需要经过聊天记录/剪贴板转手

【为什么装完要立刻发一条测试】
"配好了但其实发不出去"是这套系统最危险的状态 —— 人会开始依赖提醒, 而提醒
根本没在工作。所以宁可现在就打扰群里一次, 也不要等到某个真正要紧的日子
才发现是坏的。

用法(在服务器上跑):
    python scripts/notify_setup.py --channel wecom --webhook 'https://qyapi...' --test
    python scripts/notify_setup.py --show          # 看当前配置(隐去密钥)
    python scripts/notify_setup.py --channel stdout  # 退回影子模式
"""

import argparse
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from notify_channels import CONFIG_PATH, get_channel  # noqa: E402

WECOM_PREFIX = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send"


def _mask(url):
    """只露头尾, 中间打码 —— 打印出来的东西可能被贴进聊天或截图。"""
    if not url:
        return ""
    if "key=" in url:
        head, key = url.split("key=", 1)
        return f"{head}key={key[:6]}...{key[-4:]}" if len(key) > 12 else f"{head}key=***"
    return url[:40] + "..."


def show():
    if not CONFIG_PATH.exists():
        print(f"未配置 ({CONFIG_PATH} 不存在) -> 当前是 stdout 影子模式")
        return 0
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    print(f"配置文件: {CONFIG_PATH}")
    print(f"  权限:   {oct(CONFIG_PATH.stat().st_mode & 0o777)}")
    print(f"  通道:   {cfg.get('channel')}")
    if cfg.get("wecom_webhook"):
        print(f"  webhook: {_mask(cfg['wecom_webhook'])}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="配置推送通道")
    ap.add_argument("--channel", choices=("stdout", "queue", "wecom"))
    ap.add_argument("--webhook", help="企业微信群机器人 Webhook 地址")
    ap.add_argument("--test", action="store_true", help="写入后立刻发一条测试消息")
    ap.add_argument("--show", action="store_true", help="查看当前配置")
    args = ap.parse_args()

    if args.show or not args.channel:
        return show()

    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    cfg["channel"] = args.channel
    if args.channel == "wecom":
        wh = args.webhook or cfg.get("wecom_webhook")
        if not wh:
            print("ERROR: --channel wecom 需要 --webhook")
            return 2
        # 早点报错好过发的时候才发现。企业微信的地址形如
        # https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxxxx
        if not wh.startswith(WECOM_PREFIX):
            print(f"ERROR: 这不像企业微信群机器人地址, 应以 {WECOM_PREFIX} 开头")
            print(f"       你给的是: {_mask(wh)}")
            return 2
        if "key=" not in wh:
            print("ERROR: 地址里没有 key= 参数, 复制时可能被截断了")
            return 2
        cfg["wecom_webhook"] = wh

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    # 0600: 同机器上的其他用户读不到。这台是共享服务器, 这一步不是形式主义。
    os.chmod(CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)
    print(f"已写入 {CONFIG_PATH} (权限 0600)")
    show()

    if args.test:
        print()
        print("发送测试消息...")
        ch = get_channel(cfg=cfg)
        ok = ch.send(
            "**跟单提醒已接通**\n"
            "这是一条测试消息。今后只在【确实有操作要做】时才会发, 平时不打扰。\n\n"
            "· 执行日 14:35 催下单(会 @所有人)\n"
            "· 盘后预告下一个交易日要做什么\n"
            "· 催回填成交价\n\n"
            "http://eez041.ece.ust.hk:8737/",
            meta={"slot": "setup", "urgent": False})
        print("结果:", "成功, 去群里看看" if ok else "失败, 见上面的错误")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
