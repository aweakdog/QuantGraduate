# -*- coding: utf-8 -*-
"""推送通道抽象层。

【为什么要这一层】
判断"该不该提醒、提醒什么"是稳定的业务逻辑; "怎么把字送到人眼前"却几乎
一定会换 —— 个人微信自动化可能被封、挂微信的那台设备会下线、以后也许改用
短信或公众号。如果两者缠在一起, 每换一次通道就要重新验证一遍判定逻辑, 而
判定逻辑恰恰是最不该反复动的部分(动错了就是漏提醒或者天天误报)。

所以这里只承诺一件事: 给我一段文本, 我负责送出去, 成功返回 True。

【已实现的通道】
  stdout —— 只打印。给 --dry-run 和本地调试用, 永远可用, 不依赖任何外部条件。
  queue  —— 写进本地队列目录, 由外部发送器(登着微信的那台设备)轮询取走。
            这是【拉模式】, 有三个好处, 都是推模式做不到的:
              1. 发送器在家用 NAT 后面也能工作, 服务器不需要能连它
              2. 发送器掉线时消息留在队列里, 上线后补发, 不会丢
              3. 换设备/换实现只需换发送器, 服务器一行不用改
  wecom  —— 企业微信群机器人 webhook, 服务器直推。

【为什么发送失败不能静默】
提醒系统最坏的失效方式不是"发不出去", 而是"悄悄发不出去"——
人一旦养成"没消息就是没操作"的习惯, 静默失效比从没有提醒更危险。
所以每个通道都必须如实返回成败, 由调用方记录并暴露到看板上。
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
CONFIG_PATH = LIVE / "notify_config.json"
QUEUE_DIR = LIVE / "notify_queue"

# 队列里的消息保留多久。发送器长期掉线时, 过期的行情提醒发出去只会误导人
# ("今天尾盘买入"在三天后送达是有害的), 所以宁可丢弃并告警。
QUEUE_TTL_HOURS = 12


def load_config():
    """读通道配置。文件在 data/ 下, 已被 .gitignore 排除, 可以放 webhook。

    没有配置文件时退回 stdout —— 让人在没配任何东西的情况下也能跑通、看到
    文案长什么样, 而不是抛异常把整条链路挡住。
    """
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[notify] 配置 {CONFIG_PATH} 读取失败({e}), 退回 stdout")
    return {"channel": "stdout"}


class Channel:
    name = "base"

    def send(self, text, meta=None):
        raise NotImplementedError


class StdoutChannel(Channel):
    name = "stdout"

    def send(self, text, meta=None):
        print("─" * 60)
        print(text)
        print("─" * 60)
        return True


class QueueChannel(Channel):
    """写入待发队列, 由外部发送器取走。

    一条消息一个文件, 而不是往一个 jsonl 里追加 —— 发送器要在取走后标记
    已发, 单文件方案下"读-改-写"会和服务器的追加写抢同一个文件。一条一个
    文件时, 发送器只需把文件从 pending/ 移到 sent/, 是原子操作, 不用加锁。
    """

    name = "queue"

    def __init__(self, queue_dir=None):
        self.pending = Path(queue_dir or QUEUE_DIR) / "pending"
        self.sent = Path(queue_dir or QUEUE_DIR) / "sent"
        self.pending.mkdir(parents=True, exist_ok=True)
        self.sent.mkdir(parents=True, exist_ok=True)

    def send(self, text, meta=None):
        now = datetime.now()
        mid = f"{now.strftime('%Y%m%d_%H%M%S')}_{(meta or {}).get('slot', 'msg')}"
        payload = {
            "id": mid,
            "created_at": now.isoformat(timespec="seconds"),
            # 过期时间由服务器写死在消息里, 而不是让发送器自己算 ——
            # 发送器可能在另一台机器上, 时区/时钟不一定对得上。
            "expire_at": (now.timestamp() + QUEUE_TTL_HOURS * 3600),
            "text": text,
            "meta": meta or {},
        }
        tmp = self.pending / f".{mid}.tmp"
        dst = self.pending / f"{mid}.json"
        # 先写临时文件再改名: 发送器可能正好在这一刻扫目录, 直接写目标文件
        # 会让它读到只写了一半的 JSON。
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, dst)
        return True

    def purge_expired(self):
        """清掉过期未发的消息, 返回清掉的条数(调用方应据此告警)。"""
        n, now = 0, time.time()
        for f in self.pending.glob("*.json"):
            try:
                if json.loads(f.read_text(encoding="utf-8")).get("expire_at", 0) < now:
                    f.unlink()
                    n += 1
            except (json.JSONDecodeError, OSError):
                continue
        return n


class WecomChannel(Channel):
    """企业微信群机器人。一个 webhook URL, POST 一段 JSON 即可。"""

    name = "wecom"

    def __init__(self, webhook):
        if not webhook:
            raise ValueError("wecom 通道需要配置 wecom_webhook")
        self.webhook = webhook

    def send(self, text, meta=None):
        body = json.dumps({"msgtype": "markdown", "markdown": {"content": text}},
                          ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.webhook, data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = json.loads(r.read().decode("utf-8"))
            # 企业微信 HTTP 200 也可能是业务失败(errcode != 0), 只看状态码会误判成功
            if resp.get("errcode") != 0:
                print(f"[notify] 企业微信返回错误: {resp}")
                return False
            return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            print(f"[notify] 企业微信发送失败: {e}")
            return False


def get_channel(name=None, cfg=None):
    """按配置取通道。未知名字直接报错而不是悄悄退回 stdout ——

    配置里写错通道名却"看起来在正常运行", 是这类系统最容易漏掉的故障。
    """
    cfg = cfg if cfg is not None else load_config()
    name = name or cfg.get("channel") or "stdout"
    if name == "stdout":
        return StdoutChannel()
    if name == "queue":
        return QueueChannel(cfg.get("queue_dir"))
    if name == "wecom":
        return WecomChannel(cfg.get("wecom_webhook"))
    raise ValueError(f"未知的推送通道 {name!r} (可选: stdout / queue / wecom)")
