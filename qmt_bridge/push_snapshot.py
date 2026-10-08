"""把 QMT 只读快照签名推送到 041 (Windows 计划任务: 9:35 开盘检查 / 15:10 收盘)。

  .venv\\Scripts\\python.exe -m qmt_bridge.push_snapshot --kind close
  .venv\\Scripts\\python.exe -m qmt_bridge.push_snapshot --kind morning --dry-run

连不上 miniQMT 或资金账号没挂上时也推送一份 account_online=false 的快照, 由 041 负责告警。
签名: HMAC-SHA256(key, 时间戳 + "\\n" + body), 与 041 scripts/qmt_sync.py 同一算法。
只读: 交易对象经 ReadOnlyTrader 包装, 本脚本没有任何下单路径。
"""
import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from .readonly_probe import DEFAULT_USERDATA, ReadOnlyTrader, collect, connect_with_timeout, mask

DEFAULT_URL = "http://eez041.ece.ust.hk:8737/api/qmt/snapshot"
DEFAULT_KEY = r"D:\qmtcode\bridge.key"
SECURITY_ACCOUNT = 2


def sign(body: bytes, ts: str, key: bytes) -> str:
    return hmac.new(key, str(ts).encode() + b"\n" + body, hashlib.sha256).hexdigest()


def take_snapshot(kind, userdata=DEFAULT_USERDATA, timeout=20.0):
    """返回 (快照 dict, 交易对象或 None)。任何失败都折算成 account_online=False, 不抛异常。"""
    base = {"kind": kind, "host": os.environ.get("COMPUTERNAME", "")}
    try:
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount
    except ImportError as e:
        return {**base, "probe_at": datetime.now().isoformat(timespec="seconds"),
                "account_online": False, "error": f"xtquant 不可用: {e}"}, None
    trader = ReadOnlyTrader(XtQuantTrader(userdata, int(time.time()) % 100000 + 300000))
    trader.start()
    rc = connect_with_timeout(trader, timeout)
    if rc != 0:
        why = f"{timeout:g} 秒未连上" if rc is None else f"rc={rc}"
        return {**base, "probe_at": datetime.now().isoformat(timespec="seconds"), "account_online": False,
                "error": f"连接 miniQMT 失败({why}), QMT 未以极简模式登录?"}, trader
    infos = trader.query_account_infos() or []
    stock = [i for i in infos if getattr(i, "account_type", None) == SECURITY_ACCOUNT]
    if not stock:
        return {**base, "probe_at": datetime.now().isoformat(timespec="seconds"), "account_online": False,
                "error": "已连上 miniQMT 但资金账号未挂上 (需在柜台开放后重新登录)"}, trader
    account = StockAccount(str(stock[0].account_id), "STOCK")
    snap = collect(trader, account, masked=True)
    return {**base, "probe_at": datetime.now().isoformat(timespec="seconds"), "account_online": True,
            "account": mask(stock[0].account_id), **snap}, trader


def push(snap, url, key, timeout=20.0, opener=urllib.request.urlopen):
    body = json.dumps(snap, ensure_ascii=False, default=str).encode("utf-8")
    ts = str(int(time.time()))
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "X-QMT-Ts": ts, "X-QMT-Sig": sign(body, ts, key)})
    try:
        with opener(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        return 0, f"网络错误: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", choices=("morning", "close", "manual"), required=True)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--key-file", default=DEFAULT_KEY)
    ap.add_argument("--userdata", default=DEFAULT_USERDATA)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true", help="只打印快照, 不推送")
    args = ap.parse_args(argv)

    # 不调 trader.stop(): 它在 xtquant 里偶尔会卡住, 而推送必须先完成; 末尾 os._exit 会断开连接
    snap, _trader = take_snapshot(args.kind, args.userdata, args.timeout)
    out_dir = Path(__file__).resolve().parents[1] / "snapshots"
    out_dir.mkdir(exist_ok=True)
    stamp = snap["probe_at"].replace(":", "").replace("-", "")
    (out_dir / f"snap_{stamp}_{args.kind}.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    ok = True
    if args.dry_run:
        print(json.dumps(snap, ensure_ascii=False, indent=2, default=str))
    else:
        key = Path(args.key_file).read_text(encoding="utf-8").strip().encode()
        status, text = push(snap, args.url, key, args.timeout)
        ok = status == 200
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] kind={args.kind} online={snap['account_online']} "
              f"push={status} {text[:300]}")
    sys.stdout.flush()
    os._exit(0 if ok else 1)   # xtquant 后台线程可能卡住正常退出


if __name__ == "__main__":
    main()
