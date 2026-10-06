"""QMT 只读探针: 连接已登录的 miniQMT, 读出资金、持仓、当日委托与成交, 输出 JSON。

只读保证: 交易对象被 ReadOnlyTrader 包住, 只放行 query_* 与连接/订阅类方法;
order_* / cancel_* 等任何其他调用直接抛 PermissionError —— 即使以后有人在这里误加下单代码也会被拦。

用法 (Windows, D:\\qmtcode 下):
  .venv\\Scripts\\python.exe -m qmt_bridge.readonly_probe            # 自动选第一个股票账户
  .venv\\Scripts\\python.exe -m qmt_bridge.readonly_probe --account 资金账号 --out probe.json
前提: QMT 客户端已以「极简模式」登录 (XtMiniQmt.exe 在运行)。账号默认脱敏输出。
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_USERDATA = r"D:\迅投极速策略交易系统交易终端 华泰证券QMT实盘\userdata_mini"

# 除 query_* 外允许的方法: 只涉及连接与订阅推送, 不改变账户状态
_ALLOWED = frozenset({"start", "connect", "stop", "subscribe", "unsubscribe", "register_callback"})

ASSET_FIELDS = ("account_type", "account_id", "cash", "frozen_cash", "market_value", "total_asset")
POSITION_FIELDS = ("account_id", "stock_code", "volume", "can_use_volume", "frozen_volume", "on_road_volume",
                   "yesterday_volume", "open_price", "avg_price", "market_value")
ORDER_FIELDS = ("stock_code", "order_id", "order_sysid", "order_time", "order_type", "order_volume",
                "price_type", "price", "traded_volume", "traded_price", "order_status", "status_msg",
                "strategy_name", "order_remark")
TRADE_FIELDS = ("stock_code", "order_type", "traded_id", "traded_time", "traded_price", "traded_volume",
                "traded_amount", "order_id", "order_sysid", "strategy_name", "order_remark")


class ReadOnlyTrader:
    """只读代理: 白名单之外的属性访问一律拒绝 (下单/撤单/划转等全部不可达)。"""

    def __init__(self, trader):
        object.__setattr__(self, "_trader", trader)

    def __getattr__(self, name):
        if name.startswith("query_") or name in _ALLOWED:
            return getattr(self._trader, name)
        raise PermissionError(f"QMT 只读桥禁止调用 {name!r}")

    def __setattr__(self, name, value):
        raise PermissionError("QMT 只读桥不允许修改交易对象")


def mask(account_id):
    s = str(account_id or "")
    return ("*" * max(0, len(s) - 4) + s[-4:]) if s else ""


def to_dict(obj, fields, masked=True):
    out = {f: getattr(obj, f, None) for f in fields}
    if masked and out.get("account_id") is not None:
        out["account_id"] = mask(out["account_id"])
    return out


def pick_stock_account(infos, wanted=None, security_type=2):
    """从 query_account_infos() 里挑股票账户; wanted 给定时必须精确匹配。"""
    stock = [i for i in (infos or []) if getattr(i, "account_type", None) == security_type]
    if wanted is not None:
        stock = [i for i in stock if str(getattr(i, "account_id", "")) == str(wanted)]
        if not stock:
            raise SystemExit(f"找不到股票账户 {mask(wanted)}")
    if not stock:
        # 10-06 实测: 极简模式已连上(connect=0)、QMT 用户鉴权成功, 但客户端日志 account_num=0 ——
        # 资金账号没挂上。09-28 盘后 17:56 与 10-06 假期都是这样, 伴随柜台地址 10061/10060 连接失败。
        raise SystemExit("已连上 miniQMT, 但客户端没有加载到任何资金账号: 常见原因是非交易时段券商柜台未开放, "
                         "或资金账号没在 QMT 里登录成功 (客户端日志 refreshAccounts account_num = 0)")
    return stock[0]


def connect_with_timeout(trader, seconds):
    """connect() 在 QMT 不是极简模式时会无限阻塞(10-06 实测); 超时返回 None。"""
    import threading
    box = {}
    worker = threading.Thread(target=lambda: box.setdefault("rc", trader.connect()), daemon=True)
    worker.start()
    worker.join(seconds)
    return box.get("rc")


def collect(trader, account, masked=True):
    """读快照。trader 必须已 start/connect; 返回可 JSON 化的 dict。"""
    rc = trader.subscribe(account)
    asset = trader.query_stock_asset(account)
    positions = trader.query_stock_positions(account) or []
    orders = trader.query_stock_orders(account, False) or []
    trades = trader.query_stock_trades(account) or []
    return {"subscribe_rc": rc,
            "asset": to_dict(asset, ASSET_FIELDS, masked) if asset is not None else None,
            "positions": [to_dict(p, POSITION_FIELDS, masked) for p in positions],
            "orders": [to_dict(o, ORDER_FIELDS, masked) for o in orders],
            "trades": [to_dict(t, TRADE_FIELDS, masked) for t in trades]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--userdata", default=DEFAULT_USERDATA, help="QMT 安装目录下的 userdata_mini")
    ap.add_argument("--account", default=None, help="资金账号; 缺省取第一个股票账户")
    ap.add_argument("--out", default=None, help="另存 JSON 文件")
    ap.add_argument("--no-mask", action="store_true", help="输出完整资金账号 (只在本机看时用)")
    ap.add_argument("--timeout", type=float, default=20.0, help="连接超时秒数")
    args = ap.parse_args(argv)

    from xtquant import xtconstant
    from xtquant.xttrader import XtQuantTrader
    from xtquant.xttype import StockAccount

    if not Path(args.userdata).is_dir():
        raise SystemExit(f"userdata_mini 不存在: {args.userdata}")
    raw = XtQuantTrader(args.userdata, int(time.time()) % 100000 + 100000)
    trader = ReadOnlyTrader(raw)
    trader.start()
    rc = connect_with_timeout(trader, args.timeout)
    if rc != 0:
        why = f"{args.timeout:g} 秒未连上" if rc is None else f"rc={rc}"
        sys.stderr.write(f"连接 miniQMT 失败 ({why}): QMT 客户端需先以「极简模式」登录\n")
        sys.stderr.flush()
        os._exit(2)      # xtquant 后台线程可能卡住正常退出, 直接结束进程
    try:
        infos = trader.query_account_infos()
        info = pick_stock_account(infos, args.account, xtconstant.SECURITY_ACCOUNT)
        account = StockAccount(str(info.account_id), "STOCK")
        snap = {"probe_at": datetime.now().isoformat(timespec="seconds"),
                "accounts": [{"account_id": mask(getattr(i, "account_id", "")) if not args.no_mask else getattr(i, "account_id", ""),
                              "account_type": getattr(i, "account_type", None),
                              "login_status": getattr(i, "login_status", None)} for i in (infos or [])],
                **collect(trader, account, masked=not args.no_mask)}
    finally:
        trader.stop()
    text = json.dumps(snap, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
