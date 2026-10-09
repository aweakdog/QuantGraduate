"""QMT 下单封装 (阶段二, 2026-10-09)。

只读桥之外唯一能动账户的地方, 刻意做窄:
  - TradingTrader 只放行 query_* / 连接订阅 / order_stock / cancel_order_stock,
    转账、融资、算法单等一律 PermissionError
  - place_limit() 每笔委托前强制检查: 仅主板代码、买入整百股、限价单、价格不越涨跌停、
    单笔金额不超上限; 任何一条不满足直接抛错, 不会"修正后照下"
"""
import math

MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
_ALLOWED = frozenset({"start", "connect", "stop", "subscribe", "unsubscribe", "register_callback",
                      "order_stock", "cancel_order_stock"})
FIX_PRICE = 11          # xtconstant.FIX_PRICE (限价)
STOCK_BUY, STOCK_SELL = 23, 24
TERMINAL = {54, 56, 57, 52, 53}   # 已撤 / 已成 / 废单 / 部成部撤 / 部撤
STATUS_NAME = {48: "未报", 49: "待报", 50: "已报", 51: "已报待撤", 52: "部成待撤", 53: "部撤",
               54: "已撤", 55: "部成", 56: "已成", 57: "废单", 255: "未知"}


class OrderRejectedError(ValueError):
    pass


class TradingTrader:
    """可下单的代理: 白名单之外的调用一律拒绝。"""

    def __init__(self, trader):
        object.__setattr__(self, "_trader", trader)

    def __getattr__(self, name):
        if name.startswith("query_") or name in _ALLOWED:
            return getattr(self._trader, name)
        raise PermissionError(f"QMT 交易桥禁止调用 {name!r}")

    def __setattr__(self, name, value):
        raise PermissionError("QMT 交易桥不允许修改交易对象")


def is_main_board(stock_code):
    c = str(stock_code)
    return c[:3] in MAIN_BOARD_PREFIXES and c.endswith((".SH", ".SZ")) and len(c) == 9


def to_xt_code(code6):
    c = str(code6)[:6]
    return f"{c}.SH" if c.startswith("6") else f"{c}.SZ"


def round_tick(price, tick=0.01, up=False):
    n = price / tick
    n = math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)
    return round(n * tick, 2)


def check_order(stock_code, side, volume, price, up_stop, down_stop, max_value):
    """委托前硬检查; 不通过抛 OrderRejectedError (不做任何自动修正)。"""
    if not is_main_board(stock_code):
        raise OrderRejectedError(f"{stock_code} 不是主板股票")
    if side not in (STOCK_BUY, STOCK_SELL):
        raise OrderRejectedError(f"未知买卖方向 {side}")
    if not isinstance(volume, int) or volume <= 0:
        raise OrderRejectedError(f"{stock_code} 股数必须是正整数")
    if side == STOCK_BUY and volume % 100:
        raise OrderRejectedError(f"{stock_code} 买入必须整百股 (收到 {volume})")
    if not (up_stop and down_stop and down_stop < up_stop):
        raise OrderRejectedError(f"{stock_code} 缺涨跌停价, 不下单")
    if not (down_stop - 1e-9 <= price <= up_stop + 1e-9):
        raise OrderRejectedError(f"{stock_code} 限价 {price} 超出涨跌停 [{down_stop}, {up_stop}]")
    if side == STOCK_BUY and volume * price > max_value + 1e-6:
        raise OrderRejectedError(f"{stock_code} 买入金额 {volume * price:,.2f} 超过单笔上限 {max_value:,.2f}")


def place_limit(trader, account, stock_code, side, volume, price, up_stop, down_stop, max_value, remark):
    """检查通过后下一笔限价单, 返回 order_id (>0)。trader 必须是 TradingTrader。"""
    if not isinstance(trader, TradingTrader):
        raise PermissionError("下单必须经过 TradingTrader")
    check_order(stock_code, side, volume, price, up_stop, down_stop, max_value)
    oid = trader.order_stock(account, stock_code, side, volume, FIX_PRICE, float(price), "qmt10w", remark)
    if not oid or oid < 0:
        raise OrderRejectedError(f"{stock_code} 委托失败 (返回 {oid})")
    return oid
