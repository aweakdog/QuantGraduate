"""OCR 表格解析: 用 10-28 实盘截图在 Windows 上 OCR 出的真实文字框 (坐标、文字照抄) 测按表头分列、跳过合计行。"""
from qmt_bridge.ocr_read import _number, parse_table


def box(x, y, t, w=50):
    return (t, 1.0, (x, y), w)


HEAD_Y, ROW1, ROW2 = 522, 544, 567

POSITIONS = [box(872, 521, "资金账号"), box(956, 521, "证券代码"), box(1023, HEAD_Y, "证券名称"), box(1091, HEAD_Y, "当前拥股"),
             box(1158, HEAD_Y, "可用数量"), box(1224, HEAD_Y, "冻结数量"), box(1286, HEAD_Y, "成本价"), box(1341, HEAD_Y, "最新价"),
             box(1401, HEAD_Y, "持仓盈亏"), box(1468, HEAD_Y, "盈亏比例"), box(1536, HEAD_Y, "当日涨幅"), box(1610, HEAD_Y, "市值"),
             box(1684, HEAD_Y, "持仓成本"), box(1757, 521, "股东账号"), box(1829, HEAD_Y, "市场名称"), box(1896, 521, "资产占"),
             box(827, 499, "卖出参数"), box(898, 498, "请输入代码"),
             box(802, ROW1, "1"), box(875, 543, "666656116735"), box(947, ROW1, "888880"), box(1021, ROW1, "新标准券"),
             box(1097, ROW1, "5020张"), box(1174, ROW1, "￥0"), box(1241, ROW1, "0"), box(1295, ROW1, "0.00"), box(1350, ROW1, "1.00"),
             box(1417, 543, "0.00"), box(1480, 543, "0.00%"), box(1613, 543, "50 2000.00"), box(1699, 543, "0.00"),
             box(1760, 543, "A184506182"), box(1839, 543, "上证所"), box(1894, 543, "99.7700")]

TRADES = [box(871, HEAD_Y, "资金账号"), box(967, HEAD_Y, "成交时间"), box(1043, HEAD_Y, "证券代码"), box(1111, HEAD_Y, "证券名称"),
          box(1195, HEAD_Y, "买卖标记"), box(1281, 521, "成交数量"), box(1348, 521, "成交价格"), box(1422, HEAD_Y, "成交金额"),
          box(1490, HEAD_Y, "手续费"), box(1583, 521, "成交编号"), box(1681, 521, "合同编号"), box(1748, 521, "任务编号"),
          box(1815, 521, "投资备注"), box(1882, HEAD_Y, "账号备注"),
          box(875, 543, "666656116735"), box(980, 543, "10:52:54"), box(1052, 543, "204001"), box(1098, ROW1, "GC001"),
          box(1198, 543, "通用回购逆回购"), box(1290, ROW1, "680张"), box(1359, 543, "1.425"), box(1428, 543, "6 8000.00"),
          box(1499, 543, "0.68"), box(1586, 543, "1200801002776035"), box(1692, 543, "24433"),
          box(875, 566, "666656116735"), box(980, 566, "10:52:54"), box(1052, ROW2, "204001"), box(1098, ROW2, "GC001"),
          box(1198, ROW2, "通用回购逆回购"), box(1287, ROW2, "4340张"), box(1360, ROW2, "1.425"), box(1425, 566, "43 4000.00"),
          box(1499, 566, "4.34"), box(1586, ROW2, "1200801002776034"), box(1691, ROW2, "24433"),
          box(842, 717, "合计"), box(1295, 717, "5020"), box(1426, 717, "50 2000.00"), box(1500, 717, "5.02")]

ORDERS = [box(875, HEAD_Y, "选择"), box(978, 521, "资金账号"), box(1074, HEAD_Y, "委托时间"), box(1151, HEAD_Y, "证券代码"),
          box(1218, HEAD_Y, "证券名称"), box(1302, HEAD_Y, "买卖标记"), box(1384, HEAD_Y, "委托量"), box(1453, HEAD_Y, "委托剩余量"),
          box(1527, HEAD_Y, "成交数量"), box(1594, HEAD_Y, "委托状态"), box(1660, HEAD_Y, "委托价格"), box(1728, HEAD_Y, "成交均价"),
          box(1795, HEAD_Y, "合同编号"), box(1861, HEAD_Y, "投资备注"),
          box(982, 543, "666656116735"), box(1087, 543, "10:52:54"), box(1159, ROW1, "204001"), box(1220, ROW1, "GC001"),
          box(1302, ROW1, "通用回购逆回购"), box(1384, ROW1, "5020张"), box(1453, ROW1, "0张"), box(1527, ROW1, "5020张"),
          box(1594, ROW1, "已成"), box(1660, ROW1, "1.375"), box(1728, ROW1, "1.425"), box(1795, ROW1, "24433")]

FUNDS = [box(888, 792, "资金账号"), box(958, 792, "账号名称"), box(1028, 792, "账号备注"), box(1098, 792, "登录状态"),
         box(1205, 792, "操作"), box(1333, 792, "总资产"), box(1408, 792, "净资产"), box(1477, 792, "总负债"), box(1537, 792, "总市值"),
         box(1607, 792, "可用金额"), box(1677, 792, "冻结金额"), box(1747, 792, "持仓盈亏"), box(1817, 792, "手续费"),
         box(1883, 792, "可取金额"),
         box(888, 817, "666656116735"), box(958, 817, "李立兴"), box(1028, 817, "个人"), box(1098, 817, "登录成功"),
         box(1170, 817, "重登录"), box(1235, 817, "校准资金"), box(1333, 817, "50 3121.79"), box(1408, 817, "50 3121.79"),
         box(1477, 817, "0.00"), box(1537, 817, "1000.00"), box(1607, 817, "50 2121.79"), box(1677, 817, "0.00"),
         box(1747, 817, "0.00"), box(1817, 817, "0.01"), box(1883, 817, "50 2121.79")]


def test_number_handles_qmt_formats():
    assert _number("50 2000.00") == 502000.0 and _number("5020张") == 5020 and _number("￥0") == 0
    assert _number("1,234.5") == 1234.5 and _number("已成") is None


def test_positions_table():
    rows = parse_table(POSITIONS, "positions")["rows"]
    assert rows == [{"code": "888880", "name": "新标准券", "volume": 5020.0, "can_use": 0.0, "cost": 0.0, "last": 1.0,
                     "market_value": 502000.0}]


def test_trades_table_skips_total_row():
    rows = parse_table(TRADES, "trades")["rows"]
    assert [(r["volume"], r["amount"], r["price"], r["order_id"]) for r in rows] == [
        (680.0, 68000.0, 1.425, "24433"), (4340.0, 434000.0, 1.425, "24433")]
    assert all(r["code"] == "204001" and r["time"] == "10:52:54" and r["side"] == "通用回购逆回购" for r in rows)


def test_orders_table_volume_is_not_remaining():
    (r,) = parse_table(ORDERS, "orders")["rows"]
    assert (r["volume"], r["traded_volume"], r["status"], r["price"], r["order_id"]) == (5020.0, 5020.0, "已成", 1.375,
                                                                                         "24433")


def test_funds_row_and_account():
    f = parse_table(FUNDS, "funds")
    assert (f["total_asset"], f["available"], f["frozen"], f["market_value"], f["account_last4"]) == (
        503121.79, 502121.79, 0.0, 1000.0, "6735")
    assert f["login_status"] == "登录成功"


def test_empty_table_and_unknown_page():
    head_only = [b for b in ORDERS if b[2][1] <= 522]
    assert parse_table(head_only, "orders") == {"rows": []}
    assert parse_table([box(100, 50, "沪港通"), box(200, 50, "深港通")], "orders") is None      # 认不出表头 = 不在这一页
