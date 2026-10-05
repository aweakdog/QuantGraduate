"""QMT 桥 (Windows 侧, 华泰 QMT 实盘机 DESKTOP-NTRMANG)。

阶段一(2026-10-06 用户定): 只读对账 —— 只连 miniQMT 读资金/持仓/委托/成交, 不下任何单。
下单(收盘集合竞价)是后续阶段, 届时另立模块与风控; 本包内的只读模块必须保持零下单能力。
部署: Mac 上 scp 到 D:\\qmtcode\\qmt_bridge, 用 D:\\qmtcode\\.venv (Python 3.11 + xtquant 250807.1.2) 运行。
"""
