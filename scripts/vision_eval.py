"""视觉模型读 QMT 截图的对比评测 (10-27): 用实盘里存下的截图和人工核对过的答案, 比较各家的准确率和耗时。

  python scripts/vision_eval.py --dir /tmp/vision_eval [--providers deepseek,minimax]
截图从 Windows D:\\qmtcode\\logs\\ui_steps 拷来 (文件名见 CASES)。都是整屏原图, 不裁剪, 测的是原始能力。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmt_bridge.ui_agent import ask, png_size, to_pixels  # noqa: E402

CASES = [
    ("下单面板读数(GC001 小字)", "20261027_103632_submit_sell_check.png",
     "这是 1920x1080 的交易软件截图。只读左下角下单面板, 只返回 JSON: {code (6位证券代码), name, price (委托价), "
     "volume (总量), side (提交按钮上的文字: 买入或卖出)}",
     {"code": "204001", "name": "GC001", "price": 1.1, "volume": 10, "side": "卖出"}),
    ("委托表读列(成交数量≠剩余量)", "20261026_102459_read_orders.png",
     "这是 1920x1080 的交易软件截图。读下方「委托」表格第一行, 只返回 JSON: {code, traded_volume (成交数量列), "
     "status (委托状态), order_id (合同编号列)}",
     {"code": "600000", "traded_volume": 0, "status": "已撤", "order_id": "15730"}),
    ("资金表读数", "20261026_102459_read_orders.png",
     "这是 1920x1080 的交易软件截图。读右下「账号资金」表, 只返回 JSON: {total_asset (总资产), available (可用金额), "
     "frozen (冻结金额), market_value (总市值)}",
     {"total_asset": 503121.8, "available": 503121.8, "frozen": 0, "market_value": 0}),
    ("当前顶部页签", "20261027_093635_snap_funds.png",
     "这是 1920x1080 的交易软件截图。顶部那排页签 (股票交易/期货交易/融资融券/组合交易/股票期权/沪港通/深港通/ETF申赎) "
     "当前选中的是哪一个? 只返回 JSON: {selected}", {"selected": "沪港通"}),
    ("定位密集按钮「选中撤单」", "20261026_101433_tick_row_after.png",
     "这是 1920x1080 的交易软件截图。找委托表格上方工具栏里的「选中撤单」按钮 (旁边有 全选/反选/全部撤单), "
     "只返回 JSON: {x, y} (按钮中心, 原图像素坐标)", {"xy": (1096, 499)}),
    ("定位顶部页签「股票交易」", "20261027_093635_snap_funds.png",
     "这是 1920x1080 的交易软件截图。找顶部页签栏里的「股票交易」页签, 只返回 JSON: {x, y} (中心, 原图像素坐标)",
     {"xy": (110, 51)}),
]


def score(got, want):
    if "xy" in want:
        try:
            d = ((float(got["x"]) - want["xy"][0]) ** 2 + (float(got["y"]) - want["xy"][1]) ** 2) ** 0.5
        except (KeyError, TypeError, ValueError):
            return False, "没给坐标"
        return d <= 20, f"偏 {d:.0f}px"
    bad = []
    for k, v in want.items():
        g = got.get(k)
        if isinstance(v, (int, float)) and g is not None:
            try:
                ok = abs(float(str(g).replace(",", "").rstrip("股张元 ")) - float(v)) < 1e-6     # 「0股」「10张」算数字
            except ValueError:
                ok = False
        else:
            ok = str(g).strip() == str(v)
        if not ok:
            bad.append(f"{k}={g!r}")
    return not bad, "全对" if not bad else "错: " + ", ".join(bad)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/vision_eval")
    ap.add_argument("--providers", default="deepseek,minimax")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args(argv)
    for prov in a.providers.split(","):
        n_ok, n, secs = 0, 0, []
        print(f"== {prov}")
        for name, fn, prompt, want in CASES:
            png = (Path(a.dir) / fn).read_bytes()
            for _ in range(a.repeat):
                t0 = time.time()
                try:
                    got, _ = ask(png, prompt, provider=prov)
                    if "xy" in want and got.get("x") is not None:
                        got["x"], got["y"] = to_pixels(prov, got["x"], got["y"], *png_size(png))
                    ok, why = score(got, want)
                except Exception as e:  # noqa: BLE001
                    ok, why, got = False, f"异常 {type(e).__name__}: {str(e)[:80]}", {}
                secs.append(time.time() - t0)
                n_ok += ok
                n += 1
                print(f"  {'✓' if ok else '✗'} {name}: {why} ({secs[-1]:.1f}s) {json.dumps(got, ensure_ascii=False)[:120]}")
        print(f"  合计 {n_ok}/{n} 正确, 平均 {sum(secs) / max(1, len(secs)):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
