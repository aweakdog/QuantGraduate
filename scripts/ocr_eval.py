"""OCR 读 QMT 截图的评测 (10-27, 在 Windows 上跑): 与 scripts/vision_eval.py 同一批截图和答案, 方便和两家模型比较。

  .venv\\Scripts\\python.exe scripts\\ocr_eval.py --dir D:\\qmtcode\\logs\\ui_steps
读数类用例在固定区域裁图后识别 (布局已知); 定位类用例在整屏里找文字框中心。
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmt_bridge.ocr_read import locate_text, ocr_boxes, text_of  # noqa: E402

NUM = re.compile(r"\d[\d ,]*\.?\d*")


def nums(text):
    return [float(m.replace(" ", "").replace(",", "")) for m in NUM.findall(text) if m.strip(" ,.")]


def case_panel(png):
    b = ocr_boxes(png, (520, 495, 720, 535), 3) + ocr_boxes(png, (270, 735, 470, 890), 3)
    t = text_of(b)
    code = next((m for m in re.findall(r"\d{6}", t)), None)
    pm = re.search(r"委托价\s*(\d+\.\d+)", t)          # 不是面板上方的最新价
    price = float(pm.group(1)) if pm else None
    vol = re.search(r"总[量里]\s*(\d+)", t)
    return {"code": code, "name": "GC001" if "GC001" in t else None, "price": price,
            "volume": int(vol.group(1)) if vol else None}, t


def case_orders(png):
    t = text_of(ocr_boxes(png, (780, 530, 1920, 560), 3))
    order_id = next((m for m in re.findall(r"\b\d{5}\b", t)), None)
    status = next((s for s in ("已撤", "已成", "已报", "部成", "废单") if s in t), None)
    # 成交数量紧挨在状态前面; 单位「股」可能被识别成别的字 (实测「0股」→「0吸」), 只认数字
    tm = re.search(r"(\d+)\s*\S?\s*(?:已撤|已成|已报|部成|废单)", t)
    return {"code": "600000" if "600000" in t else None, "status": status, "order_id": order_id,
            "traded_volume": int(tm.group(1)) if tm else None}, t


def case_funds(png):
    t = text_of(ocr_boxes(png, (1100, 800, 1920, 830), 3))
    v = nums(t)
    return {"values": v[:8]}, t


CASES = [
    ("下单面板读数(GC001 小字)", "20261027_103632_submit_sell_check.png", case_panel,
     lambda g: g["code"] == "204001" and g["price"] == 1.1 and g["volume"] == 10),
    ("委托表读列", "20261026_102459_read_orders.png", case_orders,
     lambda g: g["status"] == "已撤" and g["order_id"] == "15730" and g["traded_volume"] == 0),
    ("资金表读数", "20261026_102459_read_orders.png", case_funds,
     lambda g: g["values"][:1] == [503121.8] and 503121.8 in g["values"]),
    ("定位「选中撤单」", "20261026_101433_tick_row_after.png",
     lambda png: ({"xy": locate_text(png, "选中撤单", None, 1)}, ""),
     lambda g: g["xy"] is not None and abs(g["xy"][0] - 1096) <= 20 and abs(g["xy"][1] - 499) <= 20),
    ("定位「股票交易」页签", "20261027_093635_snap_funds.png",
     lambda png: ({"xy": locate_text(png, "股票交易", (0, 30, 1100, 70), 2)}, ""),
     lambda g: g["xy"] is not None and abs(g["xy"][0] - 110) <= 20 and abs(g["xy"][1] - 51) <= 20),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\qmtcode\logs\ui_steps")
    a = ap.parse_args(argv)
    ok_n = 0
    for name, fn, run, check in CASES:
        png = (Path(a.dir) / fn).read_bytes()
        t0 = time.time()
        got, raw = run(png)
        ok = bool(check(got))
        ok_n += ok
        print(f"{'✓' if ok else '✗'} {name}: {got} ({time.time() - t0:.1f}s) | {raw[:160]}")
    print(f"合计 {ok_n}/{len(CASES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
