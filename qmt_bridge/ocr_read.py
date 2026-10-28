"""本地 OCR 读屏 (2026-10-27, Windows 上运行): RapidOCR (onnxruntime), 不联网、结果确定。

用法: 在固定区域裁图后识别, 返回 [(文字, 置信度, (屏幕x, 屏幕y))]。数字读数以 OCR 为主, 模型只做交叉核对。
"""
import io
from functools import lru_cache


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR
    return RapidOCR()


def ocr_boxes(png, box=None, scale=2):
    """png: 整屏 PNG 字节; box: (x0, y0, x1, y1) 屏幕坐标, None = 整屏。返回 [(text, score, (cx, cy))] (屏幕坐标)。"""
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB")
    x0, y0 = (box[0], box[1]) if box else (0, 0)
    if box:
        im = im.crop(box)
    if scale != 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    res = _engine()(buf.getvalue())
    out = []
    for pts, text, score in zip(res.boxes if res.boxes is not None else [], res.txts or [], res.scores or [], strict=False):
        cx = sum(p[0] for p in pts) / 4 / scale + x0
        cy = sum(p[1] for p in pts) / 4 / scale + y0
        out.append((str(text), float(score), (round(cx), round(cy))))
    return out


def find_text(boxes, target, box_width=None):
    """在 OCR 结果里找包含 target 的文字框; 正好一个返回 target 本身的中心坐标, 否则 None (没找到或有歧义)。
    相邻按钮常被识别成一个框 (如「选中撤单 全部撤单」), 这时按字符位置在框内按比例算出 target 的中心。"""
    hits = [b for b in boxes if target in b[0]]
    if len(hits) != 1:
        return None
    text, _, (cx, cy) = hits[0]
    if text == target or not box_width:
        return (cx, cy)
    i = text.index(target)
    frac = (i + len(target) / 2) / len(text)
    return (round(cx - box_width / 2 + frac * box_width), cy)


def ocr_boxes_w(png, box=None, scale=2):
    """同 ocr_boxes, 但每项多一个框宽 (屏幕像素), 供 find_text 按比例定位。"""
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB")
    x0, y0 = (box[0], box[1]) if box else (0, 0)
    if box:
        im = im.crop(box)
    if scale != 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    res = _engine()(buf.getvalue())
    out = []
    for pts, text, score in zip(res.boxes if res.boxes is not None else [], res.txts or [], res.scores or [], strict=False):
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        out.append((str(text), float(score), (round(sum(xs) / 4 / scale + x0), round(sum(ys) / 4 / scale + y0)),
                    (max(xs) - min(xs)) / scale))
    return out


def locate_text(png, target, box=None, scale=2):
    """整屏/区域里定位一段文字的中心 (屏幕坐标); 没找到或不唯一返回 None。"""
    bw = ocr_boxes_w(png, box, scale)
    hits = [b for b in bw if target in b[0]]
    if len(hits) != 1:
        return None
    t, s, c, w = hits[0]
    return find_text([(t, s, c)], target, box_width=w)


# 表头文字 -> 字段 (10-28 按 QMT 2.1.31.3 实际表头; 表头按「完全相等」匹配, 免得「委托量」认成「委托剩余量」)
COLS = {
    "positions": {"证券代码": "code", "证券名称": "name", "当前拥股": "volume", "可用数量": "can_use", "成本价": "cost",
                  "市值": "market_value"},
    "trades": {"成交时间": "time", "证券代码": "code", "证券名称": "name", "买卖标记": "side", "成交数量": "volume",
               "成交价格": "price", "成交金额": "amount", "合同编号": "order_id"},
    "orders": {"委托时间": "time", "证券代码": "code", "证券名称": "name", "买卖标记": "side", "委托量": "volume",
               "成交数量": "traded_volume", "委托状态": "status", "委托价格": "price", "合同编号": "order_id"},
    "funds": {"资金账号": "account", "登录状态": "login_status", "总资产": "total_asset", "总市值": "market_value",
              "可用金额": "available", "冻结金额": "frozen", "可取金额": "withdrawable"},
}
NUMERIC = {"volume", "can_use", "cost", "market_value", "price", "amount", "traded_volume", "total_asset", "available",
           "frozen", "withdrawable"}
ROW_TOL = 7          # 同一行的文字框中心 y 相差不超过 7 像素


def _number(text):
    """「50 2000.00」(QMT 用空格分千位)、「5020张」「￥0」「1,234.5」都转成数; 转不了返回 None。"""
    t = text.replace(" ", "").replace(",", "").replace("￥", "").replace("¥", "").rstrip("股张元")
    try:
        return float(t)
    except ValueError:
        return None


def parse_table(boxes, kind):
    """boxes: ocr_boxes_w 的结果 [(text, score, (cx, cy), width)]。认不出表头返回 None (多半不在这一页), 否则返回
    {"rows": [...]} (表格类) 或字段字典 (funds)。按表头横向中点划分列边界 (表头行上所有列都参与, 包括用不到的列),
    按 y 聚成行, 跳过「合计」行。"""
    cols = COLS[kind]
    heads = [b for b in boxes if b[0].strip() in cols]
    if len({b[0].strip() for b in heads}) < min(3, len(cols)):
        return None
    hy = sorted(b[2][1] for b in heads)[len(heads) // 2]
    row_heads = sorted((b for b in boxes if abs(b[2][1] - hy) <= ROW_TOL), key=lambda b: b[2][0])
    xs = [b[2][0] for b in row_heads]
    bounds = [(-1e9 if i == 0 else (xs[i - 1] + xs[i]) / 2, 1e9 if i == len(xs) - 1 else (xs[i] + xs[i + 1]) / 2)
              for i in range(len(xs))]
    field_of = {i: cols.get(b[0].strip()) for i, b in enumerate(row_heads)}
    body = sorted((b for b in boxes if b[2][1] > hy + ROW_TOL), key=lambda b: b[2][1])
    rows, cur = [], []
    for b in body:
        if cur and abs(b[2][1] - cur[0][2][1]) > ROW_TOL:
            rows.append(cur)
            cur = []
        cur.append(b)
    if cur:
        rows.append(cur)
    out = []
    for r in rows:
        if any("合计" in b[0] for b in r):
            continue
        cells = {}
        for b in sorted(r, key=lambda b: b[2][0]):
            i = next((k for k, (lo, hi) in enumerate(bounds) if lo <= b[2][0] < hi), None)
            f = field_of.get(i)
            if f:
                cells[f] = cells.get(f, "") + b[0].strip()
        rec = {}
        for f in cols.values():
            v = cells.get(f)
            if v is None:
                rec[f] = None
            elif f in NUMERIC:
                rec[f] = _number(v)
            elif f == "code":
                m = next((v[i:i + 6] for i in range(len(v) - 5) if v[i:i + 6].isdigit()), None)
                rec[f] = m
            else:
                rec[f] = v
        out.append(rec)
    if kind == "funds":
        if not out:
            return None
        f = out[0]
        acct = "".join(ch for ch in str(f.get("account") or "") if ch.isdigit())
        return {**{k: f.get(k) for k in ("total_asset", "available", "frozen", "market_value", "withdrawable",
                                          "login_status")}, "account_last4": acct[-4:] or None}
    return {"rows": out}


def read_table(png, box, kind, scale=2):
    """在固定区域 OCR 后按表头解析; 认不出表头返回 None。"""
    return parse_table(ocr_boxes_w(png, box, scale), kind)


def parse_panel(code_text, form_text):
    """下单面板 OCR 文字 -> {code, price, volume, est_amount}。代码取右上角第一个 6 位数 (GC001 只有 3 位数字, 不会误认)。"""
    import re
    code = next(iter(re.findall(r"\d{6}", code_text)), None)
    pm = re.search(r"委托价\s*(\d+\.\d+)", form_text)
    vm = re.search(r"总[量里]\s*(\d+)", form_text)
    em = re.search(r"预估金额\S*?\s*([\d ,]+\.\d+)\s*元", form_text)
    return {"code": code, "price": float(pm.group(1)) if pm else None, "volume": int(vm.group(1)) if vm else None,
            "est_amount": float(em.group(1).replace(" ", "").replace(",", "")) if em else None}


def available():
    """这台机器能不能用 OCR (Windows 上装了 rapidocr; Mac 上一般没装)。"""
    try:
        import rapidocr  # noqa: F401
        return True
    except ImportError:
        return False


def text_of(boxes):
    return " ".join(b[0] for b in sorted(boxes, key=lambda b: (b[2][1] // 8, b[2][0])))
