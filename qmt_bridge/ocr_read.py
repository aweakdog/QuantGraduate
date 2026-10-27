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


def text_of(boxes):
    return " ".join(b[0] for b in sorted(boxes, key=lambda b: (b[2][1] // 8, b[2][0])))
