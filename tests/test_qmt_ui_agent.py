"""界面感知原型: 请求格式、回复解析、点击前的硬检查 (不联网)。"""
import io
import json

import pytest

from qmt_bridge.ui_agent import ask, build_request, dismiss_dialog, png_size, safe_target

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (1920).to_bytes(4, "big") + (1080).to_bytes(4, "big") + b"\x08\x02"


def test_png_size_and_request_shape():
    assert png_size(PNG) == (1920, 1080)
    with pytest.raises(ValueError):
        png_size(b"GIF89a....")
    req = build_request(PNG, "找按钮")
    assert req["model"] == "deepseek-flash" and req["response_format"] == {"type": "json_object"}
    parts = req["messages"][0]["content"]
    assert parts[0]["text"] == "找按钮" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert req["messages"][0]["role"] == "user"            # DeepSeek 只接受 user 消息里的图片


def test_ask_parses_json_and_drops_echoed_type():
    reply = {"choices": [{"message": {"content": json.dumps({"type": "json_object", "found": True, "x": 1})}}],
             "usage": {"total_tokens": 10}}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = {}

    def opener(req, timeout):
        seen["auth"] = req.headers.get("Authorization")
        return Resp(json.dumps(reply).encode())
    data, meta = ask(PNG, "p", key="k-test", opener=opener)
    assert data == {"found": True, "x": 1} and meta["usage"]["total_tokens"] == 10 and seen["auth"] == "Bearer k-test"


@pytest.mark.parametrize("loc,ok", [
    ({"found": True, "label": "确定", "x": 1055, "y": 505, "confidence": 0.95, "ambiguous": False}, True),
    ({"found": False}, False),
    ({"found": True, "label": "取消", "x": 1055, "y": 505, "confidence": 0.95}, False),       # 标签不符
    ({"found": True, "label": "确定", "x": 1055, "y": 505, "confidence": 0.5}, False),        # 不够确定
    ({"found": True, "label": "确定", "x": 1055, "y": 505, "confidence": 0.9, "ambiguous": True}, False),
    ({"found": True, "label": "确定", "x": 2500, "y": 505, "confidence": 0.9}, False),        # 越界
])
def test_safe_target_refuses_anything_doubtful(loc, ok):
    assert safe_target(loc, "确定", (1920, 1080))[0] is ok


def test_dismiss_only_whitelisted_dialogs():
    with pytest.raises(SystemExit):
        dismiss_dialog("下单确认", "确定")                    # 交易相关的弹窗一律不点


def test_crop_accepts_fractional_scale():
    import io

    from PIL import Image

    from qmt_bridge.ui_agent import crop
    buf = io.BytesIO()
    Image.new("RGB", (1920, 1080), "white").save(buf, format="PNG")
    part, (x0, y0, s) = crop(buf.getvalue(), 1350, 610, w=1140, h=256, scale=2.5, mark=False)
    assert png_size(part) == (2850, 640) and (x0, y0, s) == (780, 482, 2.5)


def _fake_opener(script):
    """script: 依次返回的结果; Exception 实例就抛出, dict 就当 JSON 回复。"""
    import io
    import json as _j
    import urllib.error

    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        calls.append(req.full_url)
        item = script.pop(0)
        if isinstance(item, int):
            raise urllib.error.HTTPError(req.full_url, item, "err", {}, None)
        return _Resp(_j.dumps({"choices": [{"message": {"content": _j.dumps(item)}}]}).encode())
    return opener, calls


def test_transient_529_is_retried_then_succeeds(monkeypatch):
    import qmt_bridge.ui_agent as ua
    monkeypatch.setattr(ua.time, "sleep", lambda s: None)
    opener, calls = _fake_opener([529, {"ok": 1}])
    data, meta = ua.ask(b"\x89PNG", "q", key="k", opener=opener, provider="minimax")
    assert data == {"ok": 1} and len(calls) == 2 and "fallback" not in meta


def test_minimax_keeps_failing_falls_back_to_deepseek(monkeypatch):
    import qmt_bridge.ui_agent as ua
    monkeypatch.setattr(ua.time, "sleep", lambda s: None)
    monkeypatch.setattr(ua, "provider_key", lambda name: "mm")
    monkeypatch.setattr(ua, "load_key", lambda: "ds")
    opener, calls = _fake_opener([529, 529, 529, {"ok": 2}])
    data, meta = ua.ask(b"\x89PNG", "q", opener=opener, provider="minimax")
    assert data == {"ok": 2} and meta["fallback"] == "deepseek"
    assert "minimaxi" in calls[0] and "deepseek" in calls[-1]


def test_non_transient_error_is_not_retried(monkeypatch):
    import urllib.error

    import pytest

    import qmt_bridge.ui_agent as ua
    monkeypatch.setattr(ua.time, "sleep", lambda s: None)
    opener, calls = _fake_opener([401])
    with pytest.raises(urllib.error.HTTPError):
        ua.ask(b"\x89PNG", "q", key="k", opener=opener, provider="deepseek")
    assert len(calls) == 1
