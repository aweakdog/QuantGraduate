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
