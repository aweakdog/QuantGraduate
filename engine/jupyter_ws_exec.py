"""
JupyterHub — Python WebSocket 直连 kernel channels
"""
import asyncio
import json
import time

TOKEN = "26c4f857997e48c1a48ea59916405afb"
USER_ID = "59518255"
WS_BASE = f"wss://supermind.10jqka.com.cn/notebook/user/{USER_ID}/api/kernels"

async def main():
    import aiohttp
    import aiohttp.web
    
    # 1. 先找一个已存在的 kernel（或新建一个）
    from aiohttp import ClientSession
    
    async with ClientSession(headers={"Authorization": f"token {TOKEN}"}) as session:
        # 获取已有的 kernels
        async with session.get(f"https://supermind.10jqka.com.cn/notebook/user/{USER_ID}/api/kernels") as r:
            kernels = await r.json()
            print(f"Kernels: {len(kernels)}")
            for k in kernels:
                print(f"  {k['id'][:12]}... : {k.get('execution_state', '?')}")
            if kernels:
                kernel_id = kernels[0]['id']
            else:
                async with session.post(
                    f"https://supermind.10jqka.com.cn/notebook/user/{USER_ID}/api/kernels",
                    json={"name": "python3"}
                ) as r:
                    data = await r.json()
                    kernel_id = data['id']
                    print(f"Created kernel: {kernel_id}")
        
        # 2. 连 WebSocket
        ws_url = f"{WS_BASE}/{kernel_id}/channels"
        print(f"\nConnecting WebSocket: {ws_url}")
        
        async with session.ws_connect(ws_url, max_msg_size=0) as ws:
            print("✅ WebSocket connected!")
            
            # 3. 发送内核信息请求
            msg_id = "test-msg-1"
            msg = {
                "header": {
                    "msg_id": msg_id,
                    "msg_type": "kernel_info_request",
                    "username": "kq2026",
                    "session": "test-session-1",
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "version": "5.3"
                },
                "parent_header": {},
                "metadata": {},
                "content": {},
                "buffers": [],
                "channel": "shell"
            }
            await ws.send_json(msg)
            print("  Sent: kernel_info_request")
            
            # 4. 收回复
            responses = []
            for _ in range(10):
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("header", {}).get("msg_type", "?")
                        parent = data.get("parent_header", {}).get("msg_id", "")
                        if parent == msg_id or True:  # 接受所有
                            responses.append({
                                "msg_type": msg_type,
                                "content": data.get("content", {}),
                                "channel": data.get("channel", "?")
                            })
                            print(f"  Received: {msg_type} ({data.get('channel','?')})")
                            if msg_type == "kernel_info_reply":
                                content = data.get("content", {})
                                print(f"    Language: {content.get('language_info', {}).get('name', '?')}")
                                print(f"    Version: {content.get('language_info', {}).get('version', '?')}")
                except asyncio.TimeoutError:
                    print("  Timeout waiting for messages")
                    break
            
            print(f"\n  Total responses: {len(responses)}")
            
            # 5. 发送 execute_request
            code = """
import pandas as pd, numpy as np
print("Python WebSocket直接执行成功!")
print(f"pandas {pd.__version__}, numpy {np.__version__}")
result = {"stock": "601689.SH", "ma5": 42.5, "ma20": 40.8, "signal": "买入"}
print(json.dumps(result))
"""
            import uuid
            exec_id = str(uuid.uuid4())
            exec_msg = {
                "header": {
                    "msg_id": exec_id,
                    "msg_type": "execute_request",
                    "username": "kq2026",
                    "session": "test-session-1",
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "version": "5.3"
                },
                "parent_header": {},
                "metadata": {},
                "content": {
                    "code": code,
                    "silent": False,
                    "store_history": True,
                    "user_expressions": {},
                    "allow_stdin": False,
                    "stop_on_error": True
                },
                "buffers": [],
                "channel": "shell"
            }
            
            print(f"\n→ Sending execute_request (msg_id={exec_id[:16]}...)")
            await ws.send_json(exec_msg)
            
            # 6. 接收执行结果
            outputs = []
            for i in range(30):
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=10)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("header", {}).get("msg_type", "")
                        parent = data.get("parent_header", {}).get("msg_id", "")
                        channel = data.get("channel", "")
                        
                        if parent and parent[:16] == exec_id[:16]:
                            outputs.append((msg_type, data.get("content", {})))
                            print(f"  [{channel}] {msg_type}")
                            
                            if msg_type == "stream":
                                print(f"    STDOUT: {data['content'].get('text','')[:200]}")
                            elif msg_type == "execute_result":
                                text = data['content'].get('data', {}).get('text/plain', '')
                                print(f"    RESULT: {text[:200]}")
                            elif msg_type == "display_data":
                                text = data['content'].get('data', {}).get('text/plain', '')
                                print(f"    DISPLAY: {text[:200]}")
                            elif msg_type == "error":
                                e_name = data['content'].get('ename', '')
                                e_value = data['content'].get('evalue', '')
                                print(f"    ERROR: {e_name}: {e_value[:200]}")
                            elif msg_type == "execute_reply":
                                status = data['content'].get('status', '')
                                print(f"    STATUS: {status}")
                            elif msg_type == "status":
                                state = data['content'].get('execution_state', '')
                                print(f"    KERNEL: {state}")
                                if state == 'idle':
                                    print(f"\n  ✅ Execution complete! ({len(outputs)} messages)")
                                    break
                except asyncio.TimeoutError:
                    print(f"  Timeout after {i+1} attempts")
                    break
            
            print(f"\n  Final outputs: {len(outputs)} messages")
            for mt, content in outputs:
                if mt == "stream":
                    print(f"    STDOUT: {content.get('text','')[:300]}")
                elif mt == "execute_result":
                    print(f"    RESULT: {content.get('data',{}).get('text/plain','')[:300]}")
                elif mt == "error":
                    print(f"    ERROR: {content.get('evalue','')[:200]}")
            
            await ws.close()
    
    print("\n✅ Done")

if __name__ == "__main__":
    asyncio.run(main())
