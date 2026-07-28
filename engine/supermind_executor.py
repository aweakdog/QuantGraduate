"""SuperMind 研究环境 — WebSocket 执行引擎
全 REST + WebSocket，零浏览器依赖。
"""
import asyncio
import json
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from aiohttp import ClientSession, WSMsgType, ClientWSTimeout

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "jupyter_config.json"
CONFIG = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
TOKEN = CONFIG["token"]
USER_ID = CONFIG["user"]
WS_BASE = f"wss://supermind.10jqka.com.cn/notebook/user/{USER_ID}/api/kernels"
BASE = f"https://supermind.10jqka.com.cn/notebook/user/{USER_ID}"

class SuperMindExecutor:
    """SuperMind Jupyter 研究环境执行引擎"""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.kernel_id: Optional[str] = None
        self.ws = None
    
    async def _ensure_session(self):
        if self.session is None:
            self.session = ClientSession(headers={"Authorization": f"token {TOKEN}"})
    
    async def _get_or_create_kernel(self) -> str:
        """获取一个空闲的 kernel，或新建一个"""
        await self._ensure_session()
        async with self.session.get(f"{BASE}/api/kernels") as r:
            kernels = await r.json()
        
        # 找一个 idle kernel
        for k in kernels:
            if k.get('execution_state') == 'idle':
                print(f"  Using idle kernel: {k['id'][:12]}...")
                return k['id']
        
        # 没有则新建
        async with self.session.post(f"{BASE}/api/kernels", json={"name": "python3"}) as r:
            data = await r.json()
            kid = data['id']
            print(f"  Created kernel: {kid[:12]}...")
            # 等它 idle
            for _ in range(10):
                await asyncio.sleep(1)
                async with self.session.get(f"{BASE}/api/kernels/{kid}") as r:
                    state = (await r.json()).get('execution_state')
                    if state == 'idle':
                        break
            return kid
    
    async def connect(self, kernel_id: Optional[str] = None):
        """连接到指定或可用的 kernel"""
        await self._ensure_session()
        if kernel_id:
            self.kernel_id = kernel_id
        else:
            self.kernel_id = await self._get_or_create_kernel()
        
        ws_url = f"{WS_BASE}/{self.kernel_id}/channels"
        print(f"  WS connect: {ws_url[:60]}...")
        ws_timeout = ClientWSTimeout(ws_close=60)
        self.ws = await self.session.ws_connect(ws_url, max_msg_size=0, timeout=ws_timeout)
        print(f"  ✅ WebSocket connected")
    
    async def execute(self, code: str, timeout: int = 120) -> Dict[str, Any]:
        """
        在 kernel 中执行 Python 代码，等待完成并返回结果。
        
        Returns:
            { 'status': 'ok'|'error',
              'stdout': str,      # 所有 print 输出
              'result': str,      # 最后表达式的结果
              'error': str,       # 错误信息
              'messages': [...]   # 原始消息
            }
        """
        if not self.ws:
            raise RuntimeError("Not connected. Call connect() first.")
        
        exec_id = str(uuid.uuid4())
        exec_msg = {
            "header": {
                "msg_id": exec_id,
                "msg_type": "execute_request",
                "username": "kq2026",
                "session": "automation-session",
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
        
        await self.ws.send_json(exec_msg)
        
        result = {
            "status": "unknown",
            "stdout": "",
            "result": "",
            "error": "",
            "messages": []
        }
        
        for _ in range(timeout):
            try:
                msg = await asyncio.wait_for(self.ws.receive(), timeout=1)
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("header", {}).get("msg_type", "")
                    parent = data.get("parent_header", {}).get("msg_id", "")
                    channel = data.get("channel", "")
                    
                    if (not parent) or (parent[:16] != exec_id[:16] and parent != exec_id):
                        continue
                    
                    entry = {"msg_type": msg_type, "channel": channel}
                    content = data.get("content", {})
                    result["messages"].append(entry)
                    
                    if msg_type == "stream":
                        text = content.get("text", "")
                        result["stdout"] += text
                        entry["text"] = text
                    
                    elif msg_type == "execute_result":
                        text = content.get("data", {}).get("text/plain", "")
                        result["result"] += text
                        entry["text"] = text
                    
                    elif msg_type == "display_data":
                        text = content.get("data", {}).get("text/plain", "")
                        result["result"] += text
                        entry["text"] = text
                    
                    elif msg_type == "error":
                        e_value = content.get("evalue", "")
                        traceback = "\n".join(content.get("traceback", []))
                        result["error"] += f"{e_value}\n{traceback}"
                        entry["error"] = result["error"]
                    
                    elif msg_type == "execute_reply":
                        status = content.get("status", "")
                        result["status"] = status
                        if status != "ok":
                            entry["error"] = content.get("evalue", "")
                    
                    elif msg_type == "status":
                        state = content.get("execution_state", "")
                        if state == "idle" and result["status"] != "unknown":
                            break
            except asyncio.TimeoutError:
                continue
        
        return result
    
    async def close(self):
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


async def main():
    """测试：执行拓普集团 6因子回测"""
    
    code = """
# 拓普集团 6因子策略测试
import pandas as pd
import numpy as np
import json

stock = '601689.SH'

# 验证环境
print(f"Stock: {stock}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")

# 测试 get_price
try:
    data = get_price(stock, start_date='20260101', end_date='20260620', 
                     fre_step='1d', fields=['close', 'high', 'low', 'volume'], fq='pre')
    print(f"get_price OK: {len(data)} rows, columns={list(data.columns)}")
    print(data.tail(3).to_json(orient='records'))
except Exception as e:
    print(f"get_price error: {e}")

# 测试 query_iwencai
try:
    r = query_iwencai(f"{stock}近20日主力资金流向", df=True)
    print(f"query_iwencai OK: {type(r).__name__}")
except Exception as e:
    print(f"query_iwencai error: {e}")
"""
    
    async with SuperMindExecutor() as executor:
        await executor.connect()
        result = await executor.execute(code, timeout=60)
        
        print(f"\n{'='*60}")
        print(f"Status: {result['status']}")
        print(f"{'='*60}")
        
        if result['stdout']:
            print(f"\n--- STDOUT ---")
            print(result['stdout'])
        
        if result['error']:
            print(f"\n--- ERROR ---")
            print(result['error'])
        
        if result['result']:
            print(f"\n--- RESULT ---")
            print(result['result'])
    
    print("\n✅ Done")

if __name__ == "__main__":
    asyncio.run(main())
