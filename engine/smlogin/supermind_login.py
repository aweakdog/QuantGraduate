"""
smlogin — SuperMind 认证 & 执行引擎

依赖: aiohttp, jupyter_config.json
用法:
    from smlogin import SuperMindSession
    async with SuperMindSession() as sm:
        result = await sm.execute("print('hello')")
"""
import asyncio
import json
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from aiohttp import ClientSession, WSMsgType

# 默认配置路径
DEFAULT_CONFIG = Path(r"D:\myAI\quant-strategy\jupyter_config.json")


def load_config(config_path=None):
    """加载 jupyter_config.json，返回 {token, user, hub}"""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = json.loads(path.read_text(encoding='utf-8'))
    return cfg


class SuperMindSession:
    """
    SuperMind 研究环境认证和执行会话。

    封装 JupyterHub REST API + WebSocket kernel 通道。
    自动处理 token 鉴权、kernel 获取/创建、代码执行、结果解析。
    """

    def __init__(self, config_path: str = None, cleanup: bool = True):
        """
        Args:
            config_path: jupyter_config.json 路径
            cleanup: 退出时是否删除创建的 kernel（防止泄漏）
        """
        cfg = load_config(config_path)
        self.token = cfg["token"]
        self.user_id = cfg["user"]
        self.hub_url = cfg["hub"].rstrip("/")
        self.user_url = f"{self.hub_url}/user/{self.user_id}"
        self.ws_url = f"wss://supermind.10jqka.com.cn/notebook/user/{self.user_id}/api/kernels"
        self.cleanup = cleanup
        self._created_kernel = False  # 是否由我们创建

        self.session: Optional[ClientSession] = None
        self.kernel_id: Optional[str] = None
        self.ws = None

    # ────── Session 管理 ──────

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = ClientSession(
                headers={"Authorization": f"token {self.token}"}
            )

    async def close(self):
        """关闭 WS 和 HTTP session。如 cleanup=True 且由我们创建的 kernel，一并删除。"""
        if self.ws:
            await self.ws.close()
            self.ws = None
        if self.kernel_id and self._created_kernel and self.cleanup:
            try:
                async with self.session.delete(
                    f"{self.user_url}/api/kernels/{self.kernel_id}"
                ):
                    pass
            except:
                pass
            self.kernel_id = None
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    # ────── Kernel 管理 ──────

    async def get_kernels(self) -> List[Dict]:
        """列举所有 kernel"""
        await self._ensure_session()
        async with self.session.get(f"{self.user_url}/api/kernels") as r:
            return await r.json()

    async def create_kernel(self, name: str = "python3") -> str:
        """创建新 kernel，返回 kernel_id"""
        await self._ensure_session()
        async with self.session.post(
            f"{self.user_url}/api/kernels", json={"name": name}
        ) as r:
            if r.status != 201:
                text = await r.text()
                raise RuntimeError(f"Create kernel failed ({r.status}): {text[:200]}")
            data = await r.json()
            kid = data.get("id")
            if not kid:
                raise RuntimeError(f"Create kernel: no id in response: {data}")
        self._created_kernel = True
        # 等 idle（最多 30 秒）
        for _ in range(30):
            await asyncio.sleep(1)
            async with self.session.get(f"{self.user_url}/api/kernels/{kid}") as r:
                state = (await r.json()).get("execution_state")
                if state == "idle":
                    break
        return kid

    async def delete_kernel(self, kernel_id: str):
        """删除 kernel"""
        await self._ensure_session()
        async with self.session.delete(
            f"{self.user_url}/api/kernels/{kernel_id}"
        ) as r:
            if r.status == 204:
                return True
            raise RuntimeError(f"Delete kernel failed: {r.status} {await r.text()}")

    async def _get_or_create_kernel(self) -> str:
        """获取一个 idle kernel，没有则新建"""
        kernels = await self.get_kernels()
        for k in kernels:
            if k.get("execution_state") == "idle":
                return k["id"]
        return await self.create_kernel()

    # ────── WebSocket 连接 ──────

    async def connect(self, kernel_id: Optional[str] = None):
        """
        连接到 WebSocket kernel channel。

        Args:
            kernel_id: 指定 kernel，None=自动获取/创建
        """
        await self._ensure_session()
        self.kernel_id = kernel_id or await self._get_or_create_kernel()

        ws_url = f"{self.ws_url}/{self.kernel_id}/channels"
        self.ws = await self.session.ws_connect(ws_url, max_msg_size=0)

    # ────── 代码执行 ──────

    async def execute(self, code: str, timeout: int = 180) -> Dict[str, Any]:
        """
        在 kernel 中执行 Python 代码，等待完成。

        Args:
            code: Python 代码
            timeout: 最大等待秒数

        Returns:
            {
                "status": "ok" | "error",
                "stdout": str,   # print 输出
                "result": str,   # 表达式结果
                "error": str,    # 错误堆栈
                "messages": []   # 原始消息记录
            }
        """
        if not self.ws:
            raise RuntimeError("Not connected. Call connect() first.")

        exec_id = str(uuid.uuid4())
        msg = {
            "header": {
                "msg_id": exec_id,
                "msg_type": "execute_request",
                "username": "kq2026",
                "session": "sm-auto-" + uuid.uuid4().hex[:8],
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
        await self.ws.send_json(msg)

        result = {
            "status": "unknown",
            "stdout": "",
            "result": "",
            "error": "",
            "messages": []
        }

        for _ in range(timeout):
            try:
                ws_msg = await asyncio.wait_for(self.ws.receive(), timeout=1)
                if ws_msg.type == WSMsgType.TEXT:
                    data = json.loads(ws_msg.data)
                    mt = data.get("header", {}).get("msg_type", "")
                    parent = data.get("parent_header", {}).get("msg_id", "")

                    if not parent or (parent[:16] != exec_id[:16] and parent != exec_id):
                        continue

                    content = data.get("content", {})
                    result["messages"].append({"msg_type": mt, "channel": data.get("channel")})

                    if mt == "stream":
                        result["stdout"] += content.get("text", "")
                    elif mt == "execute_result":
                        result["result"] += content.get("data", {}).get("text/plain", "")
                    elif mt == "display_data":
                        result["result"] += content.get("data", {}).get("text/plain", "")
                    elif mt == "error":
                        result["error"] += content.get("evalue", "") + "\n"
                        result["error"] += "\n".join(content.get("traceback", []))
                    elif mt == "execute_reply":
                        result["status"] = content.get("status", "error")
                    elif mt == "status":
                        state = content.get("execution_state", "")
                        if state == "idle" and result["status"] != "unknown":
                            break

            except asyncio.TimeoutError:
                continue

        return result

    # ────── 文件操作 ──────

    async def list_files(self, path: str = "/") -> List[Dict]:
        """列举用户目录下的文件/notebook"""
        await self._ensure_session()
        async with self.session.get(f"{self.user_url}/api/contents/{path}") as r:
            data = await r.json()
            return data.get("content", [])

    async def read_notebook(self, path: str) -> Dict:
        """读取 notebook 文件内容"""
        await self._ensure_session()
        async with self.session.get(f"{self.user_url}/api/contents/{path}") as r:
            return await r.json()

    async def write_notebook(self, path: str, content: dict):
        """写入 notebook 文件"""
        await self._ensure_session()
        async with self.session.put(
            f"{self.user_url}/api/contents/{path}",
            json={"type": "notebook", "content": content, "format": "json"}
        ) as r:
            return await r.json()

    # ────── 上下文管理器 ──────

    async def __aenter__(self):
        """上下文管理器入口：创建新 kernel 并连接"""
        kid = await self.create_kernel()
        await self.connect(kid)
        return self

    async def __aexit__(self, *args):
        await self.close()

    def __repr__(self):
        return f"<SuperMindSession user={self.user_id} kernel={self.kernel_id}>"


# ────── 快捷函数 ──────

async def execute_on_kernel(code: str, kernel_id: str = None,
                            config_path: str = None, timeout: int = 180) -> Dict:
    """快捷：创建 session → 连接 → 执行 → 关闭"""
    async with SuperMindSession(config_path) as sm:
        if kernel_id:
            await sm.connect(kernel_id)
        result = await sm.execute(code, timeout=timeout)
        return result


async def main():
    """测试"""
    async with SuperMindSession() as sm:
        print(f"Session: {sm}")
        result = await sm.execute(f"""
import pandas as pd, json
data = get_price("601689.SH", start_date="20260601", end_date="20260620",
                 fre_step="1d", fields=["close", "volume"], fq="pre")
print(f"Data rows: {{len(data)}}, columns: {{list(data.columns)}}")
print(json.dumps({{"close": data['close'].iloc[-1], "volume": int(data['volume'].iloc[-1])}}))
""", timeout=60)
        print(f"Status: {result['status']}")
        if result['stdout']: print(f"STDOUT:\\n{result['stdout']}")
        if result['error']: print(f"ERROR:\\n{result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
