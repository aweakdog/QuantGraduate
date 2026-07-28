"""
JupyterHub REST API — 创建内核+执行代码+轮询结果
URL: supermind.10jqka.com.cn/notebook/user/{user_id}
"""
import requests
import json
import time
import urllib.parse

TOKEN = "26c4f857997e48c1a48ea59916405afb"
USER_ID = "59518255"
BASE = f"https://supermind.10jqka.com.cn/notebook/user/{USER_ID}"
HEADERS = {"Authorization": f"token {TOKEN}"}

# 1. 测试基本连通性
print("=== 1. Basic connectivity ===")
r = requests.get(f"{BASE}/api/", headers=HEADERS, timeout=10)
print(f"  GET /api/ → {r.status_code}")
if r.status_code == 200:
    print(f"  Response: {json.dumps(r.json(), ensure_ascii=False)[:300]}")

# 2. 获取已存在的 kernels
print("\n=== 2. Existing kernels ===")
r = requests.get(f"{BASE}/api/kernels", headers=HEADERS, timeout=10)
print(f"  GET /api/kernels → {r.status_code}")
if r.status_code == 200:
    kernels = r.json()
    print(f"  Active kernels: {len(kernels)}")
    for k in kernels:
        print(f"    ID: {k['id']}  Name: {k.get('name', 'N/A')}  Status: {k.get('execution_state', 'N/A')}")

# 3. 创建新 kernel
print("\n=== 3. Create new kernel ===")
payload = {"name": "python3"}
r = requests.post(f"{BASE}/api/kernels", headers=HEADERS, json=payload, timeout=30)
print(f"  POST /api/kernels → {r.status_code}")
if r.status_code == 201:
    kernel_id = r.json()['id']
    print(f"  Kernel ID: {kernel_id}")
else:
    # 尝试现有 kernel
    r = requests.get(f"{BASE}/api/kernels", headers=HEADERS, timeout=10)
    if r.status_code == 200 and len(r.json()) > 0:
        kernel_id = r.json()[0]['id']
        print(f"  Using existing kernel: {kernel_id}")
    else:
        print(f"  No kernel available: {r.text[:200]}")
        exit(1)

# 4. 检查 kernel 状态
print(f"\n=== 4. Kernel status ===")
r = requests.get(f"{BASE}/api/kernels/{kernel_id}", headers=HEADERS, timeout=10)
print(f"  {r.status_code}: {json.dumps(r.json(), ensure_ascii=False)[:200]}")

# 5. 尝试执行代码 via REST API
print("\n=== 5. Execute code via REST ===")
code = """
import pandas as pd
print("Hello from Jupyter REST API!")
print(f"pandas version: {pd.__version__}")
"""

# Jupyter Server 的 execute 端点
exec_url = f"{BASE}/api/kernels/{kernel_id}/execute"
print(f"  POST {exec_url}")
r = requests.post(exec_url, headers=HEADERS, json={"code": code}, timeout=15)
print(f"  Status: {r.status_code}")
if r.status_code == 200 or r.status_code == 201:
    data = r.json()
    print(f"  Response: {json.dumps(data, ensure_ascii=False)[:500]}")
else:
    print(f"  Error: {r.text[:300]}")

# 6. 也尝试其他执行方式
print("\n=== 6. Alternative: check other API endpoints ===")
for ep in ["/api/contents", "/api/sessions", "/api/terminals", 
           "/api/kernelspecs", "/api/config/", "/api/status"]:
    r = requests.get(f"{BASE}{ep}", headers=HEADERS, timeout=10)
    print(f"  GET {ep} → {r.status_code} {len(r.text)} bytes")
    if r.status_code == 200 and ep == "/api/sessions":
        sessions = r.json()
        print(f"    Sessions: {len(sessions)}")
        for s in sessions:
            print(f"      {s.get('name','N/A')} kernel={s.get('kernel',{}).get('id','N/A')}")

# 7. 获取 notebook content
print("\n=== 7. Notebook contents ===")
r = requests.get(f"{BASE}/api/contents", headers=HEADERS, timeout=10)
if r.status_code == 200:
    contents = r.json()
    print(f"  Files in root: {len(contents.get('content',[]))}")
    for item in contents.get('content', []):
        print(f"    {item['type']:8s} {item['name']}")
else:
    print(f"  GET /api/contents → {r.status_code}: {r.text[:200]}")
