import importlib
import os

MODS = ['akshare', 'thsdk', 'xtquant', 'tushare', 'baostock', 'efinance', 'yfinance', 'requests']
print("可用数据库:")
for m in MODS:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:10s} OK   {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:10s} NO   {type(e).__name__}")

print("\nMCP 配置:")
p = os.path.expanduser("~/.workbuddy/mcp.json")
print(f"  {p}: {'存在' if os.path.exists(p) else '不存在'}")
