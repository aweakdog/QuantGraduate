import sys, json
sys.path.insert(0, r"C:\Users\admin\.workbuddy\skills\ths-all-in-one\scripts")
from thsdk import THS
KQ = {"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}
for code in ["600519", "688008", "300750", "000001"]:
    print("="*50, code)
    with THS(KQ) as ths:
        sym = ths.search_symbols(code)
        if not sym.success or not sym.data:
            print("  search fail"); continue
        print("  candidates:", [(d.get("THSCODE"), d.get("MarketStr"), d.get("Name")) for d in sym.data[:5]])
        for d in sym.data:
            ms = d.get("MarketStr","")
            if ms.startswith(("USZA","USHA","UBJA")):
                tc = d.get("THSCODE")
                k = ths.klines(tc, count=3, interval="day", adjust="forward")
                print(f"  try {tc} ({ms}) success={k.success} err={getattr(k,'error','')}")
                if k.success and k.data:
                    print("  KEYS:", list(k.data[0].keys()))
                    print("  sample:", {kk: k.data[0].get(kk) for kk in list(k.data[0].keys())[:4]})
                break
