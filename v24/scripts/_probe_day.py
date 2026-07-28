import sys, time
sys.path.insert(0, r"C:\Users\admin\.workbuddy\skills\ths-all-in-one\scripts")
from thsdk import THS

KQ = {"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}
with THS(KQ) as ths:
    sym = ths.search_symbols("000001")
    if not sym.success or not sym.data:
        print("SEARCH_FAIL", sym.message if hasattr(sym,'message') else '')
        sys.exit(1)
    cand = [d for d in sym.data if d.get("MarketStr","").startswith(("USZA","USHA","UBJA"))]
    thscode = (cand[0] if cand else sym.data[0]).get("THSCODE")
    print("THSCODE:", thscode)
    k = ths.klines(thscode, count=5, interval="day", adjust="forward")
    if k.success and k.data:
        print("N rows:", len(k.data))
        r0 = k.data[0]
        print("KEYS:", list(r0.keys()))
        for row in k.data:
            print(repr(row.get("时间")), row.get("收盘价"), row.get("成交量"))
    else:
        print("KLINE_FAIL", getattr(k, "message", getattr(k, "error", "NA")))
