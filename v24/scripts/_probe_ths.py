import sys, time
sys.path.insert(0, r"C:\Users\admin\.workbuddy\skills\ths-all-in-one\scripts")
try:
    from thsdk import THS
except Exception as e:
    print("IMPORT_FAIL", e); sys.exit(1)
KQ = {"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}
try:
    with THS(KQ) as ths:
        sym = ths.search_symbols("600519")
        cand = [d for d in sym.data if d.get("MarketStr", "").startswith(("USZA", "USHA", "UBJA"))] if sym.success else []
        thscode = cand[0].get("THSCODE") if cand else (sym.data[0].get("THSCODE") if sym.success and sym.data else None)
        print("THSCODE:", thscode)
        k = ths.klines(thscode, count=12, interval="1d")
        if k.success and k.data:
            for row in k.data[-8:]:
                print(row.get("时间"), "close=", row.get("收盘价"))
        else:
            print("KLINE_FAIL", getattr(k, "message", ""))
except Exception as e:
    print("CONN_FAIL", repr(e))
