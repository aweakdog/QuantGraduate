"""探测退市股票名单可得性, 量化幸存者偏差规模"""
import signal
import time
import warnings

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd


class T(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(T()))


def go(name, fn, sec=40):
    print(f"\n[{name}]")
    signal.alarm(sec)
    try:
        d = fn()
        signal.alarm(0)
    except T:
        signal.alarm(0)
        print("    超时")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"    FAIL {type(e).__name__}: {str(e)[:70]}")
        return None
    if d is None or not len(d):
        print("    空")
        return None
    print(f"    {len(d)} 行 | 列: {list(d.columns)[:8]}")
    print(d.head(3).to_string(index=False)[:400])
    return d


res = {}
res["sh_delist"] = go("上交所终止上市 stock_info_sh_delist",
                      lambda: ak.stock_info_sh_delist())
time.sleep(0.6)
res["sz_delist"] = go("深交所终止上市 stock_info_sz_delist",
                      lambda: ak.stock_info_sz_delist())
time.sleep(0.6)
res["sh_name"] = go("上交所股票列表(含终止) stock_info_sh_name_code",
                    lambda: ak.stock_info_sh_name_code(symbol="主板A股"))
time.sleep(0.6)
res["st"] = go("风险警示板 stock_zh_a_st_em", lambda: ak.stock_zh_a_st_em())

print("\n" + "=" * 60)
print("=== 幸存者偏差规模估算 ===")
tot = 0
for k in ("sh_delist", "sz_delist"):
    if res.get(k) is not None:
        tot += len(res[k])
        print(f"  {k}: {len(res[k])} 只")
if tot:
    print(f"\n  退市股合计 {tot} 只")
    print(f"  本地 kline 覆盖 5524 只(全部在市)")
    print(f"  缺失比例 ≈ {100*tot/(5524+tot):.1f}% 的历史标的")
else:
    print("  退市名单未取到 — 需换源或人工维护")
