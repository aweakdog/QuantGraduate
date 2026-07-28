"""逐个东财主机探测, 判断封禁是主机级还是账号/IP 全局级"""
import socket
import ssl
import time
import urllib.request

HOSTS = [
    ("push2his.eastmoney.com", "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=0.000063&fields1=f1&fields2=f51&klt=101&fqt=1&beg=20260701&end=20260727", "日K线/资金流"),
    ("push2.eastmoney.com", "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5&fs=m:0+t:6&fields=f12,f14", "实时快照"),
    ("datacenter-web.eastmoney.com", "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_CUSTOM_CHINA_BOND_YIELD&columns=ALL&pageSize=5", "宏观数据中心"),
    ("np-anotice-stock.eastmoney.com", "https://np-anotice-stock.eastmoney.com/api/security/ann?page_size=5&page_index=1&ann_type=A", "公告"),
    ("search-api-web.eastmoney.com", "https://search-api-web.eastmoney.com/search/jsonp?cb=jQuery&param=%7B%22uid%22%3A%22%22%7D", "新闻搜索"),
]

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

print(f"{'主机':<34}{'用途':<12}{'DNS':<16}  HTTP")
print("-" * 84)
for host, url, use in HOSTS:
    try:
        ip = socket.gethostbyname(host)
    except Exception as e:
        print(f"{host:<34}{use:<12}{'DNS失败':<16}  {type(e).__name__}")
        continue
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=12, context=CTX) as r:
            body = r.read(200)
        print(f"{host:<34}{use:<12}{ip:<16}  OK {r.status} {len(body)}B {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"{host:<34}{use:<12}{ip:<16}  FAIL {type(e).__name__} {str(e)[:40]}")
    time.sleep(1)
