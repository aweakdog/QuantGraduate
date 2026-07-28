"""直连东财公告 API (绕过 akshare 的全市场翻页), 按股票拉取"""
import json
import ssl
import time
import urllib.parse
import urllib.request

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "Referer": "https://data.eastmoney.com/"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
BASE = "https://np-anotice-stock.eastmoney.com/api/security/ann"


def fetch(code6, page=1, size=50):
    q = {"page_size": size, "page_index": page, "ann_type": "A",
         "client_source": "web", "stock_list": code6,
         "f_node": 0, "s_node": 0}
    url = BASE + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20, context=CTX) as r:
        return json.loads(r.read().decode("utf-8"))


for code in ["000063", "600519"]:
    t0 = time.time()
    try:
        j = fetch(code)
    except Exception as e:
        print(f"{code}: FAIL {type(e).__name__} {str(e)[:80]}")
        continue
    data = (j or {}).get("data") or {}
    lst = data.get("list") or []
    print(f"\n=== {code} === {time.time()-t0:.1f}s  本页 {len(lst)} 条  "
          f"总页 {data.get('total_hits', '?')}")
    rows = []
    for it in lst[:5]:
        cols = it.get("columns") or []
        rows.append({
            "date": (it.get("notice_date") or "")[:10],
            "title": it.get("title", ""),
            "type": cols[0].get("column_name") if cols else "",
            "art_code": it.get("art_code", ""),
        })
    print(pd.DataFrame(rows).to_string(index=False)[:700])
    time.sleep(1)
