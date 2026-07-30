"""验证 /backtest 隐藏页的密码认证与数据接口

用法: python scripts/verify_backtest_page.py <base_url> <password>
"""
import http.cookiejar as cj
import json
import sys
import urllib.error as ue
import urllib.parse as up
import urllib.request as ur

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
PW = sys.argv[2] if len(sys.argv) > 2 else ""

jar = cj.CookieJar()
logged = ur.build_opener(ur.HTTPCookieProcessor(jar))
anon = ur.build_opener()


def hit(path, data=None, opener=None, headers=None):
    body = json.dumps(data).encode() if data is not None else None
    hdr = dict(headers or {})
    if body:
        hdr["Content-Type"] = "application/json"
    req = ur.Request(BASE + path, data=body, headers=hdr)
    try:
        r = (opener or anon).open(req, timeout=20)
        return r.status, r.read()
    except ue.HTTPError as ex:
        return ex.code, ex.read()


def check(label, got, want):
    ok = "OK " if got == want else "FAIL"
    print(f"  [{ok}] {label:34s} 期望 {want} 实际 {got}")
    return got == want


print("=== 认证 ===")
allok = True
allok &= check("/backtest 页面可达", hit("/backtest")[0], 200)
allok &= check("未登录取列表应拒绝", hit("/api/bt/list")[0], 401)
allok &= check("错误密码应拒绝", hit("/api/bt/login", {"password": "wrong-pw"})[0], 403)
allok &= check("伪造 cookie 应拒绝",
               hit("/api/bt/list", headers={"Cookie": "bt_token=9999999999.deadbeef"})[0], 401)
allok &= check("过期 cookie 应拒绝",
               hit("/api/bt/list", headers={"Cookie": "bt_token=1.abc"})[0], 401)
allok &= check("正确密码登录", hit("/api/bt/login", {"password": PW}, logged)[0], 200)

st, body = hit("/api/bt/list", opener=logged)
allok &= check("登录后取列表", st, 200)
runs = json.loads(body)

print(f"\n=== 可见回测 {len(runs)} 个 ===")
for r in runs:
    print(f"  {r['name'][:50]:52s} 收益{r['total_return_pct']:>7}%  "
          f"夏普{r['sharpe']:>5}  IR{r['information_ratio']:>6}  "
          f"两段都赢={r['beat_both_halves']}")

if runs:
    name = runs[0]["name"]
    q = lambda v: "?name=" + up.quote(v, safe="")
    st, body = hit("/api/bt/detail" + q(name), opener=logged)
    allok &= check("详情接口", st, 200)
    d = json.loads(body)
    print(f"\n=== 详情 {name} ===")
    print(f"  净值曲线 {len(d['curve'])} 点 | 分年度 {len(d['yearly'])} 条 | "
          f"入选特征 {len(d['selected_features'])} 个 | 交易 {d['n_trades_total']} 笔")
    print(f"  分段稳健性 {len(d['stability'])} 条 | 配置字段 {len(d['config'])} 个")

    allok &= check("目录穿越应被挡",
                   hit("/api/bt/detail" + q("../../etc/passwd"), opener=logged)[0], 404)

    st, body = hit("/api/bt/xlsx" + q(name), opener=logged)
    ok_xlsx = st == 200 and body[:2] == b"PK"
    print(f"  [{'OK ' if ok_xlsx else 'FAIL'}] Excel 下载{'':24s} "
          f"HTTP {st}, {len(body):,} 字节, zip头={body[:2]!r}")
    allok &= ok_xlsx
    allok &= check("未登录下载应拒绝", hit("/api/bt/xlsx" + q(name))[0], 401)

print("\n结论:", "全部通过" if allok else "有失败项")
sys.exit(0 if allok else 1)
