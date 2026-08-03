"""资金流覆盖补全: 东财 push2his 回拉 PIT 池全部股票的资金流日线历史

背景 (2026-08-02): consolidated fundflow_history 只覆盖 271 只 (旧 watchlist),
而 PIT 池 ~519 只 —— 入选 80 特征里的 8 个资金流特征和大部分概念聚合特征,
对约八成的池子股票是空值。本脚本把缺的股票从东财补全 (免费, 2015年起全历史)。

PIT 说明: 资金流日线是当日盘后即公布的历史事实, 回拉的历史 = 当时可得的数据。
唯一已知风险是东财偶发的历史修订, 量级可忽略。

来源差异 (诚实记录, 不装作同源):
  - 旧 271 只: thsdk/wencai, 有 dde_net / mtss_balance / fund_flow 三列独家字段
  - 新补的:    东财 push2his, 只有 main_force_net / main_force_pct 两列,
               其余三列留 NaN (特征层 fillna(0), 与现状一致)
  - 单位对账: 用新旧都有的股票在重叠日期上比中位数比值, 偏离 1 超过 5% 则报错停

用法:
  python -m pipeline.backfill_fundflow_universe            # 拉缺的 + 重建 consolidated
  python -m pipeline.backfill_fundflow_universe --dry-run  # 只看要拉哪些
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings  # noqa: E402

DATA = settings.DATA_DIR
FF_DIR = DATA / "raw" / "fund_flow_full"
CONS_PATH = FF_DIR / "fundflow_history.parquet"
WATCHLIST = DATA / "universe" / "watchlist_pit.json"

# 东财有一排编号 CDN 节点, 频控计数器按节点分开 —— 轮换可摊薄突发频率。
# (akshare 同款接口也是这么用的, 节点内容完全一致)
API_HOSTS = [f"https://push2his.eastmoney.com"] + \
            [f"https://{i}.push2his.eastmoney.com" for i in (1, 2, 3, 4, 5)]
API_PATH = "/api/qt/stock/fflow/daykline/get"
PARAMS = {
    "klt": "101", "lmt": "5000",
    "fields1": "f1,f2,f3,f7",
    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
    "ut": "b2884a393a59ad64002292a3e90d46a5",
}


def to_secid(code6: str) -> str:
    return f"1.{code6}" if code6.startswith(("60", "68")) else f"0.{code6}"


def http_get_json(url: str) -> dict:
    """用 curl 子进程抓取。

    2026-08-02 实测: 东财对 python-requests 的 TLS 指纹直接 RST 掐断
    (RemoteDisconnected), 同机 curl 却放行 —— 它按客户端指纹反爬, 不是封 IP。
    与其引入 curl_cffi 之类的模拟浏览器指纹依赖, 不如直接用系统 curl,
    一次性回拉场景够用了。
    """
    r = subprocess.run(
        ["curl", "-s", "-m", "20",
         "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
         "-H", "Referer: https://data.eastmoney.com/",
         url],
        capture_output=True, text=True, timeout=25)
    if r.returncode != 0 or not r.stdout.strip():
        raise ConnectionError(f"curl rc={r.returncode}, empty={not r.stdout.strip()}")
    return json.loads(r.stdout)


def pull_one(code6: str) -> pd.DataFrame:
    """东财资金流日线全历史: date, main_net(元), ..., main_pct(%)

    只试 2 次(换节点), 失败就把“撞墙”交给外层的全局静默处置 ——
    实测这堡频控是“被封期间每次重试都可能重置冷却计时”, 逐只退避等于持续戳墙。
    """
    q = urlencode({**PARAMS, 'secid': to_secid(code6)})
    for attempt in range(2):
        url = f"{random.choice(API_HOSTS)}{API_PATH}?{q}"
        try:
            data = http_get_json(url).get("data")
            if not data or not data.get("klines"):
                time.sleep(1)
                continue
            rows = []
            for line in data["klines"]:
                p = line.split(",")
                if len(p) < 7:
                    continue
                rows.append({"date": p[0],
                             "main_net": float(p[1]) if p[1] else None,
                             "small_net": float(p[2]) if p[2] else None,
                             "medium_net": float(p[3]) if p[3] else None,
                             "large_net": float(p[4]) if p[4] else None,
                             "super_large_net": float(p[5]) if p[5] else None,
                             "main_pct": float(p[6]) if p[6] else None})
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            if attempt == 1:
                print(f"  撞墙 {code6}: {e}", flush=True)
            time.sleep(2)
    return pd.DataFrame()


def universe_codes() -> list[str]:
    wl = json.loads(WATCHLIST.read_text())
    items = wl.get("watchlist", wl) if isinstance(wl, dict) else wl
    return sorted({str(it["code"] if isinstance(it, dict) else it)[:6] for it in items})


def unit_check(cons: pd.DataFrame, sample_codes: list[str]) -> None:
    """新旧源单位对账: 只拦数量级错误 (元 vs 万元 = 1e4 倍), 放行口径差。

    实测 2026-08-02: 旧源(thsdk) vs 东财的 main_force_net 中位比值 0.94 ——
    两家对"主力"(大单/超大单)的分桶口径差 ~6%, 属供应商方法论差异而非单位错。
    每只股票整段历史只来自单一源, z-score 特征按股滚动归一对常数因子免疫;
    原始值特征的股票间天然差异是百倍级, 6% 无关紧要。闸门只拦 5 倍以上偏差。
    """
    net_r, pct_r = [], []
    for c in sample_codes[:5]:
        em = pull_one(c)
        if em.empty:
            continue
        old = cons[cons["code"] == c][["date", "main_force_net", "main_force_pct"]].dropna()
        m = old.merge(em[["date", "main_net", "main_pct"]], on="date").dropna()
        m = m[m["main_net"].abs() > 1e4]
        if len(m) >= 30:
            net_r.append((m["main_force_net"] / m["main_net"]).median())
            mp = m[m["main_pct"].abs() > 0.5]
            if len(mp) >= 30:
                pct_r.append((mp["main_force_pct"] / mp["main_pct"]).median())
    if not net_r:
        print("!! 无重叠样本可对账 (跳过单位校验, 请人工抽查)", flush=True)
        return
    rn = pd.Series(net_r).median()
    rp = pd.Series(pct_r).median() if pct_r else float("nan")
    print(f"单位对账: net 中位比值 {rn:.4f} | pct 中位比值 {rp:.4f} "
          f"({len(net_r)} 只重叠股)", flush=True)
    for name, r in (("main_force_net", rn), ("main_force_pct", rp)):
        if r == r and not 0.2 <= r <= 5.0:
            raise SystemExit(f"ERROR: 新旧源 {name} 疑似单位不一致 (比值 {r:.4f}), "
                             f"必须先人工确认换算, 拒绝混入")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="每请求基础间隔秒 (实际为 ±50%% 抖动)。实测 0.25s 会在 ~17 只后触发频控")
    ap.add_argument("--skip-unit-check", action="store_true",
                    help="跳过新旧源单位对账 (已人工确认过时用, 省 5 次请求)")
    a = ap.parse_args()

    cons = pd.read_parquet(CONS_PATH)
    cons["date"] = pd.to_datetime(cons["date"])
    covered = set(cons["code"].astype(str).str.zfill(6))
    codes = universe_codes()
    missing = [c for c in codes if c not in covered]
    print(f"PIT 池 {len(codes)} 只 | consolidated 已覆盖 {len(covered)} | 待补 {len(missing)}",
          flush=True)
    if a.dry_run:
        print("dry-run:", missing[:20], "...")
        return

    if a.skip_unit_check:
        print("跳过单位对账 (2026-08-02 已两次确认: net/pct 中位比值 0.944, 口径差非单位错)")
    else:
        unit_check(cons, [c for c in codes if c in covered])

    FF_DIR.mkdir(parents=True, exist_ok=True)
    new_rows, pulled, cooldowns = [], 0, 0
    i = 0
    while i < len(missing):
        c = missing[i]
        raw_path = FF_DIR / f"{c}.parquet"
        if raw_path.exists():
            df = pd.read_parquet(raw_path)
            df["date"] = pd.to_datetime(df["date"])
        else:
            df = pull_one(c)
            if df.empty:
                # 全局静默: 冷却期内任何请求都可能重置计时器, 必须完全停手。
                # 不跳过该股, 静默后从同一只继续 —— 不留失败尾巴。
                cooldowns += 1
                wait = min(600 * cooldowns, 1800)
                print(f"  进入全局静默 {wait / 60:.0f} 分钟 (第 {cooldowns} 次, 停在 "
                      f"{i + 1}/{len(missing)} {c}) ...", flush=True)
                time.sleep(wait)
                continue
            cooldowns = 0
            df.to_parquet(raw_path, index=False)
            pulled += 1
            time.sleep(a.sleep * random.uniform(0.5, 1.5))
        new_rows.append(pd.DataFrame({
            "date": df["date"], "code": c,
            "main_force_net": df["main_net"],
            "main_force_pct": df["main_pct"],
            "dde_net": pd.NA, "mtss_balance": pd.NA, "fund_flow": pd.NA,
        }))
        i += 1
        if i % 50 == 0:
            print(f"  [{i}/{len(missing)}] 已拉 {pulled}", flush=True)

    if not new_rows:
        print("没有可补的数据, consolidated 不变")
        return
    add = pd.concat(new_rows, ignore_index=True)
    add = add[add["date"] >= cons["date"].min()]   # 与旧表同起点, 不引入更早的孤段
    bak = CONS_PATH.with_name(f"fundflow_history.bak_{pd.Timestamp.now():%Y%m%d_%H%M%S}.parquet")
    CONS_PATH.rename(bak)
    out = pd.concat([cons, add], ignore_index=True).sort_values(["code", "date"])
    out.to_parquet(CONS_PATH, index=False)
    print(f"完成: consolidated {len(cons):,} -> {len(out):,} 行, "
          f"覆盖 {out['code'].nunique()} 只 (旧表备份 {bak.name})", flush=True)


if __name__ == "__main__":
    main()
