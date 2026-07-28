# backfill_recent_kline.py  v4.1 (retry版)
# 仅补 universe 216 只日K到最新交易日(目标 2026-07-09)
# 正确解析 watchlist_216.json (元素是 {"code":"603256.SH",...})
# 失败 code 循环重试 + 长冷却, 绕开 thsdk 当日限流
import sys, time, os, json
import pandas as pd

sys.path.insert(0, r"C:\Users\admin\.workbuddy\skills\ths-all-in-one\scripts")
from thsdk import THS

KQ = {"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}
KL = r"D:\myAI\WorkBuddy-workspace\quant-strategy\data\raw\kline"
WL = r"D:\myAI\WorkBuddy-workspace\quant-strategy\data\universe\watchlist_216.json"
TARGET = pd.Timestamp("2026-07-08")
NEED = ["时间", "收盘价", "成交量", "总金额", "开盘价", "最高价", "最低价"]


def code_to_ths(c):
    return ("USHA." + c) if c.startswith("6") else ("USZA." + c)


def fetch(c):
    try:
        with THS(KQ) as ths:
            k = ths.klines(code_to_ths(c), count=30, interval="day", adjust="forward")
            if not k.success or not k.data:
                return None
            rows = [r for r in k.data if isinstance(r, dict) and "时间" in r]
            if not rows:
                return None
            df = pd.DataFrame(rows)
            for col in NEED:
                if col not in df.columns:
                    df[col] = None
            df = df[NEED].copy()
            df["时间"] = pd.to_datetime(df["时间"], errors="coerce")
            df = df.dropna(subset=["时间"])
            return df if not df.empty else None
    except Exception:
        return None


def merge_write(c, df):
    p = os.path.join(KL, c + ".parquet")
    if os.path.exists(p):
        old = pd.read_parquet(p)
        old["时间"] = pd.to_datetime(
            old["时间"] if "时间" in old.columns else old.iloc[:, 0], errors="coerce"
        )
        old = old.dropna(subset=["时间"])
        merged = pd.concat([old, df], ignore_index=True)
    else:
        merged = df
    merged = merged.drop_duplicates(subset=["时间"], keep="last").sort_values("时间")
    merged.to_parquet(p, index=False)


def main():
    wl = json.load(open(WL))
    items = wl["watchlist"] if isinstance(wl, dict) else wl
    codes6, seen = [], set()
    for it in items:
        c = str(it["code"]).split(".")[0]
        if c not in seen:
            seen.add(c)
            codes6.append(c)

    pending = []
    for c in codes6:
        p = os.path.join(KL, c + ".parquet")
        if not os.path.exists(p):
            pending.append(c)
            continue
        d = pd.read_parquet(p)
        t = d["时间"] if "时间" in d.columns else d.iloc[:, 0]
        mx = pd.to_datetime(t, errors="coerce").max()
        if pd.isna(mx) or mx < TARGET:
            pending.append(c)

    print(f"[v4.1] universe={len(codes6)} pending={len(pending)}", flush=True)
    failed = list(pending)
    round_n = 0
    while failed and round_n < 12:
        round_n += 1
        success_this = 0
        next_failed = []
        consec = 0
        print(f"[round {round_n}] retry {len(failed)}", flush=True)
        untested = list(failed)
        for c in failed:
            untested.remove(c)
            df = fetch(c)
            if df is None:
                next_failed.append(c)
                consec += 1
                if consec >= 3:
                    next_failed.extend(untested)  # 限流, 携带未试代码进下一轮
                    print(f"  [throttled] consec={consec}, sleep 240s, carry {len(untested)}", flush=True)
                    time.sleep(240)
                    break
            else:
                merge_write(c, df)
                success_this += 1
                consec = 0
        failed = next_failed
        print(f"[round {round_n}] success_this={success_this} remaining={len(failed)}", flush=True)
        if not failed:
            break
        if success_this == 0:
            print(f"  [no-progress] sleep 300s (likely throttled)", flush=True)
            time.sleep(300)
        else:
            time.sleep(60)
    print(f"[v4.1] FINAL remaining={len(failed)}", flush=True)


if __name__ == "__main__":
    main()
