# -*- coding: utf-8 -*-
"""探查沪深两所逐笔数据的编码差异与历史可用性

第一轮探查 (tick_probe.py) 发现沪深编码完全不同, 不能用同一套解析:
    SH  委托类型 = A(新增)/D(撤单)          撤单在【委托】文件里
    SZ  委托类型 = 0/1/U (订单类别, 非增撤)  撤单疑似在【成交】文件里(BS标志为空)
    SZ  叫买序号 对 交易所委托号 命中率 0    ID 空间不同, 要试 委托编号
搞错任何一条, 算出来的净委托流和撤单率都是错的。本脚本把两所的真实编码定下来,
并扫描逐年可用性 —— 上交所早年不发逐笔委托, 决定了委托类特征的历史起点。

用法
────
    python scripts/tick_probe2.py decode 20260807      # 解码两所字段
    python scripts/tick_probe2.py years                # 扫描逐年可用性
"""
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import py7zr

warnings.filterwarnings("ignore")

TICK = Path("/home/yliog/tickdata/----逐笔委托成交行情-明细---")
FILES = ["逐笔委托.csv", "逐笔成交.csv"]


def resolve(day: str):
    y, m = day[:4], day[4:6]
    for sub in (m, y + m):
        p = TICK / y / sub / f"{day}.7z"
        if p.exists():
            return p
    return None


def grab(day: str, codes, files=FILES):
    """把指定股票的指定文件抽到临时目录, 返回 {code: {file: DataFrame}}"""
    pack = resolve(day)
    if pack is None:
        return None, f"{day} 无包"
    tmp = tempfile.mkdtemp(dir="/tmp")
    out = {}
    try:
        with py7zr.SevenZipFile(pack, mode="r") as z:
            names = set(z.getnames())
        want, missing = [], []
        for c in codes:
            for f in files:
                k = f"{day}/{c}/{f}"
                (want if k in names else missing).append(k)
        if not want:
            return None, f"{day} 无匹配 (缺 {len(missing)})"
        with py7zr.SevenZipFile(pack, mode="r") as z:
            z.extract(path=tmp, targets=want)
        for c in codes:
            d = {}
            for f in files:
                fp = Path(tmp) / day / c / f
                if fp.exists():
                    try:
                        d[f] = pd.read_csv(fp, encoding="gbk", on_bad_lines="skip")
                    except Exception as e:
                        d[f] = f"读失败 {e}"
                else:
                    d[f] = None
            out[c] = d
        return out, f"缺 {len(missing)}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def decode(day: str):
    data, note = grab(day, ["600519.SH", "000001.SZ"])
    if data is None:
        sys.exit(note)
    print(f"=== {day} ({note}) ===")
    for code, d in data.items():
        print(f"\n######## {code}")
        o, t = d.get("逐笔委托.csv"), d.get("逐笔成交.csv")
        for nm, df in [("委托", o), ("成交", t)]:
            if df is None:
                print(f"  {nm}: 文件不存在")
            elif isinstance(df, str):
                print(f"  {nm}: {df}")
            else:
                print(f"  {nm}: {len(df)} 行")
        if not isinstance(o, pd.DataFrame) or not isinstance(t, pd.DataFrame):
            continue

        print("  --- 委托文件 ---")
        for c in ["委托类型", "委托代码"]:
            if c in o:
                print(f"    {c}: {dict(o[c].astype(str).value_counts().head(6))}")
        for c in ["委托编号", "交易所委托号"]:
            if c in o:
                v = pd.to_numeric(o[c], errors="coerce")
                print(f"    {c}: >0占比 {(v > 0).mean():.3f} "
                      f"唯一值 {v.nunique()} 范围 [{v.min():.0f},{v.max():.0f}]")

        print("  --- 成交文件 ---")
        for c in ["BS标志", "成交代码", "委托代码"]:
            if c in t:
                s = t[c].astype(str).replace({"nan": "<NaN>", " ": "<空格>", "": "<空串>"})
                print(f"    {c}: {dict(s.value_counts().head(6))}")
        # 撤单在成交文件里的证据: 这些记录的成交价格/数量是什么
        if "BS标志" in t:
            s = t["BS标志"].astype(str)
            odd = t[~s.isin(["B", "S"])]
            if len(odd):
                px = pd.to_numeric(odd["成交价格"], errors="coerce")
                qty = pd.to_numeric(odd["成交数量"], errors="coerce")
                print(f"    非B/S记录 {len(odd)} 条: 价格>0占比 {(px > 0).mean():.3f}, "
                      f"数量>0占比 {(qty > 0).mean():.3f}, 数量中位 {qty.median():.0f}")
                if "成交代码" in odd:
                    print(f"      其 成交代码: "
                          f"{dict(odd['成交代码'].astype(str).value_counts().head(4))}")

        # 成交回连委托: 两种 ID 都试
        print("  --- 成交回连委托 (决定能否做委托生命周期特征) ---")
        for oid in ["委托编号", "交易所委托号"]:
            if oid not in o:
                continue
            ids = set(pd.to_numeric(o[oid], errors="coerce").dropna().astype("int64"))
            for tid in ["叫买序号", "叫卖序号"]:
                if tid not in t:
                    continue
                v = pd.to_numeric(t[tid], errors="coerce").dropna().astype("int64")
                print(f"    {tid} -> {oid}: 命中 {v.isin(ids).mean():.3f}")


def years():
    print("扫描逐年可用性 (每年取一天, 看两所的委托/成交文件是否存在且有内容)")
    print(f"{'日期':<10} {'SH委托':>8} {'SH成交':>8} {'SZ委托':>8} {'SZ成交':>8}")
    for y in range(2017, 2027):
        day = None
        for mmdd in ["0601", "0602", "0603", "0604", "0605", "0606", "0607",
                     "0110", "0111", "0112"]:
            if resolve(f"{y}{mmdd}"):
                day = f"{y}{mmdd}"
                break
        if day is None:
            print(f"{y}       无包")
            continue
        data, _ = grab(day, ["600519.SH", "000001.SZ"])
        row = [day]
        for code in ["600519.SH", "000001.SZ"]:
            d = (data or {}).get(code, {})
            for f in FILES:
                df = d.get(f)
                row.append(str(len(df)) if isinstance(df, pd.DataFrame) else "缺")
        print(f"{row[0]:<10} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8}")


def szcancel(day: str):
    """深交所撤单记录不带 BS标志, 方向只能靠 叫买序号/叫卖序号 哪个非零来判

    如果这个猜测错了, 买方撤单率和卖方撤单率就会整个对调 —— 必须验证。
    顺带确认两所的价格单位是否都是 1/10000 元。
    """
    data, _ = grab(day, ["000001.SZ", "600519.SH"])
    if data is None:
        sys.exit("无数据")
    t = data["000001.SZ"]["逐笔成交.csv"]
    o = data["000001.SZ"]["逐笔委托.csv"]
    cx = t[t["成交代码"].astype(str) == "C"]
    b = pd.to_numeric(cx["叫买序号"], errors="coerce").fillna(0)
    s = pd.to_numeric(cx["叫卖序号"], errors="coerce").fillna(0)
    print(f"=== SZ 撤单记录 {len(cx)} 条 ===")
    print(f"  只买号非零 {((b > 0) & (s == 0)).mean():.3f}"
          f"  只卖号非零 {((s > 0) & (b == 0)).mean():.3f}"
          f"  两者都非零 {((b > 0) & (s > 0)).mean():.3f}"
          f"  都为零 {((b == 0) & (s == 0)).mean():.3f}")

    # 用委托文件的方向做交叉验证: 撤单指向的委托, 其 委托代码 应与推断方向一致
    dirn = o.set_index(pd.to_numeric(o["交易所委托号"], errors="coerce"))["委托代码"]
    dirn = dirn[~dirn.index.duplicated()]
    for nm, ser in [("买号", b), ("卖号", s)]:
        hit = ser[ser > 0].astype("int64").map(dirn).dropna()
        if len(hit):
            print(f"  由{nm}回连到的委托, 其委托代码分布: "
                  f"{dict(hit.astype(str).value_counts().head(3))}  (n={len(hit)})")

    print("\n=== 价格单位 ===")
    for code in ["000001.SZ", "600519.SH"]:
        oo = data[code]["逐笔委托.csv"]
        tt = data[code]["逐笔成交.csv"]
        op = pd.to_numeric(oo["委托价格"], errors="coerce")
        tp = pd.to_numeric(tt["成交价格"], errors="coerce")
        tp = tp[tp > 0]
        print(f"  {code}: 委托价中位 {op[op > 0].median():.0f} "
              f"成交价中位 {tp.median():.0f} -> /1e4 = {tp.median() / 1e4:.2f} 元")


def cost(day: str, n: int = 100):
    """量批量抽取成本 —— 决定全量可行性

    7z 是固实压缩, 抽 N 个小文件的代价不是 N x 单个, 必须实测。
    """
    import time

    root = Path(__file__).resolve().parents[1]
    u = pd.read_parquet(root / "data/universe/universe_pit.parquet")
    c6 = sorted(u["code"].astype(str).str.zfill(6).unique())
    codes = [f"{c}.{'SH' if c[0] == '6' else 'SZ'}" for c in c6][:n]

    pack = resolve(day)
    with py7zr.SevenZipFile(pack, mode="r") as z:
        t0 = time.time()
        names = set(z.getnames())
        t_list = time.time() - t0
    want = [f"{day}/{c}/{f}" for c in codes for f in FILES + ["行情.csv"]]
    want = [w for w in want if w in names]

    tmp = tempfile.mkdtemp(dir="/tmp")
    try:
        t0 = time.time()
        with py7zr.SevenZipFile(pack, mode="r") as z:
            z.extract(path=tmp, targets=want)
        t_ex = time.time() - t0
        sz = sum(f.stat().st_size for f in Path(tmp).rglob("*.csv"))
        t0 = time.time()
        rows = 0
        for f in Path(tmp).rglob("*.csv"):
            rows += len(pd.read_csv(f, encoding="gbk", on_bad_lines="skip"))
        t_rd = time.time() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"包 {pack.name} {pack.stat().st_size / 1e9:.2f}GB  getnames {t_list:.1f}s")
    print(f"抽 {len(codes)} 只 x 3 文件 = {len(want)} 个文件")
    print(f"  解压 {t_ex:.1f}s  体积 {sz / 1e6:.0f}MB")
    print(f"  读盘+解析 {t_rd:.1f}s  {rows} 行")
    per600 = (t_list + t_ex + t_rd) / len(codes) * 600
    print(f"推算 600 只/天 ≈ {per600:.0f}s = {per600 / 60:.1f}min")
    for nd, nm in [(970, "2022-09~2026-08"), (2331, "全量 2017 起")]:
        for nw in (8, 16, 24):
            print(f"  {nm} {nd}天 x {nw}并发 ≈ {per600 * nd / nw / 3600:.1f}h")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "decode"
    if mode == "years":
        years()
    elif mode == "cost":
        cost(sys.argv[2] if len(sys.argv) > 2 else "20260807",
             int(sys.argv[3]) if len(sys.argv) > 3 else 100)
    elif mode == "szcancel":
        szcancel(sys.argv[2] if len(sys.argv) > 2 else "20260807")
    else:
        decode(sys.argv[2] if len(sys.argv) > 2 else "20260807")
