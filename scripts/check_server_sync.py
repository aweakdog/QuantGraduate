"""核对本地与服务器的关键脚本是否一致

为什么需要这个
──────────────
2026-08-04 踩过一次: 修好了 wf_v35 里 --ic-timing 的未来函数, 提交了、也写了
测试, 但【忘了 scp 到服务器】。于是服务器仍跑着带未来函数的旧代码, 跑出
"窗口A 收益中位 +50%、夏普 0.95、20个种子无一亏损"这种好得不合理的结果 ——
而窗口A 恰恰是信号多数时间为负的那段。差点把它当成真结论。

回测跑在服务器上、代码改在本地, 这个缝隙会一直存在。所以在【解读任何回测
结果之前】先跑一遍本脚本, 比事后靠"结果好得可疑"去发现要可靠得多。

用法
────
    python scripts/check_server_sync.py            # 只核对
    python scripts/check_server_sync.py --push     # 不一致的直接推上去
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "yliog@eez041.ece.ust.hk"
REMOTE_ROOT = "/home/yliog/quant-strategy"

# 影响回测结论或线上行为的文件。新增同类文件请加进来。
WATCHED = [
    "scripts/wf_v35_breadth_alpha.py",   # 回测引擎, 结论全靠它
    "scripts/eval_grid.py",              # 评估框架
    "scripts/live_signal.py",            # 线上出信号
    "scripts/live_config.py",            # 线上参数唯一来源
    "scripts/migrate_config.py",         # 换参数不清账
    "scripts/action_page.py",
    "scripts/web_server.py",
    "scripts/daily_rebuild.py",
    "scripts/build_pit_universe.py",
    "scripts/update_kline_akshare.py",
    "scripts/ensemble_pred_caches.py",
    "pipeline/feature_engine.py",
]


def local_hash(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def remote_hashes(rels):
    """一次 ssh 取回全部哈希, 避免每个文件一次连接"""
    files = " ".join(f"'{r}'" for r in rels)
    cmd = ["ssh", "-p", "22", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST,
           f"cd {REMOTE_ROOT} && sha256sum {files} 2>/dev/null"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip()] = parts[0].strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", action="store_true", help="把不一致的文件推到服务器")
    args = ap.parse_args()

    rem = remote_hashes(WATCHED)
    bad = []
    for rel in WATCHED:
        lh, rh = local_hash(rel), rem.get(rel)
        if lh is None:
            print(f"?  本地缺失      {rel}")
            continue
        if rh is None:
            print(f"✗  服务器缺失    {rel}")
            bad.append(rel)
        elif lh != rh:
            print(f"✗  不一致        {rel}  本地={lh[:12]} 服务器={rh[:12]}")
            bad.append(rel)
        else:
            print(f"✓  一致          {rel}")

    if not bad:
        print("\n全部一致。回测结果可以放心解读。")
        return

    print(f"\n{len(bad)} 个文件不一致。"
          f"{'正在推送...' if args.push else '服务器上跑出来的结果可能不对应你以为的代码。'}")
    if not args.push:
        print("加 --push 直接同步, 或手动 scp。")
        sys.exit(1)

    for rel in bad:
        dst = f"{HOST}:{REMOTE_ROOT}/{rel}"
        r = subprocess.run(["scp", "-P", "22", str(ROOT / rel), dst],
                           capture_output=True, text=True)
        print(f"{'✓ 已推送' if r.returncode == 0 else '✗ 推送失败'}  {rel}"
              + ("" if r.returncode == 0 else f"\n   {r.stderr.strip()}"))
    # 推完再验一遍, 不能只信 scp 的返回码
    rem2 = remote_hashes(bad)
    still = [r for r in bad if local_hash(r) != rem2.get(r)]
    if still:
        print(f"\n仍不一致: {still}")
        sys.exit(1)
    print("\n推送后复验通过, 全部一致。")


if __name__ == "__main__":
    main()
