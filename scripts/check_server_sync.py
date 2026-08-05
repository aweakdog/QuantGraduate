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
# 三台机器都要核对: 分布式跑任务时 eez040/042 上的脚本版本落后, 会以"参数
# 不认识"之类的方式秒退, 而编排脚本很容易把秒退误判成跑完。实测踩过这个坑。
HOSTS = ["eez041", "eez040", "eez042"]        # eez041 是主节点, 放第一个
REMOTE_ROOT = "/home/yliog/quant-strategy"

# eez040/042 只装了轻量依赖(pandas/pyarrow/lightgbm), 不跑实盘, 所以只需要
# 分布式任务真正用到的文件保持一致; 全套核对留给主节点。
WORKER_FILES = [
    "scripts/pull_fundflow_shard.py",
    "pipeline/pull_fundflow_sina.py",
    "scripts/wf_v35_breadth_alpha.py",
    "scripts/eval_grid.py",
    "scripts/dist_caches.py",        # 分布式建缓存, 主节点调度但工人也要有
    "pipeline/feature_engine.py",
    "pipeline/config.py",
]

# 影响回测结论或线上行为的文件。新增同类文件请加进来。
MAIN_FILES = [
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
    "scripts/expand_2015_overnight.py",  # 过夜编排, 由它调度另两台
    "pipeline/feature_engine.py",
]

# 主节点必须【包含】全部工作节点文件, 否则会出现旧版本反向传播:
# 实测踩过 —— pull_fundflow_shard.py 只在 WORKER_FILES 里, 主节点那份从未被
# 核对过一直是旧的; --push 把新版推给了 eez040/042, 但编排脚本随后从主节点
# rsync 分发, 又把旧版覆盖回两台, 于是三台一起退回旧版而工具报告"全部一致"。
# 所以这里用并集自动兜住, 不靠人记得两个清单都要加。
WATCHED = MAIN_FILES + [f for f in WORKER_FILES if f not in MAIN_FILES]


def local_hash(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def remote_hashes(rels, host):
    """一次 ssh 取回全部哈希, 避免每个文件一次连接"""
    files = " ".join(f"'{r}'" for r in rels)
    cmd = ["ssh", "-p", "22", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
           f"yliog@{host}.ece.ust.hk",
           f"cd {REMOTE_ROOT} && sha256sum {files} 2>/dev/null"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip()] = parts[0].strip()
    return out


def check_host(host, rels, push):
    """核对单台机器, 返回仍不一致的文件列表"""
    rem = remote_hashes(rels, host)
    bad = []
    for rel in rels:
        lh, rh = local_hash(rel), rem.get(rel)
        if lh is None:
            print(f"  ?  本地缺失   {rel}")
            continue
        if rh is None:
            print(f"  ✗  远端缺失   {rel}")
            bad.append(rel)
        elif lh != rh:
            print(f"  ✗  不一致     {rel}  本地={lh[:12]} 远端={rh[:12]}")
            bad.append(rel)
    if not bad:
        print(f"  ✓  {len(rels)} 个文件全部一致")
        return []
    if not push:
        return bad
    for rel in bad:
        dst = f"yliog@{host}.ece.ust.hk:{REMOTE_ROOT}/{rel}"
        r = subprocess.run(["scp", "-P", "22", "-o", "BatchMode=yes",
                            str(ROOT / rel), dst], capture_output=True, text=True)
        print(f"  {'✓ 已推送' if r.returncode == 0 else '✗ 推送失败'}  {rel}"
              + ("" if r.returncode == 0 else f"\n     {r.stderr.strip()}"))
    # 推完复验, 不能只信 scp 的返回码
    rem2 = remote_hashes(bad, host)
    return [r for r in bad if local_hash(r) != rem2.get(r)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", action="store_true", help="把不一致的文件推到远端")
    ap.add_argument("--hosts", default=None,
                    help="只核对指定机器(逗号分隔), 默认三台都查")
    args = ap.parse_args()

    hosts = args.hosts.split(",") if args.hosts else HOSTS
    still = {}
    for h in hosts:
        # 主节点跑实盘, 全套文件都要一致; 计算节点只需分布式任务用到的那几个
        rels = WATCHED if h == HOSTS[0] else WORKER_FILES
        print(f"═══ {h} ({'主节点, 全套' if h == HOSTS[0] else '计算节点, 任务文件'}) ═══")
        left = check_host(h, rels, args.push)
        if left:
            still[h] = left

    if not still:
        print("\n三台全部一致。回测结果可以放心解读。")
        return
    print("\n仍不一致:")
    for h, v in still.items():
        print(f"  {h}: {v}")
    if not args.push:
        print("加 --push 直接同步。")
    sys.exit(1)


if __name__ == "__main__":
    main()
