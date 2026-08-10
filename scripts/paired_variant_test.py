"""两个变体做【配对】显著性检验。

为什么必须配对
──────────────
比较 mb_dmw 与 mb_dmw_rq 时, 两边用的是同一批种子、同一套窗口, 差异只有那
2 列融券特征。直接看"收益中位 79.2 -> 97.1"没法区分这是特征带来的, 还是 20
个种子本身的抖动 —— 这个策略的种子间标准差本来就有几十个点。

配对法把种子当区组: 对每个 (配置, 相位, 种子) 取两边之差, 再检验差值的中心
是否偏离 0。这样种子噪声被消掉, 检验的功效比独立两样本高得多。

用 Wilcoxon 符号秩(非参数)而不是 t 检验: 单次回测收益是重尾的, 均值不稳。
同时报"变好的比例", 这个量最难被单个极端种子带偏。

注意
────
不同相位之间不独立(同一段行情的错位切片), 所以把 5 个相位的 100 个配对全丢进
一次检验会高估自由度。这里【分相位各报一次】, 再看结论是否一致 —— 5 个相位
方向一致本身就是比任何 p 值更有说服力的证据。

用法
────
    python scripts/paired_variant_test.py --base mb_dmw --test mb_dmw_rq
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_grid import (  # noqa: E402
    DEFAULT_SEEDS, WINDOWS, VARIANTS, ev_tag, out_path_cap,
)

CAP = 50000.0
METRICS = [("total_return_pct", "收益%"), ("sharpe", "夏普"), ("max_dd_pct", "回撤%")]


def load(variant, cname, win, seed):
    p = out_path_cap(ev_tag(cname, win, seed, variant), win, CAP)
    if not p.exists():
        return None
    return json.load(open(p)).get("summary", {})


def wilcoxon(diffs):
    """Wilcoxon 符号秩的正态近似 p 值(双尾)。自己实现是为了不依赖 scipy。"""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 6:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:                      # 处理并列: 取平均秩
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    sd = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    z = (wp - mu) / sd if sd else 0.0
    # 双尾正态近似
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=list(VARIANTS))
    ap.add_argument("--test", required=True, choices=list(VARIANTS))
    ap.add_argument("--configs", required=True, help="逗号分隔")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()

    seeds = DEFAULT_SEEDS[:a.seeds]
    cfgs = a.configs.split(",")
    print(f"配对检验: {a.test}  vs  {a.base}   (基准={a.base}, 正值=新变体更好)")
    print(f"种子 {len(seeds)} 个 x 配置 {len(cfgs)} 个\n")

    for win in WINDOWS:
        print(f"{'=' * 88}\n窗口 {win}\n{'=' * 88}")
        print(f"{'配置':<22}{'指标':<8}{'配对数':>6}{'差值中位':>10}"
              f"{'差值均值':>10}{'变好占比':>10}{'p值':>10}")
        pooled = {k: [] for k, _ in METRICS}
        for cname in cfgs:
            for key, lab in METRICS:
                diffs = []
                for s in seeds:
                    b, t = load(a.base, cname, win, s), load(a.test, cname, win, s)
                    if not b or not t or key not in b or key not in t:
                        continue
                    diffs.append(t[key] - b[key])
                if not diffs:
                    continue
                pooled[key].extend(diffs)
                p = wilcoxon(diffs)
                win_rate = sum(1 for x in diffs if x > 0) / len(diffs) * 100
                print(f"{cname:<22}{lab:<8}{len(diffs):>6}{st.median(diffs):>10.2f}"
                      f"{st.mean(diffs):>10.2f}{win_rate:>9.0f}%"
                      f"{('%.4f' % p) if p is not None else '--':>10}")
            print()
        print("-- 全相位合并(自由度被高估, 仅看方向, 不要当 p 值用) --")
        for key, lab in METRICS:
            d = pooled[key]
            if not d:
                continue
            wr = sum(1 for x in d if x > 0) / len(d) * 100
            print(f"{'合并':<22}{lab:<8}{len(d):>6}{st.median(d):>10.2f}"
                  f"{st.mean(d):>10.2f}{wr:>9.0f}%")
        print()


if __name__ == "__main__":
    main()
