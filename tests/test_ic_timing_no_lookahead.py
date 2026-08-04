"""--ic-timing 不得使用尚未闭合的 IC

信号日 D 的 IC = "D 当天的预测" 与 "D 往后 LABEL_HORIZON 天的实际收益" 的
秩相关, 要到 D+LABEL_HORIZON 才知道。2026-08-04 之前的实现先把当天 IC 追加进
历史、再用最近 3 条判断当天是否空仓, 等于拿未来 5 天的收益做今天的决策。

这里不跑整条回测(太慢), 而是把那段判定逻辑抽出来复算一遍, 断言它在任一时刻
只可能看到已闭合的 IC。
"""
import numpy as np
import pytest

LABEL_HORIZON = 5


def decide(ic_seq_full, horizon=LABEL_HORIZON):
    """复刻 wf_v35 修好后的 ic_timing 判定, 返回每天 (是否空仓, 可见IC条数)"""
    ic_seq, ic_cash, out = [], False, []
    for i, ic in enumerate(ic_seq_full):
        ic_seq.append(ic)
        avail = [x for x in ic_seq[:max(0, i - horizon + 1)] if not np.isnan(x)]
        if len(avail) >= 10:
            if all(x < 0 for x in avail[-3:]):
                ic_cash = True
            if ic_cash and np.mean(avail[-10:]) > 0:
                ic_cash = False
        out.append((ic_cash, len(avail)))
    return out


def test_never_sees_unclosed_ic():
    """第 i 天最多只能看到下标 0..i-horizon 的 IC"""
    seq = list(np.linspace(-0.1, 0.1, 60))
    for i, (_, n_seen) in enumerate(decide(seq)):
        assert n_seen <= max(0, i - LABEL_HORIZON + 1), \
            f"第 {i} 天看到了 {n_seen} 条 IC, 超出已闭合范围"


def test_today_ic_cannot_flip_today_decision():
    """把今天(及最近 horizon-1 天)的 IC 改成任意值, 今天的决策必须不变

    这是未来函数的判定性检验: 若今天的决策依赖今天的 IC, 改它就会改决策。
    """
    rng = np.random.default_rng(0)
    base = list(rng.normal(0.01, 0.05, 80))
    ref = decide(base)
    for tamper_at in range(20, 80):
        polluted = list(base)
        # 把"尚未闭合"的那几天(含当天)全部污染成极端值
        for j in range(tamper_at - LABEL_HORIZON + 1, tamper_at + 1):
            if 0 <= j < len(polluted):
                polluted[j] = -9.0
        got = decide(polluted)
        assert got[tamper_at][0] == ref[tamper_at][0], \
            f"第 {tamper_at} 天的决策被未闭合的 IC 改变了 —— 存在未来函数"


def test_old_buggy_version_would_fail_the_same_check():
    """反向验证: 旧实现在同一检验下必须失败, 否则这个测试没有区分力"""
    def decide_buggy(ic_seq_full):
        ic_hist, ic_cash, out = [], False, []
        for ic in ic_seq_full:
            if not np.isnan(ic):
                ic_hist.append(ic)          # 旧 bug: 当天 IC 立刻可见
            if len(ic_hist) >= 10:
                if all(x < 0 for x in ic_hist[-3:]):
                    ic_cash = True
                if ic_cash and np.mean(ic_hist[-10:]) > 0:
                    ic_cash = False
            out.append(ic_cash)
        return out

    rng = np.random.default_rng(0)
    base = list(rng.normal(0.01, 0.05, 80))
    ref = decide_buggy(base)
    flipped = False
    for tamper_at in range(20, 80):
        polluted = list(base)
        for j in range(tamper_at - LABEL_HORIZON + 1, tamper_at + 1):
            if 0 <= j < len(polluted):
                polluted[j] = -9.0
        if decide_buggy(polluted)[tamper_at] != ref[tamper_at]:
            flipped = True
            break
    assert flipped, "旧实现竟然通过了检验, 说明这个检验没有区分力"


def test_nan_ic_days_are_skipped_not_counted():
    """末尾无标签的日子 IC 是 NaN, 不能被当成一条有效观测"""
    seq = [np.nan] * 20 + [0.02] * 20
    out = decide(seq)
    # 前 20 天全是 NaN, 到第 25 天时已闭合区间里的有效 IC 仍不足 10 条
    assert out[25][1] < 10


@pytest.mark.parametrize("horizon", [1, 5])
def test_horizon_respected(horizon):
    seq = list(np.linspace(-0.1, 0.1, 40))
    for i, (_, n_seen) in enumerate(decide(seq, horizon)):
        assert n_seen <= max(0, i - horizon + 1)
