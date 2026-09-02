"""线上 profile 定义的不变量

重点守住一条: 基准线与真实线在【可投资范围】上确实不同, 且文案不得再声称
"参数逐字一致"。2026-08-04 之前代码注释和网页 desc 都写着基准线与
稳妥/激进5万 参数完全相同, 但 steady5w/aggr5w 带 skip-boards="30,688"
而基准线没有 —— 于是"真实账户 - 基准线"量到的主要是板块范围差异, 不是
当初想测的"人为干预代价"。经决定保持基准线为全市场, 但文案必须说清。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MAIN_BOARD_ONLY = ("aggr2w", "aggr2w_px2", "steady5w", "aggr5w", "aggr10w",
                   "fyf100w", "bench10m")
FULL_MARKET = ("steady2w", "base5w_steady", "base5w_aggr", "bench10m_fm")


def test_main_board_lines_keep_skip_boards():
    """没开创/科板权限的线必须带 skip-boards, 否则会推荐买不了的股"""
    from live_config import PROFILES
    for pid in MAIN_BOARD_ONLY:
        assert PROFILES[pid].get("skip-boards") == "30,688", pid


def test_full_market_lines_have_no_skip_boards():
    from live_config import PROFILES
    for pid in FULL_MARKET:
        assert "skip-boards" not in PROFILES[pid], pid


def test_baseline_desc_does_not_claim_identical_params():
    """基准线与对应真实线的板块范围不同, 文案不得再声称"完全相同/逐字一致"

    这段 desc 会直接显示在网页上, 说错了会让人把板块差异误读成人为干预代价。
    """
    from live_config import PROFILES
    for pid in ("base5w_steady", "base5w_aggr"):
        desc = PROFILES[pid]["desc"]
        for bad in ("参数完全相同", "逐字一致", "参数相同"):
            assert bad not in desc, f"{pid} 的 desc 仍声称参数一致: {desc}"
        # 必须点明它是全市场口径
        assert "全市场" in desc, f"{pid} 的 desc 未说明是全市场: {desc}"


def test_baseline_and_live_lines_actually_differ_in_universe():
    """把这个差异显式断言下来 —— 将来若有人给基准线补上 skip-boards,
    这个测试会失败, 提醒他同时回头修 desc 文案(以及重置那两条线)。
    """
    from live_config import PROFILES
    pairs = (("steady5w", "base5w_steady"), ("aggr5w", "base5w_aggr"))
    for live, base in pairs:
        assert PROFILES[live].get("skip-boards") != PROFILES[base].get("skip-boards")
        # 持仓数仍应一致 —— 这才是它们成对的意义
        assert PROFILES[live]["tranche-n"] == PROFILES[base]["tranche-n"]


def test_desc_reports_distribution_not_single_run():
    """网页上展示的 desc 必须是分布口径, 不能再写单次回测的数字

    旧 desc 写的是 "回测近3年 +108% / 回撤 -29%" 这类单次跑结果, 而同一配置换
    随机种子总收益能从 -31% 到 +190%(IC 全在 0.0133~0.0170 的极窄区间) ——
    等于把抽奖结果当业绩承诺展示给用户。
    """
    from live_config import PROFILES
    for pid, p in PROFILES.items():
        desc = p["desc"]
        assert "20种子" in desc, f"{pid} 的 desc 未标注是多种子分布: {desc}"
        # 必须同时给出两个窗口, 否则又变成只报好消息
        assert "2022-09~2026-07" in desc, f"{pid} 缺强势窗口数字"
        assert "2020-07~2022-08" in desc, f"{pid} 缺弱势窗口数字"
        assert "亏损" in desc, f"{pid} 未说明亏损种子数: {desc}"
        # 旧措辞不得复活
        for bad in ("回测近3年", "两段都跑赢"):
            assert bad not in desc, f"{pid} 的 desc 仍在用单次回测措辞: {bad}"


def test_px2_is_twin_of_aggr2w():
    """PX2 的线声称"策略与激进2万相同", 那就必须真的相同 ——
    改了一边忘了另一边, 两个人会拿到不同的信号却以为在跟同一策略。"""
    from live_config import PROFILES
    a, b = PROFILES["aggr2w"], PROFILES["aggr2w_px2"]
    for k in ("capital", "tranche-n", "lot-flex", "skip-boards",
              "fill-daily", "roll-rank"):
        assert a.get(k) == b.get(k), f"aggr2w 与 aggr2w_px2 的 {k} 不一致"


def test_every_profile_has_opened_date():
    """每条线(含基准)都要标"在网站开户"日期, 格式 YYYY-MM-DD"""
    import re
    from live_config import PROFILES
    for pid, p in PROFILES.items():
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.get("opened") or ""), \
            f"{pid} 缺 opened 开户日或格式不对: {p.get('opened')!r}"


def test_plan_glob_immune_to_profile_prefix_collision():
    """aggr2w 是 aggr2w_px2 的前缀 —— 2026-08-18 线上事故:
    plan_aggr2w_*.json 扫进了 plan_aggr2w_px2_*.json, 且 px2 段字典序
    排在日期后面, aggr2w 页面把 PX2 的建仓单当成了自己的最新计划。
    glob 必须用 [0-9] 锁死日期段首字符。"""
    from fnmatch import fnmatch
    from action_page import _plan_glob
    from live_config import PROFILES
    g = _plan_glob("aggr2w")
    assert fnmatch("plan_aggr2w_2026-08-17.json", g)
    assert not fnmatch("plan_aggr2w_px2_2026-08-17.json", g)
    # 未来新增条线也不许制造"前缀 + 数字开头后缀"的碰撞:
    # 那样即使锁了日期段 glob 也救不回来
    ids = list(PROFILES)
    for a in ids:
        for b in ids:
            if a != b and b.startswith(a + "_"):
                rest = b[len(a) + 1:]
                assert not rest[:1].isdigit(), \
                    f"{b} 与 {a} 的计划文件名会互相污染, 换个 id"


def test_regime_filter_is_in_fingerprint():
    """改 regime-filter 必须触发重置, 否则会拿旧持仓跑新策略的账"""
    from live_config import FINGERPRINT_KEYS
    assert "regime_filter" in FINGERPRINT_KEYS


def test_hold_days_pinned_to_five():
    """所有回测结论都是 hold=5 口径; live_signal 的 argparse 默认是 10"""
    from live_config import BASE_PARAMS
    assert BASE_PARAMS["hold-days"] == 5


# ── T1A/T1B 分点分配 (2026-09-01) ─────────────────────────
# 证据: LSW 逐线扫描(看板 08-31) + T1AL/T1AV lag1 生产口径复核(看板 09-01)。
# 分配是逐点判决不是全线一刀切, 改任何一条前先读 factor_family_ledger「T1A/T1B」。

T1A_LINES = ("aggr5w", "aggr10w", "base5w_aggr")
T1B_LINES = ("steady5w", "fyf100w", "base5w_steady")
NO_T1_LINES = ("steady2w", "aggr2w", "aggr2w_px2", "bench10m", "bench10m_fm")


def test_t1_allocation_matches_ledger_verdict():
    """分点分配钉死: T1A=订单结构(逐笔lag1), T1B=K线情绪分解。

    steady2w 明确【不上】T1A: lag0 下 +19.4pp 但 lag1(生产可得口径)塌成
    +3.45pp/11-20 = wash —— 六个点里唯一一个 alpha 大头是当日信息的点。
    aggr2w 是 LSW 原判 wash。谁改这里谁先出 20 种子配对证据。
    """
    from live_config import PROFILES
    for pid in T1A_LINES:
        assert PROFILES[pid].get("features-from") == "features_V24PUT_T1A.json", pid
    for pid in T1B_LINES:
        assert PROFILES[pid].get("features-from") == "features_V24PUT_T1B.json", pid
    for pid in NO_T1_LINES:
        assert "features-from" not in PROFILES[pid], \
            f"{pid} 不该有分线特征集 (steady2w=lag1 wash, aggr2w=LSW wash, " \
            f"bench* 无该点位证据)"


def test_locked_baselines_mirror_latest_strategy_features():
    """08-22 用户口径: 基准线始终跟随最新策略 —— 特征集也要跟着镜像对象走"""
    from live_config import PROFILES
    assert PROFILES["base5w_steady"].get("features-from") == \
        PROFILES["steady5w"].get("features-from")
    assert PROFILES["base5w_aggr"].get("features-from") == \
        PROFILES["aggr5w"].get("features-from")


def test_per_line_features_from_beats_global():
    """分线特征表必须顶掉全局表, 且只发一次 ——
    发两次的话 argparse 后者覆盖前者, 全局表会静默把分线表顶掉(T1 白上)。"""
    from live_config import signal_args, FEATURES_FROM
    args = signal_args("aggr5w")
    assert args.count("--features-from") == 1
    assert args[args.index("--features-from") + 1] == "features_V24PUT_T1A.json"
    # 未分配的线仍用全局表
    args2 = signal_args("aggr2w")
    assert args2.count("--features-from") == 1
    assert args2[args2.index("--features-from") + 1] == FEATURES_FROM


def test_no_features_flag_when_disabled():
    """include_features=False 时分线表也不发, 语义与全局开关一致"""
    from live_config import signal_args
    assert "--features-from" not in signal_args("aggr5w", include_features=False)


def test_feature_set_lines_use_isolated_preds_cache():
    """缓存文件是单条目的(键不同即整体覆盖): 特征集不同的线共用文件会在
    夜链里循环互踩, 每条线都 cache miss 白训。同模型的线则必须继续共享。"""
    from live_config import signal_args

    def cache_of(pid):
        a = signal_args(pid)
        return a[a.index("--preds-cache") + 1]

    assert cache_of("aggr5w") == cache_of("aggr10w")       # 同主板+T1A: 共享
    assert cache_of("steady5w") == cache_of("fyf100w")     # 同主板+T1B: 共享
    assert cache_of("aggr2w") == cache_of("bench10m")      # 同主板+基线: 共享
    assert cache_of("steady2w") == cache_of("bench10m_fm")  # 同全市场+基线: 共享
    distinct = {cache_of(p) for p in
                ("aggr5w", "steady5w", "aggr2w", "steady2w",
                 "base5w_aggr", "base5w_steady")}
    assert len(distinct) == 6, f"六类模型的缓存文件必须两两不同: {distinct}"
