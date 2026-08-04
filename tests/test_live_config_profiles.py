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

MAIN_BOARD_ONLY = ("aggr2w", "steady5w", "aggr5w", "aggr10w")
FULL_MARKET = ("steady2w", "base5w_steady", "base5w_aggr")


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


def test_regime_filter_is_in_fingerprint():
    """改 regime-filter 必须触发重置, 否则会拿旧持仓跑新策略的账"""
    from live_config import FINGERPRINT_KEYS
    assert "regime_filter" in FINGERPRINT_KEYS


def test_hold_days_pinned_to_five():
    """所有回测结论都是 hold=5 口径; live_signal 的 argparse 默认是 10"""
    from live_config import BASE_PARAMS
    assert BASE_PARAMS["hold-days"] == 5
