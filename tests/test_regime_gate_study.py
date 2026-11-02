"""广度开关研究: 回放命令只改开关参数; 比较按种子和按开关不同的段统计。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import regime_gate_study as rg  # noqa: E402


def test_command_only_changes_regime_args():
    off = rg.command("/r", "OFF", 1)
    b30 = rg.command("/r", "B30", 1)
    assert off[off.index("--regime-filter") + 1] == "off" and b30[b30.index("--regime-breadth") + 1] == "0.30"
    assert b30[b30.index("--regime-filter") + 1] == "breadth" and "--load-preds" in off and "RG_OFF" in off
    assert off[off.index("--tranche-n") + 1] == "5" and off[off.index("--initial-capital") + 1] == "100000"


def _df(rets, cash):
    idx = pd.bdate_range("2025-03-10", periods=len(rets))
    return pd.DataFrame({"daily_ret": rets, "in_cash": cash}, index=idx)


def test_compare_segments_and_drop_top():
    base = {s: _df([0.0, 0.0, 0.0, 0.01, 0.01, 0.0], [True, True, True, False, False, True]) for s in (1, 2, 3)}
    var = {s: _df([0.0, 0.02, 0.01, 0.01, 0.01, -0.01], [True, False, False, False, False, False]) for s in (1, 2, 3)}
    r = rg.compare(base, var)
    assert r["n"] == 3 and r["pos"] == 3 and len(r["segments"]) == 2           # 第 2-3 天、第 6 天两段状态不同
    assert r["segments"][0]["median_pct"] > 0 and r["segments"][1]["median_pct"] < 0 and r["seg_pos_share"] == 0.5
    assert r["med_d_ret_drop_top_seg"] < 0                                       # 去掉最赚的一段就不赚了
    assert not r["pass_core"]
    assert np.isclose(r["med_d_h1"] + 0, r["med_d_h1"])
