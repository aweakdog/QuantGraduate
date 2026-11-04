"""快速回场 (RF): 回场规则的状态机 + 回放命令只加回场参数 + 邻域一致判定。

wf_v35_breadth_alpha 在模块层就解析参数、跑全流程, 不能直接 import;
这里用 ast 把 build_regime_state 原样摘出来, 配一个假的 args 执行 —— 测的是文件里的真代码。
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import regime_reentry_study as rf  # noqa: E402

DEFAULTS = dict(regime_filter="breadth", regime_ma=20, regime_breadth=0.40, regime_confirm=2, regime_reenter=0.0,
                regime_soft_low=0.0, regime_soft_n=2, regime_confirm_on=0, regime_thrust=0.0)


def _states(breadth, **kw):
    src = (ROOT / "scripts/wf_v35_breadth_alpha.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "build_regime_state")
    ns = {"pd": pd, "np": np, "args": SimpleNamespace(**(DEFAULTS | kw))}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "wf_v35", "exec"), ns)
    r = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=len(breadth)), "mkt_close": 1.0,
                      "mkt_ma20": 1.0, "breadth_above_ma": breadth})
    cash, soft = ns["build_regime_state"](r)
    assert soft is None
    return [int(x) for x in cash.to_numpy()]


B = [0.5, 0.3, 0.3, 0.3, 0.6, 0.6, 0.3, 0.45, 0.45]


def test_default_two_day_confirm_unchanged():
    assert _states(B) == [0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_confirm_on_one_day_reenters_a_day_earlier_exit_unchanged():
    assert _states(B, regime_confirm_on=1) == [0, 0, 1, 1, 0, 0, 0, 0, 0]


def test_thrust_reenters_same_day_only_if_jump_big_enough():
    assert _states(B, regime_thrust=0.25) == [0, 0, 1, 1, 0, 0, 0, 0, 0]     # 0.3 -> 0.6, 涨 30 个点
    assert _states(B, regime_thrust=0.35) == _states(B)                      # 涨幅不够, 照常确认 2 天


def test_thrust_needs_breadth_above_reentry_line():
    b = [0.5, 0.1, 0.1, 0.35, 0.36, 0.5, 0.5]                                # 0.1 -> 0.35 涨 25 个点但没到 40%
    assert _states(b, regime_thrust=0.2) == _states(b) == [0, 0, 1, 1, 1, 1, 0]


def test_command_only_adds_reentry_args():
    f1, t15 = rf.command("/r", "F1", 1), rf.command("/r", "T15", 1)
    assert f1[f1.index("--regime-confirm-on") + 1] == "1" and "--regime-thrust" not in f1
    assert t15[t15.index("--regime-thrust") + 1] == "0.15" and "--regime-confirm-on" not in t15
    assert t15[t15.index("--regime-filter") + 1] == "breadth"
    assert "RF_T15" in t15 and t15[t15.index("--tranche-n") + 1] == "5" and "--load-preds" in t15


def test_judge_requires_neighbor_not_worse():
    res = {"F1": {"pass_core": True, "med_d_ret": 6.0}, "T15": {"pass_core": True, "med_d_ret": 7.0},
           "T20": {"pass_core": False, "med_d_ret": -1.0}}
    out = rf.judge(res)
    assert out["F1"]["pass"] and not out["T15"]["pass"] and not out["T20"]["pass"]
