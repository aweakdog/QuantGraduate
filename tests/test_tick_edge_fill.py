# -*- coding: utf-8 -*-
"""回归测试: build_tick_augmented 的活缘补行语义

防的坑 (2026-08-16 设计时就已知, 不是事后追认)
──────────────────────────────────────────────
线上 17:30 重建时, 基矩阵已含当日 d 行, 逐笔面板只到 d-1 (供应商 T+1 交付)。
原实现 shift 后按 (c6,date) 精确 merge —— 行 d 的 tk_* 全 NaN, 11/80=13.75%
超过在用特征 10% NaN 拦截线, 当晚整条管线没信号。这正是 2026-08-05 mtss 事故
的镜像: 那次是"缺数据被伪造成 0", 这次要防"该有的滞后观测被丢成 NaN"。

锁住五条行为:
  1. 历史 shift 语义: 面板覆盖期内, 行 d 的 tk_xz == 面板 d-1 日的截面 z
     (轮换构造让 z 逐日不同, shift 差一位立刻爆)
  2. 正常日更 (面板落后基矩阵 1 天): 尾行拿到面板末日观测, 非 NaN
  3. 历史逐字节不变: 有无活缘补行, 面板覆盖期内所有值完全一致
  4. 断供兜底: 面板落后 2~3 天时, 尾部行全部等于面板末日观测(过期但真实)
  5. 断供超限 (--require-fresh): 落后超过阀值退出码 2 且不写输出文件

跑法: python tests/test_tick_edge_fill.py  (或 pytest)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_tick_augmented.py"

# 每所 3 只票: 2 只造不出逐日变化的截面 z (两点 z 恒为 ±0.707)
SH = ["600001", "600002", "600003"]
SZ = ["000001", "000002", "000003"]
CODES = SH + SZ
DAYS = [f"202608{d:02d}" for d in range(3, 15) if d not in (8, 9)]  # 10 个"交易日"

Z_HOT, Z_COLD = 2 / np.sqrt(3), -1 / np.sqrt(3)   # [1,0,0] 截面的 z (ddof=1)


def _val(ci, di):
    """轮换构造: 第 di 天, 所内第 (di%3) 只票为 1 其余为 0 -> z 逐日轮换"""
    return 1.0 if ci == di % 3 else 0.0


def _xz(code, di):
    ci = (SH if code in SH else SZ).index(code)
    return Z_HOT if ci == di % 3 else Z_COLD


def _build_inputs(tmp: Path, micro_days, base_days):
    micro_dir = tmp / "data" / "processed" / "tick_micro"
    micro_dir.mkdir(parents=True)
    for di, day in enumerate(micro_days):
        rows = [{"code": c, "date": int(day),
                 "spread_bp": _val((SH if c in SH else SZ).index(c), di)}
                for c in CODES]
        pd.DataFrame(rows).to_parquet(micro_dir / f"{day}.parquet", index=False)
    base = pd.DataFrame(
        [{"code": f"{c}.{'SH' if c[0] == '6' else 'SZ'}",
          "date": pd.Timestamp(day), "close": 10.0}
         for day in base_days for c in CODES])
    base.to_parquet(tmp / "data" / "processed" / "base.parquet", index=False)


def _run(tmp: Path, *args):
    """以 tmp 为仓库根跑脚本 —— 脚本用 __file__ 定位仓库根, 所以复制一份进 tmp"""
    sdir = tmp / "scripts"
    sdir.mkdir(exist_ok=True)
    (sdir / "build_tick_augmented.py").write_text(SCRIPT.read_text())
    return subprocess.run(
        [sys.executable, str(sdir / "build_tick_augmented.py"),
         "--source", "base.parquet", "--lag", "1", *args],
        capture_output=True, text=True)


def _read_out(tmp: Path):
    m = pd.read_parquet(tmp / "data" / "processed" / "base_tick1.parquet")
    c6 = m["code"].str[:6].rename("c6")
    return m.set_index([c6, "date"]).sort_index()


def test_history_shift_and_edge():
    """历史行 shift 语义逐日核对 + 活缘行 = 面板末日观测"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _build_inputs(tmp, micro_days=DAYS[:-1], base_days=DAYS)  # 面板缺最后一天
        r = _run(tmp)
        assert r.returncode == 0, r.stdout + r.stderr
        m = _read_out(tmp)
        for c in CODES:
            # 历史: 行 DAYS[k] 应等于面板 DAYS[k-1] 的 z (k=1..8), 差一位必炸
            for k in range(1, len(DAYS) - 1):
                got = m.loc[(c, pd.Timestamp(DAYS[k])), "tk_spread_bp_xz"]
                exp = _xz(c, k - 1)
                assert abs(got - exp) < 1e-9, \
                    f"{c} {DAYS[k]} shift 语义错: got {got} exp {exp}"
            # 活缘: 基矩阵末日 DAYS[-1] 应拿到面板末日 DAYS[-2](di=8) 的 z
            got = m.loc[(c, pd.Timestamp(DAYS[-1])), "tk_spread_bp_xz"]
            exp = _xz(c, len(DAYS) - 2)
            assert np.isfinite(got), f"{c} 活缘行是 NaN —— 线上当晚会没信号"
            assert abs(got - exp) < 1e-9, f"{c} 活缘值错: got {got} exp {exp}"
        # ma5 列同样不能整行缺
        assert np.isfinite(
            m.loc[(CODES[0], pd.Timestamp(DAYS[-1])), "tk_spread_bp_xz_ma5"])
    print("test_history_shift_and_edge ok")


def test_history_bytes_unchanged():
    """同一份面板, 基矩阵多一天 vs 不多天: 覆盖期内所有 tk_* 值完全一致"""
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        p1, p2 = Path(t1), Path(t2)
        _build_inputs(p1, micro_days=DAYS[:-1], base_days=DAYS[:-1])  # 回测态
        _build_inputs(p2, micro_days=DAYS[:-1], base_days=DAYS)       # 线上态
        assert _run(p1).returncode == 0
        assert _run(p2).returncode == 0
        m1, m2 = _read_out(p1), _read_out(p2)
        tk = [c for c in m1.columns if c.startswith("tk_")]
        pd.testing.assert_frame_equal(m1[tk], m2.loc[m1.index, tk],
                                      check_exact=True)
    print("test_history_bytes_unchanged ok")


def test_outage_ffill_capped():
    """面板落后 3 天: 尾部 3 行全部等于面板末日观测; --require-fresh 3 恰好放行"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _build_inputs(tmp, micro_days=DAYS[:-3], base_days=DAYS)
        r = _run(tmp, "--require-fresh", "3")
        assert r.returncode == 0, r.stdout + r.stderr
        m = _read_out(tmp)
        last_di = len(DAYS) - 4                     # 面板末日 = DAYS[-4]
        for c in CODES:
            exp = _xz(c, last_di)
            for d in DAYS[-3:]:
                got = m.loc[(c, pd.Timestamp(d)), "tk_spread_bp_xz"]
                assert np.isfinite(got) and abs(got - exp) < 1e-9, \
                    f"{c} {d} 断供兜底值错: got {got} exp {exp}"
    print("test_outage_ffill_capped ok")


def test_stale_gate_blocks():
    """面板落后 4 天 + --require-fresh 3: 退出码 2, 不写输出"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _build_inputs(tmp, micro_days=DAYS[:-4], base_days=DAYS)
        r = _run(tmp, "--require-fresh", "3")
        assert r.returncode == 2, f"应退出码 2, 实际 {r.returncode}\n{r.stdout}"
        assert not (tmp / "data" / "processed" / "base_tick1.parquet").exists(), \
            "超限时不得写出矩阵"
    print("test_stale_gate_blocks ok")


if __name__ == "__main__":
    test_history_shift_and_edge()
    test_history_bytes_unchanged()
    test_outage_ffill_capped()
    test_stale_gate_blocks()
    print("全部通过")
