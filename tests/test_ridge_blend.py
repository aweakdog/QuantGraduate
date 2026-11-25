import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ridge_blend as rb


def _make_panel(n_days=25, n_stocks=30, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    rows = []
    for di, d in enumerate(dates):
        mkt = rng.normal()  # 截面常数列, 验证不破坏计算
        for si in range(n_stocks):
            f1 = rng.normal()
            f2 = rng.normal() if rng.random() > 0.1 else np.nan
            y = 0.01 * f1 - 0.005 * (0.0 if np.isnan(f2) else f2) + rng.normal(0, 0.02)
            if di == n_days - 1:
                y = np.nan  # 末日无标签
            rows.append({"date": d, "code": f"{si:06d}.SZ", "f1": f1, "f2": f2, "mkt": mkt, "fwd_5d_ret": y})
    df = pd.DataFrame(rows)
    df["y_target"] = df.groupby("date")["fwd_5d_ret"].transform(lambda x: x - x.mean())
    return df, dates


def test_live_and_walkforward_are_bit_equivalent():
    df, dates = _make_panel(n_days=25, n_stocks=30)
    feats = ["f1", "f2", "mkt"]
    configs = [("fixed", 100.0, 0.0), ("adapt", 0.0, 0.25)]
    wf = rb.walkforward_ridge_preds(
        df, feats, "fwd_5d_ret", "y_target",
        str(dates[15].date()), str(dates[-1].date()),
        horizon=6, min_days=8, configs=configs, winsor=0.05,
    )
    for tag, al, ar in configs:
        dps = wf[tag]
        assert len(dps) > 0
        for dp in dps:
            d = dp["date"]
            gpos = list(dates).index(d)
            cutoff = dates[gpos - 6]
            tr = df[(df["date"] < cutoff) & df["fwd_5d_ret"].notna()]
            te = df[df["date"] == d].sort_values("code")
            live_p = rb.ridge_predict_day(tr, te, feats, "fwd_5d_ret", alpha=al, alpha_ratio=ar, winsor=0.05)
            live_map = dict(zip(te["code"], live_p))
            wf_map = dict(zip(dp["ranked"], dp["pred_vals"]))
            for c in te["code"]:
                assert abs(live_map[c] - wf_map[c]) < 1e-5  # wf rounds pred_vals to 6 decimals


def test_strictly_pit_no_future_leakage():
    df, dates = _make_panel(n_days=25, n_stocks=30)
    feats = ["f1", "f2", "mkt"]
    d_pred = dates[18]
    cutoff = dates[18 - 6]
    tr = df[(df["date"] < cutoff) & df["fwd_5d_ret"].notna()]
    te = df[df["date"] == d_pred]
    p1 = rb.ridge_predict_day(tr, te, feats, "fwd_5d_ret", alpha_ratio=0.1)

    # 篡改 cutoff 及之后所有标签, 结果必须逐位不变
    df2 = df.copy()
    df2.loc[df2["date"] >= cutoff, "fwd_5d_ret"] = 999.0
    tr2 = df2[(df2["date"] < cutoff) & df2["fwd_5d_ret"].notna()]
    te2 = df2[df2["date"] == d_pred]
    p2 = rb.ridge_predict_day(tr2, te2, feats, "fwd_5d_ret", alpha_ratio=0.1)
    assert np.allclose(p1, p2, atol=1e-12)


def test_blend_affine_preserves_scale_and_handles_degenerate_aux():
    main = np.array([0.005, 0.003, 0.001, -0.001, -0.003, -0.005])
    aux = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])  # 完全不同量纲
    b = rb.blend_affine(main, aux, 0.20)
    assert abs(b.mean() - main.mean()) < 1e-12
    # 退化全常数 aux 安全回落到 main
    const_aux = np.ones_like(main) * 5.0
    assert np.allclose(rb.blend_affine(main, const_aux, 0.20), main)
    # weight=0 返回原值
    assert np.allclose(rb.blend_affine(main, aux, 0.0), main)


def test_live_config_ridge_blend_cache_isolation_and_no_fingerprint(monkeypatch):
    import live_config as lc

    # 1. 不进指纹: FINGERPRINT_KEYS 不含 ridge_*
    assert not any("ridge" in k for k in lc.FINGERPRINT_KEYS)

    # 2. 开启 ridge-blend 时命令行正确发出参数且缓存路径隔离 (用默认未开的全市场线测开关切换)
    orig_args = lc.signal_args("steady2w")
    orig_cache = orig_args[orig_args.index("--preds-cache") + 1]
    assert "--ridge-blend" not in orig_args

    patched = dict(lc.PROFILES["steady2w"], **{"ridge-blend": 0.20, "ridge-alpha-ratio": 0.05})
    monkeypatch.setitem(lc.PROFILES, "steady2w", patched)
    new_args = lc.signal_args("steady2w")
    new_cache = new_args[new_args.index("--preds-cache") + 1]
    assert "--ridge-blend" in new_args and new_args[new_args.index("--ridge-blend") + 1] == "0.2"
    assert "--ridge-alpha-ratio" in new_args and new_args[new_args.index("--ridge-alpha-ratio") + 1] == "0.05"
    assert new_cache != orig_cache and "_rb20_r50" in new_cache

