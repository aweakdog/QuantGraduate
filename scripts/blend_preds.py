"""零训练异质预测合成器 (2026-11-23): 把主模型缓存与辅模型缓存按日对齐均值/标准差后加权融合。

保持主模型 (DART) 的每日预测均值与标准差尺度不变, 使 --min-pred 0.002 的阈值语义无缝对齐。
同时输出两模型逐日截面 Spearman 相关性的均值/中位数, 验证是否构成真正的异质源。
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def blend_two(main_path: Path, aux_path: Path, w_aux: float, out_path: Path, model_tag: str) -> dict:
    with open(main_path, "rb") as fh:
        main_obj = pickle.load(fh)
    with open(aux_path, "rb") as fh:
        aux_obj = pickle.load(fh)

    aux_by_d = {dp["date"]: dp for dp in aux_obj["preds"]}
    blended = []
    corrs = []
    for dp in main_obj["preds"]:
        d = dp["date"]
        adp = aux_by_d.get(d)
        codes = dp["ranked"]
        p_main = np.array(dp["pred_vals"], dtype=np.float64)
        if adp is None or not adp.get("pred_vals"):
            blended.append(dp)
            continue
        amap = dict(zip(adp["ranked"], adp["pred_vals"]))
        p_aux_raw = np.array([amap.get(c, np.nan) for c in codes], dtype=np.float64)
        valid = np.isfinite(p_aux_raw) & np.isfinite(p_main)
        if valid.sum() > 5:
            c = spearmanr(p_main[valid], p_aux_raw[valid])[0]
            if np.isfinite(c):
                corrs.append(float(c))
            # 把辅模型当日预测仿射对齐到主模型当日的均值/标准差 (保持 min-pred 尺度语义)
            m_mu, m_sd = float(p_main[valid].mean()), float(p_main[valid].std())
            a_mu, a_sd = float(p_aux_raw[valid].mean()), float(p_aux_raw[valid].std())
            if a_sd > 1e-12 and m_sd > 1e-12:
                p_aux = (p_aux_raw - a_mu) / a_sd * m_sd + m_mu
            else:
                p_aux = np.full_like(p_main, m_mu)
            p_aux = np.where(np.isfinite(p_aux), p_aux, p_main)
            p_new = (1.0 - w_aux) * p_main + w_aux * p_aux
        else:
            p_new = p_main
        order = np.argsort(-p_new, kind="mergesort")
        blended.append({
            "date": d,
            "ranked": [codes[k] for k in order],
            "pred_vals": [round(float(p_new[k]), 6) for k in order],
            "ic": dp.get("ic"),
        })

    meta = dict(main_obj["meta"])
    meta["model"] = model_tag
    with open(out_path, "wb") as fh:
        pickle.dump({"meta": meta, "preds": blended}, fh)
    return {
        "days": len(blended),
        "corr_mean": round(float(np.mean(corrs)), 4) if corrs else None,
        "corr_median": round(float(np.median(corrs)), 4) if corrs else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--w-aux", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-tag", required=True)
    a = ap.parse_args(argv)
    info = blend_two(Path(a.main), Path(a.aux), a.w_aux, Path(a.out), a.model_tag)
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
