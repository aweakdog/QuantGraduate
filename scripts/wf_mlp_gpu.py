#!/usr/bin/env python
"""GPU MLP 截面模型 walk-forward — 与 wf_v35 完全同协议的第二模型源。

动机 (2026-08-02): 前3名选股方差是当前策略最大脆弱点(同 IC 下训练集微扰
可让组合收益腰斩)。多模型集成靠方差抵消压制它, 前提是模型间预测相关性低。
GBDT 之外最自然的低相关来源就是神经网络。

协议对齐 (必须与 wf_v35_breadth_alpha.py 逐条一致, 否则对比无意义):
- 数据: 用 wf_v35 --export-matrix 导出的矩阵 (同 PIT 池/特征/demean 标签)
- 防泄漏: 训练集 = date < all_dates[gpos - LABEL_HORIZON], 与 wf_v35 同款
- 准入: 有标签训练日 >= min_train_days 才出信号
- NaN: 训练 X 按 code 先 ffill 再 fillna(0); 预测日直接 fillna(0)
- 输出: v2 预测缓存格式 (ranked + pred_vals + ic), 可直接喂
  wf_v35 --load-preds 跑执行层

训练方式: 首日全量训练, 之后每天在扩窗上微调(warm-start), 等价于逐日重训
的近似但快 ~10 倍。--full-refit-every 控制多少天做一次全量重训防漂移。

用法示例 (冒烟):
  python scripts/wf_mlp_gpu.py --matrix wf_matrix_cs80.parquet \
      --test-start 2026-05-01 --device cuda:0 --seed 0 \
      --save-preds preds_MLP_smoke.pkl
"""
import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--matrix", required=True,
                    help="wf_v35 --export-matrix 导出的 parquet (data/processed/ 下)")
parser.add_argument("--test-start", default=None, help="默认用 meta 里的 test_start")
parser.add_argument("--test-end", default=None, help="默认用 meta 里的 test_end")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hidden", type=int, nargs=2, default=[256, 64])
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--init-epochs", type=int, default=30, help="首日全量训练轮数")
parser.add_argument("--daily-epochs", type=int, default=2, help="每日扩窗微调轮数")
parser.add_argument("--full-refit-every", type=int, default=60,
                    help="每 N 个交易日重新全量训练一次 (防 warm-start 漂移)")
parser.add_argument("--feat-transform", choices=["csrank", "raw"], default="csrank",
                    help="csrank(默认): 每日每特征截面 rank 归一到[0,1], NaN→0.5中性。"
                         "NN 对特征尺度/重尾敏感, 裸特征(raw)实测 IC 接近 0; "
                         "截面 rank 只用当日同时刻数据, PIT 安全。raw 仅供对照")
parser.add_argument("--save-preds", required=True,
                    help="输出预测缓存 pickle 文件名 (data/processed/ 下, v2 格式)")
parser.add_argument("--max-days", type=int, default=0, help="只跑前 N 个信号日 (冒烟用, 0=全部)")
args = parser.parse_args()

import torch                                    # noqa: E402  (放 argparse 后: --help 不需要它)
import torch.nn as nn                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings            # noqa: E402

DATA_DIR = settings.DATA_DIR
MPATH = DATA_DIR / "processed" / args.matrix
meta = json.load(open(str(MPATH) + ".meta.json", encoding="utf-8"))
FEATURES = meta["features"]
LABEL = meta["label"]
HORIZON = int(meta["label_horizon"])
MIN_TRAIN_DAYS = int(meta["min_train_days"])
TEST_START = args.test_start or meta["test_start"]
TEST_END = args.test_end or meta["test_end"]

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 多进程共机必限: torch 默认开满所有核, 8 个进程 × 128 线程会把 CPU 踩死
# (实测负载 280+, GPU 利用率 1%)。训练数据常驻 GPU 后 CPU 只剩轻活
torch.set_num_threads(4)
dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
if dev.type == "cpu":
    print("!! CUDA 不可用, 退回 CPU (慢, 仅调试)")

print(f"加载 {MPATH.name} ...")
df = pd.read_parquet(MPATH)
df["date"] = pd.to_datetime(df["date"])
df["code"] = df["code"].astype(str)
print(f"  {len(df)} 行, {df['code'].nunique()} 只, {len(FEATURES)} 特征")

df = df.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
y_all = df[LABEL].astype(np.float32)
if args.feat_transform == "csrank":
    # 截面 rank: 每日每特征归一到[0,1], 再平移缩放到均值0方差1。
    # 训练/预测同一变换(只依赖当日截面, 无需区分 ffill 口径)。
    print("截面 rank 变换...")
    Xcs = df.groupby("date")[FEATURES].rank(pct=True)
    Xcs = ((Xcs.fillna(0.5) - 0.5) / 0.2887).astype(np.float32)
    Xfill = Xpred = Xcs
else:
    # ── 训练用 X: 按 code 先 ffill 再 fillna(0) (与 wf_v35 训练分支同口径) ──
    Xfill = df.groupby("code")[FEATURES].ffill().fillna(0.0).astype(np.float32)
    # ── 预测用 X: 当日原始值直接 fillna(0) (与 wf_v35 预测分支同口径) ──
    Xpred = df[FEATURES].fillna(0.0).astype(np.float32)

all_dates = np.array(sorted(df["date"].unique()))
date_pos = {d: i for i, d in enumerate(all_dates)}
date_vals = df["date"].values

mask = df["date"] >= pd.Timestamp(TEST_START)
if TEST_END:
    mask &= df["date"] <= pd.Timestamp(TEST_END)
sig_dates = sorted(df.loc[mask, "date"].unique())
if args.max_days:
    sig_dates = sig_dates[: args.max_days]


class MLP(nn.Module):
    def __init__(self, nin, h1, h2, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(nin),
            nn.Linear(nin, h1), nn.GELU(), nn.Dropout(p),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(p),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = MLP(len(FEATURES), *args.hidden, args.dropout).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lossf = nn.MSELoss()


def train_window(cut_ts, epochs, lr=None):
    """在 date < cut_ts 且有标签的样本上训练。
    整窗口一次性搬上 GPU (最大 28.4万行×80列×4B ≈ 91MB), 打乱/取 batch
    全在 GPU 上做 —— 否则逐 batch CPU→GPU 拷贝会让 8 卡并行时 CPU 先于 GPU 饱和。"""
    tm = (date_vals < cut_ts) & ~np.isnan(y_all.values)
    idx = np.flatnonzero(tm)
    if lr is not None:
        for g in opt.param_groups:
            g["lr"] = lr
    Xt = torch.from_numpy(Xfill.values[idx]).to(dev)
    yt = torch.from_numpy(y_all.values[idx]).to(dev)
    n = len(idx)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, args.batch_size):
            b = perm[s: s + args.batch_size]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    del Xt, yt
    return n


daily_preds = []
t0 = datetime.now()
initialized = False
days_since_refit = 0
labeled_dates = np.array(sorted(df.loc[df[LABEL].notna(), "date"].unique()))

for i, d in enumerate(sig_dates):
    gpos = date_pos[d]
    if gpos - HORIZON < 0:
        continue
    cut = all_dates[gpos - HORIZON]
    if (labeled_dates < cut).sum() < MIN_TRAIN_DAYS:
        continue

    if not initialized:
        n = train_window(cut, args.init_epochs, lr=args.lr)
        initialized = True
        days_since_refit = 0
        print(f"  [init] {pd.Timestamp(d).date()} 全量 {n} 行 x {args.init_epochs} 轮 "
              f"({(datetime.now()-t0).total_seconds():.0f}s)")
    elif days_since_refit >= args.full_refit_every:
        # 重置参数防漂移, 全量重训
        model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        n = train_window(cut, args.init_epochs, lr=args.lr)
        days_since_refit = 0
    else:
        train_window(cut, args.daily_epochs, lr=args.lr * 0.1)
        days_since_refit += 1

    tm = date_vals == d
    Xt = torch.from_numpy(Xpred.values[np.flatnonzero(tm)]).to(dev)
    model.eval()
    with torch.no_grad():
        preds = model(Xt).cpu().numpy().astype(float)
    codes = df.loc[tm, "code"].values
    yt = df.loc[tm, LABEL].values
    ok = ~np.isnan(yt)
    ic = float(spearmanr(preds[ok], yt[ok])[0]) if ok.sum() > 5 else float("nan")

    order = np.argsort(-preds)
    daily_preds.append({"date": pd.Timestamp(d), "ranked": [str(c) for c in codes[order]],
                        "pred_vals": [round(float(v), 6) for v in preds[order]],
                        "ic": ic, "blocked": set()})
    if i % 20 == 0 or i == len(sig_dates) - 1:
        el = (datetime.now() - t0).total_seconds()
        print(f"  [{i+1}/{len(sig_dates)}] {pd.Timestamp(d).date()} IC={ic:+.4f} ({el:.0f}s)")

ics = np.array([dp["ic"] for dp in daily_preds if not np.isnan(dp["ic"])])
ic_t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 2 else float("nan")
print(f"\nMLP walk-forward 完成: {len(daily_preds)} 天 | IC 均值 {ics.mean():+.4f} "
      f"t={ic_t:.2f} | 用时 {(datetime.now()-t0).total_seconds():.0f}s")

out_meta = {"model": "mlp", "feat_transform": args.feat_transform,
            "hidden": args.hidden, "dropout": args.dropout,
            "lr": args.lr, "seed": args.seed, "init_epochs": args.init_epochs,
            "daily_epochs": args.daily_epochs, "full_refit_every": args.full_refit_every,
            "matrix": args.matrix, "train_file": meta.get("train_file"),
            "pit_universe": meta.get("pit_universe"), "label": meta.get("label_raw", "")[:6] or "5d",
            "objective": "mlp_mse", "test_start": TEST_START, "test_end": TEST_END,
            "neutralize_style": False, "n_features": len(FEATURES),
            # 与 wf_v35 的 FIRST_PRED 同公式: 本次 test 区间内首个满足准入的信号日
            "feat_cutoff": f"{daily_preds[0]['date']:%Y-%m-%d}" if daily_preds else None}
# 与 wf_v35 --load-preds 的缓存校验字段对齐: label 存 '5d' 这种短格式
out_meta["label"] = "5d" if HORIZON == 5 else "1d"
opath = DATA_DIR / "processed" / args.save_preds
with open(opath, "wb") as fh:
    pickle.dump({"meta": out_meta, "preds": daily_preds}, fh)
print(f"预测缓存已保存: {opath}")
