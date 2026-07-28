"""
HSN v0.2 — GPU 满速训练版

关键设计:
  1. 所有 tensor 一次性预处理并搬到 GPU
  2. 每 epoch 只做 forward/backward, 不做数据搬运
  3. 模型 256K 参数, 让 GPU 吃饱
"""
import os, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings
from pipeline.logger import get_logger
from pipeline.hsn_model import HSN

log = get_logger("hsn")
DATA_DIR = str(settings.DATA_DIR)

TECH_COLS = ["ret_1d","ret_5d","ret_10d","ret_20d","ma5_pct","ma10_pct","ma20_pct","ma60_pct",
             "vol_ratio","vol_change_1d","atr_pct","pos_10","pos_20","rsi_14","macd","macd_signal","macd_hist"]
EVENT_COLS = ["ann_5d","ann_20d","ann_60d","has_ann","days_since_ann"]
FUND_COLS = ["mf_net_1d","mf_net_ma3","mf_net_z"]
FUNDA_COLS = ["pe","pb","roe","mcap","revenue","profit","eps","bps"]
MACRO_COLS = ["cn_pmi","us_ism_pmi"]
LEADER_COLS = ["has_leader","leader_count","leader_exp","leader_binding_sum"]

DIM_CONFIG = {
    "tech": len(TECH_COLS), "event": len(EVENT_COLS), "fund": len(FUND_COLS),
    "funda": len(FUNDA_COLS), "macro": len(MACRO_COLS), "leader": len(LEADER_COLS),
    "calendar": 24,
}
DIM_COLS = {"tech": TECH_COLS, "event": EVENT_COLS, "fund": FUND_COLS,
            "funda": FUNDA_COLS, "macro": MACRO_COLS, "leader": LEADER_COLS}


def to_device(d, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def build_calendar(dates):
    dt = pd.to_datetime(dates)
    cal = pd.DataFrame(index=dt.index)
    for m in range(1, 13):
        cal[f"m{m}"] = (dt.dt.month == m).astype(np.float32)
    for d in range(5):
        cal[f"w{d}"] = (dt.dt.weekday == d).astype(np.float32)
    cal["spring"] = dt.dt.month.isin([3,4,5]).astype(np.float32)
    cal["summer"] = dt.dt.month.isin([6,7,8]).astype(np.float32)
    cal["autumn"] = dt.dt.month.isin([9,10,11]).astype(np.float32)
    cal["winter"] = dt.dt.month.isin([12,1,2]).astype(np.float32)
    cal["pre_h"] = 0.0
    cal["two_s"] = ((dt.dt.month==3) & (dt.dt.day.between(1,15))).astype(np.float32)
    cal["pburo"] = 0.0
    return cal.values.astype(np.float32)


def prepare_tensors(df, date_mask=None):
    """一次性把训练数据变成 GPU tensor"""
    if date_mask is not None:
        df = df[date_mask].copy()
    
    x_dict = {}
    
    # 每个维度的特征
    for name, cols in DIM_COLS.items():
        arr = df[cols].values.astype(np.float32)
        # Z-score per column
        for j in range(arr.shape[1]):
            col = arr[:, j]
            mask = ~np.isnan(col)
            if mask.sum() > 5:
                m, s = col[mask].mean(), col[mask].std()
                if s > 1e-8:
                    col[mask] = (col[mask] - m) / s
            col = np.nan_to_num(col, nan=0.0)
            arr[:, j] = np.clip(col, -10, 10)
        x_dict[name] = torch.from_numpy(arr)
    
    # 日历
    cal = build_calendar(df["date"])
    x_dict["calendar"] = torch.from_numpy(cal)
    
    # 目标: fwd_5d_ret (信号更强，HSN v0.3 已验证 +0.02% 超额)
    y = df["fwd_1d_ret"].values.astype(np.float32)
    y = np.clip(np.nan_to_num(y, nan=0.0), -0.1, 0.1)
    ym, ys = float(y.mean()), float(y.std())
    if ys < 1e-6: ys = 1.0
    y = (y - ym) / ys
    
    return x_dict, torch.from_numpy(y), df[["date","code"]]


def main():
    log.info("="*50)
    log.info("HSN v0.4 — Attention + deeper fusion")
    log.info("="*50)
    
    device = torch.device("cuda")
    log.info(f"设备: {device} ({torch.cuda.get_device_name(0)})")
    
    # 1. 加载数据
    path = os.path.join(DATA_DIR, "processed", "training_data.parquet")
    df = pd.read_parquet(path)
    log.info(f"数据: {len(df)} 行")
    
    # 2. 时间分割
    dates = sorted(df["date"].unique())
    split = int(len(dates) * 0.8)
    train_mask = df["date"].isin(dates[:split]).values
    test_mask = df["date"].isin(dates[split:]).values
    
    log.info(f"训练: {train_mask.sum()} 行 ({str(dates[0])[:10]} → {str(dates[split-1])[:10]})")
    log.info(f"验证: {test_mask.sum()} 行 ({str(dates[split])[:10]} → {str(dates[-1])[:10]})")
    
    # 3. 一次性搬到 GPU
    log.info("预处理数据...")
    x_train, y_train, _ = prepare_tensors(df, train_mask)
    x_test, y_test, meta_test = prepare_tensors(df, test_mask)
    
    x_train = to_device(x_train, device)
    y_train = y_train.to(device)
    x_test = to_device(x_test, device)
    y_test = y_test.to(device)
    
    log.info(f"x_train keys: {list(x_train.keys())}")
    for k, v in x_train.items():
        log.info(f"  {k}: {v.shape}")
    
    # 4. 模型 (256K 参数, 吃满 GPU)
    model = HSN(DIM_CONFIG, hidden=256).to(device)
    params = sum(p.numel() for p in model.parameters())
    log.info(f"参数量: {params:,}")
    
    # 5. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 6. 训练
    n_epochs = 150
    batch_size = 8192
    early_stop = 30
    n_batches = math.ceil(len(y_train) / batch_size)
    
    best_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(y_train), device=device)
        total_loss = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            batch_x = {k: v[idx] for k, v in x_train.items()}
            batch_y = y_train[idx]
            
            optimizer.zero_grad()
            scores = model(batch_x)
            loss = F.mse_loss(scores, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_scores = model(x_test)
            v_loss = F.mse_loss(val_scores, y_test).item()
            from scipy.stats import spearmanr
            ic, _ = spearmanr(val_scores.cpu().numpy(), y_test.cpu().numpy())
        
        scheduler.step(v_loss)  # ReduceLROnPlateau
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"Ep {epoch+1:2d}/{n_epochs} | Train: {total_loss:.4f} | Val: {v_loss:.4f} | IC: {ic:.4f}")
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop:
            log.info(f"Early stop at epoch {epoch+1}")
            break
    
    # 7. 回测
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    
    with torch.no_grad():
        all_scores = model(x_test).cpu().numpy()
    
    bt = meta_test.copy()
    bt["pred"] = all_scores
    bt["label"] = bt["code"]  # placeholder, 我们从 db 加载实际收益?
    
    # 重新加载原始 df 获取实际收益
    actual_y = df.loc[test_mask, "fwd_1d_ret"].values
    bt["actual"] = actual_y
    
    results = []
    for dt in bt["date"].unique():
        day = bt[bt["date"] == dt]
        if len(day) < 5:
            continue
        top5 = day.nlargest(5, "pred")
        all_ret = day["actual"].mean()
        top5_ret = top5["actual"].mean()
        cost = 0.002
        results.append({"date": dt, "top5": top5_ret - cost, "all": all_ret, "excess": top5_ret - all_ret - cost})
    
    r = pd.DataFrame(results).dropna(subset=["top5", "all"])
    if len(r) > 0:
        r["cum_t"] = (1 + r["top5"].clip(-0.5, 0.5)).cumprod()
        r["cum_a"] = (1 + r["all"].clip(-0.5, 0.5)).cumprod()
        nd = len(r)
        log.info(f"\n=== HSN v0.2 回测 ===")
        log.info(f"Top5 日均超额: {r['excess'].mean()*100:.4f}%")
        log.info(f"胜率: {(r['excess']>0).mean()*100:.1f}%")
        log.info(f"Top5 累计: {r['cum_t'].iloc[-1]:.4f}")
        log.info(f"基准累计: {r['cum_a'].iloc[-1]:.4f}")
        log.info(f"Top5 年化: {r['cum_t'].iloc[-1]**(252/nd)-1:.4f}")
        log.info(f"Top5 Sharpe: {r['top5'].mean()/r['top5'].std()*252**0.5:.4f}")
        log.info(f"基准 Sharpe: {r['all'].mean()/r['all'].std()*252**0.5:.4f}")
    
    # 保存
    save_path = os.path.join(DATA_DIR, "processed", "model", "hsn_v0.2.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state": best_state, "config": DIM_CONFIG, "val_loss": best_loss}, save_path)
    log.info(f"模型保存: {save_path}")


if __name__ == "__main__":
    main()
