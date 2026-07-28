"""
XGBoost 评分引擎 — 六维特征 → 股票评分

使用 XGBRanker 训练排序模型，输出每日关注圈评分。

流程:
  1. 读取特征矩阵 (data/processed/training_data.parquet)
  2. 训练 XGBoost (验证集评估)
  3. 每日预测 → 评分输出到 data/processed/daily_scores.json

依赖: pip install xgboost scikit-learn
"""
import pandas as pd
import numpy as np
import json
import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional

from pipeline.config import settings
from pipeline.logger import get_logger as _gl

log = _gl("xgb_scorer")
MODEL_DIR = str(settings.MODEL_DIR)
SCORES_PATH = str(settings.SCORES_PATH)
SUPPLY_CHAIN_PATH = str(settings.SUPPLY_CHAIN_PATH)
DATA_DIR = str(settings.DATA_DIR)

# ─── 链主加成 ──────────────────────────────────────────────
def load_supply_chain() -> dict:
    """加载供需链映射"""
    with open(SUPPLY_CHAIN_PATH, encoding="utf-8") as f:
        sc = json.load(f)
    return sc

def build_stock_leader_map(sc: dict) -> dict:
    """stock_code → [{leader, exposure, direction}]"""
    m = {}
    for chain in sc["chains"]:
        lname = chain["chain_leader"]["name"]
        for link in chain["demand_links"]:
            for s in link["a_share_suppliers"]:
                code = s["code"]
                if code not in m:
                    m[code] = []
                m[code].append({"leader": lname, "exposure": s["exposure"]})
    return m

EXPOSURE_BOOST = {"核心": 3.0, "高": 1.5, "中": 0.5}

def chain_leader_boost(meta_df: pd.DataFrame, stock_leader_map: dict, active_leaders: Optional[set] = None, strength: float = 0.08) -> np.ndarray:
    """
    对评分 DataFrame 应用链主加成。
    active_leaders: 当前活跃的链主名称集合（如 {"NVIDIA", "Tesla"}）
    strength: 加成强度（核心供应商+strength*3.0 = +24%）
    返回加成分数（乘法因子）
    """
    if active_leaders is None:
        return np.ones(len(meta_df))
    
    boosts = []
    for _, row in meta_df.iterrows():
        boost = 1.0
        leaders = stock_leader_map.get(row["code"], [])
        for l in leaders:
            if l["leader"] in active_leaders:
                w = EXPOSURE_BOOST.get(l["exposure"], 0.5)
                boost += strength * w
        boosts.append(boost)
    return np.array(boosts)

# ─── 特征列 — 从数据动态检测(不用硬编码, 随feature_engine自动更新) ──
ALL_FEATURES = None
TARGET = "fwd_1d_ret"

# ─── 数据加载 ────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    global ALL_FEATURES
    path = os.path.join(DATA_DIR, "processed", "training_data_v7.parquet")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, "processed", "training_data_v15.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"训练数据不存在: {path}")
    df = pd.read_parquet(path)
    ALL_FEATURES = [c for c in df.columns if c not in ("date", "code") and not c.startswith("fwd_")]
    log.info(f"加载数据: {len(df)} 行, {len(df.columns)} 列, {len(ALL_FEATURES)} 特征")
    log.info(f"日期: {df['date'].min()} → {df['date'].max()}")
    log.info(f"股票数: {df['code'].nunique()}")
    return df

def prepare_train_test(df: pd.DataFrame, test_ratio: float = 0.2):
    """按时间分割（非随机）"""
    dates = sorted(df["date"].unique())
    split_idx = int(len(dates) * (1 - test_ratio))
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    train = df[df["date"].isin(train_dates)].copy()
    test = df[df["date"].isin(test_dates)].copy()
    log.info(f"训练: {len(train)} 行 ({train_dates[0]} → {train_dates[-1]})")
    log.info(f"验证: {len(test)} 行 ({test_dates[0]} → {test_dates[-1]})")
    return train, test

def prepare_xy(df: pd.DataFrame):
    """提取特征和标签，处理缺失值"""
    X = df.reindex(columns=ALL_FEATURES, fill_value=0.0).copy()
    y = df[TARGET].copy()

    # 缺失值填充
    X = X.fillna(0)
    y = y.fillna(0)

    # 替换inf
    X = X.replace([np.inf, -np.inf], 0)
    y = y.replace([np.inf, -np.inf], 0)

    # 去掉目标为 NaN 的行
    valid = y.notna()
    return X[valid], y[valid], df[valid].reset_index(drop=True)

def prepare_xy_rank(df: pd.DataFrame, n_groups: int = 5):
    """
    将连续目标转为分级标签（按日分组排序）。
    每日涨幅前 1/n = label 4, 次 1/n = label 3, ... 最后 1/n = label 0
    """
    X, y, meta = prepare_xy(df)
    labels = np.zeros(len(meta), dtype=int)
    dates = meta["date"].values  # numpy array
    for dt in np.unique(dates):
        mask = dates == dt
        day_y = y.values[mask]
        bins = np.percentile(day_y, np.linspace(0, 100, n_groups + 1))
        bins[-1] = np.inf
        labels[mask] = np.digitize(day_y, bins[:-1]) - 1
    return X, labels, meta

# ─── 训练 ────────────────────────────────────────────────────

def train_xgb(train: pd.DataFrame, test: Optional[pd.DataFrame] = None) -> Optional["xgb.XGBRegressor"]:
    """训练 XGBoost 回归模型，预测未来1日收益"""
    try:
        import xgboost as xgb
    except ImportError:
        log.info("安装 xgboost: pip install xgboost")
        return None

    X_train, y_train, meta_train = prepare_xy(train)

    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
    }

    model = xgb.XGBRegressor(**params, objective="reg:squarederror")
    model.fit(X_train, y_train, verbose=False)

    # 评估: 测试集上 Top5 持仓 vs 基准
    if test is not None:
        X_test, y_test, meta_test = prepare_xy(test)
        preds = model.predict(X_test)
        meta_test["pred"] = preds
        results = []
        for dt in sorted(meta_test["date"].unique()):
            day = meta_test[meta_test["date"] == dt]
            if len(day) < 5:
                continue
            top5 = day.nlargest(5, "pred")
            avg_top5 = top5[TARGET].mean()
            avg_all = day[TARGET].mean()
            # 交易成本（单边0.1%，双边0.2%）
            turnover_cost = 0.002
            cost = turnover_cost  # 每日全换手
            results.append({"date": dt, "top5": avg_top5 - cost, "all": avg_all, "excess": avg_top5 - avg_all - cost})
        r_df = pd.DataFrame(results)
        r_df = r_df.dropna(subset=["top5", "all"])  # 去掉没有未来收益的最后几天
        excess_mean = r_df["excess"].mean() * 100
        win_rate = (r_df["excess"] > 0).mean() * 100
        log.info(f"测试期 Top5 日均超额: {excess_mean:.4f}%")
        log.info(f"跑赢基准胜率: {win_rate:.2f}%")
        # 安全累计
        r_df["cum_top5"] = (1 + r_df["top5"].clip(-0.5, 0.5)).cumprod()
        r_df["cum_all"] = (1 + r_df["all"].clip(-0.5, 0.5)).cumprod()
        log.info(f"Top5 累计收益: {r_df['cum_top5'].iloc[-1]:.4f}")
        log.info(f"基准累计收益: {r_df['cum_all'].iloc[-1]:.4f}")
        log.info(f"超额累计: {(r_df['cum_top5']/r_df['cum_all']).iloc[-1]:.4f}")
        # 年化
        n_days = len(r_df)
        log.info(f"Top5 年化(252交易日): {r_df['cum_top5'].iloc[-1]**(252/n_days)-1:.4f}")
        log.info(f"基准年化(252交易日): {r_df['cum_all'].iloc[-1]**(252/n_days)-1:.4f}")
        # Sharpe
        log.info(f"Top5 Sharpe(日): {r_df['top5'].mean()/r_df['top5'].std()*252**0.5:.4f}")
        log.info(f"基准Sharpe(日): {r_df['all'].mean()/r_df['all'].std()*252**0.5:.4f}")
        log.info(f"日均换手{(r_df['excess']!=0).mean()*100:.1f}%")

    # 特征重要性
    imp = pd.DataFrame({
        "feature": model.feature_names_in_ if hasattr(model, "feature_names_in_") else ALL_FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    log.info("\nTop 10 特征:")
    log.info(imp.head(10).to_string(index=False))

    return model

# ─── 每日评分 ────────────────────────────────────────────────

def score_today(model, date_str: str = None) -> list:
    """对关注圈输出当日评分"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    features_dir = os.path.join(DATA_DIR, "processed", "features")
    watchlist_path = os.path.join(DATA_DIR, "universe", "watchlist.json")
    with open(watchlist_path, encoding="utf-8") as f:
        watch = json.load(f)

    stocks = watch.get("watchlist", [])
    records = []
    for s in stocks:
        code = s["code"]
        code6 = code[:6]
        feat_path = os.path.join(features_dir, f"{code6}.parquet")
        if not os.path.exists(feat_path):
            continue
        df = pd.read_parquet(feat_path).sort_values("date")
        latest = df.iloc[[-1]]
        X = latest.reindex(columns=ALL_FEATURES, fill_value=0).fillna(0).replace([np.inf, -np.inf], 0)
        pred = model.predict(X)[0]
        records.append({"code": code, "name": s["name"], "theme": s.get("theme", ""),
                        "pred": pred, "date": str(latest.iloc[0]["date"])[:10]})

    meta = pd.DataFrame(records)
    meta["score"] = meta["pred"]  # 链主特征已在训练中学习

    scores = []
    for _, r in meta.sort_values("score", ascending=False).iterrows():
        scores.append({
            "code": r["code"], "name": r["name"], "theme": r["theme"],
            "score": round(float(r["score"]) * 10000, 2),
            "date": r["date"],
        })

    return {"date": date_str, "total": len(scores), "scores": scores}

# ─── 主流程 ──────────────────────────────────────────────────

def main():
    log.info("=" * 50)
    log.info("XGBoost 评分引擎")
    log.info("=" * 50)

    # 1. 加载数据
    df = load_data()

    # 2. 分割
    train, test = prepare_train_test(df)

    # 3. 训练
    model = train_xgb(train, test)
    if model is None:
        return

    # 4. 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"\n模型保存: {model_path}")

    # 5. 今日评分
    scores = score_today(model)
    if scores:
        with open(SCORES_PATH, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        log.info(f"\n评分输出: {SCORES_PATH}")
        log.info(f"Top 5:")
        for s in scores["scores"][:5]:
            log.info(f"  {s['name']}({s['code']}): {s['score']}")

if __name__ == "__main__":
    main()
