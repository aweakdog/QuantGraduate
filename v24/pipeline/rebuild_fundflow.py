"""
重建训练集 — 将 iFinD 资金流数据合并进 training_data.parquet

当前问题: training_data.parquet 的资金面特征 99% NaN
原因: 是用旧 westock 数据建的，iFinD 数据后补但没重新 build_all()

解决: 直接 merge fund_flow parquet → 重新计算 mf_net_1d/ma3/z → 保存
"""
import pandas as pd, numpy as np, os, sys, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings

DATA = str(settings.DATA_DIR)
train_path = os.path.join(DATA, "processed", "training_data.parquet")
ff_dir = os.path.join(DATA, "raw", "fund_flow")

print("加载训练集...")
df = pd.read_parquet(train_path)
print(f"  原始: {len(df)} 行 x {len(df.columns)} 列")

# 确保 date 是 datetime
df["date"] = pd.to_datetime(df["date"])

# 提取 code6
df["code6"] = df["code"].str[:6]

# 获取有资金流文件的股票
ff_files = {f.split(".")[0]: f for f in os.listdir(ff_dir) if f.endswith(".parquet")}
print(f"资金流文件: {len(ff_files)} 只")

# 遍历每只股票，加载资金流并 merge
fund_data = []
codes_done = set()
for code6, ff_name in ff_files.items():
    if code6 not in df["code6"].values:
        continue
    codes_done.add(code6)
    
    ff = pd.read_parquet(os.path.join(ff_dir, ff_name))
    # 日期格式兼容: YYYY-MM-DD 和 YYYYMMDD
    ff["date"] = pd.to_datetime(ff["date"].astype(str).str.replace("-", ""), format="%Y%m%d", errors="coerce")
    
    # 计算特征: 和 feature_engine.py 的 calc_fund_features 一致
    if "main_force_net" in ff.columns:
        s = ff["main_force_net"].fillna(0).astype(float)
        ff["mf_net_1d"] = s
        ff["mf_net_ma3"] = s.rolling(3).mean()
        # Z-score
        rm = s.rolling(20).mean()
        rs = s.rolling(20).std().replace(0, np.nan)
        ff["mf_net_z"] = ((s - rm) / rs).fillna(0)
    
    # 防未来泄露: 资金流是 T 日全天数据，T+1 日开盘才能用
    ff["use_date"] = ff["date"] + pd.Timedelta(days=1)
    
    fund_data.append(ff[["use_date", "mf_net_1d", "mf_net_ma3", "mf_net_z"]].assign(code6=code6))

print(f"匹配到股票: {len(codes_done)} 只")

# 合并所有资金流数据
all_fund = pd.concat(fund_data, ignore_index=True)
all_fund = all_fund.rename(columns={"use_date": "date"})

# Merge 到训练集
df = df.drop(columns=["mf_net_1d", "mf_net_ma3", "mf_net_z"], errors="ignore")
df = df.merge(all_fund, on=["code6", "date"], how="left")
df = df.drop(columns=["code6"])

# 检查结果
for col in ["mf_net_1d", "mf_net_ma3", "mf_net_z"]:
    n_nan = df[col].isna().sum()
    print(f"{col}: NaN={n_nan} ({n_nan/len(df)*100:.1f}%)")

# 保存
df.to_parquet(train_path, index=False)
print(f"保存: {train_path}")
print("完成!")
