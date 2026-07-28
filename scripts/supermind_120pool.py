"""
SuperMind 120池实测: 训练→序列化→WSS上传→预测→Top股票
"""
import asyncio
import json
import uuid
import ssl
import gzip
import base64 as b64_std
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

def _latest_path():
    candidates = sorted(DATA_DIR.glob("processed/training_data_v*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"未找到 training_data_v*.parquet @ {DATA_DIR}")
    return candidates[-1]

TRAIN_PATH = _latest_path()
CONFIG_PATH = PROJECT_DIR / "jupyter_config.json"

LABEL_RAW = "fwd_1d_ret"
LABEL = "fwd_1d_excess"
LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret", "fwd_1d_excess"}
_EXCLUDED = {"mtss_1d", "mtss_z", "mtss_1d_ma5", "mtss_z_ma5", "mtss_1d_ma20", "mtss_z_ma20"}
_FUNDA_RAW = ["pe", "pb", "revenue", "profit", "eps", "bps", "debt_ratio",
              "gross_margin", "roe", "total_assets"]
_EXCLUDED.update(f"{r}_{s}" for r in _FUNDA_RAW for s in ("ma5", "ma20"))

CHUNK_SIZE = 15000  # WSS msg size limit
OUTPUT_MARKER = "___CHUNK_OK___"


def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross") and f not in _EXCLUDED


# ═══════════════════════════════════════════════════════════
# PHASE 1: Local training
# ═══════════════════════════════════════════════════════════
def train_model():
    print("=" * 60)
    print("PHASE 1: Training LightGBM on 120-pool (daily expanding)")
    print("=" * 60)

    df = pd.read_parquet(TRAIN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[LABEL_RAW])

    # 截面demean标签
    df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

    print("  Rows: %d, Codes: %d, Dates: %s ~ %s" % (
        len(df), df["code"].nunique(),
        df["date"].min().date(), df["date"].max().date()))

    # 特征列表
    all_cols = [c for c in df.columns if c not in SKIP_COLS and is_valid_feat(c)]
    features = [f for f in all_cols if f not in LEAKAGE_FEATS]
    print("  Features: %d (excluded leak: %s)" % (len(features), sorted(LEAKAGE_FEATS)))

    # 每日扩展训练: 用所有历史数据训练
    latest_date = df["date"].max()
    print("  Latest date in data: %s" % latest_date.date())
    print("  Training on ALL data up to %s" % latest_date.date())

    X = df[features].copy()
    y = df[LABEL].copy()

    # Fill NaN with ffill per stock, then bfill, then median
    medians = {}
    for c in features:
        if X[c].isna().any():
            # Per-stock ffill
            X[c] = df.groupby("code")[c].transform(lambda s: s.ffill().bfill().fillna(s.median()))
            med = X[c].median()
            medians[c] = float(med) if pd.notna(med) else 0.0
        else:
            medians[c] = 0.0

    print("  NaN features filled: %d" % sum(1 for v in medians.values() if v != 0))

    model = lgb.LGBMRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        num_leaves=15, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=50, random_state=42, n_jobs=32, verbosity=-1
    )
    model.fit(X, y)
    print("  Model trained: %d trees" % model.booster_.num_trees())

    # Feature importance top 10
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("  Top 10 features:")
    for i, (k, v) in enumerate(imp.head(10).items()):
        print("    %2d. %-30s %d" % (i + 1, k, int(v)))

    # 最新一天的特征矩阵 (120 stocks)
    latest_mask = df["date"] == latest_date
    df_latest = df[latest_mask][["code"] + features].copy()
    for c in features:
        if df_latest[c].isna().any():
            df_latest[c] = df_latest[c].fillna(medians.get(c, 0))
    df_latest = df_latest.reset_index(drop=True)

    print("  Latest day (%s): %d stocks" % (latest_date.date(), len(df_latest)))

    # 如果有缺失的股票，用该股票最近一天的数据补
    all_codes = sorted(df["code"].unique())
    missing = set(all_codes) - set(df_latest["code"])
    if missing:
        print("  Missing %d stocks, backfilling..." % len(missing))
        for code in missing:
            code_df = df[df["code"] == code].sort_values("date")
            if len(code_df) > 0:
                latest_row = code_df.iloc[-1]
                row = {"code": code}
                for c in features:
                    row[c] = latest_row[c] if pd.notna(latest_row[c]) else medians.get(c, 0)
                df_latest = pd.concat([df_latest, pd.DataFrame([row])], ignore_index=True)
        print("  After backfill: %d stocks" % len(df_latest))

    # 按code排序，保证一致性
    df_latest = df_latest.sort_values("code").reset_index(drop=True)

    return model, features, medians, df_latest, latest_date


# ═══════════════════════════════════════════════════════════
# PHASE 2: Serialize for SuperMind
# ═══════════════════════════════════════════════════════════
def serialize_model(model, features, medians, df_latest, latest_date):
    print()
    print("=" * 60)
    print("PHASE 2: Serializing model + data for SuperMind")
    print("=" * 60)

    # Model -> text -> gzip -> base64
    model_str = model.booster_.model_to_string()
    print("  Model text: %d chars" % len(model_str))

    model_bytes = model_str.encode("utf-8")
    compressed = gzip.compress(model_bytes, compresslevel=9)
    model_b64 = b64_std.b64encode(compressed).decode("ascii")
    print("  Compressed: %d -> %d bytes (%.1f%%)" % (
        len(model_bytes), len(compressed),
        100 * len(compressed) / len(model_bytes)))

    # Feature matrix: 120 stocks x N features
    codes_list = df_latest["code"].tolist()
    feat_matrix = df_latest[features].values.astype(np.float64)
    print("  Feature matrix: %d x %d" % (feat_matrix.shape[0], feat_matrix.shape[1]))

    # Serialize feature matrix: gzip + base64
    feat_bytes = feat_matrix.tobytes()
    feat_compressed = gzip.compress(feat_bytes, compresslevel=9)
    feat_b64 = b64_std.b64encode(feat_compressed).decode("ascii")
    print("  Feature bytes: %d -> %d (%.1f%%)" % (
        len(feat_bytes), len(feat_compressed),
        100 * len(feat_compressed) / len(feat_bytes)))

    # Save to disk for reference
    output = {
        "model_b64": model_b64,
        "features": features,
        "medians": medians,
        "codes": codes_list,
        "feat_b64": feat_b64,
        "feat_shape": list(feat_matrix.shape),
        "latest_date": str(latest_date.date()),
    }
    out_path = DATA_DIR / "processed" / "supermind_120pool.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print("  Saved: %s" % out_path)

    # Chunk model b64 for WSS upload
    model_chunks = [model_b64[i:i + CHUNK_SIZE] for i in range(0, len(model_b64), CHUNK_SIZE)]
    print("  Model chunks: %d" % len(model_chunks))

    # Chunk feature matrix b64
    feat_chunks = [feat_b64[i:i + CHUNK_SIZE] for i in range(0, len(feat_b64), CHUNK_SIZE)]
    print("  Feature chunks: %d" % len(feat_chunks))

    return model_b64, model_chunks, feat_b64, feat_chunks, codes_list


# ═══════════════════════════════════════════════════════════
# PHASE 3: SuperMind WSS execution
# ═══════════════════════════════════════════════════════════
async def supermind_predict(model_b64, model_chunks, feat_b64, feat_chunks,
                            features, codes_list):
    print()
    print("=" * 60)
    print("PHASE 3: SuperMind WSS upload + predict")
    print("=" * 60)

    # Load config
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    token = cfg["token"]
    user_id = cfg["user"]

    # Get kernel
    t0 = datetime.now()
    import aiohttp

    ssl_ctx = ssl.create_default_context()
    timeout_cfg = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        # List kernels
        hubs_api = cfg["hub"] + "/user/%s/api/kernels" % user_id
        headers = {"Authorization": "token %s" % token}
        async with session.get(hubs_api, headers=headers, ssl=ssl_ctx) as resp:
            kernels = await resp.json()
        print("  Kernels: %d" % len(kernels))

        # Find running or create new
        kid = None
        for k in kernels:
            if k.get("execution_state") == "idle":
                kid = k["id"]
                break
        if not kid:
            # Create new
            async with session.post(hubs_api, headers=headers, ssl=ssl_ctx) as resp:
                new_k = await resp.json()
            kid = new_k["id"]
            print("  Created kernel: %s" % kid[:16])
        else:
            print("  Using kernel: %s" % kid[:16])

    WSS_URL = "wss://supermind.10jqka.com.cn/notebook/user/%s/api/kernels/%s/channels?token=%s" % (
        user_id, kid, token)

    # ── Helper: execute code on WSS ──
    async def execute_code(code, name, timeout=60):
        msg_id = uuid.uuid4().hex
        session_id = uuid.uuid4().hex
        results = {"stdout": [], "error": None, "status": "unknown"}
        done = asyncio.Event()

        ws_timeout = aiohttp.ClientTimeout(total=timeout, sock_connect=15)
        async with aiohttp.ClientSession(timeout=ws_timeout) as ws_session:
            async with ws_session.ws_connect(WSS_URL, ssl=ssl_ctx) as ws:
                exec_msg = {
                    "header": {"msg_id": msg_id, "username": "agent",
                               "session": session_id, "msg_type": "execute_request",
                               "version": "5.3"},
                    "parent_header": {}, "metadata": {},
                    "content": {"code": code, "silent": False, "store_history": True,
                                "user_expressions": {}, "allow_stdin": False,
                                "stop_on_error": False},
                    "channel": "shell", "buffers": []
                }
                await ws.send_json(exec_msg)

                while not done.is_set():
                    try:
                        msg = await ws.receive_json(timeout=timeout)
                        hdr = msg.get("header", {})
                        mtype = msg.get("msg_type", "") or hdr.get("msg_type", "")
                        pid = msg.get("parent_header", {}).get("msg_id", "")
                        if pid and pid != msg_id:
                            continue
                        if mtype == "stream":
                            results["stdout"].append(msg.get("content", {}).get("text", ""))
                        elif mtype == "error":
                            results["error"] = msg.get("content", {})
                            print("  !! %s: %s" % (
                                results["error"].get("ename", ""),
                                str(results["error"].get("evalue", ""))[:120]))
                        elif mtype == "execute_reply":
                            results["status"] = msg.get("content", {}).get("status", "")
                        elif mtype == "status":
                            st = msg.get("content", {}).get("execution_state", "")
                            if st == "idle":
                                done.set()
                    except asyncio.TimeoutError:
                        print("  !! TIMEOUT on %s" % name)
                        break
                    except Exception as e:
                        print("  !! WS error on %s: %s" % (name, str(e))[:80])
                        break
        return "".join(results["stdout"]).strip()

    # ── Step 1: Init ──
    print("\n[Step 1] Init variables...")
    out = await execute_code("CHUNKS = []\nprint('%s')" % OUTPUT_MARKER, "Init", timeout=15)
    ok = OUTPUT_MARKER in out
    print("  %s" % ("OK" if ok else "FAIL: " + out[:80]))

    # ── Step 2: Upload model chunks ──
    total_upload = len(model_chunks)
    print("\n[Step 2] Upload model (%d chunks)..." % total_upload)
    for i, chunk in enumerate(model_chunks):
        code = 'CHUNKS.append("%s")\nprint("%s")' % (chunk, OUTPUT_MARKER)
        out = await execute_code(code, "model_%d" % i, timeout=20)
        ok = OUTPUT_MARKER in out
        if not ok and i > 0:
            print("  [%d/%d] FAIL, retrying..." % (i + 1, total_upload))
            await asyncio.sleep(1)
            out = await execute_code(code, "model_%d_retry" % i, timeout=20)
            ok = OUTPUT_MARKER in out
        if i % 5 == 0 or i == total_upload - 1:
            print("  [%d/%d] OK" % (i + 1, total_upload))
        if not ok:
            print("  [%d/%d] FAILED after retry!" % (i + 1, total_upload))
            return

    # ── Step 3: Load model ──
    print("\n[Step 3] Loading model...")
    load_code = """
import base64, gzip, lightgbm as lgb

MODEL_B64 = "".join(CHUNKS)
print("Joined model b64: " + str(len(MODEL_B64)) + " chars")

raw = base64.b64decode(MODEL_B64)
decomp = gzip.decompress(raw)
model_str = decomp.decode("utf-8")
print("Model str: " + str(len(model_str)) + " chars")

# Try model_str (lgb 4.x) then tempfile (lgb 2.x)
model = None
try:
    model = lgb.Booster(model_str=model_str)
    print("Loaded via model_str")
except TypeError:
    try:
        import tempfile as tf
        f = tf.NamedTemporaryFile(delete=False, suffix=".txt")
        f.write(model_str.encode("utf-8"))
        f.close()
        model = lgb.Booster(model_file=f.name)
        print("Loaded via tempfile")
    except Exception as e2:
        print("Fallback FAIL: " + str(e2)[:60])

if model is not None and hasattr(model, "num_trees"):
    print("Model loaded: " + str(model.num_trees()) + " trees")
else:
    print("Model: loaded (unknown trees)")
print("LOAD_OK")
"""
    out = await execute_code(load_code, "Load model", timeout=30)
    print("  " + out.replace("\n", "\n  "))
    if "LOAD_OK" not in out:
        print("!!! Model load failed!")
        return

    # ── Step 4: Upload feature matrix chunks ──
    print("\n[Step 4] Clear CHUNKS, upload feature matrix (%d chunks)..." % len(feat_chunks))
    out = await execute_code("CHUNKS = []\nprint('%s')" % OUTPUT_MARKER, "Clear chunks", timeout=10)

    for i, chunk in enumerate(feat_chunks):
        code = 'CHUNKS.append("%s")\nprint("%s")' % (chunk, OUTPUT_MARKER)
        out = await execute_code(code, "feat_%d" % i, timeout=20)
        ok = OUTPUT_MARKER in out
        if not ok and i > 0:
            await asyncio.sleep(1)
            out = await execute_code(code, "feat_%d_retry" % i, timeout=20)
            ok = OUTPUT_MARKER in out
        if i % 5 == 0 or i == len(feat_chunks) - 1:
            print("  [%d/%d] OK" % (i + 1, len(feat_chunks)))
        if not ok:
            print("  [%d/%d] FAILED!" % (i + 1, len(feat_chunks)))
            return

    # ── Step 5: Decompress feature matrix ──
    print("\n[Step 5] Decompressing feature matrix...")
    feat_shape = [len(codes_list), len(features)]
    load_feat_code = ('import base64, gzip, numpy as np\n'
        'FEAT_B64 = "".join(CHUNKS)\n'
        'print("Joined feat b64: %%d chars" %% len(FEAT_B64))\n\n'
        'raw = base64.b64decode(FEAT_B64)\n'
        'decomp = gzip.decompress(raw)\n'
        'X = np.frombuffer(decomp, dtype=np.float64).reshape(%d, %d)\n'
        'print("Feature matrix: %%d x %%d" %% X.shape)\n'
        'print("FEAT_OK")\n') % tuple(feat_shape)

    out = await execute_code(load_feat_code, "Load features", timeout=30)
    print("  " + out.replace("\n", "\n  "))
    if "FEAT_OK" not in out:
        print("!!! Feature load failed!")
        return

    # ── Step 6: Predict ──
    print("\n[Step 6] Running predictions on %d stocks..." % len(codes_list))

    codes_json = json.dumps(codes_list)
    pred_code = """
import numpy as np

CODES = %s

# Predict
preds = model.predict(X)
print("Predictions shape: %%s" %% str(preds.shape))
print("Min=%%.6f Max=%%.6f Mean=%%.6f Std=%%.6f" %% (preds.min(), preds.max(), preds.mean(), preds.std()))

# Top 10
top_idx = np.argsort(preds)[-10:][::-1]
print("\\n=== TOP 10 PREDICTIONS ===")
for rk, i in enumerate(top_idx):
    print("  Rank %%2d: %%s  score=%%+.6f" %% (rk+1, CODES[i], preds[i]))

# Bottom 10
bot_idx = np.argsort(preds)[:10]
print("\\n=== BOTTOM 10 PREDICTIONS ===")
for rk, i in enumerate(bot_idx):
    print("  Rank %%2d: %%s  score=%%+.6f" %% (rk+1, CODES[i], preds[i]))

# Score distribution
p10, p25, p50, p75, p90 = np.percentile(preds, [10, 25, 50, 75, 90])
print("\\nPercentiles: P10=%%+.6f P25=%%+.6f P50=%%+.6f P75=%%+.6f P90=%%+.6f" %% (p10, p25, p50, p75, p90))

# Long/Short ratio
long = (preds > 0).sum()
short = (preds <= 0).sum()
print("Long/Short: %%d / %%d" %% (long, short))
print("PREDICT_OK")
""" % codes_json

    out = await execute_code(pred_code, "Predict", timeout=30)
    print("  " + out.replace("\n", "\n  "))

    elapsed = (datetime.now() - t0).total_seconds()
    print()
    print("=" * 60)
    print("Completed in %.0fs" % elapsed)
    if "PREDICT_OK" in out:
        print("=== 120 POOL SUPERMIND PREDICTION DONE! ===")
    else:
        print("!!! Prediction may have failed!")
    print("=" * 60)

    return out


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
async def main():
    # Phase 1: Train
    model, features, medians, df_latest, latest_date = train_model()

    # Phase 2: Serialize
    model_b64, model_chunks, feat_b64, feat_chunks, codes_list = serialize_model(
        model, features, medians, df_latest, latest_date)

    # Phase 3: SuperMind
    await supermind_predict(model_b64, model_chunks, feat_b64, feat_chunks,
                            features, codes_list)


if __name__ == "__main__":
    asyncio.run(main())
