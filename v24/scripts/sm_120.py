"""
120池 SuperMind WSS预测 — 用 smlogin 封装
"""
import sys, asyncio, json, base64, gzip
from pathlib import Path

sys.path.insert(0, r"C:\Users\admin\.workbuddy\skills\smlogin\scripts")
from supermind_login import SuperMindSession

DATA_DIR = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")
SN_PATH = DATA_DIR / "processed" / "supermind_120pool.json"
MARKER = "___OK___"
CHUNK_SIZE = 15000


async def exec_on_ws(ws, code):
    """Send code via raw WS and collect output"""
    import aiohttp
    msg_id = "exec_" + str(hash(code) % 100000)
    exec_msg = {
        "header": {"msg_id": msg_id, "username": "agent",
                   "session": "s1", "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": False},
        "channel": "shell", "buffers": []
    }
    await ws.send_json(exec_msg)
    stdout = []
    error = None
    while True:
        try:
            msg = await ws.receive_json(timeout=60)
        except asyncio.TimeoutError:
            break
        mtype = msg.get("msg_type", "") or msg.get("header", {}).get("msg_type", "")
        pid = msg.get("parent_header", {}).get("msg_id", "")
        if pid and pid != msg_id:
            continue
        if mtype == "stream":
            stdout.append(msg.get("content", {}).get("text", ""))
        elif mtype == "error":
            error = msg.get("content", {})
        elif mtype == "status":
            if msg.get("content", {}).get("execution_state") == "idle":
                break
        elif mtype == "execute_reply":
            pass
    return "".join(stdout).strip(), error


async def main():
    print("Loading serialized model + features...")
    with open(SN_PATH) as f:
        data = json.load(f)
    model_b64 = data["model_b64"]
    feat_b64 = data["feat_b64"]
    features = data["features"]
    codes = data["codes"]
    n_stocks = len(codes)
    n_feats = len(features)
    print("  Model b64: %d chars, Feat b64: %d chars" % (len(model_b64), len(feat_b64)))
    print("  Codes: %d, Features: %d" % (n_stocks, n_feats))

    # Chunk
    model_chunks = [model_b64[i:i + CHUNK_SIZE] for i in range(0, len(model_b64), CHUNK_SIZE)]
    feat_chunks = [feat_b64[i:i + CHUNK_SIZE] for i in range(0, len(feat_b64), CHUNK_SIZE)]
    print("  Model chunks: %d, Feat chunks: %d" % (len(model_chunks), len(feat_chunks)))

    # Connect via smlogin
    cfg_path = r"D:\myAI\WorkBuddy-workspace\quant-strategy\jupyter_config.json"
    print("\nConnecting SuperMind...")

    async with SuperMindSession(config_path=cfg_path, cleanup=True) as sm:
        await sm.connect()
        print("  Kernel: %s" % sm.kernel_id[:16])

        ws = sm.ws  # raw WS for custom protocol

        # Step 1: Init
        out, err = await exec_on_ws(ws, "CHUNKS = []\nprint('%s')" % MARKER)
        print("  [Init] %s" % ("OK" if MARKER in out else "FAIL: " + str(out)[:60]))

        # Step 2: Upload model chunks
        print("  [Upload model] %d chunks..." % len(model_chunks))
        for i, ch in enumerate(model_chunks):
            code_str = 'CHUNKS.append("%s")\nprint("%s")' % (ch, MARKER)
            out, err = await exec_on_ws(ws, code_str)
            ok = MARKER in out
            if not ok:
                await asyncio.sleep(1)
                out, err = await exec_on_ws(ws, code_str)
                ok = MARKER in out
            if i % 5 == 0:
                print("    [%d/%d] %s" % (i + 1, len(model_chunks), "OK" if ok else "FAIL"))
            if not ok:
                print("    [FATAL] Chunk %d upload failed!" % i)
                return

        # Step 3: Load model
        load_model = """
import base64, gzip, lightgbm as lgb
MODEL_B64 = "".join(CHUNKS)
print("Joined: " + str(len(MODEL_B64)))
raw = base64.b64decode(MODEL_B64)
decomp = gzip.decompress(raw)
model_str = decomp.decode("utf-8")
model = lgb.Booster(model_str=model_str)
print("Model: " + str(model.num_trees()) + " trees")
print("LOAD_OK")
"""
        out, err = await exec_on_ws(ws, load_model)
        print("  [Load model]")
        for line in out.split("\n"):
            if line.strip():
                print("    " + line)

        if "LOAD_OK" not in out:
            print("  [FATAL] Model load failed!")
            return

        # Step 4: Upload feature chunks
        out, err = await exec_on_ws(ws, "CHUNKS = []\nprint('%s')" % MARKER)
        print("  [Upload feat] %d chunks..." % len(feat_chunks))
        for i, ch in enumerate(feat_chunks):
            code_str = 'CHUNKS.append("%s")\nprint("%s")' % (ch, MARKER)
            out, err = await exec_on_ws(ws, code_str)
            ok = MARKER in out
            if not ok:
                await asyncio.sleep(1)
                out, err = await exec_on_ws(ws, code_str)
                ok = MARKER in out
            if i % 5 == 0:
                print("    [%d/%d] %s" % (i + 1, len(feat_chunks), "OK" if ok else "FAIL"))
            if not ok:
                print("    [FATAL] Chunk %d failed!" % i)
                return

        # Step 5: Decompress feature matrix
        load_feat = """
import base64, gzip, numpy as np
FEAT_B64 = "".join(CHUNKS)
raw = base64.b64decode(FEAT_B64)
decomp = gzip.decompress(raw)
X = np.frombuffer(decomp, dtype=np.float64).reshape(%d, %d)
print("Feature matrix: %%d x %%d" %% X.shape)
print("FEAT_OK")
""" % (n_stocks, n_feats)

        out, err = await exec_on_ws(ws, load_feat)
        print("  [Load feat]")
        for line in out.split("\n"):
            if line.strip():
                print("    " + line)
        if "FEAT_OK" not in out:
            print("  [FATAL] Feature load failed!")
            return

        # Step 6: Predict
        codes_json = json.dumps(codes)
        predict_code = """
import numpy as np
CODES = %s
preds = model.predict(X)
print("Preds: n=%%d min=%%+.6f max=%%+.6f mean=%%+.6f std=%%+.6f" % (len(preds), preds.min(), preds.max(), preds.mean(), preds.std()))

top10 = np.argsort(preds)[-10:][::-1]
print("\\n=== TOP 10 ===")
for rk,i in enumerate(top10):
    print("  #%%d  %%s  score=%%+.6f" % (rk+1, CODES[i], preds[i]))

bot10 = np.argsort(preds)[:10]
print("\\n=== BOTTOM 10 ===")
for rk,i in enumerate(bot10):
    print("  #%%d  %%s  score=%%+.6f" % (rk+1, CODES[i], preds[i]))

p10,p25,p50,p75,p90 = np.percentile(preds, [10,25,50,75,90])
print("\\nPercentiles: P10=%%+.6f P25=%%+.6f P50=%%+.6f P75=%%+.6f P90=%%+.6f" % (p10,p25,p50,p75,p90))
print("Long/Short: %%d/%%d" % ((preds>0).sum(), (preds<=0).sum()))
print("DONE")
""" % codes_json

        out, err = await exec_on_ws(ws, predict_code)
        print("\n" + "=" * 60)
        print("=== SUPERMIND 120 POOL PREDICTION ===")
        print("=" * 60)
        print(out)
        print("=" * 60)

        if "DONE" in out:
            print("\nAll done!")
        else:
            print("\nCompleted with issues.")


if __name__ == "__main__":
    asyncio.run(main())
