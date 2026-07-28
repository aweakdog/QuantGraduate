"""
120池 SuperMind WSS预测 — 直连版 (绕过smlogin hang)
"""
import asyncio, json, uuid, aiohttp
import base64 as b64_std
from pathlib import Path

DATA_DIR = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")
SN_PATH = DATA_DIR / "processed" / "supermind_120pool.json"
# SuperMind API
TOKEN = "26c4f857997e48c1a48ea59916405afb"
USER_ID = "59518255"
# Note: JupyterHub URL format changed. Old: quant.10jqka.com.cn/notebook/user/{uid}
# New: mindgo.10jqka.com.cn/notebook/hub/user/{uid} (server restart needed)
HUB_API = "https://quant.10jqka.com.cn/notebook/hub/api/users/%s" % USER_ID
USER_API = "https://quant.10jqka.com.cn/notebook/user/%s" % USER_ID
KERNEL_API = USER_API + "/api/kernels"
WSS_TMPL = "wss://supermind.10jqka.com.cn/notebook/user/%s/api/kernels/{kid}/channels?token=%s" % (USER_ID, TOKEN)
HEADERS = {"Authorization": "token %s" % TOKEN}
MARKER = "___OK___"
CHUNK_SIZE = 15000


async def get_kernel():
    cto = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=cto, headers=HEADERS) as s:
        # Check server status via hub API
        async with s.get(HUB_API) as r:
            info = await r.json()
        server_ready = info.get("server") is not None and info.get("servers", {}).get("", {}).get("ready")
        if not server_ready:
            print("  Spawning server...")
            async with s.post(HUB_API + "/server") as r:
                print("  Spawn: status=%d" % r.status)
                if r.status not in (201, 202, 409):
                    raise RuntimeError("Spawn failed: %d" % r.status)
            # Wait for ready
            for _ in range(60):
                await asyncio.sleep(2)
                async with s.get(HUB_API) as r:
                    info = await r.json()
                    if info.get("servers", {}).get("", {}).get("ready"):
                        break
        # Now get/create kernel at user API
        for i in range(30):
            await asyncio.sleep(1)
            async with s.get(KERNEL_API) as r:
                if r.status == 200:
                    kernels = await r.json()
                    if isinstance(kernels, list):
                        for k in kernels:
                            if k.get("execution_state") == "idle":
                                return k["id"]
            # Create kernel
            async with s.post(KERNEL_API, json={"name": "python3"}) as r:
                if r.status == 201:
                    k = await r.json()
                    kid = k.get("id", "")
                    if kid:
                        return kid
            if i % 10 == 0:
                print("  Waiting for kernel... (%ds)" % i)
    raise RuntimeError("Failed to get kernel")


async def exec_on_ws(ws, code, timeout=60):
    mid = uuid.uuid4().hex
    msg = {
        "header": {"msg_id": mid, "username": "agent", "session": "s1",
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": False},
        "channel": "shell", "buffers": []
    }
    await ws.send_json(msg)
    stdout = []
    error = None
    while True:
        try:
            m = await ws.receive_json(timeout=10)
        except asyncio.TimeoutError:
            break
        mtype = m.get("msg_type", "") or m.get("header", {}).get("msg_type", "")
        pid = m.get("parent_header", {}).get("msg_id", "")
        if pid and pid != mid:
            continue
        if mtype == "stream":
            stdout.append(m.get("content", {}).get("text", ""))
        elif mtype == "error":
            error = m.get("content", {})
            ename = error.get("ename", "")
            evalue = error.get("evalue", "")
            stdout.append("[ERROR] %s: %s" % (ename, str(evalue)[:200]))
        elif mtype == "status":
            if m.get("content", {}).get("execution_state") == "idle":
                break
    return "".join(stdout).strip(), error


async def main():
    print("Loading serialized model + features...", flush=True)
    with open(SN_PATH) as f:
        data = json.load(f)
    model_b64 = data["model_b64"]
    feat_b64 = data["feat_b64"]
    features = data["features"]
    codes = data["codes"]
    n_stocks = len(codes)
    n_feats = len(features)
    print("  Codes: %d, Features: %d" % (n_stocks, n_feats), flush=True)

    # Chunks
    model_chunks = [model_b64[i:i + CHUNK_SIZE] for i in range(0, len(model_b64), CHUNK_SIZE)]
    feat_chunks = [feat_b64[i:i + CHUNK_SIZE] for i in range(0, len(feat_b64), CHUNK_SIZE)]
    print("  Model chunks: %d, Feat chunks: %d" % (len(model_chunks), len(feat_chunks)), flush=True)

    # Get kernel
    print("\nGetting kernel...", flush=True)
    kid = await get_kernel()
    print("  Kernel: %s" % kid[:16], flush=True)

    # WS connect
    ws_url = WSS_TMPL.format(kid=kid)
    print("Connecting WS...", flush=True)
    cto = aiohttp.ClientTimeout(total=60, sock_connect=15)
    async with aiohttp.ClientSession(timeout=cto, headers=HEADERS) as s:
        async with s.ws_connect(ws_url) as ws:
            print("  Connected!", flush=True)

            # Step 1: Init
            out, err = await exec_on_ws(ws, "CHUNKS = []\nprint('%s')" % MARKER)
            print("  [Init] %s" % ("OK" if MARKER in out else "FAIL"), flush=True)

            # Step 2: Upload model
            print("  [Upload model]", flush=True)
            for i, ch in enumerate(model_chunks):
                out, err = await exec_on_ws(
                    ws, 'CHUNKS.append("%s")\nprint("%s")' % (ch, MARKER))
                ok = MARKER in out
                if not ok:
                    await asyncio.sleep(2)
                    out, err = await exec_on_ws(
                        ws, 'CHUNKS.append("%s")\nprint("%s")' % (ch, MARKER))
                    ok = MARKER in out
                if i % 5 == 0 or i == len(model_chunks) - 1:
                    print("    [%d/%d] %s" % (i + 1, len(model_chunks), "OK" if ok else "FAIL"),
                          flush=True)
                if not ok:
                    print("[FATAL] Model chunk %d failed!" % i, flush=True)
                    return

            # Step 3: Load model (no f-string, no %%)
            print("  [Load model]", flush=True)
            load_model = """import base64, gzip, lightgbm as lgb
MODEL_B64 = "".join(CHUNKS)
print("Joined: " + str(len(MODEL_B64)))
raw = base64.b64decode(MODEL_B64)
decomp = gzip.decompress(raw)
model_str = decomp.decode("utf-8")

# Try multiple loading methods (lgb 4.x: model_str kwarg, lgb 2.x: tempfile)
model = None
try:
    model = lgb.Booster(model_str=model_str)
    print("Loaded via model_str")
except TypeError:
    try:
        import tempfile as tf
        f = tf.NamedTemporaryFile(delete=False, suffix=".txt")
        f.write(model_str.encode())
        f.close()
        model = lgb.Booster(model_file=f.name)
        print("Loaded via tempfile")
    except Exception as e2:
        print("Fallback FAIL: " + str(e2)[:60])

if model is not None and hasattr(model, "num_trees"):
    print("Model: " + str(model.num_trees()) + " trees")
else:
    print("Model: loaded (unknown trees)")
print("LOAD_OK")"""
            out, err = await exec_on_ws(ws, load_model)
            for line in out.split("\n"):
                if line.strip():
                    print("    " + line.strip(), flush=True)
            if "LOAD_OK" not in out:
                print("[FATAL] Model load failed!", flush=True)
                return

            # Step 4: Upload feature chunks
            out, err = await exec_on_ws(ws, "CHUNKS = []\nprint('%s')" % MARKER)
            print("  [Upload feat] %d chunks..." % len(feat_chunks), flush=True)
            for i, ch in enumerate(feat_chunks):
                out, err = await exec_on_ws(
                    ws, 'CHUNKS.append("%s")\nprint("%s")' % (ch, MARKER))
                ok = MARKER in out
                if not ok:
                    await asyncio.sleep(2)
                    out, err = await exec_on_ws(
                        ws, 'CHUNKS.append("%s")\nprint("%s")' % (ch, MARKER))
                    ok = MARKER in out
                if i % 5 == 0 or i == len(feat_chunks) - 1:
                    print("    [%d/%d] %s" % (i + 1, len(feat_chunks), "OK" if ok else "FAIL"),
                          flush=True)
                if not ok:
                    print("[FATAL] Feat chunk %d failed!" % i, flush=True)
                    return

            # Step 5: Load feature matrix
            print("  [Load feat]", flush=True)
            load_feat = """import base64, gzip, numpy as np
FEAT_B64 = "".join(CHUNKS)
raw = base64.b64decode(FEAT_B64)
decomp = gzip.decompress(raw)
X = np.frombuffer(decomp, dtype=np.float64).reshape(%d, %d)
print("Feature matrix: " + str(X.shape[0]) + " x " + str(X.shape[1]))
print("FEAT_OK")""" % (n_stocks, n_feats)

            out, err = await exec_on_ws(ws, load_feat)
            for line in out.split("\n"):
                if line.strip():
                    print("    " + line.strip(), flush=True)
            if "FEAT_OK" not in out:
                print("[FATAL] Feature load failed!", flush=True)
                return

            # Step 6: Predict
            print("\n  [Predict]", flush=True)
            codes_str = json.dumps(codes)
            predict_code = """import numpy as np
TRADE_COST_SELL = 0.0006  # 仅卖出成本，不计买入/滑点

CODES = %s
preds = model.predict(X)
print("Preds: n=" + str(len(preds)) + " min=" + str(round(preds.min(),6)) + " max=" + str(round(preds.max(),6)) + " mean=" + str(round(preds.mean(),6)) + " std=" + str(round(preds.std(),6)))

top10_idx = np.argsort(preds)[-10:][::-1]
print("\\n=== TOP 10 (RAW) ===")
for rk,i in enumerate(top10_idx):
    print("  #" + str(rk+1) + "  " + CODES[i] + "  score=" + str(round(preds[i],6)))

# Top3 trading analysis
top3_idx = top10_idx[:3]
top3_raw = sum(preds[i] for i in top3_idx) / 3.0
top3_buy_cost = 3 * TRADE_COST_SELL
top3_net = top3_raw - top3_buy_cost
top3_full_cycle = top3_raw - top3_buy_cost - 3 * TRADE_COST_SELL

print("\\n=== TOP3 TRADING (cost 0.06%%/side) ===")
print("  Raw mean return: " + str(round(top3_raw*100, 3)) + "%%")
print("  Buy cost (3x): " + str(round(top3_buy_cost*100, 3)) + "%%")
print("  Net (buy only): " + str(round(top3_net*100, 3)) + "%%")
print("  Net (full cycle): " + str(round(top3_full_cycle*100, 3)) + "%%")

bot10_idx = np.argsort(preds)[:10]
print("\\n=== BOTTOM 10 ===")
for rk,i in enumerate(bot10_idx):
    print("  #" + str(rk+1) + "  " + CODES[i] + "  score=" + str(round(preds[i],6)))

p10 = np.percentile(preds, 10)
p25 = np.percentile(preds, 25)
p50 = np.percentile(preds, 50)
p75 = np.percentile(preds, 75)
p90 = np.percentile(preds, 90)
print("\\nPercentiles: P10=" + str(round(p10,6)) + " P25=" + str(round(p25,6)) + " P50=" + str(round(p50,6)) + " P75=" + str(round(p75,6)) + " P90=" + str(round(p90,6)))
long_cnt = (preds > 0).sum()
short_cnt = (preds <= 0).sum()
print("Long/Short: " + str(long_cnt) + "/" + str(short_cnt))

top1_raw = preds[top10_idx[0]]
print("\\nTop1: " + CODES[top10_idx[0]] + " raw=" + str(round(top1_raw*100,3)) + "%% net(entry)=" + str(round((top1_raw-TRADE_COST_SELL)*100,3)) + "%%")
print("DONE")""" % codes_str

            out, err = await exec_on_ws(ws, predict_code)
            print("\n" + "=" * 60)
            print(out)
            print("=" * 60)

            if "DONE" in out:
                print("\n120 pool SuperMind prediction completed!", flush=True)
            else:
                print("\nCompleted with issues.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
