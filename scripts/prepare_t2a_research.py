import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from run_overnight_research import resources, stamp, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    status = ROOT / "prep_t2a_status.json"
    env = dict(os.environ, QUANT_DATA_DIR=str(ROOT / "data"), QUANT_MODE="backtest", NW="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_dir = ROOT / "logs"
    matrix = pd.read_parquet(ROOT / "data/processed/training_data_pit_v24_tick1.parquet", columns=["date"])
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(matrix.date).unique()))
    end, required = calendar[-1], calendar[-2]
    panel = ROOT / "data/processed/min1"
    files = sorted(panel.glob("*.parquet"))
    if not files:
        raise RuntimeError("expected existing minute panel snapshot")
    last = pd.Timestamp(files[-1].stem)
    while True:
        load, memory = resources()
        if load + 2 <= 100 and memory >= 128:
            break
        write_json(status, {"state": "waiting_resources", "updated_at": stamp(), "load": load, "available_gb": memory})
        print("WAIT preparation", load, memory, flush=True)
        time.sleep(60)
    if last < required:
        start = (last + timedelta(days=1)).strftime("%Y%m%d")
        write_json(status, {"state": "extracting_recent_minutes", "updated_at": stamp(), "start": start, "end": required})
        with (log_dir / "t2a_extract_recent.log").open("x") as log:
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/build_minute_bars.py"), start, required.strftime("%Y%m%d")],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=10800)
        missing = [str(d.date()) for d in calendar[(calendar > last) & (calendar <= required)] if not (panel / (d.strftime("%Y%m%d") + ".parquet")).exists()]
        if missing:
            raise RuntimeError("minute packages unavailable for " + ",".join(missing))
    write_json(status, {"state": "building_features", "updated_at": stamp(), "test_end": end})
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts/build_t2a_augmented.py")], cwd=ROOT, env=env, check=True, timeout=14400)
    write_json(status, {"state": "training_queue", "updated_at": stamp(), "test_end": end})
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts/run_overnight_research.py"), "t2a", "--workers", "2", "--max-load", "100", "--min-memory-gb", "128"],
                   cwd=ROOT, env=env, check=True)
    write_json(status, {"state": "completed", "updated_at": stamp(), "test_end": end})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(ROOT / "prep_t2a_status.json", {"state": "failed", "updated_at": stamp(), "error": str(error)})
        raise
