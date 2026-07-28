"""
SQLite persistence layer for the quant-strategy Web backend.

All training results, daily returns, feature importance, and settings are
stored in a single SQLite database at  Web/data/web.db (relative to the
Web directory).  This module can be imported from anywhere; call
init_db() once on startup to ensure the schema exists.

Usage (from Web/backend/ or Web/)::

    from backend.database import init_db, save_run, get_runs, get_run_detail
    init_db()
    run_id = save_run(params, results_dict)
    runs = get_runs(limit=10)
    detail = get_run_detail(run_id)
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Resolve the database path relative to the **Web** directory (one level up
# from backend/).
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent          # .../Web/backend
_WEB_DIR = _BACKEND_DIR.parent                           # .../Web
DB_PATH = _WEB_DIR / "data" / "web.db"


def _now() -> str:
    """UTC ISO timestamp string for created_at."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _conn():
    """Yield a sqlite3 connection with WAL mode, foreign keys enabled."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Schema ────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT    NOT NULL,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    train_start      TEXT    NOT NULL,
    test_start       TEXT    NOT NULL,
    buy_pct          REAL    NOT NULL,
    sell_pct         REAL    NOT NULL,
    slip_pct         REAL    NOT NULL,
    top_n            INTEGER NOT NULL,
    n_features       INTEGER NOT NULL,
    n_days           INTEGER NOT NULL,
    sharpe_raw       REAL,
    sharpe_sampled   REAL,
    max_dd           REAL,
    win_rate         REAL,
    ic_mean          REAL,
    annual_return    REAL,
    elapsed_seconds  REAL,
    model_params     TEXT,   -- JSON object
    status           TEXT    NOT NULL DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS daily_returns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    date        TEXT    NOT NULL,
    top_ret     REAL    NOT NULL,
    ic          REAL    NOT NULL,
    cum_return  REAL
);

CREATE TABLE IF NOT EXISTS feature_importance (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    feature  TEXT    NOT NULL,
    gain     REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_holdings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    date        TEXT    NOT NULL,
    rank        INTEGER NOT NULL,
    code        TEXT    NOT NULL,
    pred_score  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_daily_returns_run ON daily_returns(run_id);
CREATE INDEX IF NOT EXISTS idx_feature_importance_run ON feature_importance(run_id);
CREATE INDEX IF NOT EXISTS idx_daily_holdings_run ON daily_holdings(run_id);
"""


# ── Public API ────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables and indices if they do not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as db:
        db.executescript(SCHEMA_SQL)
        # 迁移: 旧库补新列 (幂等, 列已存在则忽略)
        for col, ctype in [
            ("universe_source", "TEXT"),
            ("test_end", "TEXT"),
            ("min_train_days", "INTEGER"),
            ("sample_interval", "INTEGER"),
        ]:
            try:
                db.execute(f"ALTER TABLE runs ADD COLUMN {col} {ctype}")
            except sqlite3.OperationalError:
                pass


def _coerce_real(v) -> float:
    """把 None / NaN 归一为 0.0, 避免 SQLite 将 NaN 当作 NULL 拒绝 NOT NULL 列。"""
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


# ── Runs ──────────────────────────────────────────────────────────────────

def save_run(
    params: dict[str, Any],
    results: dict[str, Any],
    *,
    status: str = "completed",
) -> int:
    """
    Persist a training run and its daily returns / feature importance.

    Parameters
    ----------
    params : dict
        Must contain at least: name, train_start, test_start, buy_pct,
        sell_pct, slip_pct, top_n, n_features, model_params.
        train_start / test_start can be str ("YYYY-MM-DD") or date objects.
    results : dict
        Must contain: n_days, sharpe_raw, sharpe_sampled, max_dd, win_rate,
        ic_mean, annual_return, elapsed_seconds, daily_returns (list of
        dicts with date/top_ret/ic/cum_return), feature_importance (list
        of dicts with feature/gain).
    status : str
        One of 'completed', 'failed', 'running'.

    Returns
    -------
    int
        The auto-generated run id.
    """
    _ts_start = str(params.get("train_start", ""))
    _ts_test  = str(params.get("test_start", ""))
    _ts_end   = str(params.get("test_end", "") or "")
    _mp_json  = json.dumps(params.get("model_params", {}), ensure_ascii=False)

    with _conn() as db:
        cur = db.execute(
            """
            INSERT INTO runs (name, created_at, train_start, test_start, test_end,
                              buy_pct, sell_pct, slip_pct, top_n,
                              n_features, n_days, sharpe_raw, sharpe_sampled,
                              max_dd, win_rate, ic_mean, annual_return,
                              elapsed_seconds, universe_source,
                              min_train_days, sample_interval,
                              model_params, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                params.get("name", "unnamed"),
                _now(),
                _ts_start,
                _ts_test,
                _ts_end,
                float(params.get("buy_pct", 0)),
                float(params.get("sell_pct", 0)),
                float(params.get("slip_pct", 0)),
                int(params.get("top_n", 3)),
                int(params.get("n_features", 0)),
                int(results.get("n_days", 0)),
                _coerce_real(results.get("sharpe_raw")),
                _coerce_real(results.get("sharpe_sampled")),
                _coerce_real(results.get("max_dd")),
                _coerce_real(results.get("win_rate")),
                _coerce_real(results.get("ic_mean")),
                _coerce_real(results.get("annual_return")),
                _coerce_real(results.get("elapsed_seconds")),
                str(params.get("universe_source", "关注圈")),
                int(params.get("min_train_days", 250)),
                int(params.get("sample_interval", 5)),
                _mp_json,
                status,
            ),
        )
        run_id = cur.lastrowid

        # daily_returns
        for d in results.get("daily_returns", []):
            db.execute(
                "INSERT INTO daily_returns (run_id, date, top_ret, ic, cum_return) VALUES (?,?,?,?,?)",
                (run_id, str(d.get("date", "")), _coerce_real(d.get("top_ret")),
                 _coerce_real(d.get("ic")), _coerce_real(d.get("cum_return"))),
            )

        # daily_holdings
        for d in results.get("daily_returns", []):
            holdings = d.get("holdings", [])
            for rank_idx, h in enumerate(holdings, start=1):
                db.execute(
                    "INSERT INTO daily_holdings (run_id, date, rank, code, pred_score) VALUES (?,?,?,?,?)",
                    (run_id, str(d.get("date", "")), rank_idx,
                     str(h.get("code", "")), float(h.get("pred_score", 0))),
                )

        # feature_importance
        for fi in results.get("feature_importance", []):
            db.execute(
                "INSERT INTO feature_importance (run_id, feature, gain) VALUES (?,?,?)",
                (run_id, str(fi.get("feature", "")), float(fi.get("gain", 0))),
            )

    return run_id


def get_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent runs (summary only, no daily / feature rows)."""
    with _conn() as db:
        rows = db.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_run_detail(run_id: int) -> Optional[dict[str, Any]]:
    """Return a run with its daily_returns and feature_importance."""
    with _conn() as db:
        run = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            return None
        result = dict(run)
        result["daily_returns"] = [
            dict(r) for r in db.execute(
                "SELECT * FROM daily_returns WHERE run_id = ? ORDER BY date", (run_id,)
            ).fetchall()
        ]
        result["feature_importance"] = [
            dict(r) for r in db.execute(
                "SELECT * FROM feature_importance WHERE run_id = ? ORDER BY gain DESC", (run_id,)
            ).fetchall()
        ]
        result["daily_holdings"] = [
            dict(r) for r in db.execute(
                "SELECT * FROM daily_holdings WHERE run_id = ? ORDER BY date, rank", (run_id,)
            ).fetchall()
        ]
    return result


def set_run_status(run_id: int, status: str) -> None:
    """Update the status of a run (e.g. to 'running' or 'failed')."""
    with _conn() as db:
        db.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))


def delete_run(run_id: int) -> bool:
    """Delete a run and its cascaded daily_returns / feature_importance."""
    with _conn() as db:
        cur = db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cur.rowcount > 0


# ── Settings ──────────────────────────────────────────────────────────────

def save_settings(key: str, value: str) -> None:
    """Upsert a key-value setting."""
    with _conn() as db:
        db.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def get_settings(key: str) -> Optional[str]:
    """Retrieve a setting value, or None if not found."""
    with _conn() as db:
        row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def get_all_settings() -> dict[str, str]:
    """Return all settings as a dict."""
    with _conn() as db:
        rows = db.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


def save_last_train_params(params: dict[str, Any]) -> None:
    """Convenience: store the last training params as a JSON setting."""
    save_settings("last_train_params", json.dumps(params, ensure_ascii=False, default=str))


def get_last_train_params() -> Optional[dict[str, Any]]:
    """Convenience: retrieve last training params."""
    raw = get_settings("last_train_params")
    return json.loads(raw) if raw else None
