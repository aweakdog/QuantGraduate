import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TICK_CHANNELS = ("tk_spread_bp_xz_ma5", "tk_depth_imb_1_xz_ma5", "tk_ord_amt_ratio_xz_ma5", "tk_ord_net_l_xz_ma5")
CHANNELS = ("log_close", "log_return", "overnight", "intraday", "log_range", "log_volume", "log_amount", *TICK_CHANNELS)


@dataclass
class MarketPanel:
    values: np.ndarray
    members: np.ndarray
    dates: pd.DatetimeIndex
    codes: tuple[str, ...]
    channels: tuple[str, ...]

    def __post_init__(self):
        self.dates = pd.DatetimeIndex(self.dates)
        if self.values.shape != (len(self.dates), len(self.codes), len(self.channels)):
            raise ValueError("market panel shape mismatch")
        if self.members.shape != self.values.shape[:2]:
            raise ValueError("membership shape mismatch")
        if not self.dates.is_monotonic_increasing or self.dates.has_duplicates:
            raise ValueError("market dates must be sorted and unique")
        if not self.channels or self.channels[0] != "log_close":
            raise ValueError("primary target channel must be log_close")

    def eligible(self, position, context, minimum_fraction=0.8):
        if not context - 1 <= position < len(self.dates):
            raise ValueError("insufficient context or invalid position")
        primary = self.values[position - context + 1:position + 1, :, 0]
        valid = np.isfinite(primary)
        return np.flatnonzero(self.members[position] & valid[-1] & (valid.sum(axis=0) >= np.ceil(context * minimum_fraction)))


def pit_membership(dates, codes, universe):
    dates = pd.DatetimeIndex(dates)
    universe = universe.copy()
    universe["effective_date"] = pd.to_datetime(universe["effective_date"])
    universe["_code"] = universe["code"].astype(str).str.extract(r"(\d{6})")[0]
    effective = pd.DatetimeIndex(sorted(universe.effective_date.unique()))
    period = effective.searchsorted(dates, side="right") - 1
    members = np.zeros((len(dates), len(codes)), dtype=bool)
    for i, day in enumerate(effective):
        current = set(universe.loc[universe.effective_date == day, "_code"])
        rows = np.flatnonzero(period == i)
        columns = [j for j, code in enumerate(codes) if code in current]
        members[np.ix_(rows, columns)] = True
    return members


def split_positions(dates, context, horizon, validation_start, test_start):
    dates = pd.DatetimeIndex(dates)
    validation_start, test_start = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if validation_start >= test_start or min(context, horizon) < 1:
        raise ValueError("invalid chronological split")
    positions = np.arange(context - 1, len(dates) - horizon)
    end = dates[positions + horizon]
    return {"train": positions[end < validation_start],
            "validation": positions[(dates[positions] >= validation_start) & (end < test_start)],
            "test": positions[dates[positions] >= test_start]}


def make_batch(panel, positions, context, entities, horizon, rng):
    if min(context, entities, horizon) < 1:
        raise ValueError("batch dimensions must be positive")
    values = np.full((len(positions), entities, len(panel.channels), context), np.nan, dtype=np.float32)
    targets = np.full((len(positions), entities, len(panel.channels), horizon), np.nan, dtype=np.float32)
    identities = np.full((len(positions), entities), -1, dtype=np.int64)
    for b, t in enumerate(positions):
        candidates = panel.eligible(int(t), context)
        if not len(candidates):
            raise ValueError("no PIT-eligible entities with historical context")
        selected = rng.choice(candidates, size=min(entities, len(candidates)), replace=False)
        count = len(selected)
        values[b, :count] = panel.values[t - context + 1:t + 1, selected].transpose(1, 2, 0)
        available = min(horizon, len(panel.dates) - t - 1)
        targets[b, :count, :, :available] = panel.values[t + 1:t + 1 + available, selected].transpose(1, 2, 0)
        identities[b, :count] = selected
    return {"context": values, "targets": targets, "entities": identities, "positions": np.asarray(positions)}


def fixed_entity_batch(panel, position, context, horizon, entities=None):
    if min(context, horizon) < 1:
        raise ValueError("context and horizon must be positive")
    eligible = panel.eligible(position, context)
    selected = eligible if entities is None else np.asarray(entities)
    if selected.ndim != 1 or not len(selected) or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("entities must be a nonempty integer vector")
    if len(np.unique(selected)) != len(selected) or not np.isin(selected, eligible).all():
        raise ValueError("entities must be unique and eligible at the forecast date")
    values = panel.values[position - context + 1:position + 1, selected].transpose(1, 2, 0)[None].copy()
    targets = np.full((1, len(selected), len(panel.channels), horizon), np.nan, dtype=np.float32)
    available = min(horizon, len(panel.dates) - position - 1)
    targets[0, :, :, :available] = panel.values[position + 1:position + 1 + available, selected].transpose(1, 2, 0)
    return {"context": values, "targets": targets, "entities": selected[None].copy(),
            "positions": np.asarray([position], dtype=np.int64)}


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_snapshot(root):
    path = root / "snapshot_manifest.json"
    manifest = json.loads(path.read_text())
    for relative, expected in manifest["files"].items():
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root.resolve()) or file_hash(candidate) != expected:
            raise ValueError("snapshot fingerprint mismatch: " + relative)
    return {"manifest_sha256": file_hash(path), "test_end": manifest["test_end"]}


def load_panel(root, main_board_only=True):
    root = Path(root)
    frame = pd.read_parquet(root / "data/processed/training_data_pit_v24_tick1.parquet",
                            columns=["date", "code", *TICK_CHANNELS])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["_code"] = frame.code.astype(str).str[:6]
    if frame.duplicated(["date", "_code"]).any():
        raise ValueError("duplicate matrix keys")
    dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    codes = tuple(sorted(c for c in frame._code.unique() if not main_board_only or not c.startswith(("30", "688"))))
    values = np.full((len(dates), len(codes), len(CHANNELS)), np.nan, dtype=np.float32)
    tick = frame.set_index(["date", "_code"])
    mapping = {"时间": "date", "收盘价": "close", "开盘价": "open", "最高价": "high",
               "最低价": "low", "成交量": "volume", "总金额": "amount"}
    for index, code in enumerate(codes):
        path = root / "data/raw/kline" / (code + ".parquet")
        if not path.exists():
            continue
        bars = pd.read_parquet(path).rename(columns=mapping)
        bars["date"] = pd.to_datetime(bars.date)
        bars = bars.sort_values("date").drop_duplicates("date").set_index("date").reindex(dates)
        prices = bars[["close", "open", "high", "low"]].apply(pd.to_numeric, errors="coerce")
        logged = np.log(prices.where(prices > 0))
        primary = logged["close"]
        volume = pd.to_numeric(bars["volume"], errors="coerce").where(lambda s: s >= 0)
        amount = pd.to_numeric(bars["amount"], errors="coerce").where(lambda s: s >= 0)
        channels = np.column_stack([primary, primary.diff(), logged.open - primary.shift(1),
                                    primary - logged.open, logged.high - logged.low, np.log1p(volume), np.log1p(amount)])
        values[:, index, :7] = channels.astype(np.float32)
        values[:, index, 7:] = tick.reindex(pd.MultiIndex.from_arrays([dates, [code] * len(dates)]))[list(TICK_CHANNELS)].to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    universe = pd.read_parquet(root / "data/universe/universe_pit.parquet")
    members = pit_membership(dates, codes, universe)
    return MarketPanel(values, members, dates, codes, CHANNELS)
