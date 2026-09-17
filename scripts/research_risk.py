import numpy as np
import pandas as pd


def close_returns(klines, codes, calendar):
    prices = {}
    for code in sorted({str(c)[:6] for c in codes}):
        frame = klines.get(code)
        if frame is not None:
            series = frame.drop_duplicates("date").set_index("date")["close"]
            prices[code] = pd.to_numeric(series, errors="coerce")
    prices = pd.DataFrame(prices).reindex(pd.DatetimeIndex(calendar)).sort_index()
    return prices.where(prices > 0).pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


class CorrelationGuard:
    def __init__(self, returns, window=20, min_periods=15):
        if not 2 <= min_periods <= window:
            raise ValueError("require 2 <= min_periods <= window")
        self.returns = returns.sort_index()
        self.window = window
        self.min_periods = min_periods
        self.day = None
        self.values = {}
        self.cache = {}
        self.comparisons = 0
        self.unavailable = 0

    def max_correlation(self, code, held, signal_date):
        day = pd.Timestamp(signal_date)
        if day != self.day:
            end = self.returns.index.searchsorted(day, side="right")
            frame = self.returns.iloc[max(0, end - self.window):end]
            self.values = {str(c)[:6]: frame[c].to_numpy(dtype=float) for c in frame}
            self.cache = {}
            self.day = day
        code = str(code)[:6]
        correlations = []
        for other in sorted({str(c)[:6] for c in held} - {code}):
            key = tuple(sorted((code, other)))
            if key not in self.cache:
                value = None
                x, y = self.values.get(code), self.values.get(other)
                if x is not None and y is not None:
                    valid = np.isfinite(x) & np.isfinite(y)
                    if valid.sum() >= self.min_periods:
                        a, b = x[valid] - x[valid].mean(), y[valid] - y[valid].mean()
                        norm = np.linalg.norm(a) * np.linalg.norm(b)
                        if norm > 1e-20:
                            value = float(np.clip(np.dot(a, b) / norm, -1, 1))
                self.cache[key] = value
            value = self.cache[key]
            self.comparisons += 1
            if value is None:
                self.unavailable += 1
            else:
                correlations.append(value)
        return max(correlations) if correlations else None

    def blocks(self, code, held, signal_date, threshold):
        if threshold <= 0 or not held:
            return False
        value = self.max_correlation(code, held, signal_date)
        return value is not None and value > threshold + 1e-12
