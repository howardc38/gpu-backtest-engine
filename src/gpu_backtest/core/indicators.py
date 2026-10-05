"""Precompute rolling, EMA, RSI, and ATR indicator tables."""

import numpy as np
import pandas as pd

from .grid import int_dim_values
from .strategy import MAX_TABLES


def _series(kind, source, df):
    if kind == "atr":
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr
    return df[source]


def build_table(kind, source, windows, close, open_, high, low, volume):
    df = pd.DataFrame(
        {
            "close": close.astype(np.float64),
            "open": open_.astype(np.float64),
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
            "volume": volume.astype(np.float64),
        }
    )
    s = _series(kind, source, df)
    windows = sorted(set((int(w) for w in windows)))
    n = len(df)
    tab = np.full((len(windows), n), np.nan, dtype=np.float32)
    for i, w in enumerate(windows):
        if kind in ("roll_max", "roll_min", "roll_mean", "roll_std"):
            fn = {"roll_max": "max", "roll_min": "min", "roll_mean": "mean", "roll_std": "std"}[
                kind
            ]
            tab[i] = getattr(s.rolling(w), fn)().to_numpy()
        elif kind == "ema":
            tab[i] = s.ewm(span=w, adjust=False).mean().to_numpy()
        elif kind == "rsi":
            delta = s.diff()
            gain = delta.clip(lower=0.0).ewm(alpha=1.0 / w, adjust=False).mean()
            loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / w, adjust=False).mean()
            rs = gain / loss
            rsi = 100.0 - 100.0 / (1.0 + rs)
            rsi = rsi.where(loss > 0, 100.0)
            rsi[delta.isna()] = np.nan
            tab[i] = rsi.to_numpy()
        elif kind == "atr":
            tab[i] = s.ewm(alpha=1.0 / w, adjust=False).mean().to_numpy()
        else:
            raise ValueError(f"unknown table kind: {kind}")
    row_map = np.zeros(max(windows) + 1, dtype=np.int64)
    for i, w in enumerate(windows):
        row_map[w] = i
    return (tab, row_map)


def prepare_tables(strategy, close, open_, high, low, vol):
    all_dims = list(strategy.ENTRY_DIMS) + list(strategy.EXIT_DIMS)
    tabs, rms = ([], [])
    for kind, source, dim_names in strategy.TABLES:
        windows = int_dim_values(all_dims, dim_names)
        assert windows, f"Table {kind} has no integer window values for {dim_names}"
        t, rm = build_table(kind, source, windows, close, open_, high, low, vol)
        tabs.append(t)
        rms.append(rm)
    while len(tabs) < MAX_TABLES:
        tabs.append(np.zeros((1, 1), np.float32))
        rms.append(np.zeros(1, np.int64))
    return (tabs, rms)
