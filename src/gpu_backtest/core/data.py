"""Load and validate market input without silently repairing it."""

import numpy as np
import pandas as pd

REQUIRED_MARKET_COLUMNS = ("Time", "Open", "High", "Low", "Close", "Volume")


def validate_market_data(input_csv, expected_interval=None):
    """Load and validate an OHLCV file without repairing invalid input."""
    data = pd.read_csv(input_csv)
    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{input_csv}: missing required columns: {missing}")
    if len(data) < 2:
        raise ValueError(f"{input_csv}: at least two bars are required")
    try:
        data["Time"] = pd.to_datetime(data["Time"], errors="raise")
    except Exception as exc:
        raise ValueError(f"{input_csv}: Time contains invalid timestamps") from exc
    if data["Time"].isna().any():
        raise ValueError(f"{input_csv}: Time contains missing timestamps")
    if data["Time"].duplicated().any():
        raise ValueError(f"{input_csv}: Time contains duplicate timestamps")
    if not data["Time"].is_monotonic_increasing:
        raise ValueError(f"{input_csv}: Time must be strictly increasing")
    numeric = data[list(REQUIRED_MARKET_COLUMNS[1:])].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{input_csv}: OHLCV values must all be finite numbers")
    data.loc[:, list(REQUIRED_MARKET_COLUMNS[1:])] = numeric
    prices = numeric[["Open", "High", "Low", "Close"]]
    if (prices <= 0).any().any():
        raise ValueError(f"{input_csv}: OHLC prices must be positive")
    if (numeric["Volume"] < 0).any():
        raise ValueError(f"{input_csv}: Volume must be non-negative")
    if (numeric["High"] < prices[["Open", "Low", "Close"]].max(axis=1)).any():
        raise ValueError(f"{input_csv}: High is below another OHLC price")
    if (numeric["Low"] > prices[["Open", "High", "Close"]].min(axis=1)).any():
        raise ValueError(f"{input_csv}: Low is above another OHLC price")
    if expected_interval:
        try:
            interval = pd.to_timedelta(expected_interval)
        except Exception as exc:
            raise ValueError(f"Invalid expected interval: {expected_interval}") from exc
        if interval <= pd.Timedelta(0):
            raise ValueError("Expected interval must be positive")
        deltas = data["Time"].diff().iloc[1:]
        bad = deltas != interval
        if bad.any():
            first_bad = int(np.flatnonzero(bad.to_numpy())[0]) + 1
            raise ValueError(
                f"{input_csv}: bar interval at row {first_bad} is {deltas.iloc[first_bad - 1]}, expected {interval}"
            )
    return data
