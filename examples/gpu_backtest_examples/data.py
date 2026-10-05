"""Deterministic generated OHLCV data for examples and performance measurements."""

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


def generate_csv(output, bars=128):
    if not isinstance(bars, int) or bars < 2:
        raise ValueError("Synthetic data requires at least two bars")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = 100.0
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Time", "Open", "High", "Low", "Close", "Volume"])
        for i in range(bars):
            close = 100 + 16 * math.sin(i * 0.45) + 4 * math.sin(i * 1.1) + i * 0.03
            open_ = previous + 0.4 * math.cos(i)
            time = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
            prices = (open_, max(open_, close) + 1, min(open_, close) - 1, close)
            writer.writerow([time.isoformat(), *[f"{v:.6f}" for v in prices], 1000 + 50 * (i % 7)])
            previous = close
    return output
