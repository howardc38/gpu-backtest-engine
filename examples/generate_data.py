"""Generate deterministic synthetic daily OHLCV bars; no market data is used."""

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


def generate(output, bars=128):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = 100.0
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Time", "Open", "High", "Low", "Close", "Volume"])
        for i in range(bars):
            close = 100 + 16 * math.sin(i * 0.45) + 4 * math.sin(i * 1.1) + i * 0.03
            open_ = previous + 0.4 * math.cos(i)
            high = max(open_, close) + 1.0
            low = min(open_, close) - 1.0
            time = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
            writer.writerow(
                [
                    time.isoformat(),
                    *[f"{v:.6f}" for v in (open_, high, low, close)],
                    1000 + 50 * (i % 7),
                ]
            )
            previous = close
    return output


if __name__ == "__main__":
    print(generate(Path(__file__).with_name("synthetic.csv")))
