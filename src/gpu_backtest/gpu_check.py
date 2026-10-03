"""Small, explicit numeric checks on a real NVIDIA GPU."""

import importlib.metadata
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numba import config, cuda

from .engine import run


def _matrix_device(cuda):
    @cuda.jit(device=True, inline=True)
    def algo(
        close,
        open_,
        high,
        low,
        vol,
        t0,
        t1,
        t2,
        t3,
        rm0,
        rm1,
        rm2,
        rm3,
        ep,
        xp,
        num_bars,
        buy,
        sell,
    ):
        return ep[0] * 10 + xp[0] + 1

    return algo


def check_numerics(directory):
    """Anchor both reductions and RSI execution against explicit numeric answers."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bars = [
        (100, 101, 99, 100, 10),
        (95, 96, 89, 90, 10),
        (80, 81, 79, 80, 10),
        (90, 101, 90, 100, 10),
        (110, 121, 110, 120, 10),
        (120, 121, 119, 120, 10),
        (120, 121, 119, 120, 10),
    ]
    data = pd.DataFrame(bars, columns=["Open", "High", "Low", "Close", "Volume"])
    data.insert(0, "Time", pd.date_range("2024-01-01", periods=len(data), freq="D", tz="UTC"))
    source = directory / "check.csv"
    data.to_csv(source, index=False)
    strategy = SimpleNamespace(
        NAME="matrix_check",
        ENTRY_DIMS=[("entry", 1, 2, 1, False)],
        EXIT_DIMS=[("exit", 1, 3, 1, False)],
        TABLES=[],
        make_device_fn=_matrix_device,
    )
    first = run(strategy, source, None, verbose=False)
    expected = {
        "entry_sum": [39.0, 69.0],
        "exit_sum": [34.0, 36.0, 38.0],
        "entry_sumsq": [509.0, 1589.0],
        "exit_sumsq": [628.0, 698.0, 772.0],
    }
    for key, values in expected.items():
        np.testing.assert_array_equal(first[key], values)
    second = run(strategy, source, None, verbose=False)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    e = [("p_e", 2, 2, 1, False), ("buy_lvl", 30, 30, 1, False)]
    x = [("p_x", 2, 2, 1, False), ("sell_lvl", 70, 70, 1, False)]
    for fee, expected_return in [(0.0, 50.0), (0.0015, 49.625)]:
        result = run(
            "rsi_meanrev", source, None, buy=fee, sell=fee, entry_dims=e, exit_dims=x, verbose=False
        )
        np.testing.assert_allclose(result["entry_sum"], [expected_return], rtol=0, atol=1e-4)
        np.testing.assert_allclose(result["exit_sum"], [expected_return], rtol=0, atol=1e-4)
    return {"known_matrix": "pass", "repeat_determinism": "pass", "rsi_hand_calculation": "pass"}


def check_gpu(output=None):
    if config.ENABLE_CUDASIM:
        raise ValueError(
            "gpu-check requires NUMBA_ENABLE_CUDASIM=0; simulation is not GPU validation"
        )
    if not cuda.is_available():
        raise ValueError("No usable NVIDIA GPU/driver is available")
    with tempfile.TemporaryDirectory(prefix="gpu-check-") as directory:
        result = check_numerics(directory)
    device = cuda.get_current_device()
    name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
    result.update(device=name, compute_capability=list(device.compute_capability), simulation=False)
    result["packages"] = {}
    for package in (
        "gpu-backtest-engine",
        "numba",
        "numba-cuda",
        "numpy",
        "pandas",
        "cuda-bindings",
    ):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result
