"""Synthetic fixtures and independent expected results shared by device checks."""

import json

import numpy as np
import pandas as pd
from gpu_backtest_tools.checks.reference import reference_group_sums

from gpu_backtest import run
from gpu_backtest.core.strategy import load_strategy


def write_bars(path, bars):
    data = pd.DataFrame(bars, columns=["Open", "High", "Low", "Close", "Volume"])
    data.insert(0, "Time", pd.date_range("2024-01-01", periods=len(data), freq="D", tz="UTC"))
    data.to_csv(path, index=False)
    return data


def rsi_bars(exit_open=120):
    return [
        (100, 101, 99, 100, 10),
        (95, 96, 89, 90, 10),
        (80, 81, 79, 80, 10),
        (90, 101, 90, 100, 10),
        (110, 121, 110, 120, 10),
        (exit_open, 121, min(exit_open, 120) - 1, 120, 10),
        (120, 121, 119, 120, 10),
    ]


def matrix_plugin(tmp_path, monkeypatch):
    package = tmp_path / "outside_plugins"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "matrix.py").write_text(
        '\nNAME = "matrix"\nENTRY_DIMS = [("entry", 1, 2, 1, False)]\nEXIT_DIMS = [("exit", 1, 3, 1, False)]\nTABLES = []\n\ndef make_device_fn(cuda):\n    @cuda.jit(device=True, inline=True)\n    def algo(close, open_, high, low, vol, t0, t1, t2, t3,\n             rm0, rm1, rm2, rm3, ep, xp, num_bars, buy, sell):\n        return ep[0] * 10 + xp[0] + 1\n    return algo\n\ndef reference(close, open_, high, low, vol, tabs, rms, ep, xp, buy, sell):\n    return ep[0] * 10 + xp[0] + 1\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import sys

    sys.modules.pop("outside_plugins.matrix", None)
    sys.modules.pop("outside_plugins", None)
    return "outside_plugins.matrix"


def check_matrix(tmp_path, monkeypatch):
    plugin = matrix_plugin(tmp_path, monkeypatch)
    source = tmp_path / "bars.csv"
    write_bars(source, rsi_bars())
    prefix = str(tmp_path / "result")
    first = run(plugin, source, prefix, top_n=3, threads_per_block=32, verbose=False)
    expected = {
        "entry_sum": [39.0, 69.0],
        "exit_sum": [34.0, 36.0, 38.0],
        "entry_sumsq": [509.0, 1589.0],
        "exit_sumsq": [628.0, 698.0, 772.0],
    }
    for key, values in expected.items():
        np.testing.assert_array_equal(first[key], values)
    artifacts = [tmp_path / f"result_top_{side}.csv" for side in ("entry", "exit")]
    artifacts.append(tmp_path / "result_top_manifest.json")
    before = [path.read_bytes() for path in artifacts]
    second = run(
        load_strategy(plugin), source, prefix, top_n=3, threads_per_block=32, verbose=False
    )
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert [path.read_bytes() for path in artifacts] == before
    manifest = json.loads(artifacts[-1].read_text())
    assert manifest["strategy"] == "matrix" and manifest["pair_count"] == 6
    for side, count, order in (("entry", 2, [2, 1]), ("exit", 3, [3, 2, 1])):
        data = pd.read_csv(tmp_path / f"result_top_{side}.csv")
        assert len(data) == count
        assert data[side].tolist() == order


def check_rsi_reference(tmp_path):
    source = tmp_path / "bars.csv"
    data = write_bars(source, rsi_bars())
    strategy = load_strategy("gpu_backtest_examples.rsi.strategy")
    result = run(strategy, source, None, verbose=False)
    args = [
        data[column].to_numpy(dtype=np.float32)
        for column in ("Close", "Open", "High", "Low", "Volume")
    ]
    entry, exit_ = reference_group_sums(
        strategy, *args, strategy.ENTRY_DIMS, strategy.EXIT_DIMS, 0.0015, 0.0015
    )
    np.testing.assert_allclose(result["entry_sum"], entry, rtol=1e-05, atol=0.0001)
    np.testing.assert_allclose(result["exit_sum"], exit_, rtol=1e-05, atol=0.0001)
    assert np.abs(result["entry_sum"]).max() > 1


def single_rsi(path, bars, *, buy=0.0, sell=0.0, buy_level=30):
    write_bars(path, bars)
    result = run(
        "gpu_backtest_examples.rsi.strategy",
        path,
        None,
        buy=buy,
        sell=sell,
        entry_dims=[("p_e", 2, 2, 1, False), ("buy_lvl", buy_level, buy_level, 1, False)],
        exit_dims=[("p_x", 2, 2, 1, False), ("sell_lvl", 70, 70, 1, False)],
        verbose=False,
    )
    return float(result["entry_sum"][0])
