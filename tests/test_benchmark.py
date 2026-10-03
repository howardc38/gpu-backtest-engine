from pathlib import Path
from types import SimpleNamespace

import numba
import numpy as np
import pytest

from gpu_backtest import benchmark
from gpu_backtest.benchmark_cpu import build_cpu_reducer
from gpu_backtest.engine import (
    decode_values,
    prepare_tables,
    reference_group_sums,
    space_count,
    validate_market_data,
)
from gpu_backtest.strategies import rsi_meanrev
from gpu_backtest.synthetic import generate_csv


def test_billion_profile_is_one_billion_unique_pairs():
    entry, exit_ = benchmark.profiles()["billion"]
    assert space_count(entry) == 20_000
    assert space_count(exit_) == 50_000
    assert space_count(entry) * space_count(exit_) == 1_000_000_000


def test_parallel_cpu_baseline_matches_independent_reference(tmp_path):
    data = validate_market_data(generate_csv(tmp_path / "bars.csv", 32))
    arrays = [
        data[c].to_numpy(dtype=np.float32) for c in ("Close", "Open", "High", "Low", "Volume")
    ]
    entry = [("p_e", 2, 4, 2, False), ("buy_lvl", 20, 30, 10, False)]
    exit_ = [("p_x", 2, 4, 2, False), ("sell_lvl", 70, 80, 10, False)]
    args = benchmark.cpu_args(rsi_meanrev, arrays, entry, exit_)
    previous = numba.get_num_threads()
    numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
    try:
        results = build_cpu_reducer(rsi_meanrev)(*args)
    finally:
        numba.set_num_threads(previous)
    expected_entry, expected_exit = reference_group_sums(
        rsi_meanrev, *arrays, entry, exit_, 0.0015, 0.0015
    )
    np.testing.assert_allclose(results[0], expected_entry, rtol=0, atol=1e-5)
    np.testing.assert_allclose(results[2], expected_exit, rtol=0, atol=1e-5)
    specs = SimpleNamespace(ENTRY_DIMS=entry, EXIT_DIMS=exit_, TABLES=rsi_meanrev.TABLES)
    tables, maps = prepare_tables(specs, *arrays)
    matrix = np.empty((space_count(entry), space_count(exit_)), np.float32)
    for e in range(len(matrix)):
        ep = decode_values(entry, e) + [0.0] * (4 - len(entry))
        for x in range(matrix.shape[1]):
            xp = decode_values(exit_, x) + [0.0] * (4 - len(exit_))
            matrix[e, x] = rsi_meanrev.reference(*arrays, tables, maps, ep, xp, 0.0015, 0.0015)
    np.testing.assert_array_equal(results[1], (matrix * matrix).sum(axis=1, dtype=np.float64))
    np.testing.assert_array_equal(results[3], (matrix * matrix).sum(axis=0, dtype=np.float64))
    assert np.any(np.abs(results[0]) > 0)


def test_benchmark_rejects_simulator_and_missing_hardware(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "config", SimpleNamespace(ENABLE_CUDASIM=True))
    with pytest.raises(ValueError, match="not CUDASIM"):
        benchmark.benchmark(tmp_path / "report.json")
    monkeypatch.setattr(benchmark, "config", SimpleNamespace(ENABLE_CUDASIM=False))
    monkeypatch.setattr(benchmark, "cuda", SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ValueError, match="available NVIDIA"):
        benchmark.benchmark(tmp_path / "report.json")


def test_generated_data_reproduces_public_example(tmp_path):
    generated = generate_csv(tmp_path / "bars.csv", 128)
    original = Path(__file__).resolve().parents[1] / "examples/synthetic.csv"
    assert generated.read_bytes() == original.read_bytes()
