from pathlib import Path
from types import SimpleNamespace

import numba
import numpy as np
import pytest
from gpu_backtest_examples.data import generate_csv
from gpu_backtest_examples.rsi import strategy as rsi_meanrev
from gpu_backtest_tools.benchmarks import runner as benchmark
from gpu_backtest_tools.benchmarks.cpu import build_cpu_reducer
from gpu_backtest_tools.checks.reference import reference_group_sums

from gpu_backtest.core.data import validate_market_data
from gpu_backtest.core.grid import decode_values, space_count
from gpu_backtest.core.indicators import prepare_tables


def test_billion_profile_is_one_billion_unique_pairs():
    entry, exit_ = benchmark.profiles()["billion"]
    assert space_count(entry) == 20000
    assert space_count(exit_) == 50000
    assert space_count(entry) * space_count(exit_) == 1000000000


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
    np.testing.assert_allclose(results[0], expected_entry, rtol=0, atol=1e-05)
    np.testing.assert_allclose(results[2], expected_exit, rtol=0, atol=1e-05)
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
    original = (
        Path(__file__).resolve().parents[2] / "examples/gpu_backtest_examples/rsi/synthetic.csv"
    )
    assert generated.read_bytes() == original.read_bytes()


def test_cpu_baseline_writes_complete_ranked_artifacts(tmp_path):
    import json

    source = generate_csv(tmp_path / "bars.csv", 32)
    prefix = tmp_path / "cpu"
    entry, exit_ = (rsi_meanrev.ENTRY_DIMS, rsi_meanrev.EXIT_DIMS)
    old = numba.get_num_threads()
    numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
    try:
        result = benchmark.run_cpu_baseline(source, prefix, entry, exit_)
    finally:
        numba.set_num_threads(old)
    assert set(result) == set(benchmark.RESULT_KEYS)
    manifest = json.loads((tmp_path / "cpu_top_manifest.json").read_text())
    assert manifest["engine"] == "gpu_backtest.benchmark_cpu"
    assert manifest["pair_count"] == 16
    assert (tmp_path / "cpu_top_entry.csv").is_file()
    assert (tmp_path / "cpu_top_exit.csv").is_file()


def test_full_billion_cpu_flag_requires_billion_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "config", SimpleNamespace(ENABLE_CUDASIM=False))
    monkeypatch.setattr(benchmark, "cuda", SimpleNamespace(is_available=lambda: True))
    with pytest.raises(ValueError, match="requires --billion"):
        benchmark.benchmark(tmp_path / "report.json", cpu_threads=1, billion_cpu=True)
