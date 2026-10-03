import json

import numpy as np
import pandas as pd
import pytest

from gpu_backtest import engine as harness


def _write_market_csv(path, periods=8):
    close = np.arange(periods, dtype=float) + 100.0
    pd.DataFrame(
        {
            "Time": pd.date_range("2021-01-01", periods=periods, freq="12h", tz="UTC"),
            "Open": close,
            "High": close + 2,
            "Low": close - 2,
            "Close": close + 1,
            "Volume": np.full(periods, 1000.0),
        }
    ).to_csv(path, index=False)


def test_market_data_validation_accepts_explicit_interval(tmp_path):
    source = tmp_path / "market.csv"
    _write_market_csv(source)
    data = harness.validate_market_data(source, expected_interval="12h")
    assert len(data) == 8


def test_market_data_validation_rejects_gap_and_bad_ohlc(tmp_path):
    source = tmp_path / "market.csv"
    _write_market_csv(source)
    data = pd.read_csv(source).drop(index=4).reset_index(drop=True)
    data.to_csv(source, index=False)
    with pytest.raises(ValueError, match="bar interval"):
        harness.validate_market_data(source, expected_interval="12h")
    _write_market_csv(source)
    data = pd.read_csv(source)
    data.loc[2, "High"] = data.loc[2, "Low"] - 1
    data.to_csv(source, index=False)
    with pytest.raises(ValueError, match="High is below"):
        harness.validate_market_data(source, expected_interval="12h")


def test_reduction_contract_checks_sum_and_sumsq():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    checks = harness.validate_reduction_outputs(
        matrix.sum(axis=1, dtype=np.float64),
        (matrix * matrix).sum(axis=1, dtype=np.float64),
        matrix.sum(axis=0, dtype=np.float64),
        (matrix * matrix).sum(axis=0, dtype=np.float64),
    )
    assert checks["sum"]["difference"] == 0
    assert checks["sumsq"]["difference"] == 0
    with pytest.raises(RuntimeError, match="grand sum mismatch"):
        harness.validate_reduction_outputs([3.0, 7.0], [5.0, 25.0], [4.0, 7.0], [10.0, 20.0])


def test_top_artifact_preserves_float64_effects_and_writes_manifest(tmp_path):
    matrix = np.array([[1.125, 2.25], [3.5, 8.75]], dtype=np.float32)
    entry_sum = matrix.sum(axis=1, dtype=np.float64)
    entry_sumsq = (matrix * matrix).sum(axis=1, dtype=np.float64)
    exit_sum = matrix.sum(axis=0, dtype=np.float64)
    exit_sumsq = (matrix * matrix).sum(axis=0, dtype=np.float64)
    checks = harness.validate_reduction_outputs(entry_sum, entry_sumsq, exit_sum, exit_sumsq)
    prefix = str(tmp_path / "result")
    entry_dims = [("entry_parameter", 0.1, 0.2, 0.1, True)]
    exit_dims = [("exit_parameter", 1, 2, 1, False)]
    harness._write_top_csv(
        prefix,
        "test_strategy",
        "source.csv",
        entry_dims,
        exit_dims,
        entry_sum,
        entry_sumsq,
        exit_sum,
        exit_sumsq,
        2,
        checks,
        False,
    )
    expected = harness.cs.stats_from_groups(
        entry_sum, entry_sumsq, 2, entry_sum.sum(), entry_sumsq.sum(), 4
    )["effect_size"]
    output = pd.read_csv(f"{prefix}_top_entry.csv")
    np.testing.assert_array_equal(output["effect_size"].to_numpy(), np.sort(expected)[::-1])
    assert "0.20000000000000001" not in open(f"{prefix}_top_entry.csv").read()
    manifest = json.load(open(f"{prefix}_top_manifest.json"))
    assert manifest["pair_count"] == 4
    assert manifest["outputs"]["entry"]["rows"] == 2
