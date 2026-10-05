import json

import numpy as np
import pandas as pd
import pytest

from gpu_backtest.core import data as market_data
from gpu_backtest.core import output as artifacts


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
    data = market_data.validate_market_data(source, expected_interval="12h")
    assert len(data) == 8


def test_market_data_validation_rejects_gap_and_bad_ohlc(tmp_path):
    source = tmp_path / "market.csv"
    _write_market_csv(source)
    data = pd.read_csv(source).drop(index=4).reset_index(drop=True)
    data.to_csv(source, index=False)
    with pytest.raises(ValueError, match="bar interval"):
        market_data.validate_market_data(source, expected_interval="12h")
    _write_market_csv(source)
    data = pd.read_csv(source)
    data.loc[2, "High"] = data.loc[2, "Low"] - 1
    data.to_csv(source, index=False)
    with pytest.raises(ValueError, match="High is below"):
        market_data.validate_market_data(source, expected_interval="12h")


def test_reduction_contract_checks_sum_and_sumsq():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    checks = artifacts.validate_reduction_outputs(
        matrix.sum(axis=1, dtype=np.float64),
        (matrix * matrix).sum(axis=1, dtype=np.float64),
        matrix.sum(axis=0, dtype=np.float64),
        (matrix * matrix).sum(axis=0, dtype=np.float64),
    )
    assert checks["sum"]["difference"] == 0
    assert checks["sumsq"]["difference"] == 0
    with pytest.raises(RuntimeError, match="grand sum mismatch"):
        artifacts.validate_reduction_outputs([3.0, 7.0], [5.0, 25.0], [4.0, 7.0], [10.0, 20.0])


def test_raw_artifact_preserves_float64_arrays_and_parameter_order(tmp_path):
    import hashlib

    matrix = np.array([[1.125, 2.25], [3.5, 8.75]], dtype=np.float32)
    arrays = (
        matrix.sum(axis=1, dtype=np.float64),
        (matrix * matrix).sum(axis=1, dtype=np.float64),
        matrix.sum(axis=0, dtype=np.float64),
        (matrix * matrix).sum(axis=0, dtype=np.float64),
    )
    checks = artifacts.validate_reduction_outputs(*arrays)
    prefix = str(tmp_path / "result")
    entry_dims = [("effect_size", 0.1, 0.2, 0.1, True)]
    exit_dims = [("exit_parameter", 1, 2, 1, False)]
    artifacts.write_results(
        prefix,
        "test_strategy",
        "source.csv",
        entry_dims,
        exit_dims,
        *arrays,
        checks,
        False,
        buy=0.0015,
        sell=0.002,
    )
    path = tmp_path / "result_results.npz"
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(artifacts.RESULT_KEYS)
        for key, expected in zip(artifacts.RESULT_KEYS, arrays):
            assert saved[key].dtype == np.float64
            assert saved[key].tobytes() == expected.tobytes()
    manifest = json.loads((tmp_path / "result_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["entry_dims"] == [["effect_size", 0.1, 0.2, 0.1, True]]
    assert manifest["exit_dims"] == [["exit_parameter", 1, 2, 1, False]]
    assert manifest["pair_count"] == 4
    assert manifest["buy"] == 0.0015 and manifest["sell"] == 0.002
    assert manifest["outputs"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*_top*"))


@pytest.mark.parametrize(
    "arrays",
    [
        ([[1.0]], [1.0], [1.0], [1.0]),
        ([1.0, 2.0], [1.0], [3.0], [1.0]),
    ],
)
def test_reduction_validation_rejects_invalid_shapes(arrays):
    with pytest.raises(RuntimeError, match="one-dimensional|shapes must match"):
        artifacts.validate_reduction_outputs(*arrays)


def test_raw_writer_failure_does_not_publish_partial_results(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("Injected disk error")

    monkeypatch.setattr(artifacts.np, "savez", fail_save)
    dims = [("p", 1, 1, 1, False)]
    arrays = [np.array([1.0])] * 4
    with pytest.raises(OSError, match="Injected disk error"):
        artifacts.write_results(
            tmp_path / "result",
            "test",
            "source.csv",
            dims,
            dims,
            *arrays,
            {},
            False,
            buy=0.0,
            sell=0.0,
        )
    assert not list(tmp_path.iterdir())


def test_failed_publication_invalidates_previous_manifest(tmp_path, monkeypatch):
    dims = [("p", 1, 1, 1, False)]
    prefix = tmp_path / "result"
    arrays = [np.array([1.0])] * 4
    artifacts.write_results(
        prefix, "test", "source.csv", dims, dims, *arrays, {}, False, buy=0.0, sell=0.0
    )
    original = artifacts.os.replace

    def fail_manifest(source, destination):
        if str(destination).endswith("_manifest.json"):
            raise OSError("Injected manifest publication failure")
        return original(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_manifest)
    new_arrays = [np.array([2.0])] * 4
    with pytest.raises(OSError, match="manifest publication failure"):
        artifacts.write_results(
            prefix, "test", "source.csv", dims, dims, *new_arrays, {}, False, buy=0.1, sell=0.1
        )
    assert not (tmp_path / "result_manifest.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
