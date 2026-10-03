import json

import pandas as pd
import pytest

from gpu_backtest import splits as splitter


def _market_frame(periods=10):
    time = pd.date_range("2021-01-01", periods=periods, freq="12h", tz="Asia/Hong_Kong")
    close = pd.Series(range(periods), dtype=float) + 100.0
    return pd.DataFrame(
        {
            "Time": time,
            "Open": close,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close + 1.0,
            "Volume": 1000.0,
        }
    )


def test_full_and_halves_are_disjoint_and_cover_full_range(tmp_path):
    source = tmp_path / "sample.csv"
    _market_frame().to_csv(source, index=False)
    manifest_path = splitter.split_csv_by_date_range(
        str(source), "01-01-2021", "05-01-2021", 3, "Time"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["mode"] == "full_and_halves"
    assert manifest["schema_version"] == 2
    assert manifest["split_contract_version"] == splitter.SPLIT_CONTRACT_VERSION
    assert len(manifest["source_sha256"]) == 64
    assert all((len(record["sha256"]) == 64 for record in manifest["splits"]))
    assert [record["label"] for record in manifest["splits"]] == [
        "full",
        "first_half",
        "second_half",
    ]
    split_dir = manifest_path.parent
    full, first, second = [pd.read_csv(split_dir / record["file"]) for record in manifest["splits"]]
    assert len(first) + len(second) == len(full) == manifest["filtered_rows"]
    assert set(first["Time"]).isdisjoint(second["Time"])
    assert set(first["Time"]) | set(second["Time"]) == set(full["Time"])


@pytest.mark.parametrize(
    "mutation,error", [("duplicate", "duplicate timestamps"), ("unordered", "strictly increasing")]
)
def test_split_rejects_invalid_timestamp_order(tmp_path, mutation, error):
    data = _market_frame()
    if mutation == "duplicate":
        data.loc[4, "Time"] = data.loc[3, "Time"]
    else:
        data.loc[[3, 4], "Time"] = data.loc[[4, 3], "Time"].to_numpy()
    source = tmp_path / "invalid.csv"
    data.to_csv(source, index=False)
    with pytest.raises(ValueError, match=error):
        splitter.split_csv_by_date_range(str(source), "01-01-2021", "05-01-2021", 3, "Time")
