import csv
import os
import subprocess
import sys

import pandas as pd
import pytest


def _write_top(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parameter", "effect_size"])
        writer.writeheader()
        writer.writerows(rows)


def _run(inputs, output, labels="full,first_half,second_half", env=None):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "gpu_backtest",
            "common",
            "--input",
            *map(str, inputs),
            "--keys",
            "parameter",
            "--labels",
            labels,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_common_uses_full_precision_and_writes_audit_columns(tmp_path):
    files = [tmp_path / f"top_{i}.csv" for i in range(3)]
    _write_top(
        files[0],
        [
            {"parameter": "1", "effect_size": "0.10000000000000006"},
            {"parameter": "2", "effect_size": "0.1"},
        ],
    )
    for path in files[1:]:
        _write_top(
            path,
            [
                {"parameter": "2", "effect_size": "0.10000000000000005"},
                {"parameter": "1", "effect_size": "0.1"},
            ],
        )
    output = tmp_path / "common.csv"
    output.write_text("stale output\n")
    result = _run(files, output)
    assert result.returncode == 0, result.stderr
    data = pd.read_csv(output, dtype=str)
    assert list(data["parameter"]) == ["2", "1"]
    assert list(data.columns) == [
        "parameter",
        "effect_size_full",
        "effect_size_first_half",
        "effect_size_second_half",
        "avg_effect_size",
        "min_effect_size",
    ]
    assert data.loc[0, "avg_effect_size"] != data.loc[1, "avg_effect_size"]


def test_common_ties_are_byte_deterministic_across_hash_seeds(tmp_path):
    files = [tmp_path / f"top_{i}.csv" for i in range(3)]
    rows = [
        {"parameter": "10", "effect_size": "0.5"},
        {"parameter": "2", "effect_size": "0.5"},
        {"parameter": "1", "effect_size": "0.5"},
    ]
    for path in files:
        _write_top(path, rows)
    outputs = []
    for seed in ("1", "999"):
        output = tmp_path / f"common_{seed}.csv"
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = _run(files, output, env=env)
        assert result.returncode == 0, result.stderr
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]
    assert list(pd.read_csv(tmp_path / "common_1.csv")["parameter"]) == [1, 2, 10]


@pytest.mark.parametrize("case", ["missing", "no_common", "duplicate"])
def test_common_fails_closed_without_valid_output(tmp_path, case):
    files = [tmp_path / f"top_{i}.csv" for i in range(3)]
    for path in files:
        _write_top(path, [{"parameter": "1", "effect_size": "0.5"}])
    if case == "missing":
        files[1] = tmp_path / "missing.csv"
    elif case == "no_common":
        _write_top(files[2], [{"parameter": "2", "effect_size": "0.5"}])
    else:
        _write_top(
            files[1],
            [{"parameter": "1", "effect_size": "0.5"}, {"parameter": "1", "effect_size": "0.4"}],
        )
    output = tmp_path / "common.csv"
    output.write_text("stale output\n")
    result = _run(files, output)
    assert result.returncode != 0
    assert not output.exists()
