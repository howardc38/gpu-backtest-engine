import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def invoke(*args, simulation=False, cwd=None):
    env = os.environ.copy()
    if simulation:
        env["NUMBA_ENABLE_CUDASIM"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "gpu_backtest", *map(str, args)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_config_paths_resolve_outside_checkout(tmp_path):
    prefix = tmp_path / "rsi"
    result = invoke(
        "run",
        "--config",
        ROOT / "examples/gpu_backtest_examples/rsi/config.json",
        "--out-prefix",
        prefix,
        simulation=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((tmp_path / "rsi_top_manifest.json").read_text())
    assert manifest["pair_count"] == 16 and manifest["strategy"] == "rsi_meanrev"


def test_strategy_is_always_explicit(tmp_path):
    result = invoke(
        "run",
        "--input",
        ROOT / "examples/gpu_backtest_examples/rsi/synthetic.csv",
        "--out-prefix",
        tmp_path / "x",
    )
    assert result.returncode == 2 and "explicitly" in result.stderr
    assert not (tmp_path / "x_top_entry.csv").exists()


def test_complete_public_pipeline(tmp_path):
    output = tmp_path / "pipeline"
    result = invoke(
        "pipeline",
        "--config",
        ROOT / "examples/gpu_backtest_examples/rsi/config.json",
        "--output-dir",
        output,
        simulation=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "split_manifest.json").read_text())
    assert [record["label"] for record in manifest["splits"]] == [
        "full",
        "first_half",
        "second_half",
    ]
    assert len(list(output.glob("*_top_manifest.json"))) == 3
    for side in ("entry", "exit"):
        common = pd.read_csv(output / f"common_{side}.csv")
        assert len(common) == 4
        assert {"avg_effect_size", "min_effect_size", "effect_size_full"}.issubset(common.columns)


def test_optional_charts_detect_arbitrary_parameter_names(tmp_path):
    pytest.importorskip("altair")
    source = tmp_path / "sample_top.csv"
    pd.DataFrame({"custom_threshold": [1, 2], "effect_size": [0.1, 0.2]}).to_csv(
        source, index=False
    )
    output = tmp_path / "charts.html"
    result = invoke("charts", "--input", source, "--output", output)
    assert result.returncode == 0, result.stderr
    html = output.read_text()
    assert "custom_threshold" in html and "vegaEmbed" in html
