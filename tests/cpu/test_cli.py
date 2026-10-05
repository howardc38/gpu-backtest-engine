import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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
    manifest = json.loads((tmp_path / "rsi_manifest.json").read_text())
    assert manifest["pair_count"] == 16 and manifest["strategy"] == "rsi_meanrev"
    with np.load(tmp_path / "rsi_results.npz", allow_pickle=False) as saved:
        assert saved["entry_sum"].shape == (4,)
        assert saved["entry_sum"].dtype == np.float64


def test_strategy_is_always_explicit(tmp_path):
    result = invoke(
        "run",
        "--input",
        ROOT / "examples/gpu_backtest_examples/rsi/synthetic.csv",
        "--out-prefix",
        tmp_path / "x",
    )
    assert result.returncode == 2 and "explicitly" in result.stderr
    assert not (tmp_path / "x_results.npz").exists()


@pytest.mark.parametrize("command", ["split", "common", "charts", "pipeline"])
def test_analysis_commands_are_removed(command):
    result = invoke(command)
    assert result.returncode == 2 and "invalid choice" in result.stderr


@pytest.mark.parametrize("key", ["top_n", "num_splits", "ranges", "start_date", "end_date"])
def test_removed_config_options_are_rejected(tmp_path, key):
    config = tmp_path / "job.json"
    config.write_text(json.dumps({"strategy": "unused", "input": "unused.csv", key: 3}))
    result = invoke("run", "--config", config, "--out-prefix", tmp_path / "x")
    assert result.returncode == 2 and "Unknown engine config keys" in result.stderr
    assert not (tmp_path / "x_results.npz").exists()
