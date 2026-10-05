"""Select upload files and construct the remote execution recipe."""

import hashlib
import importlib.metadata
import importlib.resources
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from gpu_backtest import __version__, resolve_strategy_name
from gpu_backtest.cli.options import CONFIG_KEYS

PACKAGE_FILES = {
    "gpu_backtest": [
        "__init__.py",
        "__main__.py",
        "cli/__init__.py",
        "cli/backtest.py",
        "cli/options.py",
        "cli/tools.py",
        "core/__init__.py",
        "core/data.py",
        "core/engine.py",
        "core/grid.py",
        "core/indicators.py",
        "core/kernels.py",
        "core/output.py",
        "core/strategy.py",
    ],
    "gpu_backtest_examples": [
        "__init__.py",
        "data.py",
        "rsi/__init__.py",
        "rsi/generate_data.py",
        "rsi/strategy.py",
    ],
    "gpu_backtest_tools": [
        "__init__.py",
        "benchmarks/__init__.py",
        "benchmarks/cpu.py",
        "benchmarks/runner.py",
        "checks/__init__.py",
        "checks/gpu.py",
        "checks/reference.py",
        "runpod/__init__.py",
        "runpod/bundle.py",
        "runpod/client.py",
        "runpod/launcher.py",
        "runpod/transport.py",
        "runpod/requirements.txt",
    ],
}
IGNORED_PARTS = {"__pycache__", ".git", ".ruff_cache", "venv", ".venv", ".pytest_cache"}


def build_bundle(archive, *, mode, config_path=None, plugin_dir=None):
    if mode not in ("run", "check", "benchmark"):
        raise ValueError("RunPod mode must be run, check, or benchmark")
    config = {}
    if mode == "run":
        if not config_path:
            raise ValueError("RunPod run requires --config")
        config_path = Path(config_path).resolve()
        config = json.loads(config_path.read_text())
        if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
            raise ValueError("RunPod config must contain only documented engine config keys")
        if not config.get("strategy") or not config.get("input"):
            raise ValueError("RunPod config must specify strategy and input")
        if not isinstance(config["strategy"], str) or not all(
            (p.isidentifier() for p in config["strategy"].split("."))
        ):
            raise ValueError("RunPod strategy must be a builtin name or dotted module name")
        source = (config_path.parent / config["input"]).resolve()
        from gpu_backtest.core.data import validate_market_data

        validate_market_data(source, expected_interval=config.get("expected_interval"))
    distribution = importlib.metadata.distribution("gpu-backtest-engine")
    base_requires = [r for r in distribution.requires or [] if "extra ==" not in r]
    with tempfile.TemporaryDirectory(prefix="runpod-bundle-") as directory:
        staging = Path(directory)
        engine = staging / "engine"
        for package_name, members in PACKAGE_FILES.items():
            package = Path(str(importlib.resources.files(package_name)))
            for relative in members:
                path = package / relative
                if path.is_symlink():
                    raise ValueError("Engine bundle cannot include symlinked source files")
                output = engine / package_name / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, output)
        project = f"""[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gpu-backtest-engine"
version = {json.dumps(__version__)}
requires-python = ">=3.11,<3.14"
dependencies = {json.dumps(base_requires)}

[project.scripts]
gpu-backtest = "gpu_backtest.cli:main"

[tool.setuptools.packages.find]
where = ["."]

[tool.setuptools.package-data]
"gpu_backtest_tools.runpod" = ["requirements.txt"]
"""
        (engine / "pyproject.toml").write_text(project)
        for entry in distribution.files or []:
            if str(entry) == "LICENSE" or str(entry).endswith("/licenses/LICENSE"):
                shutil.copyfile(distribution.locate_file(entry), engine / "LICENSE")
                break
        if mode == "run":
            (staging / "input").mkdir()
            shutil.copyfile(source, staging / "input/market.csv")
            config["input"] = "input/market.csv"
            (staging / "job.json").write_text(json.dumps(config, indent=2) + "\n")
        if plugin_dir:
            plugin_dir = Path(plugin_dir).resolve()
            if not plugin_dir.is_dir():
                raise ValueError("Plugin directory does not exist")
            copied = 0
            for path in sorted(plugin_dir.rglob("*.py")):
                relative = path.relative_to(plugin_dir)
                if set(relative.parts) & IGNORED_PARTS:
                    continue
                if path.is_symlink():
                    raise ValueError("Plugin source symlinks are not supported")
                output = staging / "plugins" / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, output)
                copied += 1
            if not copied:
                raise ValueError("Plugin directory contains no Python files")
        if mode == "run":
            config["strategy"] = resolve_strategy_name(config["strategy"])
            (staging / "job.json").write_text(json.dumps(config, indent=2) + "\n")
        if mode == "run" and config["strategy"] != "gpu_backtest_examples.rsi.strategy":
            relative = Path(*config["strategy"].split("."))
            if not (staging / "plugins" / relative.with_suffix(".py")).is_file() and (
                not (staging / "plugins" / relative / "__init__.py").is_file()
            ):
                raise ValueError("External strategy must be present in --plugin-dir")
        (staging / "job.sh").write_text(remote_script(mode))
        files = []
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "file": path.relative_to(staging).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        (staging / "bundle_manifest.json").write_text(json.dumps(files, indent=2) + "\n")
        with tarfile.open(archive, "w:gz") as handle:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    handle.add(path, arcname=path.relative_to(staging).as_posix(), recursive=False)
    return files


def remote_script(mode):
    script = """#!/bin/bash
set -euo pipefail
export NUMBA_ENABLE_CUDASIM=0
export PYTHONPATH="$PWD/plugins${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p output
python3 -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), "Python 3.11-3.13 required"'
python3 -m venv env
env/bin/python -m pip install --disable-pip-version-check ./engine -r engine/gpu_backtest_tools/runpod/requirements.txt
nvidia-smi --query-gpu=name,driver_version --format=csv > output/gpu.csv
env/bin/python -m gpu_backtest gpu-check --output output/gpu_check.json
"""
    if mode == "run":
        script += (
            "env/bin/python -m gpu_backtest run --config job.json --out-prefix output/result\n"
        )
    elif mode == "benchmark":
        script += "env/bin/python -m gpu_backtest benchmark --output output/benchmark.json --billion --billion-cpu\n"
    script += "env/bin/python -m pip freeze > output/requirements-resolved.txt\n"
    return script
