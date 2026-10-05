"""Explicit config/CLI inputs, with relative config data paths."""

import json
from pathlib import Path

from gpu_backtest import resolve_strategy_name


def job_options(args):
    config = {}
    if args.config:
        config_path = Path(args.config).resolve()
        config = json.loads(config_path.read_text())
        if not isinstance(config, dict):
            raise ValueError("Config must be a JSON object")
        if config.get("input"):
            config["input"] = str(config_path.parent / config["input"])
    strategy = args.strategy if args.strategy is not None else config.get("strategy")
    source = args.input if args.input is not None else config.get("input")
    if not strategy or not source:
        raise ValueError("Specify strategy and input explicitly, either in config or CLI flags")
    options = {}
    for name, default in (
        ("buy", 0.0015),
        ("sell", 0.0015),
        ("top_n", 100),
        ("threads_per_block", 128),
        ("expected_interval", None),
    ):
        value = getattr(args, name)
        options[name] = config.get(name, default) if value is None else value
    options.update(entry_dims=config.get("entry_dims"), exit_dims=config.get("exit_dims"))
    return (resolve_strategy_name(strategy), source, options, config)
