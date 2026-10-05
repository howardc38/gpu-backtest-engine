"""Run independently configured date splits and intersect their ranked parameters."""

import json
import shutil
from pathlib import Path

from gpu_backtest.core.engine import run
from gpu_backtest.core.strategy import load_strategy

from .common import process_data
from .splits import _read_input, split_csv_by_custom_ranges, split_csv_by_date_range


def run_pipeline(
    strategy,
    input_csv,
    output_dir,
    *,
    num_splits=3,
    start_date=None,
    end_date=None,
    ranges=None,
    **run_options,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    staged = data_dir / Path(input_csv).name
    if Path(input_csv).resolve() == staged.resolve():
        raise ValueError("Pipeline input must be outside its output data directory")
    shutil.copyfile(input_csv, staged)
    if ranges:
        manifest_path = split_csv_by_custom_ranges(staged, ranges, "Time")
    else:
        data, _ = _read_input(staged, "Time")
        start_date = start_date or data["Time"].min().strftime("%d-%m-%Y")
        end_date = end_date or data["Time"].max().strftime("%d-%m-%Y")
        manifest_path = split_csv_by_date_range(staged, start_date, end_date, num_splits, "Time")
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["splits"]) < 2:
        raise ValueError("Common analysis requires at least two splits")
    shutil.copyfile(manifest_path, output_dir / "split_manifest.json")
    strategy_module = load_strategy(strategy)
    entry_dims = run_options.get("entry_dims")
    exit_dims = run_options.get("exit_dims")
    entry_dims = strategy_module.ENTRY_DIMS if entry_dims is None else entry_dims
    exit_dims = strategy_module.EXIT_DIMS if exit_dims is None else exit_dims
    paths = {"entry": [], "exit": []}
    for record in manifest["splits"]:
        source = manifest_path.parent / record["file"]
        prefix = output_dir / source.stem
        run(strategy_module, source, str(prefix), **run_options)
        for side in paths:
            paths[side].append(f"{prefix}_top_{side}.csv")
    for side, dims in (("entry", entry_dims), ("exit", exit_dims)):
        process_data(
            {
                "input_files": paths[side],
                "key_columns": [d[0] for d in dims],
                "file_labels": [s["label"] for s in manifest["splits"]],
                "output_file": output_dir / f"common_{side}.csv",
            }
        )
    return output_dir
