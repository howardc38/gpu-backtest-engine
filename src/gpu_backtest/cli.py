"""Command-line entry point with explicit strategy selection and JSON configuration."""

import argparse
import json
from pathlib import Path


def _options(args):
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
    return strategy, source, options, config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gpu-backtest")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "pipeline"):
        command = subparsers.add_parser(name)
        command.add_argument("--config")
        command.add_argument("--strategy", help="Builtin name or installed dotted module name")
        command.add_argument("--input")
        command.add_argument("--buy", type=float)
        command.add_argument("--sell", type=float)
        command.add_argument("--top-n", dest="top_n", type=int)
        command.add_argument("--threads-per-block", dest="threads_per_block", type=int)
        command.add_argument("--expected-interval", dest="expected_interval")
        if name == "run":
            command.add_argument("--out-prefix", required=True)
        else:
            command.add_argument("--output-dir", required=True)
    split = subparsers.add_parser("split")
    split.add_argument("--input", required=True)
    split.add_argument("--start-date")
    split.add_argument("--end-date")
    split.add_argument("--num-splits", type=int, default=3)
    common = subparsers.add_parser("common")
    common.add_argument("--input", nargs="+", required=True)
    common.add_argument("--keys", required=True)
    common.add_argument("--labels")
    common.add_argument("--output", required=True)
    charts = subparsers.add_parser("charts")
    charts.add_argument("--input", nargs="+", required=True)
    charts.add_argument("--output", required=True)
    gpu_check = subparsers.add_parser("gpu-check", help="Run small numeric checks on a real GPU")
    gpu_check.add_argument("--output")
    benchmark = subparsers.add_parser("benchmark", help="Measure compiled CPU/GPU RSI sweeps")
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--bars", type=int, default=1024)
    benchmark.add_argument("--cpu-threads", type=int, default=8)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--billion", action="store_true")
    benchmark.add_argument(
        "--billion-cpu", action="store_true", help="Measure the entire billion grid on CPU as well"
    )
    runpod = subparsers.add_parser(
        "runpod", help="Lease one GPU, run a selected job, download, clean up"
    )
    runpod.add_argument("--config")
    runpod.add_argument("--output-dir", required=True)
    runpod.add_argument("--ssh-key")
    runpod.add_argument("--plugin-dir")
    runpod.add_argument(
        "--mode", choices=("run", "pipeline", "check", "benchmark"), default="pipeline"
    )
    from .runpod import DEFAULT_IMAGE

    runpod.add_argument("--image", default=DEFAULT_IMAGE)
    runpod.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    runpod.add_argument("--cloud", choices=("SECURE", "COMMUNITY"), default="SECURE")
    runpod.add_argument("--max-seconds", type=int, default=1800)
    runpod.add_argument("--max-hourly-rate", type=float, default=1.0)
    runpod.add_argument("--keep-pod", action="store_true")
    runpod.add_argument("--charts", action="store_true")
    runpod.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command in ("run", "pipeline"):
            strategy, source, options, config = _options(args)
            if args.command == "run":
                from .engine import run

                run(strategy, source, args.out_prefix, **options)
            else:
                from .pipeline import run_pipeline

                run_pipeline(
                    strategy,
                    source,
                    args.output_dir,
                    num_splits=config.get("num_splits", 3),
                    start_date=config.get("start_date"),
                    end_date=config.get("end_date"),
                    ranges=config.get("ranges"),
                    **options,
                )
        elif args.command == "split":
            from .splits import _read_input, split_csv_by_date_range

            data, column = _read_input(args.input, "Time")
            split_csv_by_date_range(
                args.input,
                args.start_date or data[column].min().strftime("%d-%m-%Y"),
                args.end_date or data[column].max().strftime("%d-%m-%Y"),
                args.num_splits,
                column,
            )
        elif args.command == "common":
            from .common import process_data

            process_data(
                {
                    "input_files": args.input,
                    "key_columns": [p.strip() for p in args.keys.split(",") if p.strip()],
                    "file_labels": args.labels.split(",") if args.labels else None,
                    "output_file": args.output,
                }
            )
        elif args.command == "charts":
            from .charts import write_charts

            print(write_charts(args.input, args.output))
        elif args.command == "gpu-check":
            from .gpu_check import check_gpu

            check_gpu(args.output)
        elif args.command == "benchmark":
            from .benchmark import benchmark

            benchmark(
                args.output,
                bars=args.bars,
                cpu_threads=args.cpu_threads,
                repeats=args.repeats,
                billion=args.billion,
                billion_cpu=args.billion_cpu,
            )
        else:
            from .runpod import launch, terminate_on_signal

            with terminate_on_signal():
                launch(
                    config_path=args.config,
                    output_dir=args.output_dir,
                    ssh_key=args.ssh_key,
                    mode=args.mode,
                    plugin_dir=args.plugin_dir,
                    image=args.image,
                    gpu=args.gpu,
                    cloud=args.cloud,
                    max_seconds=args.max_seconds,
                    max_hourly_rate=args.max_hourly_rate,
                    keep_pod=args.keep_pod,
                    charts=args.charts,
                    dry_run=args.dry_run,
                )
    except (ValueError, ImportError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"gpu-backtest: {exc}\n")
    return 0
