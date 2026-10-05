"""Run commands: the GPU engine and generic split/common pipeline."""

from .options import job_options


def register(subparsers):
    for name in ("run", "pipeline"):
        command = subparsers.add_parser(name)
        command.add_argument("--config")
        command.add_argument("--strategy", help="Installed strategy module name")
        command.add_argument("--input")
        command.add_argument("--buy", type=float)
        command.add_argument("--sell", type=float)
        command.add_argument("--top-n", dest="top_n", type=int)
        command.add_argument("--threads-per-block", dest="threads_per_block", type=int)
        command.add_argument("--expected-interval", dest="expected_interval")
        command.set_defaults(handler=execute)
        if name == "run":
            command.add_argument("--out-prefix", required=True)
        else:
            command.add_argument("--output-dir", required=True)


def execute(args):
    strategy, source, options, config = job_options(args)
    if args.command == "run":
        from gpu_backtest.core.engine import run

        run(strategy, source, args.out_prefix, **options)
    else:
        from gpu_backtest.workflows.pipeline import run_pipeline

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
