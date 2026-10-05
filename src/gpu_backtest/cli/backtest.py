"""Command-line adapter for one GPU backtest run."""

from .options import job_options


def register(subparsers):
    command = subparsers.add_parser("run", help="Run a strategy and save raw GPU results")
    command.add_argument("--config")
    command.add_argument("--strategy", help="Installed strategy module name")
    command.add_argument("--input")
    command.add_argument("--buy", type=float)
    command.add_argument("--sell", type=float)
    command.add_argument("--threads-per-block", dest="threads_per_block", type=int)
    command.add_argument("--expected-interval", dest="expected_interval")
    command.add_argument("--out-prefix", required=True)
    command.set_defaults(handler=execute)


def execute(args):
    from gpu_backtest.core.engine import run

    strategy, source, options, _ = job_options(args)
    run(strategy, source, args.out_prefix, **options)
