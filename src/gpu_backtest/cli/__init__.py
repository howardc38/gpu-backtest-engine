"""Command-line adapters for the engine and optional tools."""

import argparse

from . import analysis, backtest, tools


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gpu-backtest")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for group in (backtest, analysis, tools):
        group.register(subparsers)
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, ImportError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"gpu-backtest: {exc}\n")
    return 0
