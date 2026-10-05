"""Pluggable GPU parameter sweeps and grouped-return analysis."""

__version__ = "0.5.0"


def resolve_strategy_name(value):
    """Keep the old demo shorthand at the API edge, outside the engine core."""
    if isinstance(value, str) and value in (
        "rsi_meanrev",
        "gpu_backtest.strategies.rsi_meanrev",
    ):
        return "gpu_backtest_examples.rsi.strategy"
    return value


def run(*args, **kwargs):
    """Run a strategy module or object; see gpu_backtest.core.engine.run."""
    from .core.engine import run as run_engine

    if args:
        args = (resolve_strategy_name(args[0]), *args[1:])
    elif "strategy_name" in kwargs:
        kwargs["strategy_name"] = resolve_strategy_name(kwargs["strategy_name"])

    return run_engine(*args, **kwargs)


__all__ = ["run", "__version__"]
