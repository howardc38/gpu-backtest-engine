"""Pluggable GPU parameter sweeps and grouped-return analysis."""

__version__ = "0.2.0"


def run(*args, **kwargs):
    """Run a strategy module or object; see gpu_backtest.engine.run."""
    from .engine import run as run_engine

    return run_engine(*args, **kwargs)


__all__ = ["run", "__version__"]
