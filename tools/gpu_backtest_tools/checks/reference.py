"""Small-grid CPU oracle used by tests and hardware checks, never by the GPU runner."""

import numpy as np

from gpu_backtest.core.grid import decode_values, space_count
from gpu_backtest.core.indicators import prepare_tables
from gpu_backtest.core.strategy import MAX_DIMS, validate_strategy


def reference_group_sums(strategy, close, open_, high, low, vol, e_dims, x_dims, buy, sell):
    """Compute small-grid grouped returns with a strategy CPU reference."""
    strategy = validate_strategy(strategy, e_dims, x_dims)
    if not callable(getattr(strategy, "reference", None)):
        raise ValueError("CPU comparisons require a callable strategy.reference")

    class _S:
        ENTRY_DIMS, EXIT_DIMS, TABLES = (e_dims, x_dims, strategy.TABLES)

    tabs, rms = prepare_tables(_S, close, open_, high, low, vol)
    ec, xc = (space_count(e_dims), space_count(x_dims))
    es = np.zeros(ec)
    xs = np.zeros(xc)
    for e in range(ec):
        ep = decode_values(e_dims, e) + [0.0] * (MAX_DIMS - len(e_dims))
        for x in range(xc):
            xp = decode_values(x_dims, x) + [0.0] * (MAX_DIMS - len(x_dims))
            tr = np.float32(
                strategy.reference(close, open_, high, low, vol, tabs, rms, ep, xp, buy, sell)
            )
            es[e] += float(tr)
            xs[x] += float(tr)
    return (es, xs)
