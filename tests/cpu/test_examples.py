"""Example-specific CPU truth lives outside generic engine tests."""

import numpy as np
import pytest
from gpu_backtest_examples.rsi import strategy

from gpu_backtest.core.indicators import prepare_tables


@pytest.mark.parametrize("fee,expected", [(0.0, 50.0), (0.0015, 49.625)])
def test_rsi_cpu_hand_calculated_trade(fee, expected):
    bars = [
        (100, 101, 99, 100, 10),
        (95, 96, 89, 90, 10),
        (80, 81, 79, 80, 10),
        (90, 101, 90, 100, 10),
        (110, 121, 110, 120, 10),
        (120, 121, 119, 120, 10),
        (120, 121, 119, 120, 10),
    ]
    raw = np.array(bars, dtype=np.float32)
    arrays = [raw[:, i] for i in (3, 0, 1, 2, 4)]
    from types import SimpleNamespace

    dims = SimpleNamespace(
        ENTRY_DIMS=[("p_e", 2, 2, 1, False)],
        EXIT_DIMS=[("p_x", 2, 2, 1, False)],
        TABLES=strategy.TABLES,
    )
    tables, maps = prepare_tables(dims, *arrays)
    result = strategy.reference(
        *arrays, tables, maps, [2.0, 30.0, 0.0, 0.0], [2.0, 70.0, 0.0, 0.0], fee, fee
    )
    assert result == pytest.approx(expected, abs=1e-6)
