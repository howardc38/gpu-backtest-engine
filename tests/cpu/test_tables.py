import numpy as np

from gpu_backtest.core.indicators import build_table


def _bt(kind, source, windows, close, open_=None, high=None, low=None, volume=None):
    n = len(close)
    close = np.asarray(close, np.float64)
    open_ = np.asarray(open_ if open_ is not None else close, np.float64)
    high = np.asarray(high if high is not None else close, np.float64)
    low = np.asarray(low if low is not None else close, np.float64)
    volume = np.asarray(volume if volume is not None else [10.0] * n, np.float64)
    return build_table(kind, source, windows, close, open_, high, low, volume)


def test_roll_mean_and_std_hand():
    tab, rm = _bt("roll_std", "close", [3], [1, 2, 3, 4])
    np.testing.assert_allclose(tab[rm[3], 2:], [1.0, 1.0], atol=1e-06)
    assert np.isnan(tab[rm[3], 1])
    tab, rm = _bt("roll_mean", "close", [3], [1, 2, 3, 4])
    np.testing.assert_allclose(tab[rm[3], 2:], [2.0, 3.0], atol=1e-06)


def test_roll_max_min_hand():
    tab, rm = _bt("roll_max", "close", [2], [5, 3, 8, 1])
    np.testing.assert_allclose(tab[rm[2], 1:], [5, 8, 8], atol=1e-06)
    tab, rm = _bt("roll_min", "close", [2], [5, 3, 8, 1])
    np.testing.assert_allclose(tab[rm[2], 1:], [3, 3, 1], atol=1e-06)


def test_ema_hand():
    tab, rm = _bt("ema", "close", [3], [2, 4, 6])
    np.testing.assert_allclose(tab[rm[3]], [2.0, 3.0, 4.5], atol=1e-06)


def test_rsi_hand():
    tab, rm = _bt("rsi", "close", [2], [100, 90, 80, 100, 120])
    r = tab[rm[2]]
    assert np.isnan(r[0])
    np.testing.assert_allclose(r[1:], [0.0, 0.0, 66.666667, 85.714286], atol=0.001)


def test_atr_hand():
    high = [101, 101, 107]
    low = [99, 99, 105]
    close = [100, 100, 106]
    tab, rm = _bt("atr", None, [2], close, high=high, low=low)
    np.testing.assert_allclose(tab[rm[2]], [2.0, 2.0, 4.5], atol=1e-06)


def test_rsi_all_up_is_100():
    tab, rm = _bt("rsi", "close", [2], [1, 2, 3, 4, 5])
    np.testing.assert_allclose(tab[rm[2], 1:], [100.0] * 4, atol=1e-06)
