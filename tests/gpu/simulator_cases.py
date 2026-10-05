"""Explicitly collected only in an isolated CUDA-simulator process."""

import pytest
from fixtures import check_matrix, check_rsi_reference, rsi_bars, single_rsi


def test_known_matrix_external_module_and_determinism(tmp_path, monkeypatch):
    check_matrix(tmp_path, monkeypatch)


def test_rsi_matches_cpu_reference(tmp_path):
    check_rsi_reference(tmp_path)


def test_packaged_gpu_check_numeric_anchors(tmp_path):
    from gpu_backtest_tools.checks.gpu import check_numerics

    assert check_numerics(tmp_path) == {
        "known_matrix": "pass",
        "repeat_determinism": "pass",
        "rsi_hand_calculation": "pass",
    }


@pytest.mark.parametrize(
    "exit_open,fee,expected",
    [(120, 0.0, 50.0), (120, 0.0015, 49.625), (60, 0.0, -25.0), (60, 0.0015, -25.2625)],
)
def test_rsi_hand_calculated_round_trip(tmp_path, exit_open, fee, expected):
    actual = single_rsi(tmp_path / "bars.csv", rsi_bars(exit_open), buy=fee, sell=fee)
    assert actual == pytest.approx(expected, abs=0.0001)


def test_no_entry_returns_zero(tmp_path):
    assert single_rsi(tmp_path / "bars.csv", rsi_bars(), buy_level=0) == 0


def test_final_pending_entry_is_not_filled(tmp_path):
    bars = [
        (100, 101, 99, 100, 10),
        (100, 101, 99, 100, 10),
        (100, 101, 89, 90, 10),
        (1, 81, 0.5, 80, 10),
    ]
    assert single_rsi(tmp_path / "bars.csv", bars) == 0


def test_open_position_is_marked_at_final_open_without_sell_fee(tmp_path):
    bars = [
        (100, 101, 99, 100, 10),
        (95, 96, 89, 90, 10),
        (80, 81, 79, 80, 10),
        (70, 71, 69, 70, 10),
        (60, 71, 59, 70, 10),
    ]
    actual = single_rsi(tmp_path / "bars.csv", bars, buy=0.0015, sell=0.5)
    assert actual == pytest.approx(-25.15, abs=0.0001)
