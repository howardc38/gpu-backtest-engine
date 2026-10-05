from types import SimpleNamespace

import pytest

from gpu_backtest.core.engine import run
from gpu_backtest.core.grid import decode_values
from gpu_backtest.core.strategy import load_strategy, validate_strategy


def specification(**changes):
    result = dict(
        NAME="test",
        ENTRY_DIMS=[("entry", 1, 2, 1, False)],
        EXIT_DIMS=[("exit", 1, 2, 1, False)],
        TABLES=[],
        make_device_fn=lambda cuda: None,
    )
    result.update(changes)
    return SimpleNamespace(**result)


def test_builtin_and_module_object_loading():
    strategy = load_strategy("gpu_backtest_examples.rsi.strategy")
    assert load_strategy(strategy) is strategy
    assert load_strategy("gpu_backtest_examples.rsi.strategy") is strategy
    validate_strategy(strategy)


@pytest.mark.parametrize(
    "entry,error",
    [
        ([], "1 to 4"),
        ([("entry", 1, 2, 0, False)], "positive step"),
        ([("entry", 2, 1, 1, False)], "low <= high"),
        ([("entry", 1, 2, 0.5, False)], "integer bounds"),
        ([("entry", 0.0, 1.0, 0.3, True)], "lie on"),
        ([("effect_size", 1, 2, 1, False)], "reserved"),
        ([("entry", 1, float("inf"), 1, False)], "finite"),
        ([("entry", 1, 2, 1, "i")], "boolean"),
        ([("entry", 1, 2, 1, False)] * 5, "1 to 4"),
    ],
)
def test_invalid_dimensions_fail_before_market_or_gpu_work(entry, error):
    with pytest.raises(ValueError, match=error):
        run(specification(ENTRY_DIMS=entry), "missing.csv", None, verbose=False)


def test_override_order_and_names_are_contractual():
    with pytest.raises(ValueError, match="names and order"):
        validate_strategy(specification(), [("other", 1, 2, 1, False)])


@pytest.mark.parametrize(
    "tables,error",
    [
        ([("rsi", "close", ["unknown"])], "Unknown table window"),
        ([("unknown", "close", ["entry"])], "Unknown indicator"),
        ([("rsi", "unknown", ["entry"])], "Unknown indicator"),
        ([("rsi", "close", [])], "window dimension names"),
        ([("rsi", "close", ["entry"])] * 5, "at most 4"),
    ],
)
def test_invalid_tables_are_rejected(tables, error):
    with pytest.raises(ValueError, match=error):
        validate_strategy(specification(TABLES=tables))


def test_windows_must_be_positive_integer_dimensions():
    with pytest.raises(ValueError, match="positive integers"):
        validate_strategy(
            specification(
                ENTRY_DIMS=[("entry", 0, 1, 1, False)], TABLES=[("rsi", "close", ["entry"])]
            )
        )


def test_parameter_decode_preserves_sub_micro_precision():
    dims = [("fraction", 1e-07, 2e-07, 1e-07, True)]
    assert decode_values(dims, 0) == [1e-07]
    assert decode_values(dims, 1) == [2e-07]


@pytest.mark.parametrize("value", ["", "../strategy", "thing:callback"])
def test_invalid_import_names(value):
    with pytest.raises(ValueError, match="module name"):
        load_strategy(value)
