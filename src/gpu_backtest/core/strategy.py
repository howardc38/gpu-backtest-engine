"""Load builtin or independently installed strategy plugins and validate their specs."""

import importlib
import math
from numbers import Real

MAX_DIMS = 4
MAX_TABLES = 4
MAX_INDEX = 2**63 - 1
TABLE_KINDS = {"roll_max", "roll_min", "roll_mean", "roll_std", "ema", "rsi", "atr"}
TABLE_SOURCES = {"close", "open", "high", "low", "volume"}


def load_strategy(value):
    """Import a strategy module or accept an already loaded strategy object."""
    if isinstance(value, str):
        if not value or not all(part.isidentifier() for part in value.split(".")):
            raise ValueError("Strategy must be a Python module name")
        module_name = value
        value = importlib.import_module(module_name)
    for attribute in ("NAME", "ENTRY_DIMS", "EXIT_DIMS", "TABLES", "make_device_fn"):
        if not hasattr(value, attribute):
            raise ValueError(f"Strategy is missing {attribute}")
    if not isinstance(value.NAME, str) or not value.NAME:
        raise ValueError("Strategy NAME must be a non-empty string")
    if not callable(value.make_device_fn):
        raise ValueError("Strategy make_device_fn must be callable")
    return value


def _validate_dims(dims, side):
    if not isinstance(dims, (list, tuple)) or not 1 <= len(dims) <= MAX_DIMS:
        raise ValueError(f"{side} dimensions must contain 1 to {MAX_DIMS} dimensions")
    names = []
    count = 1
    for dim in dims:
        if not isinstance(dim, (list, tuple)) or len(dim) != 5:
            raise ValueError("Each dimension must be (name, low, high, step, is_float)")
        name, low, high, step, is_float = dim
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"Invalid parameter name: {name!r}")
        if type(is_float) is not bool:
            raise ValueError(f"{name}: is_float must be a boolean")
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in (low, high, step)):
            raise ValueError(f"{name}: bounds and step must be numbers")
        if not all(math.isfinite(v) for v in (low, high, step)) or step <= 0 or low > high:
            raise ValueError(f"{name}: require finite bounds, positive step, and low <= high")
        if not is_float and any(int(v) != v for v in (low, high, step)):
            raise ValueError(f"{name}: integer dimensions require integer bounds and step")
        ratio = (high - low) / step
        if not math.isfinite(ratio) or not math.isclose(
            ratio, round(ratio), rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError(f"{name}: high must lie on the low + index * step grid")
        count *= round(ratio) + 1
        if count > MAX_INDEX:
            raise ValueError(f"{side} parameter space exceeds int64 indexing")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError(f"{side} parameter names must be unique")
    return names, count


def validate_strategy(strategy, entry_dims=None, exit_dims=None):
    """Validate dimensions, overrides, table slots, and indicator window references."""
    strategy = load_strategy(strategy)
    original_entry, _ = _validate_dims(strategy.ENTRY_DIMS, "Entry")
    original_exit, _ = _validate_dims(strategy.EXIT_DIMS, "Exit")
    entry_dims = strategy.ENTRY_DIMS if entry_dims is None else entry_dims
    exit_dims = strategy.EXIT_DIMS if exit_dims is None else exit_dims
    entry_names, entry_count = _validate_dims(entry_dims, "Entry")
    exit_names, exit_count = _validate_dims(exit_dims, "Exit")
    if entry_names != original_entry or exit_names != original_exit:
        raise ValueError("Dimension overrides must preserve strategy parameter names and order")
    names = entry_names + exit_names
    if len(names) != len(set(names)):
        raise ValueError("Entry and exit parameter names must be distinct")
    if entry_count * exit_count > MAX_INDEX:
        raise ValueError("Combined parameter space exceeds int64 indexing")
    if not isinstance(strategy.TABLES, (list, tuple)) or len(strategy.TABLES) > MAX_TABLES:
        raise ValueError(f"Strategy TABLES must contain at most {MAX_TABLES} tables")
    dimensions = {d[0]: d for d in [*entry_dims, *exit_dims]}
    for table in strategy.TABLES:
        if not isinstance(table, (list, tuple)) or len(table) != 3:
            raise ValueError("Each table must be (kind, source, dimension_names)")
        kind, source, window_names = table
        if kind not in TABLE_KINDS or (kind != "atr" and source not in TABLE_SOURCES):
            raise ValueError(f"Unknown indicator kind/source: {kind!r}/{source!r}")
        if not isinstance(window_names, (list, tuple)) or not window_names:
            raise ValueError("Indicator tables require window dimension names")
        for name in window_names:
            if name not in dimensions:
                raise ValueError(f"Unknown table window dimension: {name}")
            dim = dimensions[name]
            if dim[4] or dim[1] < 1:
                raise ValueError(f"Table window dimension {name} must contain positive integers")
    return strategy
