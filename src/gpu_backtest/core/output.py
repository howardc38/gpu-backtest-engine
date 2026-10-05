"""Validate grouped outputs and write ranked CSVs and their manifest."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import statistics as cs
from .grid import decode_values

ROUND_TRIP_FLOAT_FORMAT = "%.17g"


def validate_reduction_outputs(entry_sum, entry_sumsq, exit_sum, exit_sumsq):
    """Check that both reduction passes have matching grand totals."""
    arrays = {
        "entry_sum": np.asarray(entry_sum, dtype=np.float64),
        "entry_sumsq": np.asarray(entry_sumsq, dtype=np.float64),
        "exit_sum": np.asarray(exit_sum, dtype=np.float64),
        "exit_sumsq": np.asarray(exit_sumsq, dtype=np.float64),
    }
    for name, values in arrays.items():
        if values.size == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"{name} is empty or contains non-finite values")
    if (arrays["entry_sumsq"] < 0).any() or (arrays["exit_sumsq"] < 0).any():
        raise RuntimeError("Reduction sum-of-squares values must be non-negative")
    checks = {}
    for name, left, right in (
        (
            "sum",
            arrays["entry_sum"].sum(dtype=np.float64),
            arrays["exit_sum"].sum(dtype=np.float64),
        ),
        (
            "sumsq",
            arrays["entry_sumsq"].sum(dtype=np.float64),
            arrays["exit_sumsq"].sum(dtype=np.float64),
        ),
    ):
        difference = abs(left - right)
        tolerance = 1e-06 + 1e-12 * max(abs(left), abs(right), 1.0)
        if difference > tolerance:
            raise RuntimeError(
                f"Entry/exit grand {name} mismatch: diff={difference:.17g}, tolerance={tolerance:.17g}"
            )
        checks[name] = {"entry": float(left), "exit": float(right), "difference": float(difference)}
    return checks


def _format_parameter(value):
    return np.format_float_positional(float(value), trim="-")


def write_top_csv(
    prefix,
    strategy_name,
    input_csv,
    e_dims,
    x_dims,
    es,
    esq,
    xs,
    xsq,
    top_n,
    reduction_checks,
    verbose,
):
    output_records = {}
    for side, dims, s, sq, n1 in (
        ("entry", e_dims, es, esq, len(xs)),
        ("exit", x_dims, xs, xsq, len(es)),
    ):
        st = cs.stats_from_groups(s, sq, n1, s.sum(), sq.sum(), len(s) * n1)
        cols = {
            d[0]: np.array([decode_values(dims, i)[j] for i in range(len(s))])
            for j, d in enumerate(dims)
        }
        df = pd.DataFrame({**cols, **st})
        if not np.isfinite(df["effect_size"].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"{side} effect_size contains non-finite values")
        key_columns = [d[0] for d in dims]
        df = df.sort_values(
            ["effect_size", *key_columns],
            ascending=[False, *[True] * len(key_columns)],
            kind="mergesort",
        ).head(min(top_n, len(s)))
        serialized = df.copy()
        for dim in dims:
            if dim[4]:
                serialized[dim[0]] = serialized[dim[0]].map(_format_parameter)
        path = f"{prefix}_top_{side}.csv"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        temp_path = f"{path}.tmp"
        serialized.to_csv(temp_path, index=False, float_format=ROUND_TRIP_FLOAT_FORMAT)
        os.replace(temp_path, path)
        output_records[side] = {
            "file": Path(path).name,
            "rows": int(len(df)),
            "key_columns": key_columns,
            "columns": list(serialized.columns),
            "sort": ["effect_size descending", "parameter columns ascending"],
        }
        if verbose:
            print(f"  -> {path} ({len(df)} rows)")
    manifest = {
        "schema_version": 1,
        "engine": "gpu_backtest.engine",
        "strategy": strategy_name,
        "source_file": Path(input_csv).name,
        "entry_count": int(len(es)),
        "exit_count": int(len(xs)),
        "pair_count": int(len(es) * len(xs)),
        "top_n": int(top_n),
        "float_serialization": "IEEE-754 float64 round-trip (17 significant digits)",
        "reduction_checks": reduction_checks,
        "outputs": output_records,
    }
    manifest_path = f"{prefix}_top_manifest.json"
    temp_manifest = f"{manifest_path}.tmp"
    Path(temp_manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    if verbose:
        print(f"  -> {manifest_path}")
