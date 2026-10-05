"""Validate GPU reductions and persist their raw arrays without ranking or analysis."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .grid import space_count

RESULT_KEYS = ("entry_sum", "entry_sumsq", "exit_sum", "exit_sumsq")


def validate_reduction_outputs(entry_sum, entry_sumsq, exit_sum, exit_sumsq):
    """Check that both reduction passes have matching grand totals."""
    arrays = {
        "entry_sum": np.asarray(entry_sum, dtype=np.float64),
        "entry_sumsq": np.asarray(entry_sumsq, dtype=np.float64),
        "exit_sum": np.asarray(exit_sum, dtype=np.float64),
        "exit_sumsq": np.asarray(exit_sumsq, dtype=np.float64),
    }
    for name, values in arrays.items():
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"{name} must be a nonempty finite one-dimensional array")
    for side in ("entry", "exit"):
        if arrays[f"{side}_sum"].shape != arrays[f"{side}_sumsq"].shape:
            raise RuntimeError(f"{side} sum/sumsq shapes must match")
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


def _dimensions(dims):
    return [
        [name, *[float(v) if is_float else int(v) for v in (low, high, step)], is_float]
        for name, low, high, step, is_float in dims
    ]


def write_results(
    prefix,
    strategy_name,
    input_csv,
    e_dims,
    x_dims,
    es,
    esq,
    xs,
    xsq,
    reduction_checks,
    verbose,
    *,
    buy,
    sell,
    engine="gpu_backtest.core.engine",
):
    """Save the four float64 arrays in parameter-index order, plus their schema."""
    arrays = dict(zip(RESULT_KEYS, (es, esq, xs, xsq)))
    entries, exits = space_count(e_dims), space_count(x_dims)
    for key, value in arrays.items():
        count = entries if key.startswith("entry_") else exits
        if value.dtype != np.float64 or value.shape != (count,):
            raise RuntimeError(f"{key} must have float64 dtype and shape ({count},)")
    path = Path(f"{prefix}_results.npz")
    manifest_path = Path(f"{prefix}_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest = {
        "schema_version": 2,
        "engine": engine,
        "strategy": strategy_name,
        "source_file": Path(input_csv).name,
        "entry_dims": _dimensions(e_dims),
        "exit_dims": _dimensions(x_dims),
        "entry_count": entries,
        "exit_count": exits,
        "pair_count": entries * exits,
        "buy": float(buy),
        "sell": float(sell),
        "parameter_order": "Cartesian index order; last dimension varies fastest",
        "reduction_semantics": {
            "entry": "Each entry group sums returns across all exit combinations",
            "exit": "Each exit group sums returns across all entry combinations",
            "return_dtype": "float32",
            "square_dtype": "float32",
            "accumulator_dtype": "float64",
        },
        "reduction_checks": reduction_checks,
        "outputs": {
            "file": path.name,
            "format": "npz",
            "arrays": {
                key: {"dtype": str(a.dtype), "shape": list(a.shape)} for key, a in arrays.items()
            },
        },
    }
    try:
        with temp_path.open("wb") as handle:
            np.savez(handle, **arrays)
        with temp_path.open("rb") as handle:
            manifest["outputs"]["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # The manifest marks a completed pair. Invalidate it before replacing data
        # so a failed publication cannot leave new arrays with stale parameters.
        manifest_path.unlink(missing_ok=True)
        os.replace(temp_path, path)
        os.replace(temp_manifest, manifest_path)
    finally:
        temp_path.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
    if verbose:
        print(f"  -> {path} (raw grouped arrays)")
        print(f"  -> {manifest_path}")
    return manifest_path
