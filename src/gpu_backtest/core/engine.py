"""Public GPU runner; orchestration delegates to the focused core modules."""

import time

import numpy as np
from numba import cuda

from .data import validate_market_data
from .grid import dim_arrays, space_count
from .indicators import prepare_tables
from .kernels import build_kernels
from .output import validate_reduction_outputs, write_top_csv
from .strategy import MAX_DIMS, load_strategy, validate_strategy


def run(
    strategy_name,
    input_csv,
    out_prefix,
    buy=0.0015,
    sell=0.0015,
    top_n=10000,
    threads_per_block=128,
    entry_dims=None,
    exit_dims=None,
    expected_interval=None,
    verbose=True,
):
    strategy = load_strategy(strategy_name)
    e_dims = strategy.ENTRY_DIMS if entry_dims is None else entry_dims
    x_dims = strategy.EXIT_DIMS if exit_dims is None else exit_dims
    if not e_dims or not x_dims or len(e_dims) > MAX_DIMS or (len(x_dims) > MAX_DIMS):
        raise ValueError(f"Entry and exit dimensions must each contain 1 to {MAX_DIMS} dimensions")
    if not np.isfinite([buy, sell]).all() or buy < 0 or sell < 0:
        raise ValueError("Commissions must be finite and non-negative")
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if (
        isinstance(threads_per_block, bool)
        or not isinstance(threads_per_block, (int, np.integer))
        or (not 1 <= threads_per_block <= 1024)
    ):
        raise ValueError("threads_per_block must be an integer between 1 and 1024")
    validate_strategy(strategy, e_dims, x_dims)
    entry_count, exit_count = (space_count(e_dims), space_count(x_dims))
    if out_prefix and (entry_count < 2 or exit_count < 2):
        raise ValueError(
            "Effect-size isolation requires at least two entry and two exit combinations"
        )
    if verbose:
        print(
            f"[{strategy.NAME}] entry={entry_count} × exit={exit_count} = {entry_count * exit_count:,} combos"
        )
    data = validate_market_data(input_csv, expected_interval=expected_interval)
    close = data["Close"].values.astype(np.float32)
    open_ = data["Open"].values.astype(np.float32)
    high = data["High"].values.astype(np.float32)
    low = data["Low"].values.astype(np.float32)
    vol = data["Volume"].values.astype(np.float32)
    for name, values in (
        ("Close", close),
        ("Open", open_),
        ("High", high),
        ("Low", low),
        ("Volume", vol),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{input_csv}: {name} cannot be represented as finite float32 values")
    num_bars = len(data)

    class _S:
        ENTRY_DIMS, EXIT_DIMS, TABLES = (e_dims, x_dims, strategy.TABLES)

    tabs, rms = prepare_tables(_S, close, open_, high, low, vol)
    d = [cuda.to_device(a) for a in (close, open_, high, low, vol)]
    dt = [cuda.to_device(t) for t in tabs]
    dr = [cuda.to_device(r) for r in rms]
    e_lo, e_st, e_ct, n_e = dim_arrays(e_dims)
    x_lo, x_st, x_ct, n_x = dim_arrays(x_dims)
    de = [cuda.to_device(a) for a in (e_lo, e_st, e_ct)]
    dx = [cuda.to_device(a) for a in (x_lo, x_st, x_ct)]
    es = cuda.device_array(entry_count, np.float64)
    esq = cuda.device_array(entry_count, np.float64)
    xs = cuda.device_array(exit_count, np.float64)
    xsq = cuda.device_array(exit_count, np.float64)
    entry_k, exit_k = build_kernels(strategy)
    args = (
        *d,
        *dt,
        *dr,
        *de,
        n_e,
        *dx,
        n_x,
        entry_count,
        exit_count,
        num_bars,
        np.float64(buy),
        np.float64(sell),
    )
    t0 = time.time()
    eb = (entry_count + threads_per_block - 1) // threads_per_block
    entry_k[eb, threads_per_block](*args, es, esq)
    cuda.synchronize()
    if verbose:
        print(f"Pass1 (entry reduce) {time.time() - t0:.2f}s")
    t1 = time.time()
    xb = (exit_count + threads_per_block - 1) // threads_per_block
    exit_k[xb, threads_per_block](*args, xs, xsq)
    cuda.synchronize()
    if verbose:
        print(f"Pass2 (exit reduce)  {time.time() - t1:.2f}s")
    es, esq, xs, xsq = (
        es.copy_to_host(),
        esq.copy_to_host(),
        xs.copy_to_host(),
        xsq.copy_to_host(),
    )
    reduction_checks = validate_reduction_outputs(es, esq, xs, xsq)
    if verbose:
        print(f"grand total check: diff={reduction_checks['sum']['difference']:.2e}")
    out = {"entry_sum": es, "entry_sumsq": esq, "exit_sum": xs, "exit_sumsq": xsq}
    if out_prefix:
        write_top_csv(
            out_prefix,
            strategy.NAME,
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
        )
    return out
