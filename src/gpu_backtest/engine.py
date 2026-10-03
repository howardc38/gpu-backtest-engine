"""Two-pass GPU parameter sweeps with deterministic grouped statistics."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import cuda, float32, float64

from . import statistics as cs
from .strategy import load_strategy, validate_strategy
from .tables import build_table

MAX_TABLES = 4
MAX_DIMS = 4
INITIAL_CAPITAL = 10000.0
REQUIRED_MARKET_COLUMNS = ("Time", "Open", "High", "Low", "Close", "Volume")
ROUND_TRIP_FLOAT_FORMAT = "%.17g"


def dim_count(d):
    name, lo, hi, step, is_float = d
    return int(round((hi - lo) / step) + 1 if is_float else (hi - lo) // step + 1)


def space_count(dims):
    n = 1
    for d in dims:
        n *= dim_count(d)
    return n


def validate_market_data(input_csv, expected_interval=None):
    """Load and validate an OHLCV file without repairing invalid input."""
    data = pd.read_csv(input_csv)
    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{input_csv}: missing required columns: {missing}")
    if len(data) < 2:
        raise ValueError(f"{input_csv}: at least two bars are required")
    try:
        data["Time"] = pd.to_datetime(data["Time"], errors="raise")
    except Exception as exc:
        raise ValueError(f"{input_csv}: Time contains invalid timestamps") from exc
    if data["Time"].isna().any():
        raise ValueError(f"{input_csv}: Time contains missing timestamps")
    if data["Time"].duplicated().any():
        raise ValueError(f"{input_csv}: Time contains duplicate timestamps")
    if not data["Time"].is_monotonic_increasing:
        raise ValueError(f"{input_csv}: Time must be strictly increasing")
    numeric = data[list(REQUIRED_MARKET_COLUMNS[1:])].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{input_csv}: OHLCV values must all be finite numbers")
    data.loc[:, list(REQUIRED_MARKET_COLUMNS[1:])] = numeric
    prices = numeric[["Open", "High", "Low", "Close"]]
    if (prices <= 0).any().any():
        raise ValueError(f"{input_csv}: OHLC prices must be positive")
    if (numeric["Volume"] < 0).any():
        raise ValueError(f"{input_csv}: Volume must be non-negative")
    if (numeric["High"] < prices[["Open", "Low", "Close"]].max(axis=1)).any():
        raise ValueError(f"{input_csv}: High is below another OHLC price")
    if (numeric["Low"] > prices[["Open", "High", "Close"]].min(axis=1)).any():
        raise ValueError(f"{input_csv}: Low is above another OHLC price")
    if expected_interval:
        try:
            interval = pd.to_timedelta(expected_interval)
        except Exception as exc:
            raise ValueError(f"Invalid expected interval: {expected_interval}") from exc
        if interval <= pd.Timedelta(0):
            raise ValueError("Expected interval must be positive")
        deltas = data["Time"].diff().iloc[1:]
        bad = deltas != interval
        if bad.any():
            first_bad = int(np.flatnonzero(bad.to_numpy())[0]) + 1
            raise ValueError(
                f"{input_csv}: bar interval at row {first_bad} is {deltas.iloc[first_bad - 1]}, expected {interval}"
            )
    return data


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


def dim_arrays(dims):
    """Return padded lower bounds, steps, counts, and dimension count."""
    lo = np.zeros(MAX_DIMS, np.float64)
    st = np.ones(MAX_DIMS, np.float64)
    ct = np.ones(MAX_DIMS, np.int64)
    for j, d in enumerate(dims):
        lo[j] = float(d[1])
        st[j] = float(d[3])
        ct[j] = dim_count(d)
    return (lo, st, ct, len(dims))


def decode_values(dims, idx):
    """Decode a flat index; the last dimension varies fastest."""
    cnts = [dim_count(d) for d in dims]
    vals = [0.0] * len(dims)
    rem = idx
    for j in range(len(dims) - 1, -1, -1):
        i = rem % cnts[j]
        rem //= cnts[j]
        v = dims[j][1] + i * dims[j][3]
        vals[j] = float(v) if dims[j][4] else int(v)
    return vals


def int_dim_values(dims, names):
    out = set()
    for d in dims:
        if d[0] in names:
            assert not d[4], f"Table window dimension {d[0]} must contain integers"
            out |= {int(d[1] + i * d[3]) for i in range(dim_count(d))}
    return sorted(out)


def prepare_tables(strategy, close, open_, high, low, vol):
    all_dims = list(strategy.ENTRY_DIMS) + list(strategy.EXIT_DIMS)
    tabs, rms = ([], [])
    for kind, source, dim_names in strategy.TABLES:
        windows = int_dim_values(all_dims, dim_names)
        assert windows, f"Table {kind} has no integer window values for {dim_names}"
        t, rm = build_table(kind, source, windows, close, open_, high, low, vol)
        tabs.append(t)
        rms.append(rm)
    while len(tabs) < MAX_TABLES:
        tabs.append(np.zeros((1, 1), np.float32))
        rms.append(np.zeros(1, np.int64))
    return (tabs, rms)


def build_kernels(strategy):
    algo = strategy.make_device_fn(cuda)

    @cuda.jit(device=True, inline=True)
    def _decode(idx, lo, st, ct, nd, out):
        rem = idx
        for j in range(nd - 1, -1, -1):
            i = rem % ct[j]
            rem //= ct[j]
            out[j] = lo[j] + i * st[j]

    @cuda.jit
    def entry_reduce(
        close,
        open_,
        high,
        low,
        vol,
        t0,
        t1,
        t2,
        t3,
        rm0,
        rm1,
        rm2,
        rm3,
        e_lo,
        e_st,
        e_ct,
        n_e,
        x_lo,
        x_st,
        x_ct,
        n_x,
        entry_count,
        exit_count,
        num_bars,
        buy,
        sell,
        out_sum,
        out_sumsq,
    ):
        e = cuda.grid(1)
        if e >= entry_count:
            return
        ep = cuda.local.array(4, float64)
        xp = cuda.local.array(4, float64)
        _decode(e, e_lo, e_st, e_ct, n_e, ep)
        s = float64(0.0)
        sq = float64(0.0)
        for x in range(exit_count):
            _decode(x, x_lo, x_st, x_ct, n_x, xp)
            tr64 = algo(
                close,
                open_,
                high,
                low,
                vol,
                t0,
                t1,
                t2,
                t3,
                rm0,
                rm1,
                rm2,
                rm3,
                ep,
                xp,
                num_bars,
                buy,
                sell,
            )
            tr = float32(tr64)
            s += float64(tr)
            sq += float64(float32(tr * tr))
        out_sum[e] = s
        out_sumsq[e] = sq

    @cuda.jit
    def exit_reduce(
        close,
        open_,
        high,
        low,
        vol,
        t0,
        t1,
        t2,
        t3,
        rm0,
        rm1,
        rm2,
        rm3,
        e_lo,
        e_st,
        e_ct,
        n_e,
        x_lo,
        x_st,
        x_ct,
        n_x,
        entry_count,
        exit_count,
        num_bars,
        buy,
        sell,
        out_sum,
        out_sumsq,
    ):
        x = cuda.grid(1)
        if x >= exit_count:
            return
        ep = cuda.local.array(4, float64)
        xp = cuda.local.array(4, float64)
        _decode(x, x_lo, x_st, x_ct, n_x, xp)
        s = float64(0.0)
        sq = float64(0.0)
        for e in range(entry_count):
            _decode(e, e_lo, e_st, e_ct, n_e, ep)
            tr64 = algo(
                close,
                open_,
                high,
                low,
                vol,
                t0,
                t1,
                t2,
                t3,
                rm0,
                rm1,
                rm2,
                rm3,
                ep,
                xp,
                num_bars,
                buy,
                sell,
            )
            tr = float32(tr64)
            s += float64(tr)
            sq += float64(float32(tr * tr))
        out_sum[x] = s
        out_sumsq[x] = sq

    return (entry_reduce, exit_reduce)


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
        or not 1 <= threads_per_block <= 1024
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
        _write_top_csv(
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


def _format_parameter(value):
    return np.format_float_positional(float(value), trim="-")


def _write_top_csv(
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
