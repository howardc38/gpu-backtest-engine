"""Reproducible RSI CPU/GPU comparison and optional billion-pair capacity run."""

import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numba
import numpy as np
from numba import config, cuda

from .benchmark_cpu import build_cpu_reducer
from .engine import (
    _write_top_csv,
    build_kernels,
    dim_arrays,
    prepare_tables,
    run,
    space_count,
    validate_market_data,
    validate_reduction_outputs,
)
from .strategies import rsi_meanrev
from .strategy import validate_strategy
from .synthetic import generate_csv

RESULT_KEYS = ("entry_sum", "entry_sumsq", "exit_sum", "exit_sumsq")


def profiles():
    """Thresholds use exact binary steps so CPU/GPU parameter decoding agrees."""
    return {
        "comparison": (
            [("p_e", 2, 65, 1, False), ("buy_lvl", 20.0, 51.5, 0.5, True)],
            [("p_x", 2, 65, 1, False), ("sell_lvl", 49.0, 80.5, 0.5, True)],
        ),
        "billion": (
            [("p_e", 2, 401, 1, False), ("buy_lvl", 10, 59, 1, False)],
            [("p_x", 2, 1001, 1, False), ("sell_lvl", 50, 99, 1, False)],
        ),
    }


def cpu_args(strategy, arrays, entry_dims, exit_dims):
    validate_strategy(strategy, entry_dims, exit_dims)
    specs = SimpleNamespace(ENTRY_DIMS=entry_dims, EXIT_DIMS=exit_dims, TABLES=strategy.TABLES)
    tabs, maps = prepare_tables(specs, *arrays)
    e_lo, e_step, e_count, n_e = dim_arrays(entry_dims)
    x_lo, x_step, x_count, n_x = dim_arrays(exit_dims)
    return (
        *arrays,
        tuple(tabs),
        tuple(maps),
        e_lo,
        e_step,
        e_count,
        n_e,
        x_lo,
        x_step,
        x_count,
        n_x,
        space_count(entry_dims),
        space_count(exit_dims),
        0.0015,
        0.0015,
    )


class GPUPlan:
    """Prepare and compile actual engine kernels once, then time both reductions."""

    def __init__(self, strategy, args, threads_per_block=128):
        arrays, tabs, maps = args[:5], args[5], args[6]
        e_lo, e_step, e_count, n_e, x_lo, x_step, x_count, n_x, entries, exits, buy, sell = args[7:]
        self.entries, self.exits = entries, exits
        self.tpb = threads_per_block
        self.kernels = build_kernels(strategy)
        self.outputs = [
            cuda.device_array(count, np.float64) for count in (entries, entries, exits, exits)
        ]
        self.args = (
            *[cuda.to_device(a) for a in arrays],
            *[cuda.to_device(a) for a in tabs],
            *[cuda.to_device(a) for a in maps],
            *[cuda.to_device(a) for a in (e_lo, e_step, e_count)],
            n_e,
            *[cuda.to_device(a) for a in (x_lo, x_step, x_count)],
            n_x,
            entries,
            exits,
            len(arrays[0]),
            np.float64(buy),
            np.float64(sell),
        )

    def execute(self):
        self.kernels[0][(self.entries + self.tpb - 1) // self.tpb, self.tpb](
            *self.args, *self.outputs[:2]
        )
        cuda.synchronize()
        self.kernels[1][(self.exits + self.tpb - 1) // self.tpb, self.tpb](
            *self.args, *self.outputs[2:]
        )
        cuda.synchronize()

    def results(self):
        return tuple(array.copy_to_host() for array in self.outputs)


def _timed(call, repeats):
    values = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        values.append(time.perf_counter() - started)
    return result, values


def _cpu_info(threads):
    model = platform.processor() or platform.machine()
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    quota = Path("/sys/fs/cgroup/cpu.max")
    return {
        "model": model,
        "numba_threads": threads,
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": quota.read_text().strip() if quota.exists() else None,
    }


def run_cpu_baseline(source, prefix, entry_dims, exit_dims, reducer=None):
    """CSV through ranked output using the compiled CPU two-pass reference."""
    data = validate_market_data(source, expected_interval="1D")
    arrays = [
        data[c].to_numpy(dtype=np.float32) for c in ("Close", "Open", "High", "Low", "Volume")
    ]
    args = cpu_args(rsi_meanrev, arrays, entry_dims, exit_dims)
    reducer = reducer or build_cpu_reducer(rsi_meanrev)
    results = reducer(*args)
    checks = validate_reduction_outputs(*results)
    if prefix:
        _write_top_csv(
            str(prefix),
            "rsi_meanrev",
            str(source),
            entry_dims,
            exit_dims,
            *results,
            100,
            checks,
            False,
        )
        manifest = Path(f"{prefix}_top_manifest.json")
        content = json.loads(manifest.read_text())
        content["engine"] = "gpu_backtest.benchmark_cpu"
        manifest.write_text(json.dumps(content, indent=2) + "\n")
    return dict(zip(RESULT_KEYS, results))


def benchmark(output, *, bars=1024, cpu_threads=8, repeats=3, billion=False, billion_cpu=False):
    if config.ENABLE_CUDASIM:
        raise ValueError("Performance benchmarking requires real CUDA, not CUDASIM")
    if not cuda.is_available():
        raise ValueError("Performance benchmarking requires an available NVIDIA GPU")
    if not isinstance(repeats, int) or repeats < 1 or not isinstance(bars, int) or bars < 2:
        raise ValueError("Require positive repeats and at least two bars")
    if not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= numba.config.NUMBA_NUM_THREADS:
        raise ValueError("CPU thread count exceeds the available Numba thread pool")
    if billion_cpu and not billion:
        raise ValueError("--billion-cpu requires --billion")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = generate_csv(output.parent / "benchmark_synthetic.csv", bars)
    data = validate_market_data(source, expected_interval="1D")
    arrays = [
        data[column].to_numpy(dtype=np.float32)
        for column in ("Close", "Open", "High", "Low", "Volume")
    ]
    original_threads = numba.get_num_threads()
    numba.set_num_threads(cpu_threads)
    try:
        e_dims, x_dims = profiles()["comparison"]
        args = cpu_args(rsi_meanrev, arrays, e_dims, x_dims)
        reducer = build_cpu_reducer(rsi_meanrev)
        gpu = GPUPlan(rsi_meanrev, args)
        print(
            f"Comparison: {args[-4]:,} entry × {args[-3]:,} exit, {bars} bars, {cpu_threads} CPU threads",
            flush=True,
        )
        started = time.perf_counter()
        reducer(*args)
        cpu_warmup = time.perf_counter() - started
        started = time.perf_counter()
        gpu.execute()
        gpu_warmup = time.perf_counter() - started
        print("Measuring warmed parallel CPU and GPU reductions...", flush=True)
        cpu_results, cpu_times = _timed(lambda: reducer(*args), repeats)
        _, gpu_times = _timed(gpu.execute, repeats)
        gpu_results = gpu.results()
        differences = {}
        for key, left, right in zip(RESULT_KEYS, cpu_results, gpu_results):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-3)
            differences[key] = float(np.max(np.abs(left - right)))
        validate_reduction_outputs(*gpu_results)
        if not np.any(np.abs(gpu_results[0]) > 1e-6):
            raise RuntimeError("Benchmark strategy produced only zero returns")
        cpu_median, gpu_median = statistics.median(cpu_times), statistics.median(gpu_times)
        device = cuda.get_current_device()
        name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
        packages = {}
        for package in (
            "gpu-backtest-engine",
            "numba",
            "numba-cuda",
            "numpy",
            "pandas",
            "llvmlite",
            "cuda-bindings",
        ):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
        ).strip()
        report = {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "strategy": "rsi_meanrev",
            "synthetic_bars": bars,
            "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "cpu": _cpu_info(cpu_threads),
            "gpu": {
                "name": name,
                "compute_capability": list(device.compute_capability),
                "driver_query": driver,
            },
            "python": platform.python_version(),
            "packages": packages,
            "simulation": False,
            "comparison": {
                "entry_dims": e_dims,
                "exit_dims": x_dims,
                "entry_count": args[-4],
                "exit_count": args[-3],
                "unique_combinations": args[-4] * args[-3],
                "strategy_evaluations": 2 * args[-4] * args[-3],
                "cpu_seconds": cpu_times,
                "gpu_seconds": gpu_times,
                "cpu_median_seconds": cpu_median,
                "gpu_median_seconds": gpu_median,
                "speedup": cpu_median / gpu_median,
                "cpu_compile_and_warmup_seconds": cpu_warmup,
                "gpu_compile_and_warmup_seconds": gpu_warmup,
                "max_absolute_differences": differences,
            },
            "timing_scope": "Comparison: warmed two-pass reductions only. Shared input/table preparation, GPU transfers, compilation, result copying, statistics and CSV writing are excluded. CPU output allocation is included.",
        }
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"CPU {cpu_median:.6f}s / GPU {gpu_median:.6f}s = {cpu_median / gpu_median:.2f}x",
            flush=True,
        )
        if billion:
            e_dims, x_dims = profiles()["billion"]
            entries, exits = space_count(e_dims), space_count(x_dims)
            if entries * exits != 1_000_000_000:
                raise RuntimeError("Billion profile must contain exactly one billion pairs")
            print("Running the normal engine API on 1,000,000,000 unique pairs...", flush=True)
            started = time.perf_counter()
            large = run(
                "rsi_meanrev",
                source,
                str(output.parent / "billion"),
                entry_dims=e_dims,
                exit_dims=x_dims,
                top_n=100,
                verbose=True,
            )
            elapsed = time.perf_counter() - started
            report["billion"] = {
                "entry_dims": e_dims,
                "exit_dims": x_dims,
                "entry_count": entries,
                "exit_count": exits,
                "unique_combinations": entries * exits,
                "strategy_evaluations": 2 * entries * exits,
                "engine_end_to_end_seconds": elapsed,
                "reduction_array_bytes": sum(a.nbytes for a in large.values()),
                "hypothetical_float32_matrix_bytes": entries * exits * 4,
                "nonzero_entry_groups": int(np.count_nonzero(large["entry_sum"])),
                "timing_scope": "Normal engine run including CSV validation, indicator tables, allocation/transfers, kernel construction/JIT, two GPU passes, copies, statistics and top CSV/manifest output. Excludes pod provisioning and dependency installation.",
            }
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Billion-pair normal engine run completed in {elapsed:.3f}s.", flush=True)
            if billion_cpu:
                print(
                    "Running the complete SAME billion-pair grid on the compiled CPU baseline...",
                    flush=True,
                )
                started = time.perf_counter()
                cpu_large = run_cpu_baseline(
                    source, output.parent / "cpu_billion", e_dims, x_dims, reducer
                )
                cpu_elapsed = time.perf_counter() - started
                differences = {}
                for key in RESULT_KEYS:
                    np.testing.assert_allclose(cpu_large[key], large[key], rtol=1e-6, atol=1e-3)
                    differences[key] = float(np.max(np.abs(cpu_large[key] - large[key])))
                report["billion"].update(
                    cpu_engine_end_to_end_seconds=cpu_elapsed,
                    end_to_end_speedup=cpu_elapsed / elapsed,
                    elapsed_seconds_saved=cpu_elapsed - elapsed,
                    cpu_gpu_max_absolute_differences=differences,
                    cpu_timing_scope="Compiled CPU reference through CSV validation, indicator preparation, both passes, allocations, checks, statistics and CSV/manifest output. Reuses the reducer already compiled for the smaller comparison. No CUDA simulation. One full measured run, not an extrapolation.",
                )
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"Full billion grid: CPU {cpu_elapsed:.3f}s / GPU {elapsed:.3f}s = {cpu_elapsed / elapsed:.2f}x; {cpu_elapsed - elapsed:.3f}s saved.",
                    flush=True,
                )
        return report
    finally:
        numba.set_num_threads(original_threads)
