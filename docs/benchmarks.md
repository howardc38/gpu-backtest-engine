# Public CPU/GPU and billion-pair benchmark

Measured on 2026-10-03 using only the public RSI example and deterministic generated
OHLCV data. The [raw JSON report](../benchmarks/results/rtx4090_rsi_20261003.json)
contains every timing sample, both grids, software versions, and the input SHA-256.

## Results at a glance

Both profiles use 1,024 bars and 0.15% fees per side.

| Measurement | CPU | RTX 4090 | Result |
|---|---|---|---|
| 16,777,216 unique pairs, warmed two-pass reductions | 6.417 s | 1.654 s | GPU 3.88× faster |
| 1,000,000,000 unique pairs, normal GPU engine API including output | Not run | 43.341 s | Billion-pair run completed |

The CPU baseline is **8-thread, nopython-compiled Numba**, on an AMD EPYC 75F3
host. It is not a Python loop or CUDA simulator. The host reported 128 logical/
affinity CPUs; this run explicitly selected eight Numba worker threads and the
pod request required at least eight vCPUs. This is not a comparison against all
cores of the host or every possible CPU implementation.

The GPU is an RTX 4090 with driver 595.91.07, using Python 3.11.10, Numba 0.65.1,
numba-cuda 0.30.4, NumPy 2.4.6, and pandas 3.0.6. The RunPod image is
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.

## Matched CPU/GPU comparison

The comparison grid has 4,096 entry and 4,096 exit parameter sets:

- Entry RSI periods 2–65, buy thresholds 20–51.5 in steps of 0.5.
- Exit RSI periods 2–65, sell thresholds 49–80.5 in steps of 0.5.

Both implementations evaluate the same complete trading loop twice per unique
pair: one entry reduction and one exit reduction. They use identical float32
market/indicator arrays, float32 returns and squares, float64 accumulation, fees,
execution boundaries, and parameter values. The CPU baseline compiles the plugin's
independent reference with `njit`, then uses `prange` across the outer group loops.
The GPU measurement launches the actual engine kernels.

Input and indicator preparation, GPU allocation/transfers, and initial compilation
are outside the comparison timer. Each side is warmed once, then measured three
times; the table uses medians. GPU synchronization is included, so the timing is
completed device work rather than asynchronous launch latency. CPU output array
allocation is included; GPU output arrays are preallocated. Result copies,
statistics, and CSV writing are excluded from this comparison.

| Side | Samples, seconds | Median |
|---|---|---|
| CPU, eight threads | 6.349090, 6.651150, 6.417157 | 6.417157 |
| GPU | 1.673979, 1.653633, 1.653962 | 1.653962 |

All four grouped arrays—entry/exit sums and sums of squares—matched the CPU
reference with maximum absolute difference **0**. Initial compile-and-warmup calls,
including one full sweep, took 8.677 s on CPU and 2.893 s on GPU; those are not
included in the reported warmed speedup.

## Billion-scale capacity

This profile contains **20,000 entry sets × 50,000 exit sets = 1,000,000,000 unique
parameter pairs**:

- Entry periods 2–401, buy thresholds 10–59 in steps of 1.
- Exit periods 2–1001, sell thresholds 50–99 in steps of 1.

It ran through the normal `gpu_backtest.engine.run` API, writing top entry/exit
CSVs and a manifest recording one billion pairs. The 43.341-second timer includes
CSV validation, indicator tables, device allocation/transfers, kernel construction/
JIT, both GPU passes, result copies, grouped statistics, and output writing. It
excludes pod provisioning, installation, and generation of the synthetic CSV.
This is a single end-to-end measurement, not a three-run median. No billion-grid
CPU run or extrapolated CPU time is claimed.

The CUDA context was already initialized by the hardware checks/comparison.
This measures the normal API call in that process, not a fresh-process GPU startup
or an entire cloud launch.

The profile made two billion strategy evaluations across the two passes. A
combination means a complete entry/exit parameter pair, **not a trade**. The grid
includes long periods and no-entry cases, as a parameter sweep normally can;
3,913 of 20,000 entry groups had nonzero aggregate returns. The same timing does
not apply to every grid or more complex strategy.

The four grouped result arrays occupied **1,120,000 bytes (1.12 MB)**. Materializing
one float32 return per pair would require **4,000,000,000 bytes (4 GB)**. The engine
does not build that matrix. Its reduction storage scales with entry sets plus exit
sets, rather than their product. The 1.12 MB figure is only the grouped arrays;
it excludes indicator tables, input, CUDA context/JIT overhead, and host outputs.

Both passes passed the engine's finite-value and grand-total consistency checks.
These totals can differ slightly because of float64 accumulation order; this is
not a claim of exact agreement for all billion individual returns. The public
GPU numeric checker also passed before the benchmark. After download, the three
highest-ranked entry groups were independently recomputed across all 50,000 exits
each with the compiled CPU reference; their group means matched with maximum
absolute difference 0. This is a sampled check, not a CPU rerun of the full billion.
The leased pod was deleted
after result download and confirmed absent through the RunPod API.

## Reproduce

On a compatible GPU machine:

```bash
NUMBA_ENABLE_CUDASIM=0 gpu-backtest benchmark \
  --output runs/benchmark/report.json --bars 1024 --cpu-threads 8 --repeats 3 --billion
```

Or from a laptop with RunPod credentials:

```bash
gpu-backtest runpod --mode benchmark --ssh-key ~/.ssh/runpod_ed25519 \
  --output-dir runs/runpod-benchmark --max-seconds 1800
```

The RunPod route uses the pinned environment and automatically downloads output
and deletes its pod. Hardware, host load, thread count, strategy, bar count, grid
shape, and cold-start costs affect results. The measured 3.88× is specific to this
published workload; it is not a universal GPU/CPU speed claim or a trading result.
