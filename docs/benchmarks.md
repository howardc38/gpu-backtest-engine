# Public CPU/GPU benchmark: minutes saved on a billion-pair sweep

Historical v0.4 measurement on 2026-10-03 with the public RSI example and deterministic generated
OHLCV data. The headline compares **the same full billion-pair job on CPU and GPU**.
The [full raw report](../benchmarks/results/rtx4090_rsi_matched_billion_20261003.json)
contains exact timings, both parameter grids, hardware/software, and input SHA-256.

## The practical comparison

| Same RSI workload | CPU, eight Numba threads | RTX 4090 | Result |
|---|---|---|---|
| **1,000,000,000 pairs × 1,024 bars, through output** | **464.603 s (7 min 44.60 s)** | **44.952 s** | **10.34× faster; 419.651 s (~7 min) saved per sweep** |

The CPU result is a complete measured run, not a projection from a smaller grid.
Both sides load/validate the CSV, prepare indicators, execute the two passes,
validate results, compute grouped statistics, and write top CSVs/manifests.
The CPU baseline uses a parallel, nopython-compiled reference—neither Python loops
nor CUDA simulation. The CPU reducer was compiled earlier; the CUDA context was
already initialized. GPU kernel construction/JIT is included in its normal API
call. CPU initialization/compilation is not included, giving the CPU a warmed start.

Both full-grid values are single-run observations, not three-run medians. Data
CSV generation, pod provisioning, dependency installation, and first-process GPU
startup are excluded. These are engine-job times, not total cloud-launch times.
The comparison uses the same two-pass algorithm, rather than claiming to beat
all possible CPU implementations.

**Use-case:** GPU is useful for repeated large grids where minutes saved per run
accumulate. If a small job already takes only a few CPU seconds, a fresh cloud
lease and setup can take longer than the compute time it saves. The small-grid
kernel-only numbers below therefore remain supporting measurements, not the
headline justification for renting a GPU.

## Hardware and input

| Component | Matched billion-grid run |
|---|---|
| CPU | AMD EPYC 7K62 48-Core Processor host; eight Numba threads selected |
| Host-reported logical/affinity CPUs | 96 / 96; the benchmark does not use all host cores |
| Allocated pod vCPUs | 12, confirmed by the API; the request required at least eight |
| GPU | NVIDIA GeForce RTX 4090, compute capability 8.9 |
| NVIDIA driver | 580.159.04 |
| Python | 3.11.10 |
| Numba / numba-cuda | 0.65.1 / 0.30.4 |
| NumPy / pandas / llvmlite | 2.4.6 / 3.0.6 / 0.47.0 |
| CUDA Python bindings | 12.9.9 |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |

Both sides use 0.15% fees per side and the same 1,024-bar generated dataset:
`e57584c2def67be06de09b52ee72463d0626f20de30d2aace06f735d823909e2`.
No private algorithm or market dataset was involved.

## Billion-pair grid and correctness

The grid is **20,000 entry sets × 50,000 exit sets**:

- Entry periods 2–401, buy thresholds 10–59 in steps of 1.
- Exit periods 2–1001, sell thresholds 50–99 in steps of 1.

Each side performs two billion strategy evaluations across the two passes. A
combination is a complete entry/exit parameter pair, not a trade. The grid includes
long periods and no-entry cases: 3,913 entry groups have nonzero aggregate returns.
These timings do not imply every grid or strategy has the same cost.

The CPU completed the entire grid and all four aggregate arrays were compared
with the GPU result. Entry/exit sums differed by at most **3.725290298461914e-9**;
both sums-of-squares arrays matched exactly. Both sides passed their finite-value
and grand-total checks. Float64 accumulation order can cause small differences
between row/column grand totals; no bit-exact claim is made for every individual pair.
Top entry/exit CSVs and manifests were emitted for both CPU and GPU. The hardware
checker passed first. Download and owned-pod deletion completed, and the pod was
confirmed absent through the API.

The four grouped arrays use **1.12 MB**. A full float32 return matrix would require
**4 GB**, and is never materialized. This is reduction-array storage only; input,
indicator tables, runtime/JIT/context memory, and host output are additional.

## Supporting smaller-grid measurements

The original v0.3 measurement is retained in its
[raw report](../benchmarks/results/rtx4090_rsi_20261003.json). On an AMD EPYC 75F3
host with eight CPU threads, 16,777,216 pairs × 1,024 bars took CPU median 6.417 s
vs GPU median 1.654 s, or 3.88×. Its GPU-only billion run took 43.341 s; that older
record did not run the full billion grid on CPU.

The matched v0.4 run also measured that smaller grid: CPU median 8.286 s vs GPU
median 1.717 s, or 4.83×. The difference between hosts illustrates why results
must state hardware, thread count, workload, and timing scope.

The smaller grid has 4,096 entry sets and 4,096 exit sets (periods 2–65; buy levels
20–51.5 and sell levels 49–80.5, in steps of 0.5). Each side is warmed once and
measured three times. Its timer includes the two reductions and GPU synchronization;
input/indicator preparation, GPU transfers/allocation, compilation, result copies,
statistics and CSV writing are excluded. CPU output array allocation is included.
All four grouped arrays matched exactly in those smaller comparisons.

## Run the current comparison

v0.6 keeps the same grids, numerical kernels and CPU/GPU checks, but writes raw
NPZ/schema-2 manifests instead of statistics and ranked CSVs. The historical JSON
reports above are unchanged. New reports describe their actual raw-output timing;
current runs are not an exact reproduction of the old output-processing workload.

On a compatible GPU machine:

```bash
NUMBA_ENABLE_CUDASIM=0 gpu-backtest benchmark \
  --output runs/benchmark/report.json --bars 1024 --cpu-threads 8 --repeats 3 \
  --billion --billion-cpu
```

From a laptop with RunPod credentials:

```bash
gpu-backtest runpod --mode benchmark --ssh-key ~/.ssh/runpod_ed25519 \
  --output-dir runs/runpod-benchmark --max-seconds 1800
```

The RunPod route now includes the full CPU sweep. It can take several minutes,
rents one paid GPU pod, downloads results, and deletes its pod. Hardware, host
load, thread count, strategy, grid shape, bar count, and cold-start overhead affect
whether GPU use is worthwhile. No universal speedup or trading-return claim is made.


## v0.6 raw-output validation

On 2026-10-05, one RTX 4090 RunPod ran the released v0.5 wheel and reviewed v0.6
wheel against the same inputs, dimensions, fees and block size. Every float64 array
matched byte-for-byte across four workloads: tiny RSI with/without fees, 16,777,216
pairs, and 1,000,000,000 pairs (both large grids used 1,024 bars). The entire CUDA
kernel file was identical. Candidate NPZ values matched API returns, ordered
parameter manifests and file hashes were checked, and default RunPod bundle output
matched the clean wheel. Five physical-GPU tests passed, including independent
single-combination profit, commission and loss anchors.

[Validation evidence](../benchmarks/results/rtx4090_raw_output_parity_20261005.json)
records grids, data/array/kernel hashes, versions, hardware and zero differences.
This validates numerical parity, not a new speedup claim. The result files are
intentionally NPZ/schema-2 rather than ranked CSV/schema-1. Download completed,
the owned pod was deleted, and the API confirmed zero active pods.
