# RunPod hardware validation — 2026-10-03

The public RSI example and its synthetic 128-bar dataset ran through the launcher
on one RTX 4090 in RunPod Secure Cloud. No external/private strategy or market
dataset was involved. The pod was deleted automatically; a subsequent API check
confirmed it was absent from the account's pod list.

## Environment

| Component | Tested value |
|---|---|
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Python | 3.11 |
| GPU | NVIDIA GeForce RTX 4090, compute capability 8.9 |
| Driver | 570.195.03 |
| Numba / numba-cuda | 0.65.1 / 0.30.4 |
| NumPy / pandas | 2.4.6 / 3.0.6 |
| cuda-bindings / cuda-core / cuda-pathfinder | 12.9.9 / 1.2.1 / 1.8.3 |
| llvmlite | 0.47.0 |
| Optional HTML renderer | Altair 6.3.0 |

The launcher now pins the resolved core packages in
[`runpod_requirements.txt`](../src/gpu_backtest/runpod_requirements.txt).

## Results

| Check | Result |
|---|---|
| Known matrix row/column sums and float32 sums of squares | Exact expected values |
| Repeated matrix execution | Identical arrays |
| Hand-calculated RSI trade, zero fees | 50% |
| Hand-calculated RSI trade, 0.15% per side | 49.625% |
| Full-period RSI grid | 128 bars × 16 combinations, passed |
| First-half RSI grid | 65 bars × 16 combinations, passed |
| Second-half RSI grid | 63 bars × 16 combinations, passed |
| Downloaded entry/exit effects vs CPU reference | Maximum absolute difference 0 across all three segments |
| Common entry and exit outputs | Generated successfully |
| HTML charts and result download | Completed |
| Owned pod deletion | Completed and verified through the API |

After download, every segment's 16 returns was independently recomputed using the
RSI CPU reference, rounded to float32, and reduced into expected group statistics.
All parameter keys and entry/exit effect sizes matched the actual GPU CSVs.
`NUMBA_ENABLE_CUDASIM=0` was forced on the pod; the hardware checker rejected
simulation and reported `simulation: false`.

This is a small hardware/integration check. It does not establish large-grid
performance, full-scale memory behavior, results on other GPUs/drivers, or the
correctness of user-supplied strategy plugins. Lifecycle failure paths are covered
by injected local tests rather than paid destructive cloud experiments.
