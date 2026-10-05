# Tests: CPU, GPU simulation, and hardware

| Directory | Runs where | What it checks |
|---|---|---|
| `cpu/` | CPU only | Core contracts, indicators, raw artifacts, CLI, example reference, mocked RunPod lifecycle, benchmark CPU baseline, import boundaries |
| `gpu/test_simulator.py` + `simulator_cases.py` | CPU CUDA simulator in a child process | Actual GPU kernels against explicit matrix/RSI answers and CPU reference |
| `gpu/test_hardware.py` | Real NVIDIA GPU | Kernel reductions, RSI reference and single-combination raw output; marked `gpu` |

```bash
python -m pytest                         # CPU + isolated simulation; no rented pod
python -m pytest tests/cpu               # CPU tests only
python -m pytest tests/gpu/test_simulator.py
NUMBA_ENABLE_CUDASIM=0 python -m pytest -m gpu -v
```

`test_simulator.py` launches `simulator_cases.py` with CUDASIM in a separate process.
The parent keeps its original environment; simulation cannot silently contaminate
later hardware checks. The case file intentionally has no `test_` prefix, so it
is collected only by the child process. Hardware tests refuse simulation and skip
if no real GPU exists; actual passes are required for hardware validation.

Hand-calculated trades anchor timing/fees independently. A known algebraic matrix
anchors sums and float32 sums of squares. GPU kernels are compared with a Python
reference; the benchmark's parallel CPU baseline is likewise checked independently.
RunPod tests inject allocation/SSH/job/download/cleanup failure, cancellation,
price rejection, ambiguous replies, metadata failure, and unsafe archives without
leasing any resources. Architecture tests prevent examples/tools from entering core.

The optional installed hardware smoke tool is `gpu-backtest gpu-check`; its code
is in `tools/gpu_backtest_tools/checks/`, not the engine. Performance measurements
live in [benchmark evidence](../docs/benchmarks.md). The original small hardware
validation is retained under [results](../benchmarks/results/runpod_validation_20261003.md).

For changes to numerical logic, run both simulation and real-GPU checks. A pure
module/layout refactor must preserve the kernel body and compare previous/current
outputs. Installing the wheel and checking its RunPod export catches packaging
errors that editable-checkout tests can miss.


Raw output tests reload NPZ with pickle disabled, assert exact float64 array values,
ordered dimensions, fees, schema-2 counts and file SHA-256. Single-combination output
is anchored against hand-calculated profitable, fee-paying and losing trades.
Tests also reject removed analysis commands/config fields before RunPod allocation.
