# Strategy contract

Pass a module/object to `gpu_backtest.run`, or an installed dotted module name
such as `my_strategies.example`. The separate example module is
`gpu_backtest_examples.rsi.strategy`. The old `rsi_meanrev` shorthand is translated
only at the public API/CLI edge; the core imports no example strategy.
The package neither copies nor uploads an external plugin.

## Required declarations

```python
NAME = "example"
ENTRY_DIMS = [("entry_period", 2, 4, 2, False)]
EXIT_DIMS = [("exit_period", 2, 4, 2, False)]
TABLES = [("rsi", "close", ["entry_period", "exit_period"])]
```

Each dimension is `(name, low, high, step, is_float)`. Endpoints are inclusive;
`high` must lie on the grid. Names must be unique across both sides. Each side
has one to four dimensions. Overrides may change ranges, but must preserve the
declared names and order. Table windows must be positive integer dimensions.
Ranked output requires at least two combinations on each side.

The last dimension varies fastest. `ep` and `xp` are four-element float64 arrays;
only the slots corresponding to declared dimensions are defined. Integer
parameters arrive as float64 values and must be converted before indexing.

## Indicator tables

`TABLES` contains up to four `(kind, source, [dimension_names])` entries. For each
entry, the union of the specified window values determines table rows.

| Kind | Definition |
|---|---|
| `roll_max`, `roll_min`, `roll_mean` | pandas rolling window, including the current bar |
| `roll_std` | pandas rolling sample standard deviation (`ddof=1`) |
| `ema` | `ewm(span=window, adjust=False)`, seeded with the first value |
| `rsi` | EWM gains/losses with `alpha=1/window`, `adjust=False`; seeded with the first delta |
| `atr` | EWM true range with `alpha=1/window`, seeded with the first high-minus-low |

Sources are `close`, `open`, `high`, `low`, and `volume`; ATR uses OHLC and ignores
the source. Tables are float32; the row maps are int64. `rm[window]` gives the
row for a configured window. Rolling warmup values and the first RSI value are
NaN. RSI uses 100 when smoothed loss is zero, including a flat-price sequence;
this is an explicit example convention, not every RSI library's initialization.
Unused table slots are 1×1 dummy arrays and must not be indexed as full tables.

## GPU callback

```python
def make_device_fn(cuda):
    @cuda.jit(device=True, inline=True)
    def algo(
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
    ):
        # Implement your complete trading loop with CUDA-compatible operations.
        return 0.0  # Replace with total return in percent.

    return algo
```

Market arrays are float32. Fees are fractions (0.0015 means 0.15%). The device
function must return a finite number and must give the same result in both
reduction passes. Do not use randomness, shared mutable state, Python callbacks,
or future market values. The callback must compile under Numba CUDA.

Trading semantics are owned by the plugin; the harness does not enforce signals,
broker accounting, or position sizing. To compare a plugin with a CPU reference,
provide:

```python
def reference(
    close, open_, high, low, vol, tabs, rms, ep, xp, buy, sell
): ...  # Independent CPU implementation returning percent return.
```

`gpu_backtest_tools.checks.reference.reference_group_sums` uses this reference on small grids.
Supply a hand-calculated case as well: matching two implementations alone does
not establish that the intended rules are correct.

## Example

See [the separate RSI example](../examples/gpu_backtest_examples/rsi/README.md)
for a complete plugin, runnable config/data, and its execution/fee conventions.
