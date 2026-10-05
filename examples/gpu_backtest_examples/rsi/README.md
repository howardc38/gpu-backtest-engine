# RSI example plugin

This example lives outside `gpu_backtest.core`. Entry is RSI below `buy_lvl`;
exit is RSI above `sell_lvl`. Entry/exit periods are independently swept.

| File | Purpose |
|---|---|
| `strategy.py` | Parameter specs, GPU device function, and CPU reference |
| `config.json` | A tiny runnable grid and fees |
| `synthetic.csv` | Generated data, not historical market data |
| `generate_data.py` | Regenerate the fixture via the shared example data generator |

```bash
NUMBA_ENABLE_CUDASIM=1 gpu-backtest run \
  --config examples/gpu_backtest_examples/rsi/config.json --out-prefix runs/rsi
python -m gpu_backtest_examples.rsi.generate_data
```

The installed module is `gpu_backtest_examples.rsi.strategy`. The engine loads
it through the same contract used by an external private strategy.

## Execution rules

Entry: RSI(`p_e`) below `buy_lvl` while flat. Exit: RSI(`p_x`) above `sell_lvl`
while holding a position. The initial capital is 10,000 units.

Signals are evaluated after processing pending orders, starting at bar 1.
Orders fill at the following bar's open, with exits processed before entries.
The loop processes bars 0 through `n-2`. Pending orders are **not filled on the
final bar**; an existing position is valued at that bar's open, with no forced
sale or final sell fee. This boundary convention is covered by tests.

Buying invests all current capital before charging the buy fee. Cash therefore
becomes negative by the fee amount; this example permits that financing rather
than reserving the fee within available cash. Selling credits proceeds minus
the sell fee. There is no slippage, funding charge, leverage model, or partial fill.

For the seven-bar hand-calculated fixture, the zero-fee trade buys 125 units at
80 and sells them at 120: return 50%. At 0.15% per side, buy fee is 15 and sell
fee is 22.5; final equity is 14,962.5, giving 49.625%. Tests also cover a losing
round trip, no-entry conditions, and ignored final-bar pending orders.
