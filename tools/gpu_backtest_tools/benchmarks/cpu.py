"""Parallel Numba CPU baseline: the same complete two-pass sweep as the GPU."""

import numpy as np
from numba import njit, prange


@njit(inline="never")
def _decode(index, lo, step, count, dimensions, out):
    for j in range(dimensions - 1, -1, -1):
        offset = index % count[j]
        index //= count[j]
        out[j] = lo[j] + offset * step[j]


def build_cpu_reducer(strategy):
    """Compile the plugin's independent CPU reference; never use CUDA simulation."""
    if not callable(getattr(strategy, "reference", None)):
        raise ValueError("CPU benchmarking requires strategy.reference")
    algorithm = njit(strategy.reference)

    @njit(parallel=True)
    def reduce(
        close,
        open_,
        high,
        low,
        vol,
        tabs,
        maps,
        e_lo,
        e_step,
        e_count,
        n_e,
        x_lo,
        x_step,
        x_count,
        n_x,
        entries,
        exits,
        buy,
        sell,
    ):
        entry_sum = np.empty(entries, np.float64)
        entry_sumsq = np.empty(entries, np.float64)
        exit_sum = np.empty(exits, np.float64)
        exit_sumsq = np.empty(exits, np.float64)
        for e in prange(entries):
            ep = np.empty(4, np.float64)
            xp = np.empty(4, np.float64)
            _decode(e, e_lo, e_step, e_count, n_e, ep)
            total = np.float64(0.0)
            squared = np.float64(0.0)
            for x in range(exits):
                _decode(x, x_lo, x_step, x_count, n_x, xp)
                result = np.float32(
                    algorithm(close, open_, high, low, vol, tabs, maps, ep, xp, buy, sell)
                )
                total += np.float64(result)
                squared += np.float64(np.float32(result * result))
            entry_sum[e] = total
            entry_sumsq[e] = squared
        for x in prange(exits):
            ep = np.empty(4, np.float64)
            xp = np.empty(4, np.float64)
            _decode(x, x_lo, x_step, x_count, n_x, xp)
            total = np.float64(0.0)
            squared = np.float64(0.0)
            for e in range(entries):
                _decode(e, e_lo, e_step, e_count, n_e, ep)
                result = np.float32(
                    algorithm(close, open_, high, low, vol, tabs, maps, ep, xp, buy, sell)
                )
                total += np.float64(result)
                squared += np.float64(np.float32(result * result))
            exit_sum[x] = total
            exit_sumsq[x] = squared
        return (entry_sum, entry_sumsq, exit_sum, exit_sumsq)

    return reduce
