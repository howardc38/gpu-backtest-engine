"""GPU kernels: two deterministic reductions over strategy returns."""

from numba import cuda, float32, float64


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
