"""Educational RSI mean reversion with independently swept entry and exit parameters."""

NAME = "rsi_meanrev"
ENTRY_DIMS = [("p_e", 2, 4, 2, False), ("buy_lvl", 20, 30, 10, False)]
EXIT_DIMS = [("p_x", 2, 4, 2, False), ("sell_lvl", 70, 80, 10, False)]
TABLES = [("rsi", "close", ["p_e", "p_x"])]
INITIAL_CAPITAL = 10000.0


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
        re = rm0[int(ep[0])]
        buy_lvl = ep[1]
        rx = rm0[int(xp[0])]
        sell_lvl = xp[1]
        capital = INITIAL_CAPITAL
        position = 0.0
        pending_entry = False
        pending_exit = False
        for i in range(num_bars - 1):
            o = open_[i]
            if pending_exit:
                sv = position * o
                capital += sv - sv * sell
                position = 0.0
                pending_exit = False
            elif pending_entry:
                inv = capital
                capital -= inv + inv * buy
                position = inv / o
                pending_entry = False
            if i > 0:
                entry = position == 0.0 and t0[re, i] < buy_lvl
                exit_ = position > 0.0 and t0[rx, i] > sell_lvl
                pending_exit = exit_
                pending_entry = entry
        final_equity = capital + position * open_[num_bars - 1]
        return (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0

    return algo


def reference(close, open_, high, low, vol, tabs, rms, ep, xp, buy, sell):
    t0, rm0 = (tabs[0], rms[0])
    re, buy_lvl = (rm0[int(ep[0])], ep[1])
    rx, sell_lvl = (rm0[int(xp[0])], xp[1])
    capital, position = (INITIAL_CAPITAL, 0.0)
    pe = px = False
    n = len(close)
    for i in range(n - 1):
        o = float(open_[i])
        if px:
            sv = position * o
            capital += sv - sv * sell
            position = 0.0
            px = False
        elif pe:
            inv = capital
            capital -= inv + inv * buy
            position = inv / o
            pe = False
        if i > 0:
            entry = position == 0.0 and bool(t0[re, i] < buy_lvl)
            exit_ = position > 0.0 and bool(t0[rx, i] > sell_lvl)
            px, pe = (exit_, entry)
    return (capital + position * float(open_[n - 1]) - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
