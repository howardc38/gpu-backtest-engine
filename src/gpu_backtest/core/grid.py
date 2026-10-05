"""Parameter counts, decoding, and indicator-window selection."""

import numpy as np

from .strategy import MAX_DIMS


def dim_count(d):
    name, lo, hi, step, is_float = d
    return int(round((hi - lo) / step) + 1 if is_float else (hi - lo) // step + 1)


def space_count(dims):
    n = 1
    for d in dims:
        n *= dim_count(d)
    return n


def dim_arrays(dims):
    """Return padded lower bounds, steps, counts, and dimension count."""
    lo = np.zeros(MAX_DIMS, np.float64)
    st = np.ones(MAX_DIMS, np.float64)
    ct = np.ones(MAX_DIMS, np.int64)
    for j, d in enumerate(dims):
        lo[j] = float(d[1])
        st[j] = float(d[3])
        ct[j] = dim_count(d)
    return (lo, st, ct, len(dims))


def decode_values(dims, idx):
    """Decode a flat index; the last dimension varies fastest."""
    cnts = [dim_count(d) for d in dims]
    vals = [0.0] * len(dims)
    rem = idx
    for j in range(len(dims) - 1, -1, -1):
        i = rem % cnts[j]
        rem //= cnts[j]
        v = dims[j][1] + i * dims[j][3]
        vals[j] = float(v) if dims[j][4] else int(v)
    return vals


def int_dim_values(dims, names):
    out = set()
    for d in dims:
        if d[0] in names:
            assert not d[4], f"Table window dimension {d[0]} must contain integers"
            out |= {int(d[1] + i * d[3]) for i in range(dim_count(d))}
    return sorted(out)
