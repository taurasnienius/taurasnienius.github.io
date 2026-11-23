"""Exact dynamic programme over (visited set, body, time) for small instances (n <= ~12)."""
import numpy as np

def exact(C):
    n, nT = C.shape[0], C.shape[1]
    full = (1 << n) - 1
    INF = np.inf
    # f[mask] : array (n, nT)
    f = {1: np.full((n, nT), INF)}
    f[1][0, :] = 0.0
    best = INF
    masks = sorted(range(1, full + 1), key=lambda m: bin(m).count('1'))
    for mask in masks:
        if not mask & 1 or mask not in f:
            continue
        F = f[mask]
        F = np.minimum.accumulate(F, axis=1)  # free waiting
        for j in range(n):
            if not (mask >> j) & 1:
                continue
            row = F[j]
            if not np.isfinite(row).any():
                continue
            # cost to go from j at k to m at l: min_k row[k] + C[j,k,m,l]
            tot = row[:, None, None] + C[j]  # (nT, n, nT)
            arr = tot.min(axis=0)            # (n, nT) best arrival at m,l
            if mask == full:
                if j != 0:
                    best = min(best, arr[0].min())
                continue
            for m in range(1, n):
                if (mask >> m) & 1:
                    continue
                nm = mask | (1 << m)
                g = f.setdefault(nm, np.full((n, nT), INF))
                np.minimum(g[m], arr[m], out=g[m])
    return best
