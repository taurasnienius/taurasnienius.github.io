import numpy as np, pykep as pk, itertools, sys
from ktsp_io import *

def leg_costs(inst, nT, mode):
    t, R, V = states_on_grid(inst, nT); n = len(R)
    C = np.full((n, nT, n, nT), np.inf)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            for k in range(nT):
                for l in range(k + 1, nT):
                    tof = (t[l] - t[k]) * DAY
                    best = np.inf
                    dirs = [False] if mode['pro_only'] else [False, True]
                    for cw in dirs:
                        lp = pk.lambert_problem(R[i, k], R[j, l], tof, inst['mu'], cw, mode['maxrev'])
                        for a, b in zip(lp.v0, lp.v1):
                            d = np.linalg.norm(np.array(a) - V[i, k])
                            if mode['arr']: d += np.linalg.norm(np.array(b) - V[j, l])
                            best = min(best, d)
                    C[i, k, j, l] = best
    return C

def dp(C):
    n, nT = C.shape[0], C.shape[1]
    full = (1 << n) - 1
    f = {}
    for k in range(nT): f[(1, 0, k)] = 0.0  # at body 0, time k (waited), visited {0}
    best = np.inf
    for mask in range(1, full + 1):
        if not mask & 1: continue
        for j in range(n):
            if not (mask >> j) & 1: continue
            for k in range(nT):
                cur = f.get((mask, j, k))
                if cur is None: continue
                if k + 1 < nT and j != 0:  # free waiting at a body
                    key = (mask, j, k + 1); f[key] = min(f.get(key, np.inf), cur)
                for m in range(n):
                    if m == 0 and mask == full and j != 0:
                        for l in range(k + 1, nT): best = min(best, cur + C[j, k, 0, l])
                    if (mask >> m) & 1: continue
                    for l in range(k + 1, nT):
                        key = (mask | 1 << m, m, l); v = cur + C[j, k, m, l]
                        if v < f.get(key, np.inf): f[key] = v
    return best

if __name__ == '__main__':
    name, nT = sys.argv[1], int(sys.argv[2])
    inst = read_ktsp(f'/home/claude/ktsp_data/0_instances/{name}.ktsp')
    for mode in [dict(pro_only=True, maxrev=0, arr=True), dict(pro_only=False, maxrev=0, arr=True),
                 dict(pro_only=True, maxrev=5, arr=True), dict(pro_only=False, maxrev=5, arr=True),
                 ]:
        print(mode, round(dp(leg_costs(inst, nT, mode)), 2))
