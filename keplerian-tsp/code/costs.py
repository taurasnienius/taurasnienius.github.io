"""Transfer-cost tensors for ACT KTSP instances.

C[i, k, j, l] = delta-v [m/s] of the cheapest Lambert arc leaving body i at grid time k and arriving at
body j at grid time l > k: |v_dep - v_i(t_k)| + |v_arr - v_j(t_l)|, minimised over all revolution counts
(pykep lambert_problem, max_revs=MAXREV) and, if BOTH_DIRS, both transfer directions.
This model reproduces ACT's published optima (Bannach, Acciarini, Izzo 2026, Table 4).
"""
import numpy as np, pykep as pk, os, sys, time
from multiprocessing import Pool
from ktsp_io import read_ktsp, states_on_grid, DAY

DATA = '/home/claude/ktsp_data/0_instances'
OUT = '/home/claude/ktsp/tensors'
MAXREV = 1000
BOTH_DIRS = True


def _rows(args):
    i, t, R, V, mu, maxrev, both = args
    n, nT = R.shape[0], R.shape[1]
    out = np.full((nT, n, nT), np.inf, dtype=np.float64)
    vi_all = V[i]
    for j in range(n):
        if j == i:
            continue
        for k in range(nT - 1):
            ri, vi = R[i, k], vi_all[k]
            for l in range(k + 1, nT):
                tof = (t[l] - t[k]) * DAY
                vj = V[j, l]; best = np.inf
                for cw in ((False, True) if both else (False,)):
                    lp = pk.lambert_problem(ri, R[j, l], tof, mu, cw, maxrev)
                    for a, b in zip(lp.v0, lp.v1):
                        d = ((a[0]-vi[0])**2+(a[1]-vi[1])**2+(a[2]-vi[2])**2)**.5 + \
                            ((b[0]-vj[0])**2+(b[1]-vj[1])**2+(b[2]-vj[2])**2)**.5
                        if d < best:
                            best = d
                out[k, j, l] = best
    return i, out


def build(name, nT, maxrev=MAXREV, both=BOTH_DIRS, procs=2, cache=True):
    tag = f'{name}_T{nT}' + ('' if (maxrev == MAXREV and both == BOTH_DIRS) else f'_r{maxrev}_{int(both)}')
    path = f'{OUT}/{tag}.npy'
    if cache and os.path.exists(path):
        return np.load(path)
    inst = read_ktsp(f'{DATA}/{name}.ktsp')
    t, R, V = states_on_grid(inst, nT)
    n = len(R)
    C = np.full((n, nT, n, nT), np.inf, dtype=np.float64)
    jobs = [(i, t, R, V, inst['mu'], maxrev, both) for i in range(n)]
    with Pool(procs) as p:
        for i, out in p.imap_unordered(_rows, jobs):
            C[i] = out
    os.makedirs(OUT, exist_ok=True)
    if cache:
        np.save(path, C)
    return C


if __name__ == '__main__':
    name, nT = sys.argv[1], int(sys.argv[2])
    t0 = time.time(); C = build(name, nT)
    print(name, nT, C.shape, f'{time.time()-t0:.1f} s')
