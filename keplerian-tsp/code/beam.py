"""Beam search over partial tours with deferred timing.

A partial tour keeps its whole availability vector A[t] (cheapest cost to be at its last body, ready at
grid time t), not a single arrival time, so the timing decision stays open until the tour is complete.
Which partial tours survive is decided by a RANKING RULE: a function of a few features of each child.
Hand-designed rules and evolved (discovered) rules plug into the same slot.
"""
import numpy as np, time
from numba import njit
from core import leg_fwd, INF

NFEAT = 8
FEATURES = ['a_sched', 'a_min', 'a_late', 'a_early', 'leg', 'depth', 'tmin', 'near']
#  a_sched : cost if the tour stays on a proportional schedule (A at t* = depth/n of the window)
#  a_min   : cheapest cost to be at the last body at any time
#  a_late  : A at t* + 10 % of the window ;  a_early : A at t* - 10 % of the window
#  leg     : cost of the leg just added (at the cheapest consistent times)
#  depth   : fraction of bodies visited
#  tmin    : fraction of the window used at the cheapest point (argmin of A)
#  near    : mean over unvisited bodies of the cheapest single leg from the last body after t*


@njit(cache=True)
def expand(As, last, masks, C, minleg, depth, n, nT):
    """All children of the beam: returns child A vectors, parent idx, new body, feature matrix."""
    W = As.shape[0]
    maxc = W * n
    cA = np.empty((maxc, nT)); par = np.empty(maxc, np.int64); nb = np.empty(maxc, np.int64)
    F = np.empty((maxc, NFEAT))
    out = np.empty(nT)
    c = 0
    d1 = depth + 1
    ts = min(nT - 1, int(round(d1 / n * (nT - 1))))
    dl = max(1, int(round(0.1 * (nT - 1))))
    for w in range(W):
        a = last[w]
        prevmin = As[w].min()
        for b in range(1, n):
            if masks[w, b]:
                continue
            leg_fwd(As[w], C, a, b, out)
            if not np.isfinite(out[nT - 1]):
                continue
            cA[c] = out; par[c] = w; nb[c] = b
            am = out.min()
            tmin = 0
            for t in range(nT):
                if out[t] == am:
                    tmin = t; break
            F[c, 0] = out[ts]
            F[c, 1] = am
            F[c, 2] = out[min(nT - 1, ts + dl)]
            F[c, 3] = out[max(0, ts - dl)]
            F[c, 4] = am - prevmin
            F[c, 5] = d1 / n
            F[c, 6] = tmin / (nT - 1)
            s = 0.0; k = 0
            for q in range(1, n):
                if q != b and not masks[w, q]:
                    s += minleg[b, q, ts]; k += 1
            F[c, 7] = s / k if k > 0 else 0.0
            c += 1
    return cA[:c], par[:c], nb[:c], F[:c]


@njit(cache=True)
def close_tours(As, last, C, nT):
    out = np.empty(nT)
    res = np.empty(As.shape[0])
    for w in range(As.shape[0]):
        leg_fwd(As[w], C, last[w], 0, out)
        res[w] = out[nT - 1]
    return res


def minleg_table(C):
    """minleg[a, b, t] = cheapest leg a->b departing at or after grid time t (any arrival)."""
    n, nT = C.shape[0], C.shape[1]
    m = np.where(np.isfinite(C), C, np.inf).min(axis=3)          # (a, k, b): best arrival for departure k
    m = np.minimum.accumulate(m[:, ::-1, :], axis=1)[:, ::-1, :]  # depart at or after k
    return np.ascontiguousarray(m.transpose(0, 2, 1))              # (a, b, t)


COSTF = [0, 1, 2, 3, 4, 7]   # cost-valued features, divided by the instance scale before ranking


def instance_scale(minleg):
    """Typical cheapest leg of the instance (median over body pairs, departing at the start)."""
    v = minleg[:, :, 0]
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v))


def beam_search(C, rank, W=100, minleg=None):
    """rank: callable F (m x NFEAT) -> scores (lower is better). Cost features are divided by the
    instance scale so a rule transfers between instances. Returns (seq, cost)."""
    n, nT = C.shape[0], C.shape[1]
    if minleg is None:
        minleg = minleg_table(C)
    scale = instance_scale(minleg)
    As = np.zeros((1, nT)); last = np.zeros(1, np.int64)
    masks = np.zeros((1, n), np.bool_); masks[0, 0] = True
    seqs = np.zeros((1, 0), np.int64)
    for depth in range(n - 1):
        cA, par, nb, F = expand(As, last, masks, C, minleg, depth, n, nT)
        if len(par) == 0:
            return None, np.inf
        Fn = F.copy(); Fn[:, COSTF] /= scale
        with np.errstate(all='ignore'):
            s = np.asarray(rank(Fn), dtype=float) * np.ones(len(Fn))
        s = np.where(np.isfinite(s), s, np.inf)
        order = np.argsort(s, kind='stable')
        # keep at most one child per (visited set, last body): the better ranked one
        keep, seen = [], set()
        for i in order:
            key = (masks[par[i]].tobytes(), int(nb[i]))
            if key in seen:
                continue
            seen.add(key); keep.append(i)
            if len(keep) >= W:
                break
        keep = np.array(keep)
        As = cA[keep]; last = nb[keep]
        masks = masks[par[keep]].copy(); masks[np.arange(len(keep)), last] = True
        seqs = np.concatenate([seqs[par[keep]], last[:, None]], axis=1)
    tot = close_tours(As, last, C, nT)
    i = int(np.argmin(tot))
    return np.concatenate(([0], seqs[i], [0])).astype(np.int64), float(tot[i])


# hand-designed ranking rules (baselines)
HAND = {
    'cost-only (min_t A)': lambda F: F[:, 1],
    'on-schedule (A at t*)': lambda F: F[:, 0],
    'on-schedule + lookahead': lambda F: F[:, 0] + F[:, 7],
}
