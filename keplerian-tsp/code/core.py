"""Core of the KTSP solver: exact schedule for a fixed visiting order, and fast local moves.

A tour is 0 -> order[0] -> ... -> order[n-2] -> 0 on the time grid. For a fixed order the best departure
and arrival times follow from a min-plus dynamic programme over the grid (exact, O(n * T^2)).

Arrays
  A[p, t]  : cheapest cost to be AT the body in position p and ready to leave at time index t
             (free waiting, so non-increasing in t). Position 0 is the start body, A[0] = 0.
  B[p, t]  : cheapest cost from being at the body in position p, ready at time t, to the end of the tour
             (back at body 0 no later than the final epoch). Position n is the end, B[n] = 0.
  total    = min_t A[p, t] + B[p, t] for any p.
"""
import numpy as np
from numba import njit

INF = np.inf


@njit(cache=True)
def leg_fwd(Aa, C, a, b, out):
    """out[t] = min over l <= t of ( min_{k<l} Aa[k] + C[a,k,b,l] ): availability at b."""
    nT = Aa.shape[0]
    run = INF
    out[0] = INF
    for l in range(1, nT):
        m = INF
        for k in range(l):
            v = Aa[k] + C[a, k, b, l]
            if v < m:
                m = v
        if m < run:
            run = m
        out[l] = run


@njit(cache=True)
def leg_bwd(Bb, C, a, b, out):
    """out[k] = min over k' >= k of ( min_{l>k'} C[a,k',b,l] + Bb[l] ): cost-to-go when ready at a."""
    nT = Bb.shape[0]
    run = INF
    out[nT - 1] = INF
    for k in range(nT - 2, -1, -1):
        m = INF
        for l in range(k + 1, nT):
            v = C[a, k, b, l] + Bb[l]
            if v < m:
                m = v
        if m < run:
            run = m
        out[k] = run


@njit(cache=True)
def full_eval(seq, C, A, B):
    """seq: bodies by position, seq[0] = seq[n] = 0 (length n+1). Fills A, B; returns total cost."""
    npos = seq.shape[0]
    nT = C.shape[1]
    for t in range(nT):
        A[0, t] = 0.0
        B[npos - 1, t] = 0.0
    for p in range(1, npos):
        leg_fwd(A[p - 1], C, seq[p - 1], seq[p], A[p])
    for p in range(npos - 2, -1, -1):
        leg_bwd(B[p + 1], C, seq[p], seq[p + 1], B[p])
    best = INF
    for t in range(nT):
        v = A[npos - 1, t]
        if v < best:
            best = v
    return best


@njit(cache=True)
def tour_cost(seq, C):
    npos = seq.shape[0]
    nT = C.shape[1]
    A = np.empty((npos, nT))
    cur = np.zeros(nT)
    nxt = np.empty(nT)
    for p in range(1, npos):
        leg_fwd(cur, C, seq[p - 1], seq[p], nxt)
        cur[:] = nxt
    return cur.min()


@njit(cache=True)
def schedule(seq, C):
    """Optimal grid times for a fixed order: returns (cost, dep_idx[n], arr_idx[n]) per leg."""
    npos = seq.shape[0]
    nT = C.shape[1]
    A = np.empty((npos, nT))
    arg = np.full((npos, nT), -1, dtype=np.int64)   # departure time used to reach (p, l)
    wait = np.full((npos, nT), -1, dtype=np.int64)  # actual arrival time behind availability at t
    for t in range(nT):
        A[0, t] = 0.0
        wait[0, t] = 0
    for p in range(1, npos):
        a = seq[p - 1]; b = seq[p]
        run = INF; runl = -1
        A[p, 0] = INF
        for l in range(1, nT):
            m = INF; mk = -1
            for k in range(l):
                v = A[p - 1, k] + C[a, k, b, l]
                if v < m:
                    m = v; mk = k
            arg[p, l] = mk
            if m < run:
                run = m; runl = l
            A[p, l] = run
            wait[p, l] = runl
    # backtrack
    dep = np.empty(npos - 1, dtype=np.int64)
    arr = np.empty(npos - 1, dtype=np.int64)
    t = nT - 1
    cost = A[npos - 1, t]
    for p in range(npos - 1, 0, -1):
        l = wait[p, t]
        arr[p - 1] = l
        k = arg[p, l]
        dep[p - 1] = k
        t = k
    return cost, dep, arr


@njit(cache=True)
def seg_eval(seq_new, lo, hi, C, A, B, tmp1, tmp2):
    """Cost of a tour equal to the current one outside positions lo..hi (inclusive) of seq_new.
    Uses A[lo-1] (unchanged prefix) and B[hi+1] (unchanged suffix). seq_new has the same length."""
    nT = C.shape[1]
    tmp1[:] = A[lo - 1]
    for p in range(lo, hi + 2):
        leg_fwd(tmp1, C, seq_new[p - 1], seq_new[p], tmp2)
        tmp1[:] = tmp2
    best = INF
    Bs = B[hi + 1]
    for t in range(nT):
        v = tmp1[t] + Bs[t]
        if v < best:
            best = v
    return best


@njit(cache=True)
def _span(a, b):
    n = a.shape[0]
    lo = -1; hi = -1
    for p in range(n):
        if a[p] != b[p]:
            if lo < 0:
                lo = p
            hi = p
    return lo, hi


@njit(cache=True)
def local_search(seq, C, W, maxL):
    """Best-improvement descent over relocate (segments of 1..maxL), swap and reversal moves whose
    positions lie within W of each other. Exact scheduling for every candidate. Returns (seq, cost)."""
    npos = seq.shape[0]
    nT = C.shape[1]
    A = np.empty((npos, nT)); B = np.empty((npos, nT))
    tmp1 = np.empty(nT); tmp2 = np.empty(nT)
    cur = full_eval(seq, C, A, B)
    cand = seq.copy()
    best_seq = seq.copy()
    improved = True
    while improved:
        improved = False
        best = cur
        last = npos - 2  # last movable position
        # relocate segment [i, i+L-1] to before position p (in the sequence without the segment)
        for L in range(1, maxL + 1):
            for i in range(1, last - L + 2):
                for p in range(max(1, i - W), min(last - L + 2, i + W + 1)):
                    if p == i:
                        continue
                    # build candidate
                    k = 0
                    tmpseq = np.empty(npos - L, dtype=seq.dtype)
                    for q in range(npos):
                        if q < i or q >= i + L:
                            tmpseq[k] = seq[q]; k += 1
                    k = 0
                    for q in range(npos - L):
                        if q == p:
                            for r in range(L):
                                cand[k] = seq[i + r]; k += 1
                        cand[k] = tmpseq[q]; k += 1
                    lo, hi = _span(seq, cand)
                    if lo < 0:
                        continue
                    v = seg_eval(cand, lo, hi, C, A, B, tmp1, tmp2)
                    if v < best - 1e-9:
                        best = v; best_seq[:] = cand
        # swap
        for i in range(1, last + 1):
            for j in range(i + 1, min(last + 1, i + W + 1)):
                cand[:] = seq
                cand[i] = seq[j]; cand[j] = seq[i]
                v = seg_eval(cand, i, j, C, A, B, tmp1, tmp2)
                if v < best - 1e-9:
                    best = v; best_seq[:] = cand
        # reversal
        for i in range(1, last + 1):
            for j in range(i + 2, min(last + 1, i + W + 1)):
                cand[:] = seq
                for r in range(j - i + 1):
                    cand[i + r] = seq[j - r]
                v = seg_eval(cand, i, j, C, A, B, tmp1, tmp2)
                if v < best - 1e-9:
                    best = v; best_seq[:] = cand
        if best < cur - 1e-9:
            seq = best_seq.copy()
            cur = full_eval(seq, C, A, B)
            improved = True
    return seq, cur


@njit(cache=True)
def perturb(seq, k, W, seed):
    """k random segment relocations within window W (positions 1..n-1)."""
    np.random.seed(seed)
    s = seq.copy()
    npos = s.shape[0]
    last = npos - 2
    for _ in range(k):
        L = np.random.randint(1, 4)
        if last - L + 1 < 2:
            L = 1
        i = np.random.randint(1, last - L + 2)
        seg = s[i:i + L].copy()
        rest = np.concatenate((s[:i], s[i + L:]))
        lo = max(1, i - W); hi = min(rest.shape[0] - 1, i + W)
        p = np.random.randint(lo, hi + 1) if hi >= lo else 1
        s = np.concatenate((rest[:p], seg, rest[p:]))
    return s


@njit(cache=True)
def local_search_fi(seq, C, W, maxL, seed):
    """First-improvement variant of local_search with moves scanned in random order: much cheaper per
    improvement on large instances. Same neighbourhood (relocate 1..maxL, swap, reversal, window W)."""
    np.random.seed(seed)
    npos = seq.shape[0]
    nT = C.shape[1]
    A = np.empty((npos, nT)); B = np.empty((npos, nT))
    tmp1 = np.empty(nT); tmp2 = np.empty(nT)
    cur = full_eval(seq, C, A, B)
    cand = seq.copy()
    last = npos - 2
    improved = True
    while improved:
        improved = False
        order = np.random.permutation(last) + 1
        for ii in range(last):
            i = order[ii]
            done = False
            # relocations of the segment starting at i
            for L in range(1, maxL + 1):
                if i + L - 1 > last or done:
                    break
                for p in range(max(1, i - W), min(last - L + 2, i + W + 1)):
                    if p == i:
                        continue
                    k = 0
                    for q in range(npos):
                        if q < i or q >= i + L:
                            tmp_q = seq[q]
                            if k == p:
                                for r in range(L):
                                    cand[k] = seq[i + r]; k += 1
                            cand[k] = tmp_q; k += 1
                    if k < npos:  # insertion at the very end of the movable part
                        for r in range(L):
                            cand[k] = seq[i + r]; k += 1
                    lo, hi = _span(seq, cand)
                    if lo < 0:
                        continue
                    v = seg_eval(cand, lo, hi, C, A, B, tmp1, tmp2)
                    if v < cur - 1e-9:
                        seq = cand.copy(); cur = full_eval(seq, C, A, B); improved = True; done = True
                        break
            if done:
                continue
            for j in range(i + 1, min(last + 1, i + W + 1)):
                cand[:] = seq; cand[i] = seq[j]; cand[j] = seq[i]
                v = seg_eval(cand, i, j, C, A, B, tmp1, tmp2)
                if v < cur - 1e-9:
                    seq = cand.copy(); cur = full_eval(seq, C, A, B); improved = True; done = True
                    break
            if done:
                continue
            for j in range(i + 2, min(last + 1, i + W + 1)):
                cand[:] = seq
                for r in range(j - i + 1):
                    cand[i + r] = seq[j - r]
                v = seg_eval(cand, i, j, C, A, B, tmp1, tmp2)
                if v < cur - 1e-9:
                    seq = cand.copy(); cur = full_eval(seq, C, A, B); improved = True
                    break
    return seq, cur
