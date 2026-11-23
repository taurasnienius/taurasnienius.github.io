"""Hand-designed baselines named in the ACT posting, sharing the same exact-timing evaluator:
  - P-ACO: population-based ant colony optimisation (after Simões et al. 2017), tours built step by step with
    deferred timing (availability vectors), heuristic desirability = on-schedule cost of the next leg.
  - MCTS: UCT over the choice of the next body, with cheap randomised rollouts scored by the exact
    scheduler (tour_cost), committing one body at a time (receding horizon).
"""
import numpy as np, time
from numba import njit
from core import leg_fwd, tour_cost, INF


@njit(cache=True)
def aco_construct(C, tau, alpha, beta, scale, seed):
    np.random.seed(seed)
    n, nT = C.shape[0], C.shape[1]
    A = np.zeros(nT); out = np.empty(nT); best_out = np.empty(nT)
    vis = np.zeros(n, np.bool_); vis[0] = True
    seq = np.zeros(n + 1, np.int64)
    a = 0
    cand_A = np.empty((n, nT))
    w = np.zeros(n)
    for d in range(1, n):
        ts = min(nT - 1, int(round(d / n * (nT - 1))))
        base = A[ts] if np.isfinite(A[ts]) else A.min()
        tot = 0.0
        for b in range(n):
            w[b] = 0.0
            if vis[b]:
                continue
            leg_fwd(A, C, a, b, out)
            if not np.isfinite(out[nT - 1]):
                continue
            cand_A[b] = out
            c = out[ts] - base if np.isfinite(out[ts]) else out.min() - A.min() + scale
            eta = 1.0 / (max(c, 0.0) / scale + 0.05)
            w[b] = (tau[a, b] ** alpha) * (eta ** beta)
            tot += w[b]
        if tot <= 0:
            return seq, INF
        r = np.random.random() * tot
        acc = 0.0; pick = -1
        for b in range(n):
            acc += w[b]
            if w[b] > 0 and acc >= r:
                pick = b; break
        if pick < 0:
            for b in range(n - 1, -1, -1):
                if w[b] > 0:
                    pick = b; break
        A[:] = cand_A[pick]; vis[pick] = True; seq[d] = pick; a = pick
    leg_fwd(A, C, a, 0, out)
    return seq, out[nT - 1]


def p_aco(C, ants=20, iters=50, k=5, alpha=1.0, beta=2.0, seed=0, time_limit=None, scale=None):
    n = C.shape[0]
    rng = np.random.default_rng(seed)
    if scale is None:
        from beam import minleg_table, instance_scale
        scale = instance_scale(minleg_table(C))
    pop = []  # (cost, seq)
    best = (np.inf, None)
    t0 = time.time()
    tau0 = 1.0
    for it in range(iters):
        tau = np.full((n, n), tau0)
        for c, s in pop:
            for p in range(n):
                tau[s[p], s[p + 1]] += tau0 * 5.0 / k
        for _ in range(ants):
            s, c = aco_construct(C, tau, alpha, beta, scale, int(rng.integers(1 << 30)))
            if c < best[0]:
                best = (c, s.copy())
            pop.append((c, s.copy()))
        pop = sorted(pop, key=lambda x: x[0])[:k]  # quality-based population update
        if time_limit and time.time() - t0 > time_limit:
            break
    return best[1], best[0]


@njit(cache=True)
def rollout(prefix_seq, plen, vis0, minleg, n, nT, temp, seed):
    """Complete a prefix by sampling next bodies from a softmin of the cheapest-leg table at the
    proportional schedule time, then score the full order exactly."""
    np.random.seed(seed)
    seq = np.zeros(n + 1, np.int64)
    seq[:plen] = prefix_seq[:plen]
    vis = vis0.copy()
    a = seq[plen - 1]
    w = np.zeros(n)
    for d in range(plen, n):
        ts = min(nT - 1, int(round((d - 1) / n * (nT - 1))))
        m = INF
        for b in range(n):
            if not vis[b] and minleg[a, b, ts] < m:
                m = minleg[a, b, ts]
        tot = 0.0
        for b in range(n):
            w[b] = 0.0
            if not vis[b] and np.isfinite(minleg[a, b, ts]):
                w[b] = np.exp(-(minleg[a, b, ts] - m) / temp)
                tot += w[b]
        r = np.random.random() * tot; acc = 0.0; pick = -1
        for b in range(n):
            acc += w[b]
            if w[b] > 0 and acc >= r:
                pick = b; break
        if pick < 0:
            for b in range(n):
                if not vis[b]:
                    pick = b; break
        seq[d] = pick; vis[pick] = True; a = pick
    return seq


def mcts(C, minleg, scale, iters_per_step=200, c_uct=0.7, temp_frac=0.3, seed=0, time_limit=None):
    """Receding-horizon UCT: at each depth, run iterations from the committed prefix, commit the child
    with the best mean, continue. Rewards are normalised tour costs (lower cost -> higher reward)."""
    n, nT = C.shape[0], C.shape[1]
    rng = np.random.default_rng(seed)
    temp = temp_frac * scale
    prefix = [0]
    best_cost, best_seq = np.inf, None
    t0 = time.time()
    for depth in range(1, n):
        stats = {}  # key: tuple(prefix extension) -> [visits, sum_reward]
        root = tuple(prefix)
        for it in range(iters_per_step):
            path = list(root)
            vis = np.zeros(n, np.bool_); vis[path] = True
            node_keys = []
            # selection / expansion (two levels below the committed prefix at most)
            for lvl in range(2):
                cand = [b for b in range(1, n) if not vis[b]]
                if not cand:
                    break
                keys = [tuple(path + [b]) for b in cand]
                unvisited = [k for k in keys if k not in stats]
                if unvisited:
                    k = unvisited[rng.integers(len(unvisited))]
                    stats[k] = [0, 0.0]
                    path = list(k); vis[path[-1]] = True; node_keys.append(k)
                    break
                parent_n = sum(stats[k][0] for k in keys)
                ucb = [stats[k][1] / stats[k][0] + c_uct * np.sqrt(np.log(parent_n + 1) / stats[k][0]) for k in keys]
                k = keys[int(np.argmax(ucb))]
                path = list(k); vis[path[-1]] = True; node_keys.append(k)
            pre = np.zeros(n + 1, np.int64); pre[:len(path)] = path
            seq = rollout(pre, len(path), vis, minleg, n, nT, temp, int(rng.integers(1 << 30)))
            cost = tour_cost(seq, C)
            if cost < best_cost:
                best_cost, best_seq = cost, seq.copy()
            reward = -cost / (scale * n)
            for k in node_keys:
                stats[k][0] += 1; stats[k][1] += reward
        children = [(k, v) for k, v in stats.items() if len(k) == len(root) + 1 and v[0] > 0]
        if not children:
            break
        kbest = max(children, key=lambda kv: kv[1][1] / kv[1][0])[0]
        prefix = list(kbest)
        if time_limit and time.time() - t0 > time_limit:
            # finish greedily from the best tour found so far
            break
    return best_seq, best_cost
