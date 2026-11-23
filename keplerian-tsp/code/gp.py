"""Automatic discovery of beam-search ranking rules by genetic programming.

An individual is an expression tree over the beam features (beam.FEATURES) and constants. Its fitness is
the mean ratio of the beam-search tour cost to the best known value over a set of TRAINING instances.
Rules are later tested, unchanged, on held-out instances (larger and finer grids).

The generate / evaluate / select / refine loop is the same one an LLM-driven method (EoH, ReEvo) uses;
here the generator is a classic tree GP (Koza; Schmidt & Lipson 2009) so that it runs on two CPU cores.
"""
import numpy as np, json, time, sys, os
from multiprocessing import Pool
from beam import beam_search, minleg_table, FEATURES, NFEAT
from costs import build

BIN = ['+', '-', '*', '/', 'min', 'max']
UN = ['neg', 'sqrt', 'log1p', 'sq']


def rand_tree(rng, depth):
    if depth <= 0 or (depth < 3 and rng.random() < 0.3):
        if rng.random() < 0.75:
            return ('f', int(rng.integers(NFEAT)))
        return ('c', float(np.round(rng.choice([0.1, 0.25, 0.5, 1, 2, 3, 5, 10]) * (1 if rng.random() < .8 else -1), 3)))
    if rng.random() < 0.8:
        return (BIN[rng.integers(len(BIN))], rand_tree(rng, depth - 1), rand_tree(rng, depth - 1))
    return (UN[rng.integers(len(UN))], rand_tree(rng, depth - 1))


def ev(t, F):
    k = t[0]
    if k == 'f': return F[:, t[1]]
    if k == 'c': return np.full(len(F), t[1])
    if k in UN:
        a = ev(t[1], F)
        if k == 'neg': return -a
        if k == 'sqrt': return np.sqrt(np.abs(a))
        if k == 'log1p': return np.log1p(np.abs(a))
        if k == 'sq': return a * a
    a, b = ev(t[1], F), ev(t[2], F)
    if k == '+': return a + b
    if k == '-': return a - b
    if k == '*': return a * b
    if k == '/': return np.where(np.abs(b) > 1e-9, a / np.where(np.abs(b) > 1e-9, b, 1), a)
    if k == 'min': return np.minimum(a, b)
    if k == 'max': return np.maximum(a, b)


def show(t):
    k = t[0]
    if k == 'f': return FEATURES[t[1]]
    if k == 'c': return f'{t[1]:g}'
    if k in UN: return f'{k}({show(t[1])})'
    if k in ('min', 'max'): return f'{k}({show(t[1])}, {show(t[2])})'
    return f'({show(t[1])} {k} {show(t[2])})'


def size(t):
    return 1 if t[0] in ('f', 'c') else 1 + sum(size(c) for c in t[1:])


def nodes(t, path=()):
    yield path, t
    if t[0] not in ('f', 'c'):
        for i, c in enumerate(t[1:], 1):
            yield from nodes(c, path + (i,))


def replace(t, path, new):
    if not path: return new
    l = list(t); l[path[0]] = replace(t[path[0]], path[1:], new); return tuple(l)


def mutate(t, rng):
    ps = list(nodes(t)); path, sub = ps[rng.integers(len(ps))]
    r = rng.random()
    if r < 0.15 and sub[0] == 'c':  # constant nudge
        return replace(t, path, ('c', float(np.round(sub[1] * np.exp(rng.normal(0, .5)), 4))))
    if r < 0.3 and sub[0] == 'f':
        return replace(t, path, ('f', int(rng.integers(NFEAT))))
    return replace(t, path, rand_tree(rng, int(rng.integers(1, 4))))


def crossover(a, b, rng):
    pa = list(nodes(a)); pb = list(nodes(b))
    path, _ = pa[rng.integers(len(pa))]; _, sub = pb[rng.integers(len(pb))]
    return replace(a, path, sub)


# ---------------- fitness -----------------
_CACHE = {}


def _load(name, nT):
    key = (name, nT)
    if key not in _CACHE:
        C = build(name, nT); _CACHE[key] = (C, minleg_table(C))
    return _CACHE[key]


def fitness_one(args):
    tree, insts, W = args
    vals = []
    for name, nT, ref in insts:
        C, ml = _load(name, nT)
        try:
            _, c = beam_search(C, lambda F: ev(tree, F), W=W, minleg=ml)
        except Exception:
            c = np.inf
        vals.append(c / ref if np.isfinite(c) else 10.0)
    return float(np.mean(vals)), vals


def run(train, pop=120, gens=25, W=20, seed=0, procs=2, seeds_in=(), log='gp_log.jsonl', parsimony=2e-4):
    rng = np.random.default_rng(seed)
    P = list(seeds_in) + [rand_tree(rng, int(rng.integers(2, 5))) for _ in range(pop - len(seeds_in))]
    hist = []
    with Pool(procs) as pool:
        for g in range(gens):
            t0 = time.time()
            res = pool.map(fitness_one, [(t, train, W) for t in P])
            fit = np.array([r[0] + parsimony * size(t) for r, t in zip(res, P)])
            order = np.argsort(fit)
            best = P[order[0]]
            rec = dict(gen=g, best_fit=float(res[order[0]][0]), per_inst=res[order[0]][1], rule=show(best),
                       size=size(best), median=float(np.median([r[0] for r in res])), secs=time.time() - t0)
            hist.append(rec)
            with open(log, 'a') as f: f.write(json.dumps(rec) + '\n')
            print(f"gen {g:2d} best {rec['best_fit']:.4f} median {rec['median']:.3f} size {rec['size']:2d} "
                  f"{rec['secs']:.0f}s  {rec['rule']}", flush=True)
            # next generation: elitism + tournament
            elite = [P[i] for i in order[:max(2, pop // 20)]]
            new = list(elite)
            def tour():
                c = rng.choice(len(P), 4, replace=False); return P[c[np.argmin(fit[c])]]
            while len(new) < pop:
                r = rng.random()
                if r < 0.6: child = crossover(tour(), tour(), rng)
                elif r < 0.9: child = mutate(tour(), rng)
                else: child = rand_tree(rng, int(rng.integers(2, 5)))
                if size(child) <= 25:
                    new.append(child)
            P = new
    return P[0], hist
