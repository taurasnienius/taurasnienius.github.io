import numpy as np, time
from core import local_search, perturb, tour_cost

def ils(C, seq0, iters=200, W=12, maxL=3, kick=3, seed=0, time_limit=None, log=False):
    t0 = time.time()
    seq, cost = local_search(seq0.astype(np.int64), C, W, maxL)
    best_seq, best = seq.copy(), cost
    rng = np.random.default_rng(seed)
    for it in range(iters):
        if time_limit and time.time() - t0 > time_limit:
            break
        s2 = perturb(seq, kick, W, int(rng.integers(1 << 30)))
        s2, c2 = local_search(s2, C, W, maxL)
        if c2 < cost - 1e-9:
            seq, cost = s2, c2
            if c2 < best - 1e-9:
                best_seq, best = s2.copy(), c2
                if log: print(f'  it {it} {best:.2f} {time.time()-t0:.0f}s', flush=True)
        elif rng.random() < 0.05:   # occasional restart from best
            seq, cost = best_seq.copy(), best
    return best_seq, best
