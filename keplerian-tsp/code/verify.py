"""Independent check of a saved tour, written separately from the solver: reads the .ktsp file, propagates
each body to the leg times, solves every Lambert arc again, checks that the arc really connects the two
bodies (propagating the departure state for the time of flight), and sums the impulses.
Does not use the cost tensors or any solver code."""
import json, sys, numpy as np, pykep as pk

def verify(path, inst_dir='/home/claude/ktsp_data/0_instances'):
    sol = json.load(open(path))
    name, nT = sol['instance'], sol['nT']
    lines = [l.split() for l in open(f'{inst_dir}/{name}.ktsp') if l.strip() and l.split()[0] != 'c']
    t0, tf, mu = map(float, lines[0][2:5]); bodies = np.array([[float(x) for x in l] for l in lines[1:]])
    n = len(bodies); grid = np.linspace(t0, tf, nT)
    order, dep, arr = sol['order'], sol['dep_idx'], sol['arr_idx']
    errs = []
    if order[0] != 0 or order[-1] != 0: errs.append('tour must start and end at body 0')
    if sorted(order[1:-1]) != list(range(1, n)): errs.append('not every body visited exactly once')
    for p in range(len(dep)):
        if not arr[p] > dep[p]: errs.append(f'leg {p}: arrival not after departure')
        if p > 0 and dep[p] < arr[p - 1]: errs.append(f'leg {p}: departs before arriving')
        if dep[p] < 0 or arr[p] > nT - 1: errs.append(f'leg {p}: outside the window')
    def state(i, tmjd):
        r, v = pk.propagate_lagrangian([bodies[i, :3], bodies[i, 3:]], (tmjd - t0) * 86400.0, mu, False)
        return np.array(r), np.array(v)
    total, worst_miss = 0.0, 0.0
    for p in range(len(dep)):
        a, b = order[p], order[p + 1]
        ta, tb = grid[dep[p]], grid[arr[p]]
        ra, va = state(a, ta); rb, vb = state(b, tb); tof = (tb - ta) * 86400.0
        best = None
        for cw in (False, True):
            lp = pk.lambert_problem(ra, rb, tof, mu, cw, 1000)
            for v1, v2 in zip(lp.v0, lp.v1):
                dv = np.linalg.norm(np.array(v1) - va) + np.linalg.norm(np.array(v2) - vb)
                if best is None or dv < best[0]:
                    best = (dv, v1)
        r_end, _ = pk.propagate_lagrangian([ra, best[1]], tof, mu, False)
        worst_miss = max(worst_miss, float(np.linalg.norm(np.array(r_end) - rb)))
        total += best[0]
    return dict(instance=name, nT=nT, claimed=sol['cost'], recomputed=total, diff=total - sol['cost'],
                worst_position_miss_m=worst_miss, legs=len(dep), errors=errs)

if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(json.dumps(verify(p)))
