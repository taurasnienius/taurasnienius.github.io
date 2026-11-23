"""Long improvement runs on the open instances. Keeps the best tour found, with its exact schedule."""
import numpy as np, json, time, sys, os, pickle
from costs import build
from beam import beam_search, minleg_table, HAND
from core import local_search, local_search_fi, perturb, schedule, tour_cost
from gp import ev
from act_ref import VLB_VUB

def save(name, nT, seq, how):
    c, dep, arr = schedule(seq, build(name, nT))
    path = f'best/{name}_T{nT}.json'
    os.makedirs('best', exist_ok=True)
    old = json.load(open(path))['cost'] if os.path.exists(path) else np.inf
    if c < old - 1e-6:
        json.dump(dict(instance=name, nT=nT, cost=float(c), order=[int(x) for x in seq], dep_idx=[int(x) for x in dep],
                       arr_idx=[int(x) for x in arr], found_by=how, time=time.strftime('%Y-%m-%d %H:%M')), open(path, 'w'), indent=1)
    return c

def attack(name, nT, minutes, W_beam=300, W_ls=15, kick=4, seed=0, rule_pkl='gp_seed1.pkl'):
    C = build(name, nT); ml = minleg_table(C); vub = VLB_VUB[(name, nT)][1]
    rule = pickle.load(open(rule_pkl, 'rb'))['best']
    starts = []
    for lab, fn in (('beam-hand', HAND['on-schedule (A at t*)']), ('beam-discovered', lambda F: ev(rule, F))):
        s, c = beam_search(C, fn, W=W_beam, minleg=ml)
        s, c = local_search(s, C, W_ls, 3)
        starts.append((c, s, lab)); print(f'{name} T{nT} start {lab}: {c:.2f} (VUB {vub:.2f}, {(c/vub-1)*100:+.2f}%)', flush=True)
    if os.path.exists(f'best/{name}_T{nT}.json'):
        prev = json.load(open(f'best/{name}_T{nT}.json'))
        ps = np.array(prev['order'], dtype=np.int64)
        starts.append((tour_cost(ps, C), ps, prev['found_by'].split('+')[0]))
        print(f'{name} T{nT} resume from saved best {starts[-1][0]:.2f}', flush=True)
    c, seq, lab = min(starts, key=lambda x: x[0])
    best, best_seq = c, seq.copy(); save(name, nT, best_seq, lab + '+LS')
    rng = np.random.default_rng(seed); t0 = time.time(); it = 0; cur, cur_seq = best, best_seq.copy()
    while time.time() - t0 < minutes * 60:
        it += 1
        s2 = perturb(cur_seq, kick, W_ls, int(rng.integers(1 << 30)))
        s2, c2 = local_search_fi(s2, C, W_ls, 3, int(rng.integers(1 << 30)))
        if c2 < cur - 1e-9 or rng.random() < 0.02:
            cur, cur_seq = c2, s2
        if c2 < best - 1e-9:
            best, best_seq = c2, s2.copy(); save(name, nT, best_seq, lab + '+ILS')
            print(f'  {time.time()-t0:6.0f}s it {it}: {best:.2f}  ({(best/vub-1)*100:+.3f}% vs VUB)', flush=True)
        if rng.random() < 0.01:
            cur, cur_seq = best, best_seq.copy()
    print(f'{name} T{nT} final {best:.2f} VUB {vub:.2f} ({(best/vub-1)*100:+.3f}%) after {it} kicks', flush=True)

if __name__ == '__main__':
    attack(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), seed=int(sys.argv[4]) if len(sys.argv) > 4 else 0)
