"""Level-1/2 comparison on every instance with 20 or more bodies (the smaller ones are solved exactly).

For each instance: constructive methods alone, then each followed by the same local search:
  beam (hand rule 'on-schedule'), beam (discovered rule), P-ACO and MCTS (same wall-clock budget as the
  discovered beam + 20 s, at least 20 s), compared with ACT's published beam search and B-Swan.
Writes results_compare.json.
"""
import numpy as np, json, time, pickle, sys
from costs import build
from beam import beam_search, minleg_table, instance_scale, HAND
from baselines import p_aco, mcts
from core import local_search, tour_cost
from gp import ev, show
from act_ref import VLB_VUB, TABLE3

INST = [('asteroids_20', 21), ('asteroids_20', 41), ('asteroids_20', 81), ('asteroids_20', 161),
        ('jovianmoons_20', 21), ('jovianmoons_20', 41), ('jovianmoons_20', 81), ('jovianmoons_20', 161),
        ('asteroids_40', 41), ('asteroids_40', 81), ('asteroids_40', 161),
        ('asteroids_80', 81), ('asteroids_80', 161)]
TRAIN = {('asteroids_10', 81), ('asteroids_20', 41), ('asteroids_20', 81), ('jovianmoons_10', 81),
         ('jovianmoons_20', 41), ('jovianmoons_20', 81)}


def main(rule_pkl, W=100, out='results_compare.json'):
    rule = pickle.load(open(rule_pkl, 'rb'))['best']
    disc = lambda F: ev(rule, F)
    res = {}
    for name, nT in INST:
        C = build(name, nT); ml = minleg_table(C); sc = instance_scale(ml)
        lb, vub = VLB_VUB[(name, nT)]; b10k, b1k, bswan = TABLE3[(name, nT)]
        r = dict(VLB=lb, VUB=vub, act_beam10000=b10k, act_beam1000=b1k, act_bswan=bswan,
                 train=(name, nT) in TRAIN)
        for label, fn in (('beam_hand', HAND['on-schedule (A at t*)']), ('beam_cost_only', HAND['cost-only (min_t A)']),
                          ('beam_discovered', disc)):
            t = time.time(); s, c = beam_search(C, fn, W=W, minleg=ml); dt = time.time() - t
            s2, c2 = local_search(s, C, 12, 3)
            r[label] = dict(cost=c, secs=dt, ls_cost=float(c2))
        budget = max(20.0, r['beam_discovered']['secs'] + 20)
        t = time.time(); s, c = p_aco(C, ants=20, iters=100000, scale=sc, time_limit=budget); dt = time.time() - t
        s2, c2 = local_search(s, C, 12, 3); r['p_aco'] = dict(cost=float(c), secs=dt, ls_cost=float(c2))
        t = time.time(); s, c = mcts(C, ml, sc, iters_per_step=max(20, int(4000 / C.shape[0])), time_limit=budget); dt = time.time() - t
        s2, c2 = local_search(s, C, 12, 3); r['mcts'] = dict(cost=float(c), secs=dt, ls_cost=float(c2))
        res[f'{name}_T{nT}'] = r
        print(f"{name} T{nT}: VUB {vub:.0f} | ACT beam10k {b10k:.0f} B-Swan {bswan:.0f} | "
              + ' | '.join(f"{k} {r[k]['cost']:.0f}->{r[k]['ls_cost']:.0f}" for k in
                           ('beam_cost_only', 'beam_hand', 'beam_discovered', 'p_aco', 'mcts')), flush=True)
        json.dump(dict(rule=show(rule), W=W, results=res), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
