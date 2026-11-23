"""Held-out test of discovered ranking rules against the hand rule, at several beam widths (beam only, no
local search), on instances not used in training."""
import pickle, json, sys, time, numpy as np
from costs import build
from beam import beam_search, minleg_table, HAND
from gp import ev, show
from act_ref import VLB_VUB
rules = {'hand: on-schedule': HAND['on-schedule (A at t*)'], 'hand: cost-only': HAND['cost-only (min_t A)']}
for f in sys.argv[1:]:
    d = pickle.load(open(f, 'rb')); tr = d['best']
    rules[f'{f}: {show(tr)}'] = (lambda t: (lambda F: ev(t, F)))(tr)
TEST = [('asteroids_20', 21), ('asteroids_20', 161), ('jovianmoons_20', 21), ('jovianmoons_20', 161),
        ('asteroids_40', 81), ('asteroids_40', 161), ('asteroids_80', 81), ('asteroids_80', 161), ('jovianmoons_20', 41)]
out = {}
for W in (50, 200):
    for name, nT in TEST:
        C = build(name, nT); ml = minleg_table(C); vub = VLB_VUB[(name, nT)][1]
        row = {}
        for k, r in rules.items():
            _, c = beam_search(C, r, W=W, minleg=ml); row[k] = c / vub
        out[f'W{W} {name}_T{nT}'] = row
        print(f'W{W} {name} T{nT}: ' + ' | '.join(f'{v:.3f}' for v in row.values()), flush=True)
    json.dump(dict(rules=list(rules), results=out), open('results_rules.json', 'w'), indent=1)
for W in (50, 200):
    rows = [v for k, v in out.items() if k.startswith(f'W{W} ')]
    print(f'W{W} geometric mean ratio to best known:')
    for k in rules:
        print(f'   {np.exp(np.mean([np.log(r[k]) for r in rows])):.4f}  {k}')
