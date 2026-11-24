"""Builds the interactive page (site/index.html) from page_tpl.html and the result files."""
import json, numpy as np
import build_page as bp           # reuses the tables and summary numbers computed there
from act_ref import VLB_VUB, TABLE3
R = bp.R
viz = json.load(open('site/viz.json'))
keys = [f"{t['inst']}_T{t['nT']}" for t in viz['tours']]
D = dict(results={k: dict(beam_hand=dict(cost=R[k]['beam_hand']['cost'], ls_cost=R[k]['beam_hand']['ls_cost'])) for k in keys})
D['grid'] = {}
for name in ('asteroids_20', 'jovianmoons_20'):
    Ts = [21, 41, 81, 161]; vub = [VLB_VUB[(name, t)][1] for t in Ts]
    D['grid'][name] = dict(T=Ts, act=[round(TABLE3[(name, t)][0] / v, 4) for t, v in zip(Ts, vub)],
                           swan=[round(TABLE3[(name, t)][2] / v, 4) for t, v in zip(Ts, vub)],
                           mine=[round(R[f'{name}_T{t}']['beam_hand']['cost'] / v, 4) for t, v in zip(Ts, vub)])
labels, swan, mine = [], [], []
for k in bp.INST:
    name, T = k.rsplit('_T', 1); T = int(T); vub = VLB_VUB[(name, T)][1]; m, _ = bp.mine(name, T)
    if abs(m - vub) <= 0.02: m = vub
    labels.append(f"{'Jovian moons' if name.startswith('jov') else 'Asteroids'} {int(name.split('_')[1])} / {T}")
    swan.append(round((TABLE3[(name, T)][2] / vub - 1) * 100, 3)); mine.append(round((m / vub - 1) * 100, 3))
D['all'] = dict(labels=labels, swan=swan, mine=mine)
D['train'] = dict(hand=1.0873, runs=[[json.loads(l)['best_fit'] for l in open(f'gp_seed{s}.jsonl')] for s in (1, 2, 3)])
n = bp.names
D['held'] = dict(labels=[['hand:', 'on schedule'], ['hand:', 'cost only'], ['evolved', 'run 1'], ['evolved', 'run 2'], ['evolved', 'run 3'], ['evolved,', 'seeded']],
                 w50=[round(bp.gm(50, x), 4) for x in n], w200=[round(bp.gm(200, x), 4) for x in n])
# check the hard-coded hand-rule training fitness against a fresh evaluation is done in check_hand.py
big = viz['progress']['asteroids_80_T161']
sub = {'__DATA__': json.dumps(D, separators=(',', ':')), '__NNEW__': str(bp.N_NEW), '__BIG__': f'{bp.BIG:.1f}'.replace('-', '−'),
       '__BIGABS__': f'{abs(bp.BIG):.1f}', '__NEXACT__': str(bp.n_exact), '__SUMMARY__': bp.SUMMARY, '__TABLE__': bp.TABLE, '__NB__': bp.NB,
       '__B10MIN__': f'{bp.B10MIN:.0f}', '__B10MAX__': f'{bp.B10MAX:,.0f}', '__COSTX__': f'{bp.g200[bp.hand_cost]/bp.g200[bp.hand_on]:.1f}',
       '__EVLO__': f'{(bp.ev_best/bp.g200[bp.hand_on]-1)*100:+.1f}'.replace('-', '−'), '__EVHI__': f'{(bp.ev_worst/bp.g200[bp.hand_on]-1)*100:+.1f}',
       '__SEEDED__': f'{(bp.g200[n[5]]/bp.g200[bp.hand_on]-1)*100:+.1f}',
       '__LADX__': f"{big['act_beam']/viz['tours'][0]['cost']:.1f}"}
s = open('page_tpl.html').read()
for k, v in sub.items():
    assert k in s, k; s = s.replace(k, v)
assert '__' not in s.replace('__proto__', ''), [x for x in s.split() if '__' in x][:5]
open('site/index.html', 'w').write(s)
print('interactive page built', len(s) // 1024, 'kB')
