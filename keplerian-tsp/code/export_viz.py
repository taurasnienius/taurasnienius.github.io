"""Export data for the interactive page: body orbits (one period, uniform in time), tour legs as 3D
polylines, per-leg delta-v, and the search-progress traces parsed from the run logs."""
import json, re, glob, numpy as np, pykep as pk
from ktsp_io import read_ktsp
from act_ref import VLB_VUB, TABLE3
AU, DAY = 1.495978707e11, 86400.0
NEW = [('asteroids_80', 161), ('asteroids_40', 161), ('asteroids_80', 81), ('asteroids_40', 81)]
NR, NL = 72, 20
out = dict(bodies={}, tours=[])
for name in sorted({n for n, _ in NEW}):
    I = read_ktsp(f'/home/claude/ktsp_data/0_instances/{name}.ktsp'); mu = I['mu']
    B = []
    for rv in I['rv']:
        r0, v0 = rv[:3], rv[3:]
        a = 1 / (2 / np.linalg.norm(r0) - v0 @ v0 / mu); P = 2 * np.pi * np.sqrt(a ** 3 / mu)
        ring = [np.array(pk.propagate_lagrangian([r0, v0], P * k / NR, mu, False)[0]) / AU for k in range(NR)]
        B.append(dict(P=round(P / DAY, 2), ring=[round(float(x), 3) for p in ring for x in p]))
    out['bodies'][name] = dict(t0=I['t0'], days=I['tf'] - I['t0'], list=B)
for name, nT in NEW:
    I = read_ktsp(f'/home/claude/ktsp_data/0_instances/{name}.ktsp'); mu = I['mu']
    S = json.load(open(f'best/{name}_T{nT}.json'))
    grid = np.linspace(0, I['tf'] - I['t0'], nT)
    st = lambda i, d: pk.propagate_lagrangian([I['rv'][i, :3], I['rv'][i, 3:]], d * DAY, mu, False)
    legs = []; tot = 0
    for p in range(len(S['dep_idx'])):
        a, b = S['order'][p], S['order'][p + 1]; td, ta = grid[S['dep_idx'][p]], grid[S['arr_idx'][p]]
        ra, va = st(a, td); rb, vb = st(b, ta); tof = (ta - td) * DAY; best = None
        for cw in (False, True):
            lp = pk.lambert_problem(ra, rb, tof, mu, cw, 1000)
            for v1, v2 in zip(lp.v0, lp.v1):
                dv = np.linalg.norm(np.array(v1) - va) + np.linalg.norm(np.array(v2) - vb)
                if best is None or dv < best[0]: best = (dv, v1)
        pts = [np.array(ra) / AU] + [np.array(pk.propagate_lagrangian([ra, best[1]], tof * k / NL, mu, False)[0]) / AU for k in range(1, NL + 1)]
        legs.append(dict(a=a, b=b, td=round(float(td), 2), ta=round(float(ta), 2), dv=round(best[0], 1),
                         pts=[round(float(x), 3) for q in pts for x in q])); tot += best[0]
    assert abs(tot - S['cost']) < 0.01, (tot, S['cost'])
    lb, vub = VLB_VUB[(name, nT)]
    out['tours'].append(dict(inst=name, nT=nT, cost=round(S['cost'], 2), vub=vub, lb=lb, legs=legs))
    print(name, nT, round(tot, 2), 'legs', len(legs))

# search progress traces
DUR = {('asteroids_80', 161): [40, 35], ('asteroids_80', 81): [40], ('asteroids_40', 161): [20], ('asteroids_40', 81): [20]}
runs = {}
for f in ['attack_80_161.log', 'q1.log', 'q2.log']:
    cur = None
    for line in open(f):
        m = re.match(r'(\w+) T(\d+) start beam-hand: ([\d.]+)', line)
        if m:
            cur = (m.group(1), int(m.group(2))); runs.setdefault(cur, []).append(dict(starts=[float(m.group(3))], pts=[])); continue
        m = re.match(r'(\w+) T(\d+) start beam-discovered: ([\d.]+)', line)
        if m: runs[cur][-1]['starts'].append(float(m.group(3))); continue
        m = re.match(r'\s+(\d+)s it \d+: ([\d.]+)', line)
        if m and cur: runs[cur][-1]['pts'].append((int(m.group(1)), float(m.group(2))))
out['progress'] = {}
for key in NEW:
    tr = []; off = 0; rs = runs[key]
    tr.append([0, min(rs[0]['starts'])])
    for r, d in zip(rs, DUR[key]):
        for t, c in r['pts']: tr.append([round((off + t) / 60, 2), c])
        off += d * 60
    tr.append([round(off / 60, 2), tr[-1][1]])
    out['progress'][f'{key[0]}_T{key[1]}'] = dict(trace=tr, vub=VLB_VUB[key][1], bswan=TABLE3[key][2], act_beam=TABLE3[key][0])
    print(key, len(tr), tr[0], tr[-1])
json.dump(out, open('site/viz.json', 'w'), separators=(',', ':'))
import os; print(os.path.getsize('site/viz.json') // 1024, 'kB')
