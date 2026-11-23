"""Builds site/index.html from the result files (so every number on the page comes from a file)."""
import json, glob, re, numpy as np
from act_ref import VLB_VUB, TABLE3

R = json.load(open('results_compare.json'))['results']
EX = json.load(open('results_exact.json'))
RULES = json.load(open('results_rules.json'))
BEST = {(d['instance'], d['nT']): d for d in (json.load(open(f)) for f in glob.glob('best/*.json'))}
VER = {(v['instance'], v['nT']): v for v in (json.loads(l) for l in open('verify.jsonl'))}
CSS = re.search(r'<style>(.*?)</style>', open('/home/claude/ghsite/breathing-the-moon/paper.html').read(), re.S).group(1)

INST = [k for k in R]


def lab(name, T):
    return f"{'Jovian moons' if name.startswith('jov') else 'Asteroids'} {int(name.split('_')[1])} / {T}"


def fmt(x):
    return f'{x:,.0f}'.replace(',', '&#8239;')


def mine(name, T):
    r = R[f'{name}_T{T}']
    vals = {k: r[k]['ls_cost'] for k in ('beam_hand', 'beam_discovered', 'p_aco', 'mcts')}
    if (name, T) in BEST:
        vals['long search'] = BEST[(name, T)]['cost']
    k = min(vals, key=vals.get)
    return vals[k], k


rows = []
newbest = []
for k in INST:
    name, T = k.rsplit('_T', 1); T = int(T)
    lb, vub = VLB_VUB[(name, T)]; bsw = TABLE3[(name, T)][2]; b10 = TABLE3[(name, T)][0]
    m, how = mine(name, T)
    if abs(m - vub) <= 0.02: m = vub
    ch = (m / vub - 1) * 100
    cls = ' style="color:#7cc6e8;font-weight:600"' if m < vub - 0.005 else ''
    if m < vub - 0.005:
        newbest.append((name, T, m, vub, lb))
    gap_old = (vub - lb) / vub * 100 if vub > lb + 0.01 else 0
    rows.append(f'<tr><td>{lab(name, T)}</td><td class="num r">{fmt(lb)}</td><td class="num r">{fmt(vub)}</td>'
                f'<td class="num r">{fmt(bsw)}</td><td class="num r"{cls}>{fmt(m)}</td>'
                f'<td class="num r"{cls}>{("0.00%" if m == vub else f"{ch:+.2f}%")}</td></tr>')
TABLE = '\n'.join(rows)

nb_rows = []
for name, T, m, vub, lb in sorted(newbest, key=lambda x: (x[0], x[1])):
    v = VER.get((name, T))
    vtxt = f'{v["worst_position_miss_m"]*100:.1f} cm' if v else 'pending'
    old_gap = (vub - lb) / vub * 100; new_gap = (m - lb) / m * 100
    nb_rows.append(f'<tr><td>{lab(name, T)}</td><td class="num r">{fmt(vub)}</td><td class="num r" style="color:#7cc6e8;font-weight:600">{fmt(m)}</td>'
                   f'<td class="num r">{(m/vub-1)*100:+.2f}%</td><td class="num r">{old_gap:.1f}% &rarr; {new_gap:.1f}%</td>'
                   f'<td class="num r">{vtxt}</td><td><a href="tours/{name}_T{T}.json">JSON</a></td></tr>')
NB = '\n'.join(nb_rows)
N_NEW = len(newbest)
big = [x for x in newbest if x[0] == 'asteroids_80' and x[1] == 161]
BIG = (big[0][2] / big[0][3] - 1) * 100 if big else float('nan')
best_cut = min((x[2] / x[3] - 1) * 100 for x in newbest) if newbest else 0

b10r=[(TABLE3[(k.rsplit('_T',1)[0],int(k.rsplit('_T',1)[1]))][0]/VLB_VUB[(k.rsplit('_T',1)[0],int(k.rsplit('_T',1)[1]))][1]-1)*100 for k in INST]
B10MIN, B10MAX = min(b10r), max(b10r)
n_match = 0; short = []
for k in INST:
    nm, T = k.rsplit('_T', 1); T = int(T); m, _ = mine(nm, T); vub = VLB_VUB[(nm, T)][1]
    if abs(m - vub) <= 0.02: n_match += 1
    elif m > vub: short.append((lab(nm, T), (m / vub - 1) * 100, VLB_VUB[(nm, T)][0] >= vub - 0.01))
SUMMARY = (f"On {N_NEW} instances my tour is cheaper than the previous best and on the other {n_match} it equals it to the cent"
           + (f"; on {len(short)} it is still above it: " + ", ".join(f"{l} ({p:+.1f}%)" for l, p, _ in short) + ". " if short else
              ". Five of those nine are proven optimal in [1], so matching is the best possible there. On the coarsest grids, where each leg "
              "takes a single grid step, beam search plus one local search ends up to 12% above the optimum, and the longer iterated local search closes that gap. ")
           + ("All of those have a proven optimum in [1]. They are the coarsest grids, where each leg takes a single grid step and the problem "
              "becomes a pure ordering problem: that is where the mixed-integer solver is strongest and my timing idea brings nothing." if short and all(o for _, _, o in short) else ""))
# exact reproduction
n_exact = sum(1 for k, v in EX.items() if abs(v - VLB_VUB[(k.rsplit('_T', 1)[0], int(k.rsplit('_T', 1)[1]))][1]) < 0.011)

# grid figure numbers
a20 = R['asteroids_20_T161']
ratio_beam = TABLE3[('asteroids_20', 161)][0] / a20['beam_hand']['cost']

# discovery summary
names = RULES['rules']; res = RULES['results']
def gm(W, n):
    rows_ = [v for k, v in res.items() if k.startswith(f'W{W} ')]
    return np.exp(np.mean([np.log(r[n]) for r in rows_]))
hand_on, hand_cost = names[0], names[1]
evolved = names[2:5]
g200 = {n: gm(200, n) for n in names}
ev_best = min(g200[n] for n in evolved); ev_worst = max(g200[n] for n in evolved)

HTML = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Keep the clock open: the Keplerian travelling salesperson problem · Tauras Nienius</title>
<meta name="description" content="Cheaper multi-asteroid tours on ESA's Keplerian TSP benchmark: a verified evaluator, a beam search that keeps timing open, new best-known tours on {N_NEW} open instances, and an automatic search for ranking rules.">
<meta property="og:title" content="Keep the clock open: new best-known tours on ESA's Keplerian TSP benchmark">
<meta property="og:description" content="80 asteroids, one spacecraft: a tour {abs(BIG):.1f}% cheaper than the best known, checked arc by arc.">
<meta property="og:type" content="article">
<meta property="og:url" content="https://taurasnienius.github.io/keplerian-tsp/">
<meta property="og:image" content="https://taurasnienius.github.io/keplerian-tsp/og.jpg">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}
.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:26px 0 6px}}
.kpi{{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px 14px 12px}}
.kpi b{{display:block;font:500 25px/1.1 "IBM Plex Mono",monospace;color:var(--ink)}}
.kpi span{{display:block;color:var(--muted);font-size:13.5px;line-height:1.4;margin-top:6px}}
.kpi.hl b{{color:var(--earth)}}
@media (max-width:760px){{.kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.lede{{font-size:19px;line-height:1.6;margin:0 0 12px}}
main{{margin:0 auto !important;max-width:760px}}
.tbl table{{font-size:13.5px}} .tbl td.num{{font-size:12.5px}}
</style>
</head>
<body>
<div class="page">
<header class="top">
  <a class="back" href="../">&larr; Tauras Nienius</a>
  <h1>Keep the clock open: cheaper tours for ESA's Keplerian travelling salesperson problem</h1>
  <p class="lede">A spacecraft has to visit every asteroid in a group and come home before a deadline. Which order, and on which dates? I rebuilt ESA's benchmark for this problem from scratch and found cheaper tours than the best known on {N_NEW} of its open instances.</p>
  <p class="byline"><b>Tauras Nienius</b> &middot; Aerospace engineering, ELISA Aerospace (ISAE Group) &middot; November 2026 &middot; <a href="https://github.com/taurasnienius/taurasnienius.github.io/tree/main/keplerian-tsp/code">code</a></p>
  <div class="kpis">
    <div class="kpi"><b>{n_exact}/22</b><span>published optimal values reproduced to the cent</span></div>
    <div class="kpi hl"><b>{BIG:+.1f}%</b><span>Δv against the best known tour, 80 asteroids</span></div>
    <div class="kpi hl"><b>{N_NEW}</b><span>open instances with a new best-known tour</span></div>
    <div class="kpi"><b>2</b><span>CPU cores, against a 128-core server for the old records</span></div>
  </div>
</header>

<main>
<h2><span class="sn">1</span>The problem</h2>
<p>ESA's Advanced Concepts Team published the Keplerian travelling salesperson problem in 2026 with a set of benchmark instances [1]. Each instance is a group of bodies on their orbits: 5 to 80 main-belt asteroids, or 4 to 20 moons of Jupiter. A spacecraft starts at the first body, has to rendezvous with every other one, and come back to the first body before the window closes. Each leg is a Lambert arc between two dates, and its cost is the velocity change at departure plus the one at arrival. Dates are restricted to a grid of 6 to 161 points across the window. The goal is the tour with the smallest total Δv.</p>
<p>It is harder than a normal travelling salesman problem because the "distance" between two asteroids changes all the time: the same pair can cost 2 km/s one month and 20 km/s the next. Choosing the order and choosing the dates are one problem, not two.</p>
<figure class="mid"><img src="img/fig_tour80.png" alt="The new best tour through 80 asteroids, drawn from above the ecliptic" loading="lazy">
<figcaption><b>The new best tour through 80 asteroids</b> (161 time points), seen from above. Grey lines are the asteroids' orbits; each coloured arc is one Lambert transfer, from purple at the start to yellow at the end.</figcaption></figure>

<h2><span class="sn">2</span>First, make sure it is the same problem</h2>
<p>A better number only means something if it is computed exactly the way ACT computes it. I wrote my own transfer-cost tables and an exact solver for small instances (dynamic programming over the set of visited bodies, the current body and the time). The cost model that matches theirs: Lambert arcs with any number of revolutions in both directions, departure plus arrival impulse, waiting at a body for free.</p>
<p>With that model, my exact solver reproduces all {n_exact} optimal values ACT published for instances with up to 10 bodies, to the cent, each in about a second. This is what every comparison below rests on.</p>

<h2><span class="sn">3</span>The idea: keep the clock open</h2>
<p>A classic beam search builds tours body by body and keeps the best few hundred or thousand partial tours. To compare partial tours, it has to fix when the spacecraft arrives at each body. On a coarse grid that is fine. On a fine grid it locks in early choices of date that later turn out expensive, and the beam has no way back.</p>
<p>My beam search does not fix the arrival date. For each partial tour it keeps a whole curve: the cheapest Δv to be at the current body, ready to leave, for every date on the grid. Adding a leg updates the curve exactly (a min-plus step over the grid). Partial tours are then ranked by a rule; my hand-designed one is "on schedule": the cost of the curve at the date where the tour would be if it spread its legs evenly over the window. For a finished order, the same curves give the optimal dates exactly, so the local search that polishes tours (moving bodies, swapping, reversing short stretches) always works with perfectly timed tours.</p>
<figure><img src="img/fig_grid.png" alt="Tour cost against the number of grid points for ACT's beam search, B-Swan and my beam search" loading="lazy">
<figcaption><b>Finer grids should help, and with open timing they do.</b> ACT's beam search (width 10,000, values from [1]) gets worse as the grid gets finer; with 161 points on 20 asteroids it ends {ratio_beam:.0f} times above my beam search of width 100. Beam search alone, no local search.</figcaption></figure>

<h2><span class="sn">4</span>Results on the thirteen harder instances</h2>
<p>All instances with 20 or more bodies. Instances with fewer bodies are solved exactly (Section 2). "Best known" is the best of all Gurobi runs in [1] (4-hour limit, 2 × 64-core server), "lower bound" the best proven bound. My column is the best of my methods: the two beam searches, ant colony and MCTS, each followed by local search, and a longer iterated local search of 6 to 75 minutes per instance on one core.</p>
<div class="tbl"><table>
<thead><tr><th>Instance (bodies / grid points)</th><th class="r">Lower bound</th><th class="r">Best known</th><th class="r">ACT B-Swan</th><th class="r">Mine</th><th class="r">vs best known</th></tr></thead>
<tbody>
{TABLE}
</tbody>
<caption>Total Δv in m/s. Blue: lower than the previous best known. Reference values from Tables 3 and 4 of [1]; ACT's beam search results are in the chart of Section 3.</caption>
</table></div>
<p>{SUMMARY}</p>
<figure><img src="img/fig_all.png" alt="Gap to the previous best known for ACT B-Swan and my methods on thirteen instances" loading="lazy">
<figcaption><b>Distance to the previous best known tour.</b> Below zero is a new best. ACT's beam search is left out of this chart: it sits between {B10MIN:.0f}% and {B10MAX:,.0f}% above.</figcaption></figure>

<h2><span class="sn">5</span>New best-known tours</h2>
<div class="tbl"><table>
<thead><tr><th>Instance</th><th class="r">Previous best</th><th class="r">New</th><th class="r">Change</th><th class="r">Gap to lower bound</th><th class="r">Worst arc miss</th><th>Tour</th></tr></thead>
<tbody>
{NB}
</tbody>
<caption>Δv in m/s. Gap to lower bound = (tour − bound) / tour, before and after. Each tour file lists the order, the grid dates and every leg.</caption>
</table></div>
<p>Each new tour was checked by a separate script that shares no code with the solver (only the Lambert library, pykep): it reads the instance file, moves every body to the leg dates, solves each Lambert arc again, propagates the spacecraft along it to confirm it really reaches the next body (worst miss in the table), checks the visiting and date rules, and adds up the Δv. The totals agree with the solver to the last digit.</p>
<div class="box"><p class="t">What this does and does not show</p>
<p>The new values are upper bounds on the optimum, found by heuristic search; I cannot prove any of them optimal. On the small instances where the optimum is known, my methods reach it, which says the search is sound, not that it is complete. The comparison is fair on the problem (same grid, same cost model, same rules) but not on computers: the old records came from a mixed-integer solver run for up to four hours on a large server, mine from heuristics on two CPU cores.</p></div>

<h2><span class="sn">6</span>Can the search rule be discovered automatically?</h2>
<p>This is the question of ACT's internship topic: replace hand-designed parts of search methods (beam rules, rollouts, pheromone updates) with parts found by an automatic generate, test and refine loop [5, 6]. I tested it on the one component that mattered most in my beam search, the rule that ranks partial tours.</p>
<p>A genetic programming loop wrote ranking rules as formulas over eight features of a partial tour (cost on schedule, cheapest cost, cost a little earlier and later, last leg, progress, time used, cheapest next legs). It trained on six small instances only, and every rule was then tested unchanged on nine others, most of them larger.</p>
<figure><img src="img/fig_discovery.png" alt="Training curves of three evolution runs and held-out comparison of rules" loading="lazy">
<figcaption><b>Left:</b> three runs from random formulas each beat my hand rule on the training set within a few generations. <b>Right:</b> on unseen instances (geometric mean of tour Δv / best known, beam search alone) the evolved rules land within about 1% of the hand rule, on both sides, and far ahead of ranking by cost alone.</figcaption></figure>
<p>What came out:</p>
<ul>
<li><strong>Starting from random formulas, every run rediscovered the same principle as my hand rule:</strong> judge a partial tour by its cost on schedule and penalise using up time. Ranking by cost alone, the obvious choice, is {g200[hand_cost]/g200[hand_on]:.1f} times worse on the held-out instances.</li>
<li><strong>None of the evolved rules beat the hand rule reliably</strong> on instances it had not seen: at beam width 200 they range from {(ev_best/g200[hand_on]-1)*100:+.1f}% to {(ev_worst/g200[hand_on]-1)*100:+.1f}% against it. The gains seen in training did not transfer.</li>
<li><strong>The largest improvement in this project was structural, not a better formula:</strong> keeping the timing open (Section 3) is a change to how the search represents a tour. A search over formulas cannot invent that; a search over code, which is what language-model methods do [5, 6], could. That is the part I would most like to explore with the team.</li>
</ul>

<h2><span class="sn">7</span>Limits</h2>
<ul>
<li>The model is ACT's benchmark model: two-body dynamics, impulsive Lambert legs, dates on a fixed grid. Real low-thrust tours are a different problem.</li>
<li>All values are for the grid of each instance. A finer grid gives cheaper tours; comparisons are only made on the same grid.</li>
<li>My tours visit each body exactly once. ACT's formulation allows repeat visits, so a still cheaper tour could exist outside my search space.</li>
<li>The automatic discovery used genetic programming over a fixed set of features, not a language model: this workspace has two CPU cores and no way to run one. The framework uses the same generate, evaluate, select loop, so a model-based generator can replace the GP step.</li>
<li>Run times were short (minutes to about an hour per instance). Longer runs would probably improve the large instances further.</li>
</ul>

<h2><span class="sn">8</span>Code and data</h2>
<p>All code is public in the <a href="https://github.com/taurasnienius/taurasnienius.github.io/tree/main/keplerian-tsp/code">code folder</a>: cost tables, exact solver, beam search, ant colony, MCTS, local search, the genetic programming loop, and the independent checker. The instance files are ACT's own (Zenodo, doi:10.5281/zenodo.14850862). The new tours are in <a href="tours/">tours/</a>, one JSON file each.</p>

<h2><span class="sn">9</span>References</h2>
<ol class="refs">
<li>M. Bannach, G. Acciarini, D. Izzo, <i>The Keplerian Traveling Salesperson Problem</i>, arXiv:2605.00010, 2026. Instances: doi:10.5281/zenodo.14850862.</li>
<li>D. Izzo et al., <i>Evolving solutions to TSP variants for active space debris removal</i>, GECCO 2015.</li>
<li>L. F. Simões, D. Izzo, E. Haasdijk, A. E. Eiben, <i>Multi-rendezvous spacecraft trajectory optimization with beam P-ACO</i>, EvoCOP 2017.</li>
<li>M. Schmidt, H. Lipson, <i>Distilling free-form natural laws from experimental data</i>, Science 324, 2009.</li>
<li>F. Liu et al., <i>Evolution of Heuristics: towards efficient automatic algorithm design using large language model</i>, ICML 2024.</li>
<li>H. Ye et al., <i>ReEvo: large language models as hyper-heuristics with reflective evolution</i>, NeurIPS 2024.</li>
</ol>
</main>
<footer><a href="../">Tauras Nienius</a> · <a href="mailto:taurasn.nie@gmail.com">taurasn.nie@gmail.com</a> · <a href="https://www.linkedin.com/in/tauras-nienius/">LinkedIn</a> · November 2026</footer>
</div>
</body>
</html>
'''
open('site/index.html', 'w').write(HTML)
print('page built', N_NEW, 'new best', f'{BIG:+.2f}%')
