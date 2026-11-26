"""Builds site/paper.html (technical paper) from the result files."""
import json, numpy as np
import build_page as bp
from act_ref import VLB_VUB, TABLE3

R, EX, RULES = bp.R, bp.EX, bp.RULES
viz = json.load(open('site/viz.json'))
f0 = lambda x: f'{x:,.0f}'.replace(',', '&#8239;')
f2 = lambda x: f'{x:,.2f}'.replace(',', '&#8239;')

# Table 1: exact reproduction
t1 = ''
for k in sorted(EX, key=lambda s: (s.split('_')[0], int(s.split('_')[1]), int(s.rsplit('T', 1)[1]))):
    name, T = k.rsplit('_T', 1); T = int(T); ref = VLB_VUB[(name, T)][1]
    t1 += f'<tr><td>{bp.lab(name, T)}</td><td class="num r">{f2(ref)}</td><td class="num r">{f2(EX[k])}</td><td class="num r">{abs(EX[k]-ref):.2f}</td></tr>\n'
maxdiff = max(abs(EX[k] - VLB_VUB[(k.rsplit('_T', 1)[0], int(k.rsplit('_T', 1)[1]))][1]) for k in EX)

# Table 3: methods on 13 instances, ratio to previous best known
t3 = ''
cols = ('beam_cost_only', 'beam_hand', 'p_aco', 'mcts')
agg = {c: [] for c in cols}; aggl = {c: [] for c in cols}
for k in bp.INST:
    name, T = k.rsplit('_T', 1); T = int(T); vub = VLB_VUB[(name, T)][1]; r = R[k]; m, _ = bp.mine(name, T)
    cells = ''
    for c in cols:
        a, b = r[c]['cost'] / vub, r[c]['ls_cost'] / vub; agg[c].append(a); aggl[c].append(b)
        cells += f'<td class="num r">{a:.3f} &rarr; {b:.3f}</td>'
    t3 += f'<tr><td>{bp.lab(name, T)}</td>{cells}<td class="num r">{min(m, vub + (0 if abs(m-vub) > .02 else 0))/vub:.4f}</td></tr>\n'
gmean = lambda v: float(np.exp(np.mean(np.log(v))))
t3 += '<tr><td><b>Geometric mean</b></td>' + ''.join(f'<td class="num r"><b>{gmean(agg[c]):.3f} &rarr; {gmean(aggl[c]):.3f}</b></td>' for c in cols) + '<td></td></tr>'

# Table 5: rules
names = RULES['rules']
pretty = ['a_sched (hand rule, "on schedule")', 'a_min (hand rule, "cost only")'] + [n.split(': ', 1)[1] for n in names[2:]]
origin = ['hand', 'hand', 'evolved, run 1', 'evolved, run 2', 'evolved, run 3', 'evolved, seeded with a_sched']
t5 = ''.join(f'<tr><td>{o}</td><td class="num" style="white-space:normal">{p}</td><td class="num r">{bp.gm(50, n):.3f}</td><td class="num r">{bp.gm(200, n):.3f}</td></tr>\n'
             for o, p, n in zip(origin, pretty, names))

big = viz['progress']['asteroids_80_T161']
cross = next(m for m, c in big['trace'] if c < big['vub'])

HTML = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deferred-timing search for the Keplerian TSP: technical paper · Tauras Nienius</title>
<meta name="description" content="Technical paper: cost model verification, exact scheduling of a fixed visiting order, beam search with deferred timing, local search, automatic rule discovery and new best-known tours on ESA ACT's Keplerian TSP benchmark.">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{bp.CSS}
main{{margin:0 auto !important;max-width:780px}}
.tbl table{{font-size:13.5px}} .tbl td.num{{font-size:12.5px}}
</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$','$'],['\\\\(','\\\\)']], displayMath: [['$$','$$'],['\\\\[','\\\\]']], tags: 'none' }},
  svg: {{ fontCache: 'global' }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }}
}};
</script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js"></script>
</head>
<body>
<div class="page">
<header class="top">
  <a class="back" href="./">&larr; A cheaper way around</a>
  <h1>Deferred-timing search for the Keplerian travelling salesperson problem</h1>
  <p class="byline"><b>Tauras Nienius</b> &middot; Aerospace engineering, ELISA Aerospace (ISAE Group) &middot; November 2026 &middot; technical paper &middot; <a href="https://github.com/taurasnienius/taurasnienius.github.io/tree/main/keplerian-tsp/code">code</a></p>
</header>
<main>
<div class="box"><p class="t">Abstract</p>
<p>The Keplerian travelling salesperson problem (KTSP) asks for the cheapest closed tour, in total Δv, that rendezvous with every body of a set moving on Keplerian orbits, with departure and arrival dates on a time grid [1]. I rebuild the cost model of the benchmark published by ESA's Advanced Concepts Team and verify it by reproducing all {bp.n_exact} published optima for instances with up to 10 bodies (to the printed cent). I then describe a search in which the timing of a partial tour is never fixed: every partial tour carries its whole cost-versus-time curve, which is updated exactly when a leg is added. This gives an exact schedule for any fixed visiting order, a beam search that does not degrade on fine grids, and a local search that always compares optimally timed tours. On the 13 instances with 20 or more bodies the method finds {bp.N_NEW} tours cheaper than the best published ones (up to {abs(bp.BIG):.1f} % on 80 asteroids and 161 time points) and equals the published value on the other {bp.n_match}. A genetic-programming search over the beam's ranking rule rediscovers the hand-designed rule but does not beat it on held-out instances. The technique of propagating time-dependent cost functions is known from time-dependent routing [7]; the contribution here is its application to this benchmark, the verified results and the negative result on rule discovery.</p></div>

<h2><span class="sn">1</span>Problem</h2>
<p>An instance gives $n$ bodies (body 0 is the depot), their Cartesian states at an initial epoch, a gravitational parameter $\\mu$ and a time window $[t_0, t_f]$. Time is restricted to a uniform grid of $K$ points $\\tau_0=t_0,\\dots,\\tau_{{K-1}}=t_f$. A tour is a permutation $\\pi$ of the bodies $1..n-1$ together with departure and arrival indices $(d_p, a_p)$ for each of the $n$ legs, with</p>
$$ d_p < a_p, \\qquad a_p \\le d_{{p+1}}, \\qquad 0 \\le d_0, \\quad a_{{n-1}} \\le K-1 . \\tag{{1}}$$
<p>Waiting at a body between an arrival and the next departure is free. The cost of the tour is the sum of the leg costs,</p>
$$ J(\\pi, d, a) = \\sum_{{p=0}}^{{n-1}} c\\big(\\pi_p, d_p, \\pi_{{p+1}}, a_p\\big), \\qquad \\pi_0=\\pi_n=0 . \\tag{{2}}$$
<p>I consider tours that visit every body exactly once. The formulation of [1] works on a time-expanded network and also admits repeated visits; this restriction is discussed in Section 10.</p>

<h2><span class="sn">2</span>Cost model and its verification</h2>
<p>The leg cost is the two-impulse rendezvous cost of a Lambert arc. With $\\mathbf r_i(\\tau), \\mathbf v_i(\\tau)$ the two-body state of body $i$ at time $\\tau$,</p>
$$ c(i,k,j,l) = \\min_{{s}} \\Big( \\lVert \\mathbf v^{{\\rm dep}}_s - \\mathbf v_i(\\tau_k) \\rVert + \\lVert \\mathbf v^{{\\rm arr}}_s - \\mathbf v_j(\\tau_l) \\rVert \\Big), \\tag{{3}}$$
<p>where $s$ runs over all solutions of Lambert's problem from $\\mathbf r_i(\\tau_k)$ to $\\mathbf r_j(\\tau_l)$ in time $\\tau_l-\\tau_k$: every admissible number of revolutions, both branches, and both transfer directions. Bodies are propagated with <code>pykep.propagate_lagrangian</code> and arcs solved with <code>pykep.lambert_problem</code>. All $c(i,k,j,l)$ are precomputed into a tensor $C$ of size $n \\times K \\times n \\times K$ (about 1.3 GB for $n=80$, $K=161$; 7 minutes on two cores).</p>
<p>The paper [1] does not need to spell out these choices because it distributes the time-expanded networks. I did not use those files; I identified the model by brute force on the smallest instances. Multi-revolution arcs are required: on Jovian moons 4/11 the optimum is 33&#8239;268.31 m/s with zero-revolution arcs only and 20&#8239;096.59 m/s with all revolutions, the published value. Allowing the retrograde direction changed nothing on the instances I tested and is kept for completeness.</p>
<p><strong>Exact solver.</strong> For small $n$ a dynamic programme over (visited set $S$, current body $j$, time index $t$) is exact:</p>
$$ F(S \\cup \\{{m\\}}, m, l) = \\min_{{j \\in S}} \\; \\min_{{k \\lt l}} \\Big[ \\min_{{k' \\le k}} F(S, j, k') + c(j,k,m,l) \\Big], \\tag{{4}}$$
<p>with $F(\\{{0\\}},0,\\cdot)=0$ and the closing leg back to body 0. It runs in $O(2^n n^2 K^2)$; 10 bodies and 161 time points take 1.6 s. Table 1 compares it with every optimum published in [1] for $n \\le 10$.</p>
<div class="tbl"><table>
<thead><tr><th>Instance (bodies / grid points)</th><th class="r">Optimum in [1]</th><th class="r">This work</th><th class="r">Difference</th></tr></thead>
<tbody>{t1}</tbody>
<caption>Table 1. Reproduction of the published optima, m/s. Absolute differences; 0.01 is rounding in the last printed digit.</caption></table></div>

<h2><span class="sn">3</span>Exact timing of a fixed order</h2>
<p>Fix the order $\\pi$. Let $A_p(t)$ be the cheapest cost of the first $p$ legs such that the spacecraft is at body $\\pi_p$ and free to leave at grid index $t$. Because waiting is free, $A_p$ is non-increasing in $t$. Then $A_0 \\equiv 0$ and</p>
$$ A_p(t) = \\min_{{l \\le t}} \\; \\min_{{k < l}} \\Big[ A_{{p-1}}(k) + c(\\pi_{{p-1}}, k, \\pi_p, l) \\Big]. \\tag{{5}}$$
<p>This is a min-plus product of the vector $A_{{p-1}}$ with the $K \\times K$ slice of $C$ for that pair of bodies, followed by a running minimum. The optimal cost of the order is $A_n(K-1)$ and the dates follow by backtracking. One leg costs $O(K^2)$, a whole tour $O(nK^2)$: about 20 &micro;s per leg for $K=161$ in compiled code.</p>
<p>The same recursion run backwards gives $B_p(t)$, the cheapest cost of legs $p..n-1$ when the spacecraft is at $\\pi_p$ and free to leave at $t$. For any position $p$,</p>
$$ J^*(\\pi) = \\min_t \\big[ A_p(t) + B_p(t) \\big]. \\tag{{6}}$$
<p>Equation (6) makes local search cheap: if a move changes the order only between positions $lo$ and $hi$, the prefix curve $A_{{lo-1}}$ and the suffix curve $B_{{hi+1}}$ are still valid, so the new tour is evaluated exactly with $hi-lo+2$ applications of (5) instead of $n$.</p>

<h2><span class="sn">4</span>Beam search with deferred timing</h2>
<p>A beam search over (order prefix, arrival index) pairs has to commit to dates as it goes. In [1] the beam search of width 10&#8239;000 ends between {bp.B10MIN:.0f} % and {bp.B10MAX:,.0f} % above the best known value on the 13 larger instances, and gets worse as $K$ grows. Here a beam state is a prefix together with its whole curve $A_p(\\cdot)$. Expanding a state by body $b$ applies (5) once. Two states with the same visited set and the same last body are duplicates for ranking purposes and only the better ranked is kept. After $n-1$ expansions each surviving state is closed with the leg back to the depot.</p>
<p>What remains is how to rank states, since a curve is not a number. Each child is summarised by eight features (cost features are divided by an instance scale, the median over body pairs of the cheapest leg departing at $t_0$):</p>
<div class="tbl"><table>
<thead><tr><th>Feature</th><th>Meaning</th></tr></thead><tbody>
<tr><td class="num">a_sched</td><td>$A_p(t^*)$ at the proportional date $t^* = (p/n)(K-1)$: the cost if the tour stays on schedule</td></tr>
<tr><td class="num">a_min</td><td>$\\min_t A_p(t)$, the cheapest way to have visited the prefix at all</td></tr>
<tr><td class="num">a_late, a_early</td><td>$A_p$ at $t^* \\pm 0.1(K-1)$</td></tr>
<tr><td class="num">leg</td><td>increase of $\\min_t A$ caused by the last leg</td></tr>
<tr><td class="num">depth</td><td>$p/n$</td></tr>
<tr><td class="num">tmin</td><td>$\\arg\\min_t A_p(t) / (K-1)$, the share of the window already used at the cheapest point</td></tr>
<tr><td class="num">near</td><td>mean over unvisited bodies of the cheapest leg from the last body departing at or after $t^*$</td></tr>
</tbody><caption>Table 2. Features available to a ranking rule.</caption></table></div>
<p>The hand-designed rule is simply <code>a_sched</code>. Ranking by <code>a_min</code>, the obvious choice, favours prefixes that are cheap because they have used most of the window, and leaves no time for the rest of the tour.</p>

<h2><span class="sn">5</span>Local search and iterated local search</h2>
<p>Neighbourhood: relocation of a segment of 1 to 3 consecutive bodies, swap of two bodies, and reversal of a stretch, all restricted to positions at most $W$ apart ($W$ = 12 to 15). Every candidate is evaluated exactly with (5) and (6). Two descent variants are used: best improvement (for the comparisons of Section 7) and first improvement with a random scan order (for long runs, several times cheaper per improvement on 80 bodies).</p>
<p>Iterated local search perturbs the current tour with 4 random segment relocations inside the same window, descends, accepts the result if it is better (or with probability 0.02 otherwise), and returns to the best tour found with probability 0.01 per iteration. Runs start from the better of two beam-search tours (width 200 or 300) after one descent.</p>

<h2><span class="sn">6</span>Baselines</h2>
<p>Two of the methods named in ACT's internship topic were implemented on the same evaluator, so that the comparison isolates the search and not the cost model. <strong>P-ACO</strong> (a plain population-based ant colony optimisation of my own, simpler than the Beam P-ACO hybrid of [3], which I did not reimplement): 20 ants per iteration, population of the 5 best tours, pheromone on body-to-body arcs, $\\alpha=1$, $\\beta=2$, heuristic desirability from the on-schedule cost of the candidate leg; ants also carry the curve $A$. <strong>MCTS</strong>: UCT ($c=0.7$) over the next two bodies from the committed prefix, rollouts that complete the order by sampling from a softmin of the cheapest-leg table, each rollout scored exactly with (5), one body committed per step. Both get the wall-clock time of the beam search plus 20 s.</p>

<h2><span class="sn">7</span>Results</h2>
<div class="tbl"><table>
<thead><tr><th>Instance</th><th class="r">Beam, cost-only</th><th class="r">Beam, on-schedule</th><th class="r">P-ACO</th><th class="r">MCTS</th><th class="r">Final</th></tr></thead>
<tbody>{t3}</tbody>
<caption>Table 3. Tour Δv divided by the best value published in [1] (1 = equal, below 1 = new best). Each cell: constructive method alone &rarr; after one local-search descent. Beam width 100. "Final" is the best tour of this work, including iterated local search of 6 to 75 minutes on one core.</caption></table></div>
<p>The on-schedule beam is the strongest constructive method by a wide margin, and one descent brings it to 7 % above the best known on average. The coarsest grids ($K = n+1$) are the weak spot: each leg must take exactly one grid step, timing carries no freedom, and the problem is a pure ordering problem. There the beam plus one descent is still up to 35 % above the best known (80 bodies, 81 points), and it is the iterated local search that closes the gap.</p>
<div class="tbl"><table>
<thead><tr><th>Instance</th><th class="r">Previous best [1]</th><th class="r">New</th><th class="r">Change</th><th class="r">Gap to lower bound</th><th class="r">Worst arc miss</th><th>Tour</th></tr></thead>
<tbody>{bp.NB}</tbody>
<caption>Table 4. New best-known tours, m/s. Gap = (tour − bound)/tour with the lower bounds of [1], before &rarr; after.</caption></table></div>
<p>On 80 asteroids and 161 time points the search passes the previous best after {cross:.0f} minutes on one core and reaches {f0(viz['tours'][0]['cost'])} m/s after 75 minutes. ACT's public project page lists the same previous value for this instance (checked 25 November 2026); I am not aware of other published results on these instances, but cannot exclude unpublished ones.</p>

<h2><span class="sn">8</span>Automatic discovery of the ranking rule</h2>
<p>The ranking rule is the one hand-designed component of the beam search, which makes it a small test case for automated heuristic design [5, 6]. Rules are expression trees over the features of Table 2 and constants, with operators $+,-,\\times,\\div,\\min,\\max$, negation, $\\sqrt{{|\\cdot|}}$, $\\log(1+|\\cdot|)$ and square. Genetic programming, in the spirit of symbolic search over expressions [4]: population 200, 40 generations, tournament selection of size 4, 60 % crossover, 30 % mutation, 10 % fresh trees, 5 % elitism, trees of at most 25 nodes, parsimony pressure $2\\times10^{{-4}}$ per node. Fitness is the mean ratio to the best known value of a width-20 beam search on six training instances (10 and 20 bodies). Three independent runs start from random trees; a fourth starts from the hand rule with width 50 on five instances.</p>
<div class="tbl"><table>
<thead><tr><th>Origin</th><th>Rule (lower score is better)</th><th class="r">Held-out, W = 50</th><th class="r">Held-out, W = 200</th></tr></thead>
<tbody>{t5}</tbody>
<caption>Table 5. Geometric mean of tour Δv / best known on eight instances not used in training (20, 40 and 80 bodies), beam search alone.</caption></table></div>
<p>All three unseeded runs converge on rules dominated by <code>a_sched</code>, with a penalty on <code>tmin</code> or a product with a neighbouring sample of the curve. On the training set they improve on the hand rule by about 3 % (fitness 1.087 down to between 1.051 and 1.058). On held-out instances the differences are within about 1 % in both directions, and the seeded run, which was allowed to specialise at width 50 on fewer instances, is 3 to 6 % worse. With eight held-out instances a 1 % difference is not significant. I read this as: the feature set already contains the right quantity, a formula search finds it, and there is little left to gain by recombining these features. The improvement that mattered, replacing a date by a curve in the state, is a change of representation that lies outside a search over formulas.</p>

<h2><span class="sn">9</span>Verification</h2>
<ul>
<li>The cost model is tied to [1] by Table 1.</li>
<li>The scheduler (5) was checked against brute-force enumeration of all orders with the exact solver (4) on four instances, and the identity (6) holds to $10^{{-6}}$ at every position.</li>
<li>All compiled routines were run once with array-bounds checking enabled.</li>
<li>Every tour in Table 4 was re-evaluated by a separate script that shares no code with the solver: it reads the instance file, propagates the bodies, solves each Lambert arc again, propagates the spacecraft along the arc and measures the distance to the target at arrival ("worst arc miss"), checks constraints (1) and the visit-once rule, and sums the impulses. Totals agree with the solver to $10^{{-6}}$ m/s. The checker uses the same Lambert library (pykep), so an error inside that library would not be caught.</li>
</ul>

<h2><span class="sn">10</span>Limits</h2>
<ul>
<li><strong>Upper bounds only.</strong> No optimality proof; the lower bounds are those of [1].</li>
<li><strong>Single visits.</strong> The formulation of [1] allows a body to be visited more than once. All its published optima for $n \\le 10$ are reproduced by single-visit tours, but for larger instances a cheaper tour with revisits could exist.</li>
<li><strong>Compute is not comparable.</strong> The values of [1] come from a mixed-integer solver with a 4-hour limit on a 2 &times; 64-core server and include bounds; mine from heuristics on two cores, minutes to 75 minutes, without bounds.</li>
<li><strong>One run per instance.</strong> The long runs were not repeated with different seeds, so no variance is reported.</li>
<li><strong>Benchmark model.</strong> Two-body dynamics, impulsive legs, grid dates. Nothing here addresses low thrust.</li>
<li><strong>Rule discovery.</strong> Genetic programming over fixed features, not a language model; small training and test sets.</li>
<li><strong>Memory.</strong> The dense tensor $C$ grows as $n^2K^2$; beyond a few hundred bodies it would have to be computed lazily or sparsified.</li>
</ul>

<h2><span class="sn">11</span>Relation to earlier work</h2>
<p>Carrying time-dependent cost information along a partial route is a known idea in time-dependent routing; dynamic programming with labels for the time-dependent TSP with time windows [7] is one example, and I do not claim the idea as new. TSP formulations of multi-target space missions go back to at least [2], and the hybrid of beam search and population-based ant colony optimisation for multi-rendezvous trajectories is from [3]. The KTSP benchmark, its exact methods and the B-Swan heuristic are from [1]. What this work adds is the combination applied to that benchmark, a verified independent implementation of its cost model, improved upper bounds on four open instances, and a first, negative, data point on automatic discovery of a search component for it.</p>

<h2><span class="sn">12</span>Reproducibility</h2>
<p>Code, result files and tours are in the <a href="https://github.com/taurasnienius/taurasnienius.github.io/tree/main/keplerian-tsp/code">repository</a> (Python, numpy, numba, pykep 3). The new tours are provided as JSON and in ACT's <code>.sol</code> format in <a href="sol/">sol/</a>. Instance files: doi:10.5281/zenodo.14850862.</p>

<h2><span class="sn">13</span>References</h2>
<p style="font-size:13.5px;color:var(--muted)">I worked from [1] in full. References [2] to [7] are cited for context and credit: I rely on their abstracts, not on a detailed reading, and I make no claim about their results beyond what is stated above.</p>
<ol class="refs">
<li>M. Bannach, G. Acciarini, D. Izzo, <i>The Keplerian Traveling Salesperson Problem</i>, arXiv:2605.00010, 2026.</li>
<li>D. Izzo, I. Getzner, D. Hennes, L. F. Simões, <i>Evolving solutions to TSP variants for active space debris removal</i>, GECCO 2015.</li>
<li>L. F. Simões, D. Izzo, E. Haasdijk, A. E. Eiben, <i>Multi-rendezvous spacecraft trajectory optimization with beam P-ACO</i>, EvoCOP 2017.</li>
<li>M. Schmidt, H. Lipson, <i>Distilling free-form natural laws from experimental data</i>, Science 324, 2009.</li>
<li>F. Liu et al., <i>Evolution of Heuristics: towards efficient automatic algorithm design using large language model</i>, ICML 2024.</li>
<li>H. Ye et al., <i>ReEvo: large language models as hyper-heuristics with reflective evolution</i>, NeurIPS 2024.</li>
<li>G. Lera-Romero, J. J. Miranda-Bront, F. J. Soulignac, <i>Dynamic programming for the time-dependent traveling salesman problem with time windows</i>, INFORMS Journal on Computing 34(6), 3292&ndash;3308, 2022.</li>
</ol>
</main>
<footer><a href="./">A cheaper way around</a> · <a href="../">Tauras Nienius</a> · <a href="mailto:taurasn.nie@gmail.com">taurasn.nie@gmail.com</a> · November 2026</footer>
</div>
</body>
</html>
'''
open('site/paper.html', 'w').write(HTML)
print('paper built', len(HTML) // 1024, 'kB', 'maxdiff', maxdiff, 'cross', cross)
print({c: (round(gmean(agg[c]), 3), round(gmean(aggl[c]), 3)) for c in cols})
print([(o, round(bp.gm(50, n), 3), round(bp.gm(200, n), 3)) for o, n in zip(origin, names)])
