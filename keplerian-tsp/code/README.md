# Keep the clock open: code

Code behind the [project page](https://taurasnienius.github.io/keplerian-tsp/). Python 3, numpy, numba, pykep.

```
pip install numpy numba pykep matplotlib
```

Put ACT's instance files (`0_instances/*.ktsp`, from doi:10.5281/zenodo.14850862) where `costs.DATA` points, then:

```
python build_all.py                 # transfer-cost tables for every instance and grid (about 15 min on 2 cores)
python -c "from costs import build; from exact import exact; print(exact(build('asteroids_10', 161)))"   # 24718.40
python compare.py gp_seed1.pkl      # all methods on the 13 harder instances
python attack.py asteroids_80 161 40   # long iterated local search, keeps the best tour in best/
python verify.py best/*.json        # independent check of saved tours
```

| File | What it does |
|---|---|
| `ktsp_io.py` | Reads `.ktsp` files, propagates the bodies to the grid (two-body). |
| `costs.py` | Cost tables `C[i, k, j, l]`: cheapest Lambert leg from body i at grid time k to body j at time l (all revolutions, both directions, departure + arrival impulse). |
| `calib.py` | The brute-force check used to pin down the cost model on the smallest instances. |
| `exact.py` | Exact dynamic programme over (visited set, body, time) for up to about 12 bodies. |
| `core.py` | Exact optimal dates for a fixed order (min-plus dynamic programme), segment re-evaluation, local search (relocate, swap, reversal) in best- and first-improvement versions, perturbation. |
| `ils.py`, `attack.py` | Iterated local search; long runs on the open instances. |
| `beam.py` | Beam search that keeps the timing open (availability curves), with pluggable ranking rules and the features they see. |
| `baselines.py` | Population-based ant colony optimisation and MCTS on the same evaluator. |
| `gp.py`, `run_gp.py`, `run_gp2.py`, `test_rules.py` | Genetic programming over ranking rules: training, and testing on held-out instances. |
| `compare.py` | All methods on the 13 instances with 20 or more bodies. |
| `verify.py` | Independent checker: re-solves every leg from the instance file, propagates each arc, checks the rules. Uses no solver code. |
| `act_ref.py` | Reference values from Tables 3 and 4 of Bannach, Acciarini and Izzo (2026). |
| `figs.py`, `build_page.py` | Figures and the page, built from the result files. |

`results/` holds the result files the page is built from.

Licence: MIT (repository root).
