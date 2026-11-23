"""Discovery run v2: fitness at beam width 50 on a more varied training set, population seeded with the
hand-designed rule (as in LLM-based heuristic evolution, which starts from a seed heuristic)."""
import sys, json, pickle
from gp import run
from act_ref import VLB_VUB
TRAIN = [('asteroids_10', 161), ('asteroids_20', 81), ('asteroids_40', 41), ('jovianmoons_10', 161), ('jovianmoons_20', 81)]
train = [(n, t, VLB_VUB[(n, t)][1]) for n, t in TRAIN]
seed = int(sys.argv[1])
best, hist = run(train, pop=100, gens=int(sys.argv[2]), W=50, seed=seed, procs=1, seeds_in=[('f', 0)],
                 log=f'gp2_seed{seed}.jsonl')
pickle.dump(dict(best=best, hist=hist, train=TRAIN), open(f'gp2_seed{seed}.pkl', 'wb'))
