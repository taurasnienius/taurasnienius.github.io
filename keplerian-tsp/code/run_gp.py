"""Discovery run: evolve beam ranking rules on small training instances only."""
import sys, json, pickle
from gp import run
from act_ref import VLB_VUB
TRAIN = [('asteroids_10', 81), ('asteroids_20', 41), ('asteroids_20', 81), ('jovianmoons_10', 81), ('jovianmoons_20', 41), ('jovianmoons_20', 81)]
train = [(n, t, VLB_VUB[(n, t)][1]) for n, t in TRAIN]
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
best, hist = run(train, pop=int(sys.argv[3]) if len(sys.argv) > 3 else 200, gens=int(sys.argv[2]) if len(sys.argv) > 2 else 25, W=20, seed=seed, procs=2,
                 log=f'gp_seed{seed}.jsonl')
pickle.dump(dict(best=best, hist=hist), open(f'gp_seed{seed}.pkl', 'wb'))
