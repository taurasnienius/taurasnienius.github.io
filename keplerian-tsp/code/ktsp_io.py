"""Read ACT .ktsp instances and propagate bodies (two-body, from Cartesian state at t0)."""
import numpy as np, pykep as pk

DAY = 86400.0

def read_ktsp(path):
    rows, t0 = [], None
    for line in open(path):
        s = line.split()
        if not s or s[0] == 'c':
            continue
        if s[0] == 'p':
            t0, tf, mu = float(s[2]), float(s[3]), float(s[4])
            continue
        rows.append([float(x) for x in s])
    return dict(t0=t0, tf=tf, mu=mu, rv=np.array(rows))

def states_on_grid(inst, nT):
    """r, v of every body at every grid time: arrays (n, nT, 3)."""
    t = np.linspace(inst['t0'], inst['tf'], nT)
    n = len(inst['rv']); R = np.zeros((n, nT, 3)); V = np.zeros((n, nT, 3))
    for i in range(n):
        r0, v0 = inst['rv'][i, :3], inst['rv'][i, 3:]
        for k, tk in enumerate(t):
            r, v = pk.propagate_lagrangian([r0, v0], (tk - inst['t0']) * DAY, inst['mu'], False)
            R[i, k], V[i, k] = r, v
    return t, R, V
