"""Prior sensitivity of the calibrated gravity penalty.

The baseline calibration (sim_table.py) draws the mechanism parameters from bounded priors and weights them
by agreement with the Lomax et al. 2022 water data at lunar g. Two posteriors sit near a prior bound
(r0 median 23 um against a 10 um bound; m mean 1.27 against a bound of 1). This script widens each prior in
turn, and all together, and reports how far the molten-salt prediction at the Moon (j = 0.14 A/cm2) moves.
"""
import numpy as np, json
from sim_table_core import chain, wq, G0, SALT, WATER, MEAS

BASE = dict(r0=(10e-6, 200e-6), m=(1, 2), alpha=(0.5, 1.0), lf=(0.7, 1.5), thw=(30, 50), sig=0.4)
CASES = [
    ("baseline", {}),
    ("r0 1 to 500 um", dict(r0=(1e-6, 500e-6))),
    ("r0 0.1 um to 1 mm", dict(r0=(0.1e-6, 1000e-6))),
    ("m 0.5 to 3", dict(m=(0.5, 3))),
    ("m 0 to 4", dict(m=(0.0, 4))),
    ("alpha 0.25 to 2", dict(alpha=(0.25, 2.0))),
    ("f_L 0.3 to 3", dict(lf=(0.3, 3.0))),
    ("water angle 20 to 70 deg", dict(thw=(20, 70))),
    ("sigma_ln 0.25", dict(sig=0.25)),
    ("sigma_ln 0.6", dict(sig=0.6)),
    ("all widened", dict(r0=(1e-6, 500e-6), m=(0.5, 3), alpha=(0.25, 2.0), lf=(0.3, 3.0), thw=(20, 70))),
    ("all widened, sigma_ln 0.6", dict(r0=(1e-6, 500e-6), m=(0.5, 3), alpha=(0.25, 2.0), lf=(0.3, 3.0), thw=(20, 70), sig=0.6)),
]


def run(over, N=400000, seed=11):
    c = dict(BASE); c.update(over)
    rng = np.random.default_rng(seed)
    P = dict(fritz=rng.random(N) < 0.5, r0=np.exp(rng.uniform(np.log(c["r0"][0]), np.log(c["r0"][1]), N)),
             m=rng.uniform(*c["m"], N), alpha=rng.uniform(*c["alpha"], N), lf=rng.uniform(*c["lf"], N))
    thw = rng.uniform(*c["thw"], N)
    ll = np.zeros(N); pw = {}
    for j, b in MEAS.items():
        dv = chain(0.166 * G0, j, P, thw, WATER)[0] - chain(G0, j, P, thw, WATER)[0]
        pw[j] = dv
        ll += -(np.log(np.maximum(dv, 1e-6) / b)) ** 2 / (2 * c["sig"] ** 2)
    w = np.exp(ll - ll.max()); w /= w.sum()
    ths = rng.uniform(105, 130, N)
    out = {}
    for name, gv in (("moon", 0.1652), ("mars", 0.3783)):
        dv, th = chain(gv * G0, 1400, P, ths, SALT); e, thE = chain(G0, 1400, P, ths, SALT)
        x = (dv - e) * 1e3
        out[name] = [round(wq(x, w, q), 1) for q in (.5, .05, .95)]
        out[name + "_cov"] = round(wq(th, w, .5), 3)
    out.update(ess=int(1 / np.sum(w ** 2)), p_fritz=float(w[P["fritz"]].sum()),
               m=round(float(np.sum(w * P["m"])), 2), alpha=round(float(np.sum(w * P["alpha"])), 2),
               lf=round(float(np.sum(w * P["lf"])), 2), r0_um=round(wq(P["r0"], w, .5) * 1e6, 1),
               water=[round(wq(pw[j], w, .5) * 1e3, 1) for j in MEAS])
    return out


if __name__ == "__main__":
    res = []
    for name, over in CASES:
        o = run(over); o["case"] = name; res.append(o)
        print(f"{name:28s} Moon {o['moon'][0]:5.1f} [{o['moon'][1]:5.1f},{o['moon'][2]:5.1f}]  Mars {o['mars'][0]:4.1f}  cov {o['moon_cov']:.2f}"
              f"  m {o['m']:.2f} a {o['alpha']:.2f} fL {o['lf']:.2f} r0 {o['r0_um']:6.1f} um  ESS {o['ess']:6d}  P(Fritz) {o['p_fritz']:.1e}  water {o['water']}")
    json.dump(res, open("data/prior_sensitivity.json", "w"), indent=1)
