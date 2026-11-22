"""Tables for the in-page electrolysis simulator, from the calibrated bubble chain (vectorised).

1. Draw mechanism parameters (detachment law, pinning radius r0, coverage exponent m, transfer coefficient
   alpha, bubble-layer factor) and weight them by agreement with Lomax et al. 2022 water data at lunar g
   (Fig. 7 fits: +15.2, +30.0, +23.4 mV at 50, 75, 100 mA/cm2), as in bubble_calib.py.
2. For molten CaCl2 (contact angle 105-130 deg), compute the extra voltage relative to Earth and the anode
   coverage on a grid of gravity (0.1-1 g) and anode current density (0.05-0.5 A/cm2): weighted median, 5 %, 95 %.
3. Calibration check: posterior predictive for the water points.
"""
import numpy as np, json

F, R = 96485.33, 8.314462
G0 = 9.80665
SALT = dict(rho=2009.0, sigma=0.1429, kappa=278.0, T=1223.0)
WATER = dict(rho=1000.0, sigma=0.072, kappa=30.0, T=298.0)
MEAS = {500: 0.0152, 750: 0.0300, 1000: 0.0234}


def d_dep(g, theta, mode, r0, med):
    fr = 0.0208 * theta * np.sqrt(med["sigma"] / (g * med["rho"]))
    pin = 2 * (3 * r0 * med["sigma"] * np.sin(np.radians(theta)) / (2 * med["rho"] * g)) ** (1 / 3)
    return np.where(mode, fr, pin)


def chain(g, j, P, theta, med):
    d = d_dep(g, theta, P["fritz"], P["r0"], med)
    dE = d_dep(G0, theta, P["fritz"], P["r0"], med)
    thE = np.minimum(0.023 * j ** 0.3, 0.95)
    th = 1 - np.exp(np.log(1 / (1 - thE)) * -(d / dE) ** P["m"])
    dv = j / med["kappa"] * P["lf"] * d * (1 / (1 - th) - 1) + R * med["T"] / (P["alpha"] * F) * np.log(1 / (1 - th))
    return dv, th


def wq(x, w, q):
    i = np.argsort(x); c = np.cumsum(w[i]); c /= c[-1]
    return float(np.interp(q, c, x[i]))


rng = np.random.default_rng(11)
N = 60000
P = dict(fritz=rng.random(N) < 0.5, r0=np.exp(rng.uniform(np.log(10e-6), np.log(200e-6), N)),
         m=rng.uniform(1, 2, N), alpha=rng.uniform(0.5, 1.0, N), lf=rng.uniform(0.7, 1.5, N))
thw = rng.uniform(30, 50, N)
ll = np.zeros(N)
pred_w = {}
for j, b in MEAS.items():
    dv = chain(0.166 * G0, j, P, thw, WATER)[0] - chain(G0, j, P, thw, WATER)[0]
    pred_w[j] = dv
    ll += -(np.log(np.maximum(dv, 1e-6) / b)) ** 2 / (2 * 0.4 ** 2)
w = np.exp(ll - ll.max()); w /= w.sum()
print(f"ESS {1/np.sum(w**2):.0f}; P(Fritz) {w[P['fritz']].sum():.3f}")
print("posterior means:", {k: round(float(np.sum(w * P[k])), 4) for k in ("r0", "m", "alpha", "lf")})

calib = [dict(j_mA=j / 10, measured_mV=b * 1e3, pred_med=wq(pred_w[j], w, .5) * 1e3, pred_lo=wq(pred_w[j], w, .05) * 1e3,
              pred_hi=wq(pred_w[j], w, .95) * 1e3, prior_med=float(np.median(pred_w[j])) * 1e3) for j, b in MEAS.items()]
for c in calib:
    print(c)

ths = rng.uniform(105, 130, N)
g_grid = np.round(np.logspace(np.log10(0.1), 0, 13), 4)
j_grid = [500, 800, 1400, 2500, 5000]
tab = {"g": [float(x) for x in g_grid], "j": j_grid, "med": [], "lo": [], "hi": [], "cov": [], "covE": []}
for jj in j_grid:
    e, thE = chain(G0, jj, P, ths, SALT)
    rm, rl, rh, rc = [], [], [], []
    for gg in g_grid:
        dv, th = chain(gg * G0, jj, P, ths, SALT)
        x = dv - e
        rm.append(round(wq(x, w, .5) * 1e3, 2)); rl.append(round(wq(x, w, .05) * 1e3, 2)); rh.append(round(wq(x, w, .95) * 1e3, 2))
        rc.append(round(wq(th, w, .5), 3))
    tab["med"].append(rm); tab["lo"].append(rl); tab["hi"].append(rh); tab["cov"].append(rc); tab["covE"].append(round(wq(thE, w, .5), 3))
i14 = j_grid.index(1400)
print("j=0.14 A/cm2: Moon", tab["med"][i14][list(g_grid).index(min(g_grid, key=lambda v: abs(v - 0.165)))], "mV")
json.dump(dict(table=tab, calib=calib), open("data/sim_table.json", "w"))
print("saved")
