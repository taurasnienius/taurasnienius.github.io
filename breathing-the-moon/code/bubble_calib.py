"""Calibrate the bubble-mechanism parameters on Lomax et al. 2022 (water, centrifuge + parabolic flight),
then transfer them to molten CaCl2.

Measured (Lomax 2022, Fig. 7, galvanostatic fits over 0.166-8 g, % change = -a ln(g) + c):
a = 0.410, 0.784, 0.594 % at 50, 75, 100 mA/cm2 with cell potentials ~2.06, 2.13, 2.19 V (Fig. 3),
so lunar minus 1 g = a ln(1/0.166) % of V = 15.2, 30.0, 23.4 mV.
Shared 'mechanism' parameters: detachment law (Fritz vs pinned), pinning radius r0, coverage exponent m,
transfer coefficient alpha, bubble-layer factor. Medium-specific: contact angle, sigma, rho, kappa, T, current density.
Approximate Bayesian weighting: log-normal error, sigma_ln = 0.4 per data point.
"""
import numpy as np
from bubbles import penalty, WATER, SALT, G

MEAS = {500: 0.0152, 750: 0.0300, 1000: 0.0234}  # V, lunar minus Earth


def run(n=40000, seed=7, s_ln=0.4, j_salt=(500, 5000)):
    rng = np.random.default_rng(seed)
    W, salt_moon, salt_mars, rows = [], [], [], []
    for _ in range(n):
        p = dict(mode="fritz" if rng.random() < 0.5 else "pinned", r0=np.exp(rng.uniform(np.log(10e-6), np.log(200e-6))),
                 m=rng.uniform(1, 2), alpha=rng.uniform(0.5, 1.0), layer_factor=rng.uniform(0.7, 1.5))
        thw = rng.uniform(30, 50)
        ll = 0.0
        for j, b in MEAS.items():
            dv = penalty(9.80665 * 0.166, j, theta_deg=thw, med=WATER, **p)["dv"] - penalty(G["Earth"], j, theta_deg=thw, med=WATER, **p)["dv"]
            ll += -(np.log(max(dv, 1e-6) / b)) ** 2 / (2 * s_ln ** 2)
        ths = rng.uniform(105, 130); js = np.exp(rng.uniform(np.log(j_salt[0]), np.log(j_salt[1])))
        e = penalty(G["Earth"], js, theta_deg=ths, **p)["dv"]
        salt_moon.append(penalty(G["Moon"], js, theta_deg=ths, **p)["dv"] - e)
        salt_mars.append(penalty(G["Mars"], js, theta_deg=ths, **p)["dv"] - e)
        W.append(ll); rows.append(dict(p, js=js))
    W = np.exp(np.array(W) - max(W)); W /= W.sum()
    return np.array(salt_moon), np.array(salt_mars), W, rows


def wq(x, w, q):
    i = np.argsort(x); c = np.cumsum(w[i]); return np.interp(q, c, x[i])


if __name__ == "__main__":
    moon, mars, w, rows = run()
    ess = 1 / np.sum(w ** 2)
    print(f"effective sample size after calibration: {ess:.0f} of {len(w)}")
    pf = sum(wi for wi, r in zip(w, rows) if r["mode"] == "fritz")
    print(f"posterior P(Fritz detachment) = {pf:.2f} (prior 0.50)")
    print("posterior mean m = %.2f (prior 1.5)" % sum(wi * r["m"] for wi, r in zip(w, rows)))
    for name, v in (("Moon", moon), ("Mars", mars)):
        prior = np.percentile(v, [50, 5, 95]) * 1e3
        post = [wq(v, w, q) * 1e3 for q in (0.5, 0.05, 0.95)]
        print(f"CaCl2 extra voltage on {name}: prior median {prior[0]:.0f} mV [{prior[1]:.0f}, {prior[2]:.0f}] -> "
              f"calibrated median {post[0]:.0f} mV [{post[1]:.0f}, {post[2]:.0f}]  = {post[0]/3200*100:.1f}% of 3.2 V [{post[1]/3200*100:.1f}, {post[2]/3200*100:.1f}]")
    # at the estimated Lomax 2025 cell current density ~1400 A/m2
    sel = np.array([abs(np.log(r["js"] / 1400)) < 0.25 for r in rows])
    print("at j ~ 0.14 A/cm2 (estimated ESA lab cell): calibrated median %.0f mV" % (wq(moon[sel], w[sel] / w[sel].sum(), 0.5) * 1e3))
