"""Module C: anode gas bubbles in molten CaCl2 versus gravity.

Chain (each link is a published relation; sources in research/bubbles.md):
 1. Departure diameter.  Fritz (spreading contact line):  d = 0.0208 theta sqrt(sigma / (g drho)),   d ~ g^-1/2
    Pinned contact line (Angulo et al. 2020, Eq. 10):        r = (3 r0 sigma sin(theta) / (2 drho g))^(1/3), d ~ g^-1/3
 2. Coverage at Earth gravity: Vogt & Balzer 2005, Theta_E = 0.023 j^0.3 (j in A/m2; aqueous-derived).
    Gravity scaling: the projected footprint per nucleation site ~ d^2 at fixed site density,
    combined with Poisson overlap:  Theta(g) = 1 - exp(-lambda_E * (d/d_E)^m),  lambda_E = -ln(1-Theta_E), m in [1, 2].
 3. Ohmic penalty of the bubble layer (Hyde & Welch 1997, used for molten-salt Al cells):
        dV_ohm = (j / kappa) * delta_b * (1/(1-Theta) - 1),   bubble-layer thickness delta_b ~ departure diameter.
 4. Activation penalty from the reduced free area (Angulo et al.):  dV_act = (R T / (alpha F)) ln(1/(1-Theta)).
Outputs are reported as the change relative to Earth gravity, which cancels part of the unknowns.
"""
import numpy as np

F, Rg = 96485.33, 8.314462
G = {"Earth": 9.80665, "Mars": 3.71, "Moon": 1.62}
T = 1223.0
MELT = dict(rho=2009.0, sigma=0.1429, kappa=278.0, mu=2.2e-3)   # CaCl2 at 1223 K (NSRDS-NBS 15, Janz 1975)
RHO_O2 = 0.315                                                   # ideal gas, 1223 K, 1 bar


def d_fritz(g, theta_deg, sigma=MELT["sigma"], drho=MELT["rho"] - RHO_O2):
    return 0.0208 * theta_deg * np.sqrt(sigma / (g * drho))


def d_pinned(g, theta_deg, r0, sigma=MELT["sigma"], drho=MELT["rho"] - RHO_O2):
    return 2 * (3 * r0 * sigma * np.sin(np.radians(theta_deg)) / (2 * drho * g)) ** (1 / 3)


WATER = dict(rho=1000.0, sigma=0.072, kappa=30.0, T=298.0)   # 0.75 M H2SO4, room temperature (Lomax 2022 cell)
SALT = dict(rho=MELT["rho"], sigma=MELT["sigma"], kappa=MELT["kappa"], T=T)


def penalty(g, j, theta_deg=120.0, mode="fritz", r0=50e-6, m=2.0, alpha=0.5, layer_factor=1.0, med=None):
    """Bubble-induced extra cell voltage [V] at gravity g and anode current density j [A/m2]."""
    med = med or SALT
    kw = dict(sigma=med["sigma"], drho=med["rho"])
    d = d_fritz(g, theta_deg, **kw) if mode == "fritz" else d_pinned(g, theta_deg, r0, **kw)
    dE = d_fritz(G["Earth"], theta_deg, **kw) if mode == "fritz" else d_pinned(G["Earth"], theta_deg, r0, **kw)
    thE = min(0.023 * j ** 0.3, 0.95)
    lam = -np.log(1 - thE) * (d / dE) ** m
    th = 1 - np.exp(-lam)
    delta_b = layer_factor * d
    dv_ohm = j / med["kappa"] * delta_b * (1 / (1 - th) - 1)
    dv_act = Rg * med["T"] / (alpha * F) * np.log(1 / (1 - th))
    return dict(d_mm=d * 1e3, coverage=th, dv_ohm=dv_ohm, dv_act=dv_act, dv=dv_ohm + dv_act)


def kwh_per_kg_O2(volts, ce=1.0):
    """Energy for a voltage V over 1 kg O2: 3.35 kWh per volt at 100 % current efficiency."""
    return volts * (1000 / 31.998 * 4 * F) / 3.6e6 / ce


def monte_carlo(n=20000, seed=2, V_cell=3.2):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        p = dict(j=np.exp(rng.uniform(np.log(500), np.log(5000))),        # 0.05-0.5 A/cm2 (FFC anodes)
                 theta_deg=rng.uniform(105, 130),                         # cryolite/carbon at detachment
                 mode="fritz" if rng.random() < 0.5 else "pinned",
                 r0=np.exp(rng.uniform(np.log(10e-6), np.log(200e-6))),
                 m=rng.uniform(1, 2), alpha=rng.uniform(0.5, 1.0), layer_factor=rng.uniform(0.7, 1.5))
        e = penalty(G["Earth"], **p)
        res = {k: penalty(gv, **p) for k, gv in G.items()}
        rows.append(dict(p, **{f"dV_{k}": res[k]["dv"] - e["dv"] for k in G},
                         **{f"d_{k}": res[k]["d_mm"] for k in G},
                         **{f"cov_{k}": res[k]["coverage"] for k in G}))
    return rows


if __name__ == "__main__":
    print("Nominal (Fritz, theta 120 deg, j = 0.2 A/cm2, m = 2, alpha 0.5):")
    for k, gv in G.items():
        r = penalty(gv, 2000)
        print(f"  {k:5s} d = {r['d_mm']:5.1f} mm  coverage = {r['coverage']:.2f}  dV_ohm = {r['dv_ohm']*1e3:5.0f} mV  dV_act = {r['dv_act']*1e3:5.0f} mV  total = {r['dv']*1e3:5.0f} mV")
    rows = monte_carlo(8000)
    for k in ("Mars", "Moon"):
        v = np.array([r[f"dV_{k}"] for r in rows])
        print(f"\nExtra voltage on {k} vs Earth: median {np.median(v)*1e3:.0f} mV, 90% [{np.percentile(v,5)*1e3:.0f}, {np.percentile(v,95)*1e3:.0f}] mV"
              f"  = {np.median(v)/3.2*100:.1f}% of a 3.2 V cell [{np.percentile(v,5)/3.2*100:.1f}, {np.percentile(v,95)/3.2*100:.1f}]")
    # which input drives the lunar penalty? Spearman rank correlation
    from scipy.stats import spearmanr
    v = np.array([r["dV_Moon"] for r in rows])
    for key in ("j", "theta_deg", "r0", "m", "alpha", "layer_factor"):
        x = np.array([r[key] for r in rows])
        print(f"  rank corr with {key:12s}: {spearmanr(x, v).correlation:+.2f}")
    fr = np.array([r["mode"] == "fritz" for r in rows])
    print(f"  median by detachment law: fritz {np.median(v[fr])*1e3:.0f} mV, pinned {np.median(v[~fr])*1e3:.0f} mV")
