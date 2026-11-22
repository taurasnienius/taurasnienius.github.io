"""Shared functions of the calibrated bubble chain (same code as sim_table.py)."""
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


