"""Module B: reduction kinetics of regolith in an FFC cell.

Shrinking-core conversion laws for a sphere of radius R:
  reaction (interface) control : t/tau = 1 - (1-X)^(1/3),                tau ~ R
  diffusion (product layer)    : t/tau = 1 - 3(1-X)^(2/3) + 2(1-X),      tau ~ R^2
Mixed control adds the two times.

Calibration data: Lomax et al. 2020, 75 % of oxygen at 15 h and 96 % at 50 h.
"""
import numpy as np
from scipy.optimize import brentq, fsolve

T_DATA = np.array([15.0, 50.0])
X_DATA = np.array([0.75, 0.96])

f_r = lambda X: 1 - (1 - X) ** (1 / 3)
f_d = lambda X: 1 - 3 * (1 - X) ** (2 / 3) + 2 * (1 - X)


def X_diff(tn):
    if tn >= 1:
        return 1.0
    if tn <= 0:
        return 0.0
    return brentq(lambda X: f_d(X) - tn, 0, 1 - 1e-15)


def X_react(tn):
    if tn >= 1:
        return 1.0
    return 1 - (1 - max(tn, 0)) ** 3


def fit_mixed_single_size(xmax=1.0):
    """Mixed control, one particle size. Two points fix tau_r and tau_d exactly."""
    A = np.array([[f_r(x / xmax), f_d(x / xmax)] for x in X_DATA])
    tau_r, tau_d = np.linalg.solve(A, T_DATA)
    return tau_r, tau_d


_z, _w = np.polynomial.hermite_e.hermegauss(80)
_w = _w / _w.sum()


def X_psd(t, tau50, sigma, mode):
    """Conversion of a log-normal size population (sigma = std of ln R)."""
    p, f = (2, X_diff) if mode == "diffusion" else (1, X_react)
    return float(sum(w * f(t / (tau50 * np.exp(p * sigma * z))) for z, w in zip(_z, _w)))


def fit_psd(mode):
    def eq(v):
        tau, s = np.exp(v)
        return [X_psd(T_DATA[0], tau, s, mode) - X_DATA[0], X_psd(T_DATA[1], tau, s, mode) - X_DATA[1]]

    sol, info, ier, msg = fsolve(eq, [np.log(25), np.log(0.5)], full_output=True)
    tau, s = np.exp(sol)
    return tau, s, ier == 1


def time_to(Xtarget, tau50, sigma, mode, size_factor=1.0, tmax=2000):
    """Hours to reach Xtarget when every particle is scaled by size_factor."""
    p = 2 if mode == "diffusion" else 1
    tau = tau50 * size_factor ** p
    g = lambda t: X_psd(t, tau, sigma, mode) - Xtarget
    if g(tmax) < 0:
        return np.inf
    return brentq(g, 1e-6, tmax)


def time_trimmed(Xtarget, tau50, sigma, mode, cut_sigma):
    """Hours to reach Xtarget if particles larger than exp(cut_sigma*sigma)*R50 are sieved out."""
    p, f = (2, X_diff) if mode == "diffusion" else (1, X_react)
    keep = _z <= cut_sigma
    w = _w[keep] / _w[keep].sum()
    zz = _z[keep]
    g = lambda t: sum(wi * f(t / (tau50 * np.exp(p * sigma * zi))) for zi, wi in zip(zz, w)) - Xtarget
    return brentq(g, 1e-6, 2000)


if __name__ == "__main__":
    print("Single size, mixed control:")
    for xm in (1.0, 0.98, 0.9666):
        tr, td = fit_mixed_single_size(xm)
        print(f"  Xmax={xm:.4f}: tau_r={tr:7.1f} h  tau_d={td:7.1f} h  -> {'physical' if tr >= 0 and td >= 0 else 'NOT physical'}")
    for mode in ("diffusion", "reaction"):
        tau, s, ok = fit_psd(mode)
        curve = [round(X_psd(t, tau, s, mode), 3) for t in (5, 10, 15, 24, 36, 50, 72)]
        t2 = time_to(0.96, tau, s, mode, 2.0)
        th = time_to(0.96, tau, s, mode, 0.5)
        tc = time_trimmed(0.96, tau, s, mode, 1.0)
        print(f"{mode:9s}: tau50={tau:5.1f} h sigma_lnR={s:.2f} ok={ok} curve(5..72h)={curve}")
        print(f"           t96 x2 size={t2:6.1f} h, x0.5 size={th:5.1f} h, coarsest 16% removed={tc:5.1f} h")
