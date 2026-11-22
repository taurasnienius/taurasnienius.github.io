"""Shape and departure size of an O2 bubble on an anode in molten CaCl2, from the Young-Laplace equation.

Axisymmetric gas bubble sitting on an upward-facing electrode, contact line pinned on a circle of radius r0
(a nucleation site / surface defect). Coordinates scaled by the apex radius b; s = arc length from the apex,
phi = angle of the tangent, z measured downward from the apex towards the electrode. Pressure balance:

    dx/ds = cos(phi),  dz/ds = sin(phi),  dphi/ds = 2 - beta*z - sin(phi)/x,   beta = drho g b^2 / sigma

(the same equation as a pendant drop). The profile is integrated past the equator until it necks back to
x = r0: that is where it meets the electrode. For a given g, the bubble volume as a function of b has a maximum:
no static shape exists for a larger volume, so this is the departure volume. At g = 0 there is no maximum:
the bubble keeps growing on the electrode until it merges with its neighbours.

Melt: sigma = 0.1429 N/m, rho = 2009 kg/m3 (CaCl2 at 1223 K, Janz / NSRDS-NBS 15).
"""
import numpy as np, json
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

SIGMA, DRHO = 0.1429, 2009.0 - 0.3
G = {"Earth": 9.80665, "Mars": 3.71, "Moon": 1.62}


def profile(b, g, r0, smax=12.0):
    """Dimensional profile (x, z) in metres from apex to the pinned contact line, and its volume."""
    beta = DRHO * g * b * b / SIGMA
    rr0 = r0 / b

    def f(s, y):
        x, z, p = y
        sx = np.sin(p) / x if x > 1e-9 else 1.0  # sin(phi)/x -> 1 at the apex
        return [np.cos(p), np.sin(p), 2 - beta * z - sx]

    def hit(s, y):  # contact line: back at x = r0 after passing the equator
        return y[0] - rr0 if y[2] > np.pi / 2 else 1.0
    hit.terminal, hit.direction = True, -1

    s0 = 1e-6
    sol = solve_ivp(f, (s0, smax), [s0, s0 * s0 / 2, s0], events=hit, max_step=0.01, rtol=1e-8, atol=1e-10)
    if sol.status != 1:
        return None
    x, z = sol.y[0] * b, sol.y[1] * b
    V = np.trapezoid(np.pi * x ** 2, z)
    return x, z, V


def departure(g, r0):
    """Largest static volume over apex radii (the departure bubble)."""
    lc = np.sqrt(SIGMA / (DRHO * g))
    bs = np.linspace(0.05 * lc, 3.0 * lc, 220)
    Vs = []
    for b in bs:
        p = profile(b, g, r0)
        Vs.append(p[2] if p else np.nan)
    Vs = np.array(Vs)
    i = np.nanargmax(Vs)
    res = minimize_scalar(lambda b: -(profile(b, g, r0) or (0, 0, 0))[2],
                          bounds=(bs[max(i - 2, 0)], bs[min(i + 2, len(bs) - 1)]), method="bounded")
    x, z, V = profile(res.x, g, r0)
    return dict(b=res.x, x=x, z=z, V=V, d_eq=(6 * V / np.pi) ** (1 / 3), height=z[-1], width=2 * x.max(), lc=lc)


if __name__ == "__main__":
    out = {}
    for r0 in (50e-6, 100e-6, 200e-6):
        line = []
        for k, g in G.items():
            d = departure(g, r0)
            line.append(f"{k}: d_eq {d['d_eq']*1e3:5.2f} mm (h {d['height']*1e3:4.2f} x w {d['width']*1e3:4.2f}, V {d['V']*1e9:6.2f} mm3)")
            if r0 == 100e-6:
                # resample the profile to ~60 points for the 3D lathe
                n = len(d["x"]); idx = np.linspace(0, n - 1, 60).astype(int)
                out[k] = dict(d_eq_mm=d["d_eq"] * 1e3, V_mm3=d["V"] * 1e9, h_mm=d["height"] * 1e3, w_mm=d["width"] * 1e3,
                              x_mm=[round(float(v) * 1e3, 4) for v in d["x"][idx]], z_mm=[round(float(v) * 1e3, 4) for v in d["z"][idx]])
        print(f"r0 = {r0*1e6:.0f} um | " + " | ".join(line))
    E, Ma, Mo = out["Earth"]["d_eq_mm"], out["Mars"]["d_eq_mm"], out["Moon"]["d_eq_mm"]
    print(f"ratios (r0 100 um): Mars {Ma/E:.2f}, Moon {Mo/E:.2f}; volume Moon/Earth {out['Moon']['V_mm3']/out['Earth']['V_mm3']:.2f}")
    # departure size vs g for the plot
    gs = np.logspace(np.log10(0.05), np.log10(20), 25) * 9.80665
    out["sweep"] = dict(g_rel=[float(g / 9.80665) for g in gs], d_eq_mm=[float(departure(g, 100e-6)["d_eq"] * 1e3) for g in gs])
    json.dump(out, open("data/bubble_shapes.json", "w"))
    print("saved")
