"""Growth sequence of static bubble shapes (increasing volume up to departure) for the 3D page."""
import json, numpy as np
from bubble_shape import profile, departure, G

R0 = 100e-6
out = {}
for k, g in G.items():
    dep = departure(g, R0)
    seq = []
    for f in np.linspace(0.12, 1.0, 14):
        b = dep["b"] * f
        # find b with volume fraction: walk b from small to b*, volumes increase monotonically before the maximum
        p = profile(b, g, R0)
        if not p:
            continue
        x, z, V = p
        n = len(x); idx = np.linspace(0, n - 1, 48).astype(int)
        h = z[-1]
        seq.append(dict(V=float(V * 1e9), x=[round(float(v) * 1e3, 4) for v in x[idx]], y=[round(float(h - v) * 1e3, 4) for v in z[idx]]))
    out[k] = dict(d_eq_mm=float(dep["d_eq"] * 1e3), V_mm3=float(dep["V"] * 1e9), h_mm=float(dep["height"] * 1e3),
                  w_mm=float(dep["width"] * 1e3), seq=seq)
    print(k, len(seq), [round(s["V"], 2) for s in seq][-4:], round(out[k]["V_mm3"], 2))
json.dump(out, open("data/bubble_growth.json", "w"), separators=(",", ":"))
