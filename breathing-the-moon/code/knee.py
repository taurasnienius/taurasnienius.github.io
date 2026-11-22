"""Is the ~60 % knee in Lomax et al. 2025 chemistry or geometry?

Null model (no chemistry): a reduction front moving at constant speed v from all surfaces of the
46 x 10 mm disc (Meurisse 2022 geometry). X(t) = 1 - (R - vt)^2 (H - 2vt) / (R^2 H).
Fit v to the cumulative curve (Fig. 2), then compare its RATE with the measured hourly rates (Fig. 3).
"""
import json, numpy as np
from scipy.optimize import minimize_scalar

R, H = 23.0, 10.0  # mm
d = json.load(open('data/lomax2025_fig2.json')); t = np.array(d['t_h']); X = np.array(d['X_pct']) / 100
rates = json.load(open('data/lomax2025_fig3_rates.json'))['blue']
tr = np.array([p[0] for p in rates]); rr = np.array([p[1] for p in rates])
O_BASIS_G = 24 * 0.445  # oxygen basis implied by Fig. 2 vs integrated Fig. 3 (9.54 g at 89.3 %)


def Xgeo(t, v):
    a = np.clip(R - v * t, 0, None); b = np.clip(H - 2 * v * t, 0, None)
    return 1 - a ** 2 * b / (R ** 2 * H)


res = minimize_scalar(lambda v: np.sum((Xgeo(t, v) - X) ** 2), bounds=(0.01, 1), method='bounded')
v = res.x; rms = np.sqrt(np.mean((Xgeo(t, v) - X) ** 2))
tt = np.arange(0.5, 30.5, 1.0)
rate_geo = (Xgeo(tt + 0.5, v) - Xgeo(tt - 0.5, v)) * O_BASIS_G  # g O per hour
print(f"geometric null model: v = {v:.3f} mm/h ({v/3.6*1e3:.0f} nm/s), cumulative rms = {rms*100:.1f} %")
print("hour  measured g/h  null g/h")
for h in (1, 3, 5, 10, 15, 17, 18, 20, 24, 30):
    m = np.interp(h, tr, rr); g = np.interp(h, tt + 0.5, rate_geo)
    print(f"{h:4d}  {m:8.3f}      {g:6.3f}")
# shape statistics: does the measured rate step? largest one-hour drop vs null
md = np.diff(np.interp(np.arange(1, 31), tr, rr)); gd = np.diff(np.interp(np.arange(1, 31), tt + 0.5, rate_geo))
print(f"largest 1-h drop: measured {md.min():.3f} g/h at {np.argmin(md)+1}->{np.argmin(md)+2} h; null model {gd.min():.3f} g/h")
json.dump({'v_mm_h': float(v), 'rms_pct': float(rms * 100), 't_h': list(range(1, 31)),
           'rate_measured': [float(np.interp(h, tr, rr)) for h in range(1, 31)],
           'rate_null': [float(np.interp(h, tt + 0.5, rate_geo)) for h in range(1, 31)],
           'X_t': [float(x) for x in t], 'X_measured': [float(x) for x in X * 100], 'X_null': [float(x) for x in Xgeo(t, v) * 100]}, open('data/knee.json', 'w'))
