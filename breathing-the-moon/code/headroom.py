"""How much does the cell current respond to an extra anode-side voltage at a fixed 3.2 V?

Rate loss at fixed voltage = dV / H, with H = I dV/dI at the operating point ("headroom").
The paper assumed H = 1.65 V (1.4 to 1.9). Three independent estimates:

 A. Thermodynamic chord: H = V - E, with E the decomposition voltage on a graphite anode at 1223 K
    (my ladder from thermo.py minus the carbon shift). Exact only for a purely ohmic cell.
 B. Electrolyte ohmic drop alone (coaxial estimate): a floor showing how little of the 3.2 V the salt takes.
 C. Natural experiment in Lomax et al. 2025, Fig. 3: the same cell at the same 3.2 V with a tin oxide anode
    (O2 evolving) instead of graphite (CO/CO2). Thermodynamically this removes 1.03 to 1.14 V of anode
    depolarisation, and the oxygen rate falls. The ratio of rates gives H over that step.
    Green points digitised from Fig. 3: 40 of 48 confirmed by pixel extraction to +-0.002 g/h; hours 23, 25, 27, 42,
    44, 45, 47 and 48 overlap other markers and were read by eye (about +-0.01 g/h).
"""
import numpy as np, json

V = 3.2
SHIFT = dict(CO2=1.026, CO=1.140)          # graphite depolarisation at 1223 K (thermo.py)
E_INERT = {"Fe (FeO)": 0.999, "Si (via CaSiO3)": 2.021, "Si (SiO2)": 1.792, "Ti (via CaTiO3)": 2.101,
           "Al (Al2O3)": 2.224, "Mg (MgO)": 2.421, "Ca (CaO)": 2.626}
print("A. Thermodynamic chord H = 3.2 - E(graphite):")
A = {}
for k, e in E_INERT.items():
    lo, hi = V - (e - SHIFT["CO2"]), V - (e - SHIFT["CO"])
    A[k] = (round(lo, 2), round(hi, 2)); print(f"   {k:18s} E = {e-SHIFT['CO']:.2f} to {e-SHIFT['CO2']:.2f} V   H = {lo:.2f} to {hi:.2f} V")

# B. electrolyte resistance, coaxial: rod radius 6.35 mm, counter-surface at 25 to 50 mm, immersion 3 to 6 cm
kappa, r1 = 278.0, 6.35e-3
B = []
for r2 in (0.025, 0.05):
    for L in (0.03, 0.06):
        R = np.log(r2 / r1) / (2 * np.pi * kappa * L)
        for I in (2.2, 3.4):
            B.append(I * R)
print(f"B. Electrolyte ohmic drop: {min(B)*1e3:.0f} to {max(B)*1e3:.0f} mV  (R = "
      f"{np.log(0.025/r1)/(2*np.pi*kappa*0.06)*1e3:.0f} to {np.log(0.05/r1)/(2*np.pi*kappa*0.03)*1e3:.0f} mOhm)")

# C. tin oxide vs graphite
blue = dict(json.load(open("data/lomax2025_fig3_rates.json"))["blue"])
green = {1: .005, 2: .018, 3: .043, 4: .107, 5: .106, 6: .062, 7: .048, 8: .160, 9: .134, 10: .243, 11: .299, 12: .323,
         13: .256, 14: .285, 15: .240, 16: .197, 17: .208, 18: .194, 19: .220, 20: .237, 21: .237, 22: .223, 23: .165,
         24: .198, 25: .217, 26: .209, 27: .204, 28: .146, 29: .191, 30: .246, 31: .300, 32: .217, 33: .286, 34: .266,
         35: .243, 36: .225, 37: .218, 38: .202, 39: .163, 40: .141, 41: .123, 42: .115, 43: .100, 44: .072, 45: .072,
         46: .067, 47: .064, 48: .056}
O = 24 * 0.445


def window_rate(d, lo, hi):
    """mean rate (g/h) while cumulative extraction is between lo and hi (fractions of O)."""
    ts = np.arange(1, max(d) + 1); r = np.interp(ts, sorted(d), [d[k] for k in sorted(d)])
    cum = np.cumsum(r) / O
    t_lo, t_hi = np.interp([lo, hi], cum, ts)
    return (hi - lo) * O / (t_hi - t_lo), t_lo, t_hi, cum[-1]


res = {}
print("C. Tin oxide against graphite, same cell, same 3.2 V:")
for name, (lo, hi) in {"20-60 % extracted": (0.2, 0.6), "30-50 % extracted": (0.3, 0.5)}.items():
    rb, *tb = window_rate(blue, lo, hi); rg, *tg = window_rate(green, lo, hi)
    res[name] = rg / rb
    print(f"   {name}: graphite {rb:.3f} g/h (h {tb[0]:.1f}-{tb[1]:.1f}), tin oxide {rg:.3f} g/h (h {tg[0]:.1f}-{tg[1]:.1f}), ratio {rg/rb:.2f}")
pk = max(green.values()) / max(blue.values()); res["peak hour"] = pk
print(f"   peak hourly rate: graphite {max(blue.values()):.3f}, tin oxide {max(green.values()):.3f}, ratio {pk:.2f}")
print(f"   tin oxide total integrated: {sum(green.values())/O*100:.0f} % of the oxygen basis in 48 h")
Hs = []
for name, r in res.items():
    for gas, dv in SHIFT.items():
        Hlin, Hexp = dv / (1 - r), dv / np.log(1 / r)
        Hs += [Hlin, Hexp]
        print(f"   {name:18s} {gas}: H = {Hexp:.2f} V (exponential I-V) to {Hlin:.2f} V (linear I-V)")
print(f"   => H between {min(Hs):.1f} and {max(Hs):.1f} V; every known bias (tin oxide adds resistance, and its "
      "current efficiency was lower) makes the true H larger")
print("Rate loss on the Moon for dV = 41 mV [29, 63]:")
for H in (1.4, 1.65, 1.9, 2.2, 2.4, 3.0, 4.0):
    print(f"   H = {H:.2f} V: {41/H/10:.1f} %  [{29/H/10:.1f}, {63/H/10:.1f}]")
json.dump(dict(chord=A, electrolyte_drop_mV=[round(min(B)*1e3), round(max(B)*1e3)], tin_ratio={k: round(v, 3) for k, v in res.items()},
               H_tin_range=[round(min(Hs), 2), round(max(Hs), 2)], green_digitised=green), open("data/headroom.json", "w"), indent=1)
