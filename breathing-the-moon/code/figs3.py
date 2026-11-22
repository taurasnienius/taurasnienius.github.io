"""Figure set for the site, one consistent style: deep warm-dark ground, pastel series, no frame, soft grid."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.ticker
from matplotlib.patches import Circle
from bubble_shape import profile, G

OUT = "figures/"
import os; os.makedirs(OUT, exist_ok=True)
BG, INK, MUTED, GRID = "#17120f", "#efe9e2", "#a39a91", "#2a221e"
E, M, L = "#7cc6e8", "#f2836b", "#e9e1d3"
PEACH, LILAC, TEAL, PINK = "#f6c28b", "#b8a1e3", "#6cc4b8", "#f28bb0"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
    "axes.edgecolor": BG, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 14, "axes.labelsize": 11.5,
    "axes.grid": True, "grid.color": GRID, "grid.linestyle": (0, (2, 3)), "grid.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "legend.frameon": False, "legend.fontsize": 10.5, "lines.solid_capstyle": "round"})


def finish(fig, name):
    fig.tight_layout(pad=1.4)
    fig.savefig(OUT + name, dpi=200)
    plt.close(fig)


S = json.load(open("data/bubble_shapes.json"))
T = json.load(open("data/sim_table.json"))

# 1. Bubble sizes to scale
fig, ax = plt.subplots(figsize=(8, 5))
x0 = 0.4
for k, c in (("Earth", E), ("Mars", M), ("Moon", L)):
    d = S[k]["d_eq_mm"]; r = d / 2; cx = x0 + r
    for rr, a in ((r * 1.18, 0.05), (r * 1.08, 0.08)):
        ax.add_patch(Circle((cx, r), rr, facecolor=mc.to_rgba(c, a), edgecolor="none"))
    ax.add_patch(Circle((cx, r), r, facecolor=mc.to_rgba(c, 0.22), edgecolor=c, lw=2.6))
    ax.add_patch(Circle((cx - r * 0.35, r * 1.35), r * 0.18, facecolor=mc.to_rgba("white", 0.35), edgecolor="none"))
    ax.text(cx, d + 0.38, k, ha="center", va="bottom", color=c, fontsize=13, fontweight="bold")
    ax.text(cx, d + 0.12, f"{d:.1f} mm", ha="center", va="bottom", color=MUTED, fontsize=11)
    x0 += d + 1.1
ax.fill_between([-1, 20], -0.6, 0, color="#3a302b")
ax.text(0.4, -0.33, "electrode", color=MUTED, fontsize=10, va="center")
ax.set_aspect("equal"); ax.set_xlim(0, x0 - 0.5); ax.set_ylim(-0.6, 5.0)
ax.set_xticks(range(0, int(x0), 2)); ax.set_yticks([]); ax.grid(False)
ax.set_xlabel("millimetres, to scale")
ax.set_title("O₂ bubble size at departure in molten CaCl₂, 950 °C")
finish(fig, "fig_round.png")

# 2. Departure size vs gravity
sw = S["sweep"]; g = np.array(sw["g_rel"]); d = np.array(sw["d_eq_mm"]); dE = np.interp(1.0, g, d)
fig, ax = plt.subplots(figsize=(8, 5))
ax.axvspan(1, 20, color=E, alpha=0.07, lw=0); ax.axvspan(0.166, 1, color=M, alpha=0.07, lw=0)
ax.text(1.12, 0.72, "centrifuge (ESA LDC, up to 20 g)", color=E, fontsize=10)
ax.text(0.175, 0.72, "parabolic flight", color=M, fontsize=10)
ax.fill_between(g, d * 0.8, d * 1.25, color=PEACH, alpha=0.10, lw=0)
ax.plot(g, d, color=PEACH, lw=3, label="my model: Young–Laplace, pin 0.1 mm (band: pin 0.05–0.2 mm)")
bg_ = np.array([1.0, 3.68 / 9.81, 1.62 / 9.81]); bv = np.array([0.0895, 0.1791, 0.3402])
ax.plot(bg_, dE * (bv / bv[0]) ** (1 / 3), "o", ms=9, mfc=BG, mec=LILAC, mew=2.2, label="Burke et al. 2024 CFD, ratio to 1 g")
for k, gr, c in (("Moon", 0.1652, L), ("Mars", 0.3783, M), ("Earth", 1.0, E)):
    ax.plot(gr, S[k]["d_eq_mm"], "o", ms=12, color=c, mec=BG, mew=2, zorder=5)
    ax.annotate(k, (gr, S[k]["d_eq_mm"]), xytext=(10, 6), textcoords="offset points", color=c, fontsize=11.5, fontweight="bold")
ax.set_xscale("log"); ax.set_xlim(0.14, 22); ax.set_ylim(0.5, 5.2); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([0.166, 0.378, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.17", "0.38", "1", "2", "5", "10", "20"])
ax.set_xlabel("gravity (g)"); ax.set_ylabel("departure diameter (mm)")
ax.legend(loc="upper right")
ax.set_title("Bubble size against gravity: what Earth facilities can reach")
finish(fig, "fig_size_vs_g.png")

# 3. Method (wide)
R0 = 100e-6
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.3]})
lc = np.sqrt(0.1429 / (2009 * G["Earth"])); bs = np.linspace(0.06 * lc, 2.2 * lc, 120); Vs = []
for b in bs:
    p = profile(b, G["Earth"], R0); Vs.append(p[2] if p else np.nan)
ib = np.nanargmax(np.array(Vs))
for f in (0.25, 0.45, 0.65, 0.85, 1.0):
    p = profile(bs[ib] * f, G["Earth"], R0)
    if not p: continue
    x, z, V = p; h = z[-1]
    xs = np.concatenate([-x[::-1], x]) * 1e3; ys = (h - np.concatenate([z[::-1], z])) * 1e3
    if f == 1:
        a1.fill(xs, ys, color=E, alpha=0.15, lw=0); a1.plot(xs, ys, color=E, lw=3)
    else:
        a1.plot(xs, ys, color=E, lw=1.3, alpha=0.5)
a1.fill_between([-2, 2], -0.4, 0, color="#3a302b")
a1.text(0.0, -0.22, "contact line pinned on a 0.1 mm site", color=MUTED, fontsize=10, ha="center", va="center")
a1.text(-1.45, 2.42, "last shape that can exist:\nit lets go", color=E, fontsize=10.5)
a1.set_aspect("equal"); a1.set_xlim(-1.5, 1.5); a1.set_ylim(-0.4, 2.9); a1.set_yticks([]); a1.grid(False); a1.set_xlabel("mm")
a1.set_title("Earth: shapes as the bubble grows")
for k, gk, c in (("Earth", G["Earth"], E), ("Mars", G["Mars"], M), ("Moon", G["Moon"], L)):
    lck = np.sqrt(0.1429 / (2009 * gk)); bk = np.linspace(0.06 * lck, 2.2 * lck, 110); Vk = []
    for b in bk:
        p = profile(b, gk, R0); Vk.append(p[2] * 1e9 if p else np.nan)
    Vk = np.array(Vk); i = np.nanargmax(Vk)
    a2.plot(bk * 1e3, Vk, color=c, lw=3, label=k)
    a2.fill_between(bk * 1e3, 0, Vk, color=c, alpha=0.06, lw=0)
    a2.plot(bk[i] * 1e3, Vk[i], "o", color=c, ms=11, mec=BG, mew=2)
a2.set_xlabel("radius of curvature at the top (mm)"); a2.set_ylabel("bubble volume (mm³)")
a2.set_xlim(0, None); a2.set_ylim(0, None); a2.legend(loc="upper left")
a2.set_title("Each curve ends at the departure bubble")
finish(fig, "fig_method.png")

# 4. Calibration
C = T["calib"]
fig, ax = plt.subplots(figsize=(8, 5)); ax.set_axisbelow(True)
xs = np.arange(len(C))
ax.bar(xs - 0.2, [c["measured_mV"] for c in C], width=0.34, color=L, alpha=0.9, label="measured (Lomax et al. 2022, Fig. 7 fits)", zorder=2)
ax.errorbar(xs + 0.2, [c["pred_med"] for c in C], yerr=[[c["pred_med"] - c["pred_lo"] for c in C], [c["pred_hi"] - c["pred_med"] for c in C]],
            fmt="o", color=E, ms=11, mec=BG, mew=2, capsize=5, lw=2, label="my model after calibration (median, 90 %)", zorder=3)
ax.plot(xs + 0.2, [c["prior_med"] for c in C], "D", color=M, ms=9, mec=BG, mew=1.5, label="my model before calibration", zorder=3)
for x, c in zip(xs, C):
    ax.plot([x + 0.2, x + 0.2], [c["pred_hi"], c["prior_med"]], color=M, lw=1, ls=(0, (2, 3)), alpha=0.6)
ax.set_xticks(xs); ax.set_xticklabels([f"{c['j_mA']:.0f} mA/cm²" for c in C])
ax.set_ylabel("extra voltage at 0.166 g vs 1 g (mV)"); ax.set_ylim(0, 125); ax.grid(axis="x", visible=False)
ax.legend(loc="upper left")
ax.set_title("Accuracy check on real reduced-gravity electrolysis data")
finish(fig, "fig_calibration.png")

# 5. Knee
K = json.load(open("data/knee.json"))
fig, ax = plt.subplots(figsize=(8, 5))
t = np.array(K["t_h"]); rm = np.array(K["rate_measured"]); rn = np.array(K["rate_null"])
ax.axvspan(16.2, 18.4, color=TEAL, alpha=0.12, lw=0)
ax.text(16.5, 0.47, "Fe + Si + Ti oxygen\nused up (~63 %)", color=TEAL, fontsize=10, va="top")
ax.fill_between(t, 0, rm, color=PINK, alpha=0.10, lw=0, step="mid")
ax.plot(t, rn, color=PEACH, lw=2.6, ls=(0, (5, 3)), label="front moving inward, no chemistry")
ax.plot(t, rm, "o", color=PINK, ms=8, mec=BG, mew=1.5, label="measured hourly rate (Lomax et al. 2025)")
ax.annotate("−21 % in one hour", xy=(18, 0.30), xytext=(20.5, 0.37), color=INK, fontsize=10.5, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_xlim(0, 30.5); ax.set_ylim(0, 0.5); ax.set_xlabel("hours at 3.2 V"); ax.set_ylabel("oxygen extracted (g per hour)")
ax.legend(loc="lower left")
ax.set_title("A step, not a slope: the 60 % slowdown")
finish(fig, "fig_knee.png")

# 6. Calcium
P = json.load(open("data/predominance.json")); a = np.array(P["a_CaO"])
fig, ax = plt.subplots(figsize=(8, 5))
ax.axvspan(0.3, 1, color=M, alpha=0.08, lw=0); ax.text(0.31, 1.70, "CaO-rich salt\ninside the pellet", color=M, fontsize=10)
for lab, key, c, lw, ls in [("Mg into Mg₂Si", "2MgO + Si -> Mg2Si", TEAL, 1.6, (0, (4, 3))), ("Al from Al₂O₃", "Al2O3 -> 2Al (simple oxide)", "#c9ced6", 1.6, (0, (4, 3))),
                            ("Ti via CaTiO₃", "CaTiO3 -> Ti + CaO (via intermediate)", LILAC, 1.6, (0, (4, 3))), ("Si into FeSi", "SiO2 + Fe -> FeSi", "#5b8fb0", 1.6, (0, (1, 2))),
                            ("Ca into CaSi₂", "CaO + 2Si -> CaSi2", PEACH, 3.2, "-"), ("Si via CaSiO₃", "CaSiO3 -> Si + CaO (via intermediate)", E, 3.2, "-")]:
    ax.plot(a, P["steps"][key], color=c, lw=lw, ls=ls, label=lab)
ax.annotate("17 mV apart", xy=(1, 2.03), xytext=(0.05, 2.36), color=INK, fontsize=11, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_ylim(1.5, 2.5)
ax.set_xticks([1e-3, 1e-2, 0.1, 1]); ax.set_xticklabels(["0.001", "0.01", "0.1", "1"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("activity of CaO dissolved in the salt"); ax.set_ylabel("voltage needed (V, vs O₂ anode)")
ax.legend(loc="upper left", ncol=2, fontsize=9.8)
ax.set_title("Why calcium follows silicon into the metal")
finish(fig, "fig_calcium.png")

# 7. Current-density lever (Moon)
tab = T["table"]; gg = np.array(tab["g"]); js = np.array(tab["j"]) / 1e4
f = lambda arr, i, gv: float(np.interp(np.log(gv), np.log(gg), arr[i]))
fig, ax = plt.subplots(figsize=(8, 5))
for gv, c, lab in ((0.1652, L, "Moon"), (0.3783, M, "Mars")):
    med = [f(tab["med"], i, gv) for i in range(len(js))]; lo = [f(tab["lo"], i, gv) for i in range(len(js))]; hi = [f(tab["hi"], i, gv) for i in range(len(js))]
    ax.fill_between(js, lo, hi, color=c, alpha=0.13, lw=0)
    ax.plot(js, med, "-o", color=c, lw=3, ms=8, mec=BG, mew=1.5, label=lab)
ax.axvline(0.14, color=MUTED, lw=1, ls=(0, (2, 3))); ax.text(0.145, 125, "ESA lab cell (est.)", color=MUTED, fontsize=10)
ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks([0.05, 0.08, 0.14, 0.25, 0.5]); ax.set_xticklabels(["0.05", "0.08", "0.14", "0.25", "0.5"])
ax.set_xlabel("current density at the anode (A/cm²)"); ax.set_ylabel("extra cell voltage vs Earth (mV)"); ax.set_ylim(0, 140)
ax.legend(loc="upper left")
ax.set_title("A design lever: spread the current, shrink the penalty")
finish(fig, "fig_jlever.png")
print("figures written")
