"""Figures for the project page, same style as the Breathing the Moon page."""
import json, glob, numpy as np, matplotlib, pickle
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from act_ref import VLB_VUB, TABLE3
from ktsp_io import read_ktsp, states_on_grid
import os

OUT = "site/img/"
os.makedirs(OUT, exist_ok=True)
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


R = json.load(open("results_compare.json"))["results"]
BEST = {(d["instance"], d["nT"]): d["cost"] for d in (json.load(open(f)) for f in glob.glob("best/*.json"))}


def mine_best(name, nT):
    r = R.get(f"{name}_T{nT}")
    vals = [BEST.get((name, nT), np.inf)]
    if r:
        vals += [r[k]["ls_cost"] for k in ("beam_hand", "beam_discovered", "p_aco", "mcts")]
    return min(vals)


# 1. Grid refinement: their beam gets worse, mine gets better
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
for ax, name, title in ((axs[0], "asteroids_20", "20 asteroids"), (axs[1], "jovianmoons_20", "20 Jovian moons")):
    Ts = [21, 41, 81, 161]
    vub = np.array([VLB_VUB[(name, t)][1] for t in Ts])
    act = np.array([TABLE3[(name, t)][0] for t in Ts]) / vub
    bsw = np.array([TABLE3[(name, t)][2] for t in Ts]) / vub
    mb = np.array([R[f"{name}_T{t}"]["beam_hand"]["cost"] for t in Ts]) / vub
    ax.plot(Ts, act, "o-", color=M, lw=2.6, label="ACT beam search, width 10,000")
    ax.plot(Ts, bsw, "s-", color=LILAC, lw=2.2, label="ACT B-Swan")
    ax.plot(Ts, mb, "D-", color=E, lw=2.8, label="my beam, width 100, timing kept open")
    ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(150, 0.93, "best known", color=MUTED, fontsize=9.5, va="top", ha="right")
    ax.set_xscale("log", base=2); ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts])
    ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10]); ax.set_yticklabels(["1", "2", "5", "10"])
    ax.set_xlabel("time-grid points (finer →)"); ax.set_title(title)
axs[0].set_ylabel("tour Δv / best known")
axs[1].legend(loc="upper left", fontsize=9.5)
finish(fig, "fig_grid.png")

# 2. Thirteen instances: % above previous best known, B-Swan vs mine (horizontal, diverging)
inst = [k for k in R]
labels, bsw, mine = [], [], []
for k in inst:
    name, T = k.rsplit("_T", 1); T = int(T)
    vub = VLB_VUB[(name, T)][1]
    labels.append(f"{'Jovian' if name.startswith('jov') else 'Asteroids'} {int(name.split('_')[1])} / {T}")
    bsw.append((TABLE3[(name, T)][2] / vub - 1) * 100); mine.append((mine_best(name, T) / vub - 1) * 100)
y = np.arange(len(labels))[::-1]; h = 0.38
fig, ax = plt.subplots(figsize=(10, 6.4))
ax.barh(y + h / 2, bsw, h, color=LILAC, label="ACT B-Swan heuristic")
ax.barh(y - h / 2, mine, h, color=[E if v < -0.005 else "#4f7f96" for v in mine], label="mine")
for yy, v in zip(y, mine):
    ax.text(v + (0.4 if v > -0.005 else -0.4), yy - h / 2, ("matches" if abs(v) < 0.005 else f"{v:+.1f}%"), va="center", ha="left" if v > -0.005 else "right",
            color=E if v < -0.005 else MUTED, fontsize=9.5, fontweight="bold" if v < -0.005 else "normal")
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
ax.set_xlabel("% above the previous best known tour (left of zero = new best)")
ax.grid(axis="y", visible=False); ax.legend(loc="lower right", fontsize=10)
lo = min(mine) - 3; ax.set_xlim(lo, max(bsw) + 3)
ax.set_title("Thirteen harder instances, same grid and cost model as ACT")
finish(fig, "fig_all.png")

# 3. Discovery: training curves and held-out
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for s, c in ((1, E), (2, PEACH), (3, TEAL)):
    h = [json.loads(l) for l in open(f"gp_seed{s}.jsonl")]
    axs[0].plot([r["gen"] for r in h], [r["best_fit"] for r in h], color=c, lw=2.4, label=f"run {s}, from random rules")
axs[0].axhline(1.0873, color=M, lw=1.6, ls=(0, (4, 3)))
axs[0].text(0.5, 1.089, "my hand-designed rule", color=M, fontsize=9.5, va="bottom")
axs[0].set_xlabel("generation"); axs[0].set_ylabel("mean Δv / best known (training)")
axs[0].set_title("Evolving the ranking rule"); axs[0].legend(fontsize=9.5, loc="center right")
rr = json.load(open("results_rules.json"))
names = rr["rules"]; res = rr["results"]
for W, mk in ((50, "o"), (200, "s")):
    rows = [v for k, v in res.items() if k.startswith(f"W{W} ")]
    gm = [np.exp(np.mean([np.log(r[n]) for r in rows])) for n in names]
    axs[1].scatter(range(len(names)), gm, marker=mk, s=70, color=E if W == 50 else PEACH, label=f"beam width {W}", zorder=3)
axs[1].set_xticks(range(len(names)))
axs[1].set_xticklabels(["hand:\non-schedule", "hand:\ncost-only", "evolved\nrun 1", "evolved\nrun 2", "evolved\nrun 3", "evolved,\nseeded"], fontsize=9.5)
axs[1].set_ylabel("geo-mean Δv / best known (held-out)"); axs[1].set_title("Tested on unseen, larger instances")
axs[1].legend(fontsize=9.5)
finish(fig, "fig_discovery.png")

# 4. The new 80-asteroid tour, seen from above
sol = json.load(open("best/asteroids_80_T161.json"))
inst = read_ktsp("/home/claude/ktsp_data/0_instances/asteroids_80.ktsp")
t, Rr, V = states_on_grid(inst, 161)
AU = 1.495978707e11
fig, ax = plt.subplots(figsize=(7.6, 7.6))
cmap = plt.get_cmap("plasma")
ax.scatter([0], [0], s=140, color="#ffd27a", zorder=4)
for i in range(len(Rr)):
    ax.plot(Rr[i, :, 0] / AU, Rr[i, :, 1] / AU, color="#3a302b", lw=0.6, zorder=1)
order, dep, arr = sol["order"], sol["dep_idx"], sol["arr_idx"]
import pykep as pk
for p in range(len(dep)):
    a, b = order[p], order[p + 1]
    tof = (t[arr[p]] - t[dep[p]]) * 86400.0
    best = None
    for cw in (False, True):
        lp = pk.lambert_problem(Rr[a, dep[p]], Rr[b, arr[p]], tof, inst["mu"], cw, 1000)
        for v1, v2 in zip(lp.v0, lp.v1):
            dv = np.linalg.norm(np.array(v1) - V[a, dep[p]]) + np.linalg.norm(np.array(v2) - V[b, arr[p]])
            if best is None or dv < best[0]: best = (dv, v1)
    pts = [Rr[a, dep[p]]]
    for f in np.linspace(0, 1, 60)[1:]:
        r, _ = pk.propagate_lagrangian([Rr[a, dep[p]], best[1]], tof * f, inst["mu"], False); pts.append(r)
    pts = np.array(pts) / AU
    ax.plot(pts[:, 0], pts[:, 1], color=cmap(0.1 + 0.8 * dep[p] / 160), lw=1.3, alpha=0.95, zorder=2)
    ax.scatter(*(Rr[b, arr[p], :2] / AU), s=9, color=cmap(0.1 + 0.8 * arr[p] / 160), zorder=3)
p0 = Rr[0, dep[0], :2] / AU; p1 = Rr[0, arr[-1], :2] / AU
ax.scatter(*p0, s=40, color=cmap(0.1), zorder=5); ax.scatter(*p1, s=40, color=cmap(0.9), zorder=5)
ax.annotate("start", p0, xytext=(p0[0] + 0.35, p0[1] - 0.45), color=INK, fontsize=10.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate("home again", p1, xytext=(p1[0] - 1.3, p1[1] - 0.55), color=INK, fontsize=10.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
lim = 3.8; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_title("80 asteroids, one spacecraft, 5.5 years (colour = time)")
finish(fig, "fig_tour80.png")
print("figures done")
