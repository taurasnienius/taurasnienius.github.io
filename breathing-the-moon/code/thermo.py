"""Module A: decomposition-voltage ladder for real regolith phases in molten CaCl2.

Each step is an electrochemical reaction at the cathode with O2 evolved at an inert anode:
    E = dG_r / (z F),  dG_r = sum(nu * dfG(products)) - sum(nu * dfG(reactants)),  elements and O2 = 0.
Activities: CaO produced or consumed is dissolved in the melt at activity a_CaO (pure solid CaO = 1).
    E(a) = E0 + (R T / z F) * ln( prod a_products^nu / prod a_reactants^nu ), O2 at 1 bar.
Monte Carlo: dfG values drawn from N(value, sd); a_CaO log-uniform over a range; T fixed.
Simple oxides are kept beside the mineral and intermediate routes so the difference is visible.
"""
import numpy as np, yaml, os

F, R = 96485.33, 8.314462
HERE = os.path.dirname(os.path.abspath(__file__))
D = yaml.safe_load(open(os.path.join(HERE, "data", "thermo.yaml")))["species"]

# (label, group, reactants {species: nu}, products {species: nu}, electrons z)
# elements and O2 omitted (dfG = 0). 'CaO' in products/reactants gets the melt activity.
STEPS = [
    ("FeO -> Fe",                          "Fe", {"FeO": 1}, {}, 2),
    ("Ilmenite FeTiO3 -> Fe + TiO2",       "Fe", {"ilmenite": 1}, {"TiO2": 1}, 2),
    ("Fayalite Fe2SiO4 -> 2Fe + SiO2",     "Fe", {"fayalite": 1}, {"SiO2": 1}, 4),
    ("MnO -> Mn",                          "Mn", {"MnO": 1}, {}, 2),
    ("SiO2 -> Si (simple oxide)",          "Si", {"SiO2": 1}, {}, 4),
    ("SiO2 + Fe -> FeSi",                  "Si", {"SiO2": 1}, {"FeSi": 1}, 4),
    ("CaSiO3 -> Si + CaO (via intermediate)", "Si", {"wollastonite": 1}, {"CaO": 1}, 4),
    ("Anorthite -> 2Si + CaO + Al2O3",     "Si", {"anorthite": 1}, {"CaO": 1, "Al2O3": 1}, 8),
    ("Forsterite Mg2SiO4 -> Si + 2MgO",    "Si", {"forsterite": 1}, {"MgO": 2}, 4),
    ("Diopside -> Si + CaSiO3 + MgO",      "Si", {"diopside": 1}, {"wollastonite": 1, "MgO": 1}, 4),
    ("TiO2 -> Ti (simple oxide)",          "Ti", {"TiO2": 1}, {}, 4),
    ("CaTiO3 -> Ti + CaO (via intermediate)", "Ti", {"perovskite": 1}, {"CaO": 1}, 4),
    ("Al2O3 -> 2Al (simple oxide)",        "Al", {"Al2O3": 1}, {}, 6),
    ("Spinel MgAl2O4 -> 2Al + MgO",        "Al", {"spinel": 1}, {"MgO": 1}, 6),
    ("CaO + 2Si -> CaSi2",                 "Ca", {"CaO": 1}, {"CaSi2": 1}, 2),
    ("CaO + Si -> CaSi",                   "Ca", {"CaO": 1}, {"CaSi": 1}, 2),
    ("CaO -> Ca (pure metal)",             "Ca", {"CaO": 1}, {}, 2),
    ("2MgO + Si -> Mg2Si",                 "Mg", {"MgO": 2}, {"Mg2Si": 1}, 4),
    ("MgO -> Mg (simple oxide)",           "Mg", {"MgO": 1}, {}, 2),
    ("Forsterite Mg2SiO4 -> 2Mg + SiO2",   "Mg", {"forsterite": 1}, {"SiO2": 1}, 4),
    ("Diopside -> Mg + CaSiO3 + SiO2",     "Mg", {"diopside": 1}, {"wollastonite": 1, "SiO2": 1}, 2),
    ("CaCl2 -> Ca + Cl2 (salt breaks down)", "ceiling", {"CaCl2": 1}, {}, 2),
]


def dfG(sp, T, draw=None):
    e = D[sp]["dfG"]
    v = float(np.interp(T, sorted(e), [e[k] for k in sorted(e)]))
    return v + (draw[sp] if draw else 0.0)


def E_step(step, T, a_CaO=1.0, draw=None):
    _, _, reac, prod, z = step
    dG = sum(n * dfG(s, T, draw) for s, n in prod.items()) - sum(n * dfG(s, T, draw) for s, n in reac.items())
    E = dG * 1e3 / (z * F)
    # activity term: products CaO at a_CaO (makes it easier), reactant CaO at a_CaO (makes it harder)
    lnQ = prod.get("CaO", 0) * np.log(a_CaO) - reac.get("CaO", 0) * np.log(a_CaO)
    return E + R * T / (z * F) * lnQ


def ladder(T=1223.0, a_CaO=1.0):
    return [(s[0], s[1], E_step(s, T, a_CaO)) for s in STEPS]


def monte_carlo(T=1223.0, n=20000, a_range=(0.02, 0.5), seed=1):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, len(STEPS)))
    for i in range(n):
        draw = {k: rng.normal(0, v["sd"]) for k, v in D.items()}
        a = np.exp(rng.uniform(np.log(a_range[0]), np.log(a_range[1])))
        out[i] = [E_step(s, T, a, draw) for s in STEPS]
    return out


def predominance(T=1223.0, labels=None, a_grid=None):
    """Voltage of selected steps versus dissolved-CaO activity (log grid)."""
    a_grid = np.logspace(-3, 0, 61) if a_grid is None else a_grid
    steps = [s for s in STEPS if labels is None or s[0] in labels]
    return a_grid, {s[0]: [E_step(s, T, a) for a in a_grid] for s in steps}


if __name__ == "__main__":
    T = 1223.0
    print("Unit activities, 1223 K:")
    for lab, g, e in sorted(ladder(T), key=lambda r: r[2]):
        print(f"  {e:6.3f} V  [{g:7s}] {lab}")
    mc = monte_carlo(T, n=5000)
    print("\nMonte Carlo (dfG sd, a_CaO log-uniform 0.02-0.5): median [5%, 95%]")
    med = np.median(mc, 0); lo = np.percentile(mc, 5, 0); hi = np.percentile(mc, 95, 0)
    for i in np.argsort(med):
        print(f"  {med[i]:6.3f} [{lo[i]:5.3f}, {hi[i]:5.3f}]  [{STEPS[i][1]:7s}] {STEPS[i][0]}")
    # probability Ca silicide forms at a lower voltage than Si from CaSiO3
    iSi = [s[0] for s in STEPS].index("CaSiO3 -> Si + CaO (via intermediate)")
    iCa = [s[0] for s in STEPS].index("CaO + 2Si -> CaSi2")
    print("\nP(CaSi2 formation needs less voltage than Si via CaSiO3) =", np.mean(mc[:, iCa] < mc[:, iSi]).round(3))
    print("Median gap (CaSi2 - Si via CaSiO3) = %.3f V, 90%% range [%.3f, %.3f]" % (
        np.median(mc[:, iCa] - mc[:, iSi]), *np.percentile(mc[:, iCa] - mc[:, iSi], [5, 95])))
