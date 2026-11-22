"""Oxygen 'staircase': share of a feedstock's oxygen bound to each element, ordered by the voltage
needed to release it (thermo.py ladder). Stage 1 = Fe, Mn, Si, Ti (all below ~2.05 V thermodynamic),
plus Na/K/P oxygen which is expected to leave by exchange with the melt rather than by reduction (flagged)."""
M = {'O': 15.999, 'Si': 28.085, 'Ti': 47.867, 'Al': 26.982, 'Fe': 55.845, 'Mn': 54.938, 'Mg': 24.305, 'Ca': 40.078,
     'Na': 22.990, 'K': 39.098, 'P': 30.974, 'Cr': 51.996, 'Ni': 58.693, 'Sr': 87.62, 'S': 32.06}
OX = {'SiO2': ('Si', 1, 2), 'TiO2': ('Ti', 1, 2), 'Al2O3': ('Al', 2, 3), 'FeO': ('Fe', 1, 1), 'Fe2O3': ('Fe', 2, 3),
      'MnO': ('Mn', 1, 1), 'MgO': ('Mg', 1, 1), 'CaO': ('Ca', 1, 1), 'Na2O': ('Na', 2, 1), 'K2O': ('K', 2, 1),
      'P2O5': ('P', 2, 5), 'Cr2O3': ('Cr', 2, 3), 'NiO': ('Ni', 1, 1), 'SrO': ('Sr', 1, 1), 'SO3': ('S', 1, 3)}
FEEDS = {
    'LHS-1 (Dec 2023 sheet)': {'SiO2': 49.12, 'TiO2': 0.63, 'Al2O3': 26.29, 'FeO': 3.20, 'MnO': 0.06, 'MgO': 2.86, 'CaO': 13.52, 'Na2O': 2.55, 'K2O': 0.34, 'P2O5': 0.17},
    'LMS-1 (Dec 2023 sheet)': {'SiO2': 48.22, 'TiO2': 2.70, 'Al2O3': 12.40, 'FeO': 8.79, 'MnO': 0.06, 'MgO': 15.97, 'CaO': 7.65, 'Na2O': 1.73, 'K2O': 0.42, 'P2O5': 0.23},
    'LMS-1 (Meurisse 2022 Table 2, ESA lab batch)': {'SiO2': 42.81, 'MgO': 18.89, 'Al2O3': 14.13, 'FeO': 7.87, 'CaO': 5.94, 'Na2O': 4.92, 'TiO2': 4.62, 'K2O': 0.57, 'P2O5': 0.44, 'Cr2O3': 0.21, 'MnO': 0.15, 'SO3': 0.11},
    'LHS-1 (2021 sheet)': {'SiO2': 48.1, 'Al2O3': 25.8, 'CaO': 18.4, 'Fe2O3': 3.7, 'K2O': 0.7, 'MgO': 0.3, 'MnO': 0.1, 'P2O5': 1.0, 'TiO2': 1.1, 'SO3': 0.3, 'SrO': 0.1},
    'LMS-1 (2021 sheet)': {'SiO2': 40.2, 'Al2O3': 14.0, 'CaO': 9.8, 'Fe2O3': 13.9, 'K2O': 0.6, 'MgO': 12.0, 'MnO': 0.3, 'P2O5': 1.0, 'TiO2': 7.3, 'Cr2O3': 0.3, 'NiO': 0.2, 'SrO': 0.1},
}
STAGE1 = {'Fe', 'Mn', 'Si', 'Ti', 'Ni', 'Cr'}
EXCH = {'Na', 'K', 'P', 'S', 'Sr'}


def oxygen_by_element(c):
    tot = sum(c.values()); o = {}
    for ox, w in c.items():
        e, n, m = OX[ox]
        o[e] = o.get(e, 0) + w / tot * m * M['O'] / (n * M[e] + m * M['O'])
    return o


def stage1_share(c):
    o = oxygen_by_element(c); O = sum(o.values())
    s1 = sum(v for k, v in o.items() if k in STAGE1) / O
    ex = sum(v for k, v in o.items() if k in EXCH) / O
    return O, s1, ex, {k: v / O for k, v in o.items()}


def share_on_basis(c, O_measured=None):
    """Stage-1 (+ exchangeable) oxygen as % of total, on the composition's own O or on a measured O wt% basis."""
    O, s1, ex, _ = stage1_share(c)
    base = O if O_measured is None else O_measured
    return (s1 + ex) * O / base


if __name__ == '__main__':
    print('LMS-1 stage 1 incl. exchangeable Na/K/P, on sheet O basis and on measured 41.6 wt% (Lomax 2025 Table 2):')
    for k, c in FEEDS.items():
        if 'LMS' in k:
            print(f'   {k:45s} {share_on_basis(c)*100:5.1f}%   {share_on_basis(c, 0.416)*100:5.1f}%')
    for k, c in FEEDS.items():
        O, s1, ex, parts = stage1_share(c)
        print(f"{k:24s} O={O*100:5.2f} wt%  stage1={s1*100:5.1f}%  (+exchangeable {ex*100:.1f}%)  "
              + ' '.join(f"{e}:{v*100:.1f}" for e, v in sorted(parts.items(), key=lambda x: -x[1])))
