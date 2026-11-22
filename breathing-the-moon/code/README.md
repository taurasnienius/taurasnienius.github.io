# Breathing the Moon: code

Python scripts behind the [project page](https://taurasnienius.github.io/breathing-the-moon/) and the [technical paper](https://taurasnienius.github.io/breathing-the-moon/paper.html). Every number in the paper comes from running these unchanged.

```
pip install -r requirements.txt
cd breathing-the-moon/code
python sim_table.py
```

Run each script from this folder. Random seeds are fixed (thermodynamic Monte Carlo seed 1, calibration seed 11), so the output is reproducible.

| Script | What it does | Paper |
|---|---|---|
| `thermo.py` | Reversible decomposition voltages of regolith minerals and Ca-intermediates at 1223 K, Monte Carlo over Gibbs energies and CaO activity. Inputs in `data/thermo.yaml`. | Section 4 |
| `stages.py` | Oxygen bound to each element for five simulant compositions. | Section 5 |
| `knee.py` | Geometric (no-chemistry) front model against the digitised 2025 pellet data; tests the slowdown near 60 %. | Section 5 |
| `kinetics.py` | Shrinking-core fits to the 2020 powder run (set aside in the paper, kept for the record). | Section 2 |
| `bubble_shape.py` | Young-Laplace static bubble shapes and departure sizes against gravity (slow, several minutes). | Section 6 |
| `bubble_growth.py` | Sequence of static shapes up to departure for Earth, Mars and Moon. | Section 6 |
| `bubbles.py` | The gravity chain (departure size, coverage, ohmic and activation terms) and the original uncalibrated Monte Carlo. | Section 7 |
| `bubble_calib.py` | First calibration against the Lomax et al. 2022 water data. | Section 8 |
| `sim_table.py` | Main calibration (60,000 draws) and the penalty tables for the interactive simulator. **All gravity results come from this script.** | Sections 8, 9 |
| `sim_table_core.py` | Shared functions of the chain, imported by `prior_sens.py`. | |
| `prior_sens.py` | Repeats the calibration with widened prior ranges. | Section 8.7, Table S1 |
| `headroom.py` | Three estimates of the voltage headroom at fixed 3.2 V. | Section 9.3, Table S2 |
| `environment.py` | Vacuum evaporation of the salt and lunar-night heat balance. | Section 10 |
| `figs3.py` | Regenerates the figures into `figures/`. | |

`data/` holds the inputs and outputs: digitised curves from Lomax et al. 2025 (Figs. 2 and 3), the thermodynamic data with sources, and the JSON tables the page reads.

## Limits

This is a first-order model calibrated on aqueous data and transferred to molten CaCl2. Read Section 12 of the paper before reusing any number.

## Licence

Code under the MIT licence (see `LICENSE` at the repository root). Digitised data from published figures are given for reproducibility; the original figures belong to their authors.
