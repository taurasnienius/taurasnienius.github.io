"""Module D: lunar environment edge cases for a molten CaCl2 cell.

1. Vacuum exposure: Hertz-Knudsen evaporation of CaCl2 from a free melt surface.
   p_sat from JANAF-derived Clausius-Clapeyron fit (1200-1300 K): ln(p/bar) = 15.58 - 32940/T  (research/bubbles.md)
   J = alpha * p * sqrt(M / (2 pi R T)),  alpha = 1 is the upper bound.
2. Lunar night: keep the salt molten (radiative/insulation loss) versus let it freeze and remelt.
"""
import numpy as np

R, M_CACL2 = 8.314462, 0.11098
H_FUS, CP_L, CP_S = 28.5e3 / 0.11098, 102.5 / 0.11098, 0.75e3   # J/kg ; CP_S approximate solid value (assumption)
T_MELT = 1045.0


def p_sat(T):
    return np.exp(15.58 - 32940 / T) * 1e5  # Pa


def evap_flux(T, alpha=1.0):
    return alpha * p_sat(T) * np.sqrt(M_CACL2 / (2 * np.pi * R * T))  # kg/m2/s


def night_tradeoff(m_salt, A_out, U, T_op=1223.0, T_amb=250.0, night_h=14.8 * 24, T_cold=300.0):
    """Energy [kWh] to hold the salt molten through a night vs to remelt it afterwards.
    Holding: U*A*(T_op-T_amb)*t. Remelting from T_cold: sensible + fusion + sensible."""
    hold = U * A_out * (T_op - T_amb) * night_h * 3600 / 3.6e6
    remelt = m_salt * (CP_S * (T_MELT - T_cold) + H_FUS + CP_L * (T_op - T_MELT)) / 3.6e6
    return hold, remelt


if __name__ == "__main__":
    for T in (1173, 1223, 1273):
        print(f"T={T} K  p_sat={p_sat(T):.2f} Pa  evaporation into vacuum <= {evap_flux(T)*3600:.1f} kg/m2/h")
    A_free = np.pi * 0.05 ** 2  # 10 cm diameter melt surface (lab scale)
    print(f"Lab cell, 10 cm melt surface, 800 g salt: inventory gone in >= {0.8/(evap_flux(1223)*A_free)/60:.0f} min if open to vacuum")
    for m, A in ((0.8, 0.3), (100.0, 3.0), (1000.0, 12.0)):
        for U in (0.5, 2.0):
            h, r = night_tradeoff(m, A, U)
            print(f"salt {m:6.1f} kg, outer area {A:4.1f} m2, U={U} W/m2K: hold through night {h:7.1f} kWh vs remelt {r:6.2f} kWh")
