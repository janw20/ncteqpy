from __future__ import annotations

import sympy as sp

pT = sp.Symbol("pT")
pT_min = sp.Symbol("pT_min")
pT_max = sp.Symbol("pT_max")
y = sp.Symbol("y")
y_min = sp.Symbol("y_min")
y_max = sp.Symbol("y_max")
eta = sp.Symbol("eta")
eta_min = sp.Symbol("eta_min")
eta_max = sp.Symbol("eta_max")
x = sp.Symbol("x")
Q2 = sp.Symbol("Q2")
W2 = sp.Symbol("W2")
sqrt_s = sp.Symbol("sqrt_s")
E_had = sp.Symbol("E_had")

label_to_kinvar: dict[str, sp.Symbol] = {
    "pT": pT,
    "pT_min": pT_min,
    "pT_max": pT_max,
    "y": y,
    "y_min": y_min,
    "y_max": y_max,
    "eta": eta,
    "eta_min": eta_min,
    "eta_max": eta_max,
    "x": x,
    "Q2": Q2,
    "W2": W2,
    "sqrt_s": sqrt_s,
    "E_had": E_had,
}

m_proton = 0.938

# relations used for Q2 and W2 cuts
W2_dis = Q2 * (1 - x) / x + m_proton**2
Q2_disdimu = 2 * m_proton * E_had * x * y  # E_had is the neutrino energy in DISDIMU
W2_disdimu = m_proton**2 + 2 * m_proton * (1 - x) * y * E_had

# approximate relations for x and Q2 used in the kinematic coverage plot
Q2_dis = x / (1 - x) * (W2 - m_proton**2)

Q2_sih = sp.Max(pT / 2, 1.3) ** 2
x_sih = sp.sqrt(Q2) / sqrt_s * sp.exp(-y)

x_wzprod = Q2 / sqrt_s * sp.exp(-eta)
x_wzprod_bin = sp.sqrt(Q2) / sqrt_s * sp.exp(-(eta_min + eta_max) / 2)

m_charm = 1.3
Q2_hq_pT = pT**2 + m_charm**2
Q2_hq_pT_bin = ((pT_min + pT_max) / 2) ** 2 + m_charm**2
Q2_hq_y = sqrt_s**2 * x**2 * sp.exp(2 * y)
x_hq = sp.sqrt(Q2) / sqrt_s * sp.exp(-y)
x_hq_bin = sp.sqrt(Q2) / sqrt_s * sp.exp(-(y_min + y_max) / 2)
