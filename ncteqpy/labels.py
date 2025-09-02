from __future__ import annotations

from io import StringIO
from typing import cast

import numpy as np
import pandas as pd

chi2fcn_per_point_breakdown_yaml_to_py = {
    "IdPoint": "id_point",
    "IDDataSet": "id_dataset",
    "Chi2": "chi2",
    "Data": "data",
    "Theo": "theory",
    "KinVarVals": "kinetic_variables",
    "TypeExp": "type_experiment",
    "FSHadron": "final_state_hadron",
}

kinvars_yaml_to_py = {
    "BIN": "id_bin",
    "RS": "sqrt_s",
    "X": "x",
    "Y": "y",
    "Q2": "Q2",
    "W2": "W2",
    "EHAD": "E_had",
    "EPS": "eps",
    "PT": "pT",
    "PTMIN": "pT_min",
    "PTMAX": "pT_max",
    "Y": "y",
    "YMIN": "y_min",
    "YMAX": "y_max",
    "ETA": "eta",
    "ETAMIN": "eta_min",
    "ETAMAX": "eta_max",
    "XI_R": "xi_r",
    "XI_F": "xi_f",
}

kinvars_py_to_tex = {
    "id_bin": r"\text{bin}",
    "sqrt_s": r"\sqrt{s}",
    "Q2": r"Q^2",
    "x": r"x",
    "y": r"y",
    "E_had": r"E_{\text{hadron}}",
    "pT": r"p_{\text{T}}",
    "pT_min": r"p_{\text{T}}",
    "pT_max": r"p_{\text{T}}",
    "y": r"y",
    "y_min": r"y",
    "y_max": r"y",
    "eta": r"\eta",
    "eta_min": r"\eta",
    "eta_max": r"\eta",
    "xi_r": r"\xi_{\text{r}}",
    "xi_f": r"\xi_{\text{f}}",
}

theory_yaml_to_py = {
    "SIG": "sigma",
    "RSIG": "ratio_sigma",
    "SIGIPT": "sigma_pT_integrated",
    "F2": "F2",
    "RF2": "ratio_F2",
    "APPLGRID": "applgrid",
    "RAPPLGRID": "rapplgrid",
}

theory_py_to_tex = {
    "sigma": r"\sigma",
    "ratio_sigma": r"\dfrac{\sigma^{A}}{\sigma^{\text{D}}}",
    "F2": r"F_2",
    "ratio_F2": r"\dfrac{F_2^{A}}{F_2^{\text{D}}}",
}

uncertainties_yaml_to_py = {
    "STAT": "unc_stat",
    "SYSU": "unc_sys_uncorr",
    "COR": "unc_sys_corr",
    "CORP": "unc_sys_corr_percent",
    "SYST": "unc_sys_corr_tot",
    "THEO": "unc_theo",
}

data_yaml_to_py = {
    **kinvars_yaml_to_py,
    **theory_yaml_to_py,
    **uncertainties_yaml_to_py,
}

parameters_cj15_py_to_tex = {
    "kappa": r"\kappa",
    "c": "c",
    "uv_p1": r"p_1^{u_v}",
    "uv_p2": r"p_2^{u_v}",
    "uv_p3": r"p_3^{u_v}",
    "uv_p4": r"p_4^{u_v}",
    "uv_p5": r"p_5^{u_v}",
    "uv_a1": r"a_1^{u_v}",
    "uv_a2": r"a_2^{u_v}",
    "uv_a3": r"a_3^{u_v}",
    "uv_a4": r"a_4^{u_v}",
    "uv_a5": r"a_5^{u_v}",
    "uv_b1": r"b_1^{u_v}",
    "uv_b2": r"b_2^{u_v}",
    "uv_b3": r"b_3^{u_v}",
    "uv_b4": r"b_4^{u_v}",
    "uv_b5": r"b_5^{u_v}",
    "dv_p1": r"p_1^{d_v}",
    "dv_p2": r"p_2^{d_v}",
    "dv_p3": r"p_3^{d_v}",
    "dv_p4": r"p_4^{d_v}",
    "dv_p5": r"p_5^{d_v}",
    "dv_p6": r"p_6^{d_v}",
    "dv_a1": r"a_1^{d_v}",
    "dv_a2": r"a_2^{d_v}",
    "dv_a3": r"a_3^{d_v}",
    "dv_a4": r"a_4^{d_v}",
    "dv_a5": r"a_5^{d_v}",
    "dv_a6": r"a_6^{d_v}",
    "dv_b1": r"b_1^{d_v}",
    "dv_b2": r"b_2^{d_v}",
    "dv_b3": r"b_3^{d_v}",
    "dv_b4": r"b_4^{d_v}",
    "dv_b5": r"b_5^{d_v}",
    "dv_b6": r"b_6^{d_v}",
    "ubdb_p1": r"p_1^{u̅+d̅}",
    "ubdb_p2": r"p_2^{u̅+d̅}",
    "ubdb_p3": r"p_3^{u̅+d̅}",
    "ubdb_p4": r"p_4^{u̅+d̅}",
    "ubdb_p5": r"p_5^{u̅+d̅}",
    "ubdb_a1": r"a_1^{u̅+d̅}",
    "ubdb_a2": r"a_2^{u̅+d̅}",
    "ubdb_a3": r"a_3^{u̅+d̅}",
    "ubdb_a4": r"a_4^{u̅+d̅}",
    "ubdb_a5": r"a_5^{u̅+d̅}",
    "ubdb_b1": r"b_1^{u̅+d̅}",
    "ubdb_b2": r"b_2^{u̅+d̅}",
    "ubdb_b3": r"b_3^{u̅+d̅}",
    "ubdb_b4": r"b_4^{u̅+d̅}",
    "ubdb_b5": r"b_5^{u̅+d̅}",
    "ssbsum_p0": r"p_0^{s+s̅}",
    "ssb_p1": r"p_1^{s+s̅}",
    "ssb_p2": r"p_2^{s+s̅}",
    "ssb_p3": r"p_3^{s+s̅}",
    "ssb_p4": r"p_4^{s+s̅}",
    "ssb_p5": r"p_5^{s+s̅}",
    "ssbsum_a0": r"a_0^{s+s̅}",
    "ssb_a1": r"a_1^{s+s̅}",
    "ssb_a2": r"a_2^{s+s̅}",
    "ssb_a3": r"a_3^{s+s̅}",
    "ssb_a4": r"a_4^{s+s̅}",
    "ssb_a5": r"a_5^{s+s̅}",
    "ssbsum_b0": r"b_0^{s+s̅}",
    "ssb_b1": r"b_1^{s+s̅}",
    "ssb_b2": r"b_2^{s+s̅}",
    "ssb_b3": r"b_3^{s+s̅}",
    "ssb_b4": r"b_4^{s+s̅}",
    "ssb_b5": r"b_5^{s+s̅}",
    "gsum_p0": r"p_0^{g}",
    "g_p1": r"p_1^{g}",
    "g_p2": r"p_2^{g}",
    "g_p3": r"p_3^{g}",
    "g_p4": r"p_4^{g}",
    "g_p5": r"p_5^{g}",
    "gsum_a0": r"a_0^{g}",
    "g_a1": r"a_1^{g}",
    "g_a2": r"a_2^{g}",
    "g_a3": r"a_3^{g}",
    "g_a4": r"a_4^{g}",
    "g_a5": r"a_5^{g}",
    "gsum_b0": r"b_0^{g}",
    "g_b1": r"b_1^{g}",
    "g_b2": r"b_2^{g}",
    "g_b3": r"b_3^{g}",
    "g_b4": r"b_4^{g}",
    "g_b5": r"b_5^{g}",
    "dboub_p0": r"p_0^{d̅/u̅}",
    "dboub_p1": r"p_1^{d̅/u̅}",
    "dboub_p2": r"p_2^{d̅/u̅}",
    "dboub_p3": r"p_3^{d̅/u̅}",
    "dboub_p4": r"p_4^{d̅/u̅}",
    "dboub_p5": r"p_5^{d̅/u̅}",
    "dboub_a0": r"a_0^{d̅/u̅}",
    "dboub_a1": r"a_1^{d̅/u̅}",
    "dboub_a2": r"a_2^{d̅/u̅}",
    "dboub_a3": r"a_3^{d̅/u̅}",
    "dboub_a4": r"a_4^{d̅/u̅}",
    "dboub_a5": r"a_5^{d̅/u̅}",
    "dboub_b0": r"b_0^{d̅/u̅}",
    "dboub_b1": r"b_1^{d̅/u̅}",
    "dboub_b2": r"b_2^{d̅/u̅}",
    "dboub_b3": r"b_3^{d̅/u̅}",
    "dboub_b4": r"b_4^{d̅/u̅}",
    "dboub_b5": r"b_5^{d̅/u̅}",
}


def nucleus_to_latex(
    Z: int | float | None = None,
    A: int | float | None = None,
    long: bool = False,
    superscript: bool = False,
) -> str:
    """Convert atomic number and mass number to element symbol.

    Parameters
    ----------
    A : int | float | None, optional
        Mass number of the element. By default None.
    Z : int | float | None, optional
        Atomic number of the element. By default None.
    long : bool, optional
        If the name of the element should be returned instead of its name (e.g. "Lead" instead of "Pb"). By default False.

    Returns
    -------
    str
        Symbol or name of the element.
    """

    if Z is not None and A is not None:
        if np.isnan(Z) and np.isnan(A) or np.isinf(Z) and np.isinf(A):
            return "\\dots"

        row = _elements.iloc[
            (_elements[["AtomicNumber", "AtomicMass"]] - [Z, A])
            .abs()
            .sort_values(by=["AtomicNumber", "AtomicMass"])
            .index[0]
        ]
    elif Z is not None and not np.isnan(Z):
        row = _elements.iloc[(_elements["AtomicNumber"] - Z).abs().argsort().iloc[0]]
    elif A is not None and not np.isnan(A):
        row = _elements.iloc[(_elements["AtomicMass"] - A).abs().argsort().iloc[0]]
    else:
        return ""

    if long:
        res = cast(str, row["Name"])
    else:
        res = cast(str, row["Symbol"])

    res = r"\mathrm{" + res + r"}"

    if superscript and not row["AtomicNumber"] == 1:
        res = f"^{{{round(A if A is not None and not np.isnan(A) else row['AtomicMass'])}}}{res}"

    return res


_elements_csv = """"AtomicNumber","Symbol","Name","AtomicMass"
1,"p","Proton",1.0073
1,"H","Hydrogen",1.0080
1,"D","Deuterium",2.0141
2,"He","Helium",4.00260
3,"Li","Lithium",7.0
4,"Be","Beryllium",9.012183
5,"B","Boron",10.81
6,"C","Carbon",12.011
7,"N","Nitrogen",14.007
8,"O","Oxygen",15.999
9,"F","Fluorine",18.99840316
10,"Ne","Neon",20.180
11,"Na","Sodium",22.9897693
12,"Mg","Magnesium",24.305
13,"Al","Aluminum",26.981538
14,"Si","Silicon",28.085
15,"P","Phosphorus",30.97376200
16,"S","Sulfur",32.07
17,"Cl","Chlorine",35.45
18,"Ar","Argon",39.9
19,"K","Potassium",39.0983
20,"Ca","Calcium",40.08
21,"Sc","Scandium",44.95591
22,"Ti","Titanium",47.867
23,"V","Vanadium",50.9415
24,"Cr","Chromium",51.996
25,"Mn","Manganese",54.93804
26,"Fe","Iron",55.84
27,"Co","Cobalt",58.93319
28,"Ni","Nickel",58.693
29,"Cu","Copper",63.55
30,"Zn","Zinc",65.4
31,"Ga","Gallium",69.723
32,"Ge","Germanium",72.63
33,"As","Arsenic",74.92159
34,"Se","Selenium",78.97
35,"Br","Bromine",79.90
36,"Kr","Krypton",83.80
37,"Rb","Rubidium",85.468
38,"Sr","Strontium",87.62
39,"Y","Yttrium",88.90584
40,"Zr","Zirconium",91.22
41,"Nb","Niobium",92.90637
42,"Mo","Molybdenum",95.95
43,"Tc","Technetium",96.90636
44,"Ru","Ruthenium",101.1
45,"Rh","Rhodium",102.9055
46,"Pd","Palladium",106.42
47,"Ag","Silver",107.868
48,"Cd","Cadmium",112.41
49,"In","Indium",114.818
50,"Sn","Tin",118.71
51,"Sb","Antimony",121.760
52,"Te","Tellurium",127.6
53,"I","Iodine",126.9045
54,"Xe","Xenon",131.29
55,"Cs","Cesium",132.9054520
56,"Ba","Barium",137.33
57,"La","Lanthanum",138.9055
58,"Ce","Cerium",140.116
59,"Pr","Praseodymium",140.90766
60,"Nd","Neodymium",144.24
61,"Pm","Promethium",144.91276
62,"Sm","Samarium",150.4
63,"Eu","Europium",151.964
64,"Gd","Gadolinium",157.2
65,"Tb","Terbium",158.92535
66,"Dy","Dysprosium",162.500
67,"Ho","Holmium",164.93033
68,"Er","Erbium",167.26
69,"Tm","Thulium",168.93422
70,"Yb","Ytterbium",173.05
71,"Lu","Lutetium",174.9668
72,"Hf","Hafnium",178.49
73,"Ta","Tantalum",180.9479
74,"W","Tungsten",183.84
75,"Re","Rhenium",186.207
76,"Os","Osmium",190.2
77,"Ir","Iridium",192.22
78,"Pt","Platinum",195.08
79,"Au","Gold",196.96657
80,"Hg","Mercury",200.59
81,"Tl","Thallium",204.383
82,"Pb","Lead",207.97665
83,"Bi","Bismuth",208.98040
84,"Po","Polonium",208.98243
85,"At","Astatine",209.98715
86,"Rn","Radon",222.01758
87,"Fr","Francium",223.01973
88,"Ra","Radium",226.02541
89,"Ac","Actinium",227.02775
90,"Th","Thorium",232.038
91,"Pa","Protactinium",231.03588
92,"U","Uranium",238.0289
93,"Np","Neptunium",237.048172
94,"Pu","Plutonium",244.06420
95,"Am","Americium",243.061380
96,"Cm","Curium",247.07035
97,"Bk","Berkelium",247.07031
98,"Cf","Californium",251.07959
99,"Es","Einsteinium",252.0830
100,"Fm","Fermium",257.09511
101,"Md","Mendelevium",258.09843
102,"No","Nobelium",259.10100
103,"Lr","Lawrencium",266.120
104,"Rf","Rutherfordium",267.122
105,"Db","Dubnium",268.126
106,"Sg","Seaborgium",269.128
107,"Bh","Bohrium",270.133
108,"Hs","Hassium",269.1336
109,"Mt","Meitnerium",277.154
110,"Ds","Darmstadtium",282.166
111,"Rg","Roentgenium",282.169
112,"Cn","Copernicium",286.179
113,"Nh","Nihonium",286.182
114,"Fl","Flerovium",290.192
115,"Mc","Moscovium",290.196
116,"Lv","Livermorium",293.205
117,"Ts","Tennessine",294.211
118,"Og","Oganesson",295.216 """
_elements = pd.read_csv(StringIO(_elements_csv))
