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
    "XI_F": "xi_f"
}

kinvars_py_to_tex = {
    "id_bin": r"\text{bin}",
    "sqrt_s": r"\sqrt{s}",
    "Q2": r"Q^2",
    "x": r"x",
    "y": r"y",
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
    "F2": "F2",
    "RF2": "ratio_F2",
    "APPLGRID": "applgrid",
    "RAPPLGRID": "rapplgrid"
}

theory_py_to_tex = {
    "sigma": r"\sigma",
    "ratio_sigma": r"\dfrac{\sigma^{A}}{\sigma^{\text{D}}}",
    "F2": r"F_2",
    "ratio_F2": r"\dfrac{F_2^{A}}{F_2^{\text{D}}}",
}

uncertainties_yaml_to_py = {
    "STAT": "unc_stat",
    "SYSU": "unc_sys",
    "SYST": "unc_tot",
    "THEO": "unc_theo",
}