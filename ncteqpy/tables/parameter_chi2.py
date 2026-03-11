from __future__ import annotations
from itertools import zip_longest
import numpy as np
import pandas as pd
from typing_extensions import Hashable, Literal, Mapping, Sequence, List

from ncteqpy._typing import SequenceNotStr
from ncteqpy.labels import nucleus_to_latex
from ncteqpy.util import update_kwargs
from ncteqpy.settings import Settings

FlavourTypes = Literal[
    "uv",
    "dv",
    "g",
    "ssb",
    "ubdb",
    "dboub",
]

def table_parameter_settings(
    flavour: FlavourTypes,
    settings: Settings,
    minimum: Chi2,
    column_types: SequenceNotStr[str] | None = None,
    format_columns: str | list[str | None] | dict[Literal["j", "p","a","b"], str] | None = None,
    format_data: str | None = None,
    sort_ascending: bool | Sequence[bool | None] | None = None,
    highlight: dict[Literal["j", "p", "a", "b"], int | List[int]] | Literal["open"] | None = "open",
    labels: dict[str, str] | None = None,
    alignment: Literal["r", "l", "c"] = "r",
    width:float= 0.45,
) -> tuple[pd.DataFrame, str]:
    
    columns=["j", "p", "a", "b"]
    
    # update format_columns
    format_columns_default = {c: "{}" for c in columns}

    if format_columns is None:
        format_columns_updated = format_columns_default
    elif isinstance(format_columns, list):
        if len(format_columns) < len(columns):
            format_columns = format_columns + (len(columns) - len(format_columns)) * [
                None
            ]
        format_columns_updated = {
            c: (f if f is not None else format_columns_default[c])
            for f, c in zip(format_columns, columns)
        }
    elif isinstance(format_columns, dict):
        format_columns_updated: dict[str, str] = update_kwargs(
            format_columns_default, format_columns
        )  # pyright: ignore[reportArgumentType]
    else:
        raise ValueError("format_columns must be list, dict or None")

    table = pd.DataFrame(
        columns=columns, dtype=str
    )
    parameters_min=minimum.minimum_parameters
    parameters_input=settings.parameters_input
    vals_p, vals_a, vals_b = [], [], []
    highlight_p, highlight_a, highlight_b = [], [], []
    is_sum = is_kappa = is_zero = False

    # Zähler für dict-basiertes Highlighting
    idx_counter = {"p": 0, "a": 0, "b": 0}

    # Zuordnung Typ → Listen
    type_map = {
        "p": (vals_p, highlight_p),
        "a": (vals_a, highlight_a),
        "b": (vals_b, highlight_b),
    }

    filtered_params = [par for par in parameters_input.index if flavour in par]
    needs_sum_rule = not any("sum" in par or "kappa" in par or "0" in par
                         for par in filtered_params)
    for p in filtered_params:
        in_min = p in parameters_min.index

        # --- Flags setzen ---
        if "sum" in p:
            is_sum = True
        if "kappa" in p:
            is_kappa = True
        if "0" in p:
            is_zero = True

        # --- Typ bestimmen (p, a, b) ---
        ptype = None
        for t in ("p", "a", "b"):
            if t in p:
                ptype = t
                break
        if ptype is None:
            continue

        vals_list, hl_list = type_map[ptype]

        # --- Wert holen ---
        raw = parameters_min[p] if in_min else parameters_input[p]

        vals_list.append(raw)

        # --- Highlighting bestimmen ---
        if highlight == "open":
            hl_list.append(in_min)
        elif isinstance(highlight, dict):
            hl_list.append(ptype in highlight and idx_counter[ptype] in highlight[ptype])
        # else: kein Highlight → Liste bleibt leer

        idx_counter[ptype] += 1

    # --- Sum-Rule einfügen ---
    if needs_sum_rule:
        for vals_list, hl_list in type_map.values():
            vals_list.insert(0, "sum rule")
            if highlight == "open":
                hl_list.insert(0, False)
            elif isinstance(highlight, dict):
                # Dict-Index 0 → sum rule highlighten?
                hl_list.insert(0, ptype in highlight and 0 in highlight[ptype])

    # --- Formatierung ---
    if format_data:
        for vals_list in (vals_p, vals_a, vals_b):
            for i, v in enumerate(vals_list):
                if v == "sum rule":
                    continue
                vals_list[i] = f"${v:.0f}$" if v == 0.0 else f"${v:.{format_data}}$"
    
    # --- Bold-Highlighting (LaTeX) ---
    if highlight:
        for vals_list, hl_list in type_map.values():
            for i, (val, bold) in enumerate(zip(vals_list, hl_list)):
                if bold:
                    if isinstance(val, str) and val.startswith("$") and val.endswith("$"):
                        # Already in math mode → use \mathbf inside $...$
                        inner = val[1:-1]  # strip $ $
                        vals_list[i] = rf"$\mathbf{{{inner}}}$"
                    else:
                        vals_list[i] = rf"\textbf{{{val}}}"


    for col in columns:
        if col == "j":
            if is_sum:
                table[col] = ["sum"]+list(np.arange(1,len(vals_p)))
            elif is_kappa:
                table[col] = ["kappa"]+list(np.arange(1,len(vals_p)))
            elif is_zero:
                table[col] = np.arange(len(vals_p))
            else: 
                table[col] = np.arange(len(vals_p))
        elif col == "p":
            table[col] = vals_p
        elif col == "a":
            table[col] = vals_a
        elif col == "b":
            table[col] = vals_b    



    if flavour=="uv":
        flv_str=r"{u_\mathrm{v}}"
    elif flavour=="dv":
        flv_str=r"{d_\mathrm{v}}"
    elif flavour=="ssb":
        flv_str=r"{s+\bar{s}}"
    elif flavour=="ubdb":
        flv_str=r"{\bar{u}+\bar{d}}"
    elif flavour=="dboub":
        flv_str=r"{\bar{d}/\bar{u}}"
    else:
        flv_str="g"

    labels_default = {
        "j": "$j$",
        "p": rf"$p^{flv_str}$",
        "a": rf"$a^{flv_str}$",
        "b": rf"$b^{flv_str}$",
    }

    labels_updated = update_kwargs(labels_default, labels)

    if column_types is None:
        
        if alignment.startswith("|"):
            column_spec = len(columns) * f"{alignment}" + "|"
        elif alignment.endswith("|"):
            column_spec = "|" + len(columns) * f"{alignment}"
        else:
            column_spec = "|" + len(columns) * f"{alignment}" + "|"
    else:
        if len(column_types) < len(columns):
            raise ValueError("len(column_spec) must not be less than len(columns)")
        else:
            column_types = column_types[: len(columns)]

        column_spec_list = [c for c in column_types]
        column_spec = "".join(column_spec_list)

    table_latex = "\\begin{subtable}[t]{"+f"{width}"+"}\n \\centering \n"
    table_latex += "\\begin{tabular}{" + column_spec + "} \n"

    labels_formatted = [
        format_columns_updated[c].format(labels_updated[c]) for c in columns
    ]

    # add label row
    table_latex += "\\hline\n" + " & ".join(labels_formatted) + " \\\\\n\\hline\n"

    first_row = True
    for row in table.astype(str).itertuples(index=False):
        if needs_sum_rule and first_row:

            table_latex += "$0$ & \\multicolumn{3}{c|}{sum rule} "  + " \\\\\n\\cline{2-4}\n"
            first_row = False
        else:

            table_latex += " & ".join(row) + " \\\\\n"

    table_latex += "\\hline\n"


    table_latex += "\\end{tabular}"
    table_latex += "\n\\end{subtable}%\n"
    # table_latex += "\n\\vspace{0pt}\n"


    return table, table_latex