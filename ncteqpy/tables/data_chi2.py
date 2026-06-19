from __future__ import annotations

import numpy as np
import pandas as pd
from typing_extensions import Hashable, Literal, Mapping, Sequence

from ncteqpy._typing import SequenceNotStr
from ncteqpy.labels import nucleus_to_latex
from ncteqpy.util import update_kwargs

ColumnType = Literal[
    "id_dataset",
    "type_experiment",
    "experiment",
    "collaboration",
    "observable",
    "sqrt_s [TeV]",
    "reference",
    "N_data before cuts",
    "N_data after cuts",
    "N_data before/after cuts",
    "N_data after/before cuts",
    "A",
    "chi2",
    "chi2/point",
    "normalization",
]


def table_data_chi2(
    columns: Sequence[ColumnType],
    datasets_index: pd.DataFrame,
    id_dataset: Sequence[int] | None = None,
    chi2: pd.Series[float] | None = None,
    normalization: pd.Series[float] | None = None,
    num_points_after_cuts: pd.Series[int] | None = None,
    column_types: SequenceNotStr[str] | None = None,
    format_columns: str | list[str | None] | dict[ColumnType, str] | None = None,
    format_total: (
        str | SequenceNotStr[str | None] | Mapping[ColumnType, str | None] | None
    ) = None,
    sort_by: str | SequenceNotStr[str] | None = None,
    sort_ascending: bool | Sequence[bool | None] | None = None,
    sort_order: (
        list[Hashable] | list[tuple[Hashable, ...]] | dict[str, list[Hashable]] | None
    ) = None,
    sparse_columns: Literal["all"] | Sequence[ColumnType] | None = "all",
    hlines: ColumnType | Sequence[ColumnType] | None = None,
    highlight: int | Sequence[int] | None = None,
    labels: dict[ColumnType, str] | None = None,
    title: str | None = None,
    tabular_options: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Generate a data set table as LaTeX `tabular` environment with a row for each data set.

    Parameters
    ----------
    columns : Sequence[ColumnType]
        The columns of the table.
    datasets_index : pd.DataFrame
        Datasets in the format of `nc.Datasets.index`.
    id_dataset : Sequence[int] | None, optional
        IDs of the data sets, by default None, i.e., all IDs in `datasets_index`.
    chi2 : pd.Series[float] | None, optional
        Total χ² of each data set, by default None.
    normalization : pd.Series[float] | None, optional
        Normalization factor for each data set, by default None.
    num_points_after_cuts : pd.Series[int] | None, optional
        Number of points after cuts for each data set, by default None. The number of points before cuts is taken from `datasets_index`.
    column_types : SequenceNotStr[str] | None, optional
        LaTeX column type to be inserted into `\\begin{tabular}{...}`, e.g., `c` or `p{1cm}`, by default None. The default column type that is used is `l`.
    format_columns : str | list[str | None] | dict[ColumnType, str] | None, optional
        Format string for the column labels, by default None.
    format_total : str | SequenceNotStr[str | None] | Mapping[ColumnType, str   None] | None, optional
        Format string for the cells in the row for total values, by default None.
    sort_by : str | SequenceNotStr[str] | None, optional
        Column(s) by which the rows are sorted, by default None, i.e., no sorting.
    sort_ascending : bool | Sequence[bool  |  None] | None, optional
        Sort order of the column(s), by default None, i.e., ascending.
    sort_order : list[Hashable] | list[tuple[Hashable, ...]] | dict[str, list[Hashable]] | None, optional
        Order of the columns, by default None. Not implemented yet.
    sparse_columns : Literal["all"] | Sequence[ColumnType] | None, optional
        Columns in which duplicate adjacent values are to be wrapped with `\\multirow`, by default "all".
    hlines : ColumnType | Sequence[ColumnType] | None, optional
        Column(s) for which a horizontal line is inserted after each value, by default None, i.e., not horizontal lines are inserted. If a column is in `hlines` and `sparse_columns`, the line is inserted after the multi-row cell. The horizontal lines extend from the column(s) only to right. The actual command that is used is `\\cline`.
    highlight : int | Sequence[int] | None, optional
        Data set IDs which are to be bold-faced, by default None, i.e., no highlights.
    labels : dict[str, str] | None, optional
        Labels of the columns, by default None, i.e., default values are chosen for the labels.
    title : str | None, optional
        Title to be inserted in the first row of the table, by default None, i.e., no title.
    tabular_options : str | None, optional
        Options to pass to `\\tabular`, by default None. Not implemented yet.

    Returns
    -------
    table : pd.DataFrame
        The table in DataFrame format.
    table_latex : str
        The table as LaTeX code.
    """
    columns = list(columns)

    datasets_index_filtered = (
        datasets_index[datasets_index["id_dataset"].isin(id_dataset)].copy()
        if id_dataset is not None
        else datasets_index.copy()
    ).set_index("id_dataset", drop=False)

    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        else:
            sort_by = list(sort_by)

        if sort_ascending is not None:
            if isinstance(sort_ascending, bool):
                sort_ascending = [sort_ascending]

            sort_ascending = [(True if s is None else s) for s in sort_ascending]
        else:
            sort_ascending = True

        datasets_index_filtered.reset_index(drop=True, inplace=True)
        datasets_index_filtered.sort_values(
            by=sort_by, ascending=sort_ascending, ignore_index=True, inplace=True
        )

        datasets_index_filtered.set_index("id_dataset", drop=False, inplace=True)

    if num_points_after_cuts is not None:
        # set num_points to 0 for data sets that are in the datasets_index but not known by the chi2 function
        num_points_after_cuts = num_points_after_cuts.align(
            pd.Series(0, index=datasets_index_filtered["id_dataset"]),
            axis=0,
            fill_value=0,
            join="right",
        )[0]

    if chi2 is not None:
        chi2 = chi2.align(
            pd.Series(0, index=datasets_index_filtered["id_dataset"]),
            axis=0,
            join="right",
        )[0]

    if highlight is not None:
        if isinstance(highlight, int):
            highlight = [highlight]
        else:
            highlight = list(highlight)

    merge_data_first_index = None

    if column_types is None:
        column_spec = "|" + len(columns) * r">{\rowmacro}l" + r"<{\clearrowstyle}|"
    else:
        if len(column_types) < len(columns):
            raise ValueError("len(column_spec) must not be less than len(columns)")
        else:
            column_types = column_types[: len(columns)]

        i1 = (
            columns.index("N_data before cuts")
            if "N_data before cuts" in columns
            else None
        )
        i2 = (
            columns.index("N_data after cuts")
            if "N_data after cuts" in columns
            else None
        )

        column_spec_list = [r">{\rowmacro}" + c for c in column_types]

        if i1 is not None and i2 is not None and abs(i1 - i2) == 1:
            merge_data_first_index = min(i1, i2)
            column_spec_list[merge_data_first_index] += r"@{\rowmacro /}"

        column_spec = "|" + "".join(column_spec_list) + r"<{\clearrowstyle}|"

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

    # update format_total
    format_total_default = {c: "{}" for c in columns}
    if format_total is None:
        format_total_updated = format_total_default
    elif isinstance(format_total, list):
        if len(format_total) < len(columns):
            format_total = format_total + (len(columns) - len(format_total)) * [None]

        format_total_updated = {
            c: (f if f is not None else format_total_default[c])
            for f, c in zip(format_total, columns)
        }
    elif isinstance(format_total, dict):
        format_total_updated: dict[str, str] = update_kwargs(
            format_total_default, format_total
        )  # pyright: ignore[reportArgumentType]
    else:
        raise ValueError("format_total must be list, dict or None")

    # DataFrame that represents the table, i.e., containing only strings
    table = pd.DataFrame(
        columns=columns, index=datasets_index_filtered.index, dtype=str
    )

    total: list[str] = []

    # determine the columns of the table and the total row
    for col in columns:
        if col == "id_dataset":
            if highlight is not None:
                highlighted = datasets_index_filtered[col].isin(highlight)
                table[col] = (
                    np.strings.multiply(r"\textbf{", highlighted.astype(int))
                    + datasets_index_filtered[col].astype(str)
                    + np.strings.multiply("}", highlighted.astype(int))
                )
            else:
                table[col] = datasets_index_filtered[col].astype(str)
            total.append("")
        elif col in ("type_experiment", "experiment", "collaboration"):
            table[col] = datasets_index_filtered[col].astype(str)
            total.append("")
        elif col == "observable":
            table[col] = "$" + datasets_index_filtered["observable"] + "$"
            total.append("")
        elif col == "sqrt_s [TeV]":
            table[col] = (datasets_index_filtered["sqrt_s"] / 1000).astype(str)
            total.append("")
        elif col == "N_data before cuts":
            table[col] = datasets_index_filtered["num_points"].astype(str)
            total.append(str(datasets_index_filtered["num_points"].sum()))
        elif col == "N_data after cuts":
            table[col] = (
                num_points_after_cuts[datasets_index_filtered["id_dataset"]].astype(str)
                if num_points_after_cuts is not None
                else ""
            )
            if num_points_after_cuts is not None:
                table[col] = num_points_after_cuts[
                    datasets_index_filtered["id_dataset"]
                ].astype(str)
                total.append(
                    str(
                        num_points_after_cuts[
                            datasets_index_filtered["id_dataset"]
                        ].sum()
                    )
                )
            else:
                table[col] = ""
                total.append("")
        elif col == "N_data before/after cuts":
            table[col] = (
                datasets_index_filtered["num_points"].astype(str)
                + "/"
                + (
                    num_points_after_cuts[datasets_index_filtered["id_dataset"]].astype(
                        str
                    )
                    if num_points_after_cuts is not None
                    else ""
                )
            )
            total.append(
                str(datasets_index_filtered["num_points"].sum())
                + "/"
                + (
                    str(
                        num_points_after_cuts[
                            datasets_index_filtered["id_dataset"]
                        ].sum()
                    )
                    if num_points_after_cuts is not None
                    else ""
                )
            )
        elif col == "N_data after/before cuts":
            table[col] = (
                (
                    num_points_after_cuts[datasets_index_filtered["id_dataset"]].astype(
                        str
                    )
                    if num_points_after_cuts is not None
                    else ""
                )
                + "/"
            ) + datasets_index_filtered["num_points"].astype(str)
            total.append(
                (
                    str(
                        num_points_after_cuts[
                            datasets_index_filtered["id_dataset"]
                        ].sum()
                    )
                    if num_points_after_cuts is not None
                    else ""
                )
                + "/"
                + str(datasets_index_filtered["num_points"].sum())
            )
        elif col == "A":
            table[col] = [
                (
                    "$"
                    + nucleus_to_latex(A=A1, Z=Z1, show_A=True)
                    + "/"
                    + nucleus_to_latex(A=A2, Z=Z2, show_A=True)
                    + "$"
                    if "/" in reaction
                    else "$"
                    + nucleus_to_latex(A=A_heavier, Z=Z_heavier, show_A=True)
                    + "$"
                )
                for A1, Z1, A2, Z2, A_heavier, Z_heavier, reaction in datasets_index_filtered[
                    ["A1", "Z1", "A2", "Z2", "A_heavier", "Z_heavier", "reaction"]
                ].itertuples(
                    index=False, name=None
                )
            ]
            total.append("")
        elif col == "reference":
            table[col] = "\\cite{" + datasets_index_filtered["cite_key"] + "}"
            total.append("")
        elif col == "chi2":
            table[col] = (
                chi2[datasets_index_filtered["id_dataset"]]
                .apply("{:.2f}".format)
                .fillna(r"--\;\;")
                if chi2 is not None
                else ""
            )
            total.append(
                "{:.2f}".format(chi2[datasets_index_filtered["id_dataset"]].sum())
                if chi2 is not None
                else ""
            )
        elif col == "chi2/point":
            table[col] = (
                (
                    chi2[datasets_index_filtered["id_dataset"]]
                    / num_points_after_cuts[datasets_index_filtered["id_dataset"]]
                ).apply(lambda x: "{:.2f}".format(x) if not pd.isna(x) else r"--\;\;")
                if chi2 is not None and num_points_after_cuts is not None
                else ""
            )
            total.append(
                "{:.2f}".format(
                    chi2[datasets_index_filtered["id_dataset"]].sum()
                    / num_points_after_cuts[datasets_index_filtered["id_dataset"]].sum()
                )
                if chi2 is not None and num_points_after_cuts is not None
                else ""
            )
        elif col == "normalization":
            table[col] = (
                normalization[
                    datasets_index_filtered[
                        datasets_index_filtered["id_dataset"].isin(normalization.index)
                    ]["id_dataset"]
                ].apply(lambda x: "{:.2f}".format(x) if not pd.isna(x) else "n/a")
                if normalization is not None
                else ""
            )
            total.append("")

    # deduplicate adjacent row values in each column and replace each first occurrence with a \multirow invocation
    if sparse_columns is not None:
        if sparse_columns == "all":
            all_sparse_columns = [
                "id_dataset",
                "type_experiment",
                "experiment",
                "observable",
                "sqrt_s",
                "reference",
                "A",
            ]

            sparse_columns = [col for col in all_sparse_columns if col in columns]

        sparse_columns = list(sparse_columns)

        sparse_columns.sort(key=columns.index)

        table_dedup = table.copy()

        for i, col in enumerate(sparse_columns):
            if col == "id_dataset":
                continue

            num_occurrences = (
                table[col]
                .groupby(
                    (
                        table[sparse_columns[: (i + 1)]].apply(tuple, axis=1)
                        != table[sparse_columns[: (i + 1)]]
                        .apply(tuple, axis=1)
                        .shift(1)
                    ).cumsum()
                )
                .transform("size")
            )
            mask_all_duplicates = num_occurrences > 1

            mask_last_duplicates = table[sparse_columns[: (i + 1)]].apply(
                tuple, axis=1
            ) == table[sparse_columns[: (i + 1)]].apply(tuple, axis=1).shift(1)
            mask_first_duplicates = mask_all_duplicates & ~mask_last_duplicates

            table_dedup.loc[mask_first_duplicates, col] = (
                "\\multirow{"
                + num_occurrences[mask_first_duplicates].astype(str)
                + "}{*}{"
                + table[col][mask_first_duplicates].astype(str)
                + "}"
            )
            table_dedup.loc[mask_last_duplicates, col] = ""

        table = table_dedup

    labels_default = {
        "id_dataset": "ID",
        "type_experiment": "Process",
        "experiment": "Experiment",
        "collaboration": "Collaboration",
        "observable": "Observable",
        "sqrt_s [TeV]": r"$\sqrt{s}~[\mathrm{TeV}]$",
        "reference": "Ref.",
        "N_data before cuts": r"$N_{\mathrm{data}}$",
        "N_data after cuts": r"$N_{\mathrm{data}}$",
        "N_data before/after cuts": r"$N_{\mathrm{data}}$",
        "N_data after/before cuts": r"$N_{\mathrm{data}}$",
        "A": "Nucleus",
        "chi2": r"$\chi^2$",
        "chi2/point": r"$\chi^2/N_{\mathrm{data}}$",
        "normalization": "Norm.",
    }
    labels_updated = update_kwargs(labels_default, labels)

    table_latex = "\\begin{tabular}{" + column_spec + "}\n"

    # add title row
    if title is not None:
        table_latex += (
            "\\hline\n\\multicolumn{"
            + str(len(columns))
            + "}{|c|}{"
            + title
            + "}\\\\\n\\hline\n"
        )

    labels_formatted = [
        format_columns_updated[c].format(labels_updated[c]) for c in columns
    ]

    if merge_data_first_index is not None:
        labels_formatted[merge_data_first_index] = (
            r"\multicolumn{2}{c}{" + labels_formatted[merge_data_first_index] + "}"
        )
        labels_formatted.pop(merge_data_first_index + 1)

    # add label row
    table_latex += "\\hline\n" + " & ".join(labels_formatted) + " \\\\\n\\hline\n"

    if hlines is not None:
        if isinstance(hlines, str):
            hlines = [hlines]
        else:
            hlines = list(hlines)

        hlines.sort(key=columns.index)

    # for the columns in `hlines`, add a horizontal line after the deduplicated values
    first_row = True
    for row in table.astype(str).itertuples(index=False):
        if not first_row and hlines is not None:
            for hline in hlines:
                index = columns.index(hline)
                if row[index] != "":
                    table_latex += (
                        "\\cline{"
                        + str(columns.index(hline) + 1)
                        + "-"
                        + str(len(columns))
                        + "}\n"
                    )
                    break

        first_row = False

        table_latex += " & ".join(row) + " \\\\\n"

    table_latex += "\\hline\n"

    # add row for total

    total_formatted = [
        format_total_updated[c].format(t) if t != "" else ""
        for c, t in zip(columns, total)
    ]

    if merge_data_first_index is not None:
        total_formatted[merge_data_first_index] = (
            r"\!" + total_formatted[merge_data_first_index]
        )

    table_latex += (
        "\\hline\n\\setrowstyle{\\bfseries}\nTotal"
        + " & ".join(total_formatted)
        + "\\\\\n\\hline\n"
    )

    table_latex += "\\end{tabular}"

    return table, table_latex
