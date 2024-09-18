from __future__ import annotations

from typing import Any, Literal, Sequence

import matplotlib.cm as m_cm
import matplotlib.colors as m_colors
import matplotlib.pyplot as plt
import matplotlib.transforms as m_transforms
import numpy as np
import pandas as pd
from matplotlib.container import ErrorbarContainer
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class AvoidingLegend(Legend):

    avoid: list[m_transforms.BboxBase]

    def __init__(
        self, parent, handles, labels, avoid: list[m_transforms.BboxBase], **kwargs: Any
    ):
        super().__init__(parent, handles, labels, **kwargs)
        self.avoid = avoid

    def _auto_legend_data(self) -> Any:
        bboxes, lines, offsets = super()._auto_legend_data()
        bboxes.extend(self.avoid)
        return bboxes, lines, offsets


def _update_kwargs(
    kwargs: dict[str, Any],
    kwargs_user: dict[str, Any] | list[dict[str, Any] | None],
    i: int | None = None,
) -> dict[str, Any]:
    if isinstance(kwargs_user, dict):
        return kwargs | kwargs_user
    elif isinstance(kwargs_user, list):
        if i is not None:
            if i < len(kwargs_user) and kwargs_user[i] is not None:
                return kwargs | kwargs_user[i]  # type: ignore[operator] # this is correct but mypy complains
            else:
                return kwargs
        else:
            raise ValueError("i must be given if kwargs_user is list")
    else:
        raise ValueError("kwargs_user must be dict or list")


def plot(
    type_experiment: str,
    data: pd.DataFrame | None = None,
    theory: pd.DataFrame | None = None,
    kinematic_variable: str | None = None,
    ax: plt.Axes | None = None,
    xlabel: Literal["fallback"] | str | dict[str, str] | None = "fallback",
    ylabel: Literal["fallback"] | str | None = "fallback",
    title: str | None = None,
    legend: bool = True,
    chi2_label: bool = True,
    kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_xlabel: dict[str, Any] = {},
    kwargs_ylabel: dict[str, Any] = {},
    **kwargs: Any,
) -> None:

    match type_experiment:
        case "DIS":
            plot_DIS(
                data=data,
                theory=theory,
                kinematic_variable=kinematic_variable,
                ax=ax,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                legend=legend,
                chi2_label=chi2_label,
                kwargs_data=kwargs_data,
                kwargs_theory=kwargs_theory,
                kwargs_xlabel=kwargs_xlabel,
                kwargs_ylabel=kwargs_ylabel,
                **kwargs,
            )

        case "HQ" | "OPENHEAVY" | "QUARKONIA":
            plot_HQ(
                data=data,
                theory=theory,
                kinematic_variable=kinematic_variable,
                ax=ax,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                legend=legend,
                chi2_label=chi2_label,
                kwargs_data=kwargs_data,
                kwargs_theory=kwargs_theory,
                kwargs_xlabel=kwargs_xlabel,
                kwargs_ylabel=kwargs_ylabel,
                **kwargs,
            )


def plot_basic(
    x: str | Sequence[str],  # sequence for binned variables
    y: str,
    ax: plt.Axes,
    data: pd.DataFrame | None = None,
    theory: pd.DataFrame | None = None,
    xlabel: Literal["fallback"] | str | None = "fallback",
    ylabel: Literal["fallback"] | str | None = "fallback",
    title: str | None = None,
    legend: bool = True,
    chi2_label: bool = True,
    kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_xlabel: dict[str, Any] = {},
    kwargs_ylabel: dict[str, Any] = {},
    kwargs_title: dict[str, Any] = {},
    kwargs_annotate_chi2: dict[str, Any] = {},
) -> None:
    if data is not None:
        kwargs_default: dict[str, Any] = {
            "capsize": 2,
            "marker": ".",
            "markersize": 6,
            "ls": "",
            "color": "black",
            "label": "Data",
        }
        kwargs = _update_kwargs(kwargs_default, kwargs_data)
        ax.errorbar(x=data[x], y=data[y], **kwargs)

    if theory is not None:
        kwargs_default = {"label": "Theory"}
        kwargs = _update_kwargs(kwargs_default, kwargs_data)
        ax.plot(x=theory[x], y=theory[y], **kwargs)


def plot_DIS(
    data: pd.DataFrame | None = None,
    theory: pd.DataFrame | None = None,
    x_variable: str = "fallback",
    y_variable: str = "fallback",
    ax: plt.Axes | None = None,
    xlabel: Literal["fallback"] | str | dict[str, str] | None = "fallback",
    ylabel: Literal["fallback"] | str | None = "fallback",
    title: str | None = None,
    legend: bool = True,
    curve_label: (
        Literal[
            "annotate above",
            "annotate right",
            "ticks",
            "colorbar",
            "legend",
        ]
        | None
    ) = "annotate above",
    chi2_label: bool = True,
    curve_groupby: str | Sequence[str] = "Q2",
    kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_xlabel: dict[str, Any] = {},
    kwargs_ylabel: dict[str, Any] = {},
    kwargs_annotate_chi2: dict[str, Any] = {},
    kwargs_annotate_curves: dict[str, Any] = {},
    **kwargs: Any,
) -> None:
    if ax is None:
        ax = plt.gca()

    legend_handles: list[Line2D | ErrorbarContainer] = []

    if x_variable == "fallback":
        x_variable = "x"

    if y_variable == "fallback":
        y_variable = "ratio_sigma"

    if data is not None:
        for i, (Q2_i, data_i) in enumerate(data.groupby(curve_groupby, sort=True)):
            data_i.sort_values(x_variable, inplace=True)

            kwargs_default: dict[str, Any] = {
                "capsize": 2,
                "marker": ".",
                "markersize": 6,
                "ls": "",
                "color": "black",
            }
            if "unc_tot" in data.columns:
                kwargs_default["yerr"] = data_i["unc_tot"]

            if legend and i == 0:
                kwargs_default["label"] = "Data"

            kwargs = _update_kwargs(
                kwargs_default, kwargs_data, i
            )  # TODO: if kwargs_data is dict, the label should only be set once

            e = ax.errorbar(
                data_i[x_variable],
                data_i[y_variable],
                **kwargs,
            )

            if legend and i == 0:
                legend_handles.append(e)

    ticks = []
    tick_labels = []

    if theory is not None:
        kwargs_index = 0
        for i, (Q2_i, theo_i) in enumerate(
            gb := theory.groupby(curve_groupby, sort=True)
        ):
            theo_i.sort_values(x_variable, inplace=True)

            kwargs_default = {}

            if legend and kwargs_index == 0:
                kwargs = _update_kwargs(
                    kwargs_default
                    | {
                        "color": (
                            "black"
                            if len(gb) > 1 or curve_label is not None
                            else plt.rcParams["axes.prop_cycle"].by_key()["color"][
                                0
                            ]  # we have to set the color manually so that the dummy call to ax.plot for the legend does not advance the prop cycle
                        ),
                        "label": "Theory",
                    },
                    kwargs_theory,
                    kwargs_index,
                )
                kwargs_index += 1

                l = ax.plot([], [], **kwargs)

                legend_handles.append(l[0])

            label_curve = (
                f"${Q2_i} = {xlabel.get('x_variable', 'x').replace('$', '') if isinstance(xlabel, dict) else 'x'}$"
                # + (f" $(\\times 10^{{{i}}})$" if len(gb) > 1 else "")
            )

            if curve_label is not None and curve_label == "legend":
                label = {"label": label_curve}
            else:
                label = {}

            print(label_curve)

            kwargs = _update_kwargs(kwargs_default | label, kwargs_theory, kwargs_index)

            l = ax.plot(
                theo_i[x_variable],
                theo_i["theory"],
                **kwargs,
            )

            matched_data_i = (
                data.query(f"abs({curve_groupby} - @Q2_i) < 1e-6").sort_values(
                    x_variable
                )
                if data is not None
                else None
            )

            if curve_label is not None:
                if curve_label == "legend":
                    legend_handles.append(l[0])
                elif curve_label == "annotate above":
                    kwargs = {
                        "xytext": (0, 0.3),
                        "textcoords": "offset fontsize",
                        "ha": "right",
                    } | kwargs_annotate_curves

                    ax.annotate(
                        label_curve,
                        (theo_i[x_variable].iloc[-1], theo_i["theory"].iloc[-1]),
                        **kwargs,
                    )
                elif curve_label == "annotate right":
                    kwargs = {
                        "xytext": (0.3, 0),
                        "textcoords": "offset fontsize",
                        "ha": "left",
                        "va": "center",
                    } | kwargs_annotate_curves

                    if data_i is not None:
                        pos = (
                            theo_i["theory"].iloc[-1] + data_i[y_variable].iloc[-1]
                        ) / 2
                    else:
                        pos = theo_i["theory"].iloc[-1]

                    ax.annotate(
                        label_curve,
                        (theo_i[x_variable].iloc[-1], pos),
                        **kwargs,
                    )
                elif curve_label == "ticks":
                    if data_i is not None:
                        pos = (
                            theo_i["theory"].iloc[-1] + data_i[y_variable].iloc[-1]
                        ) / 2
                    else:
                        pos = theo_i["theory"].iloc[-1]

                    ticks.append(pos)
                    tick_labels.append(label_curve)

            if chi2_label and "chi2" in theo_i.columns:
                if data_i is not None:
                    pos = (
                        np.maximum(
                            theo_i["theory"].to_numpy(),
                            (data_i[y_variable] + data_i["unc_tot"]).to_numpy(),
                        )
                        * 10**i
                    )
                else:
                    pos = theo_i["theory"] * 10**i

                kwargs = {
                    "xytext": (0, 0.25),
                    "textcoords": "offset fontsize",
                    "ha": "center",
                } | kwargs_annotate_chi2

                for chi2_i, pos_i, pT_i in zip(theo_i["chi2"], pos, data_i[x_variable]):
                    ax.annotate(f"{chi2_i:.1f}", (pT_i, pos_i), **kwargs)

            kwargs_index += 1

        if curve_label == "colorbar":
            colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
            cmap = m_colors.LinearSegmentedColormap.from_list(
                "cmap", colors, N=len(colors)
            )
            norm = m_colors.BoundaryNorm(
                theory[y_variable],  # TODO
                ncolors=len(gb),
            )
            c = plt.colorbar(mappable=m_cm.ScalarMappable(norm, cmap), ax=ax)

            for i in range(len(gb)):
                c.ax.annotate(
                    f"$\\times 10^{i}$",
                    (0.5, i / 8 + 1 / 16),
                    xycoords="axes fraction",
                    xytext=(0.15, 0.0),
                    textcoords="offset fontsize",
                    rotation="vertical",
                    rotation_mode="anchor",
                    ha="center",
                    va="center",
                )
            if xlabel is not None:
                if isinstance(xlabel, dict):
                    c.set_label(xlabel["y_min"])
                else:
                    c.set_label("y")

    if xlabel is not None:
        if isinstance(xlabel, str):
            if xlabel == "fallback":
                ax.set_xlabel(x_variable, **kwargs_xlabel)
            else:
                ax.set_xlabel(xlabel, **kwargs_xlabel)
        elif isinstance(xlabel, dict):
            ax.set_xlabel(xlabel[x_variable], **kwargs_xlabel)
        else:
            raise ValueError(f"xlabel must be str or dict but given was {type(xlabel)}")

    if ylabel is not None:
        if isinstance(ylabel, str):
            if ylabel == "fallback":
                ax.set_ylabel(
                    "",
                    **kwargs_ylabel,
                )
            else:
                ax.set_ylabel(ylabel, **kwargs_ylabel)
        else:
            raise ValueError(f"ylabel must be str or dict but given was {type(ylabel)}")

    if title is not None:
        ax.set_title(title, fontsize="medium")  # TODO: kwargs for the title

    ax.set_yscale("log")
    ax.grid()

    if legend or bin_label is not None and bin_label == "legend":
        leg = ax.legend(handles=legend_handles, **kwargs_legend)

    plt.draw()

    if chi2_legend and theory is not None and "chi2" in theory.columns:
        bbox = leg.get_window_extent()
        leg2 = AvoidingLegend(
            ax,
            handles=[Patch(), Patch(), Patch()],
            labels=[
                f"$N_{{\\text{{points}}}} = {len(theory)}$",
                f"$\\chi^2_{{\\text{{total}}}} = {theory['chi2'].sum():.3f}$",
                f"$\\chi^2_{{\\text{{total}}}}\\,/\\, N_{{\\text{{points}}}} = {theory['chi2'].sum() / len(theory):.3f}$",
            ],
            avoid=[bbox],
            labelspacing=0,
            handlelength=0,
            handleheight=0,
            handletextpad=0,
        )
        ax.add_artist(leg2)

    # for some reason the ticks don't work if we set them at an earlier stage
    if bin_label == "ticks" and ticks and tick_labels:
        ax2 = ax.twinx()
        ax2.set_yticks(ticks=ticks, labels=tick_labels)
        ax2.set_ybound(*ax.get_ybound())
        ax2.set_yscale(ax.get_yscale())
        ax2.set_yticks(ticks=ticks, labels=tick_labels)


def plot_HQ(
    data: pd.DataFrame | None = None,
    theory: pd.DataFrame | None = None,
    kinematic_variable: str | None = None,
    ax: plt.Axes | None = None,
    xlabel: Literal["fallback"] | str | dict[str, str] | None = "fallback",
    ylabel: Literal["fallback"] | str | None = "fallback",
    title: str | None = None,
    legend: bool = True,
    chi2_label: bool = True,
    chi2_legend: bool = True,
    bin_label: (
        Literal[
            "annotate above",
            "annotate right",
            "ticks",
            "colorbar",
            "legend",
        ]
        | None
    ) = "annotate above",
    kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_xlabel: dict[str, Any] = {},
    kwargs_ylabel: dict[str, Any] = {},
    kwargs_legend: dict[str, Any] = {},
    kwargs_annotate_bins: dict[str, Any] = {},
    kwargs_annotate_chi2: dict[str, Any] = {},
    kwargs_annotate_colorbar: dict[
        str, Any
    ] = {},  # TODO: better kwargs for the bin_label options # TODO define overloads for the different bin_label possibilites and the respective kwargs
    kwargs_label_colorbar: dict[str, Any] = {},
) -> None:
    if ax is None:
        ax = plt.gca()

    legend_handles = []

    if data is not None:
        for i, ((y_min, y_max), data_y) in enumerate(
            data.groupby(["y_min", "y_max"], sort=True)
        ):
            data_y.sort_values(["pT_min", "pT_max"], inplace=True)

            kwargs_default: dict[str, Any] = {
                "xerr": (data_y["pT_max"] - data_y["pT_min"]) / 2,
                "capsize": 2,
                "marker": ".",
                "markersize": 6,
                "ls": "",
                "color": "black",
            }
            if "unc_tot" in data.columns:
                kwargs_default["yerr"] = data_y["unc_tot"] * 10**i

            if legend and i == 0:
                kwargs_default["label"] = "Data"

            kwargs = _update_kwargs(
                kwargs_default, kwargs_data, i
            )  # TODO: if kwargs_data is dict, the label should only be set once

            e = ax.errorbar(
                (data_y["pT_min"] + data_y["pT_max"]) / 2,
                data_y["sigma"] * 10**i,
                **kwargs,
            )

            if legend and i == 0:
                legend_handles.append(e)

    ticks = []
    tick_labels = []

    if theory is not None:
        kwargs_index = 0
        for i, ((y_min, y_max), theo_y) in enumerate(
            gb := theory.groupby(["y_min", "y_max"], sort=True)
        ):
            theo_y.sort_values(["pT_min", "pT_max"], inplace=True)

            kwargs_default = {}

            if legend and kwargs_index == 0:
                kwargs = _update_kwargs(
                    kwargs_default
                    | {
                        "color": (
                            "black"
                            if len(gb) > 1 or bin_label is not None
                            else plt.rcParams["axes.prop_cycle"].by_key()["color"][
                                0
                            ]  # we have to set the color manually so that the dummy call to ax.plot for the legend does not advance the prop cycle
                        ),
                        "label": "Theory",
                    },
                    kwargs_theory,
                    kwargs_index,
                )
                kwargs_index += 1

                l = ax.plot([], [], **kwargs)

                legend_handles.append(l[0])

            label_rapidity = (
                f"${y_min} < {xlabel.get('y_min', 'y').replace('$', '') if isinstance(xlabel, dict)else 'y'} < {y_max}$"
                + (f" $(\\times 10^{{{i}}})$" if len(gb) > 1 else "")
            )

            if bin_label is not None and bin_label == "legend":
                label = {"label": label_rapidity}
            else:
                label = {}

            kwargs = _update_kwargs(kwargs_default | label, kwargs_theory, kwargs_index)

            l = ax.plot(
                np.append(theo_y["pT_min"].iloc[0], theo_y["pT_max"]),
                np.append(theo_y["theory"], theo_y["theory"].iloc[-1]) * 10**i,
                drawstyle="steps-post",
                **kwargs,
            )

            data_y = (
                data.query(
                    "abs(y_min - @y_min) < 1e-6 and abs(y_max - @y_max) < 1e-6"
                ).sort_values(["pT_min", "pT_max"])
                if data is not None
                else None
            )

            if bin_label is not None:
                if bin_label == "legend":
                    legend_handles.append(l[0])
                elif bin_label == "annotate above":
                    kwargs = {
                        "xytext": (0, 0.3),
                        "textcoords": "offset fontsize",
                        "ha": "right",
                    } | kwargs_annotate_bins

                    ax.annotate(
                        label_rapidity,
                        (theo_y["pT_max"].iloc[-1], theo_y["theory"].iloc[-1] * 10**i),
                        **kwargs,
                    )
                elif bin_label == "annotate right":
                    kwargs = {
                        "xytext": (0.3, 0),
                        "textcoords": "offset fontsize",
                        "ha": "left",
                        "va": "center",
                    } | kwargs_annotate_bins

                    if data_y is not None:
                        pos = (
                            (theo_y["theory"].iloc[-1] + data_y["sigma"].iloc[-1])
                            / 2
                            * 10**i
                        )
                    else:
                        pos = theo_y["theory"].iloc[-1] * 10**i

                    ax.annotate(
                        label_rapidity,
                        (theo_y["pT_max"].iloc[-1], pos),
                        **kwargs,
                    )
                elif bin_label == "ticks":
                    if data_y is not None:
                        pos = (
                            (theo_y["theory"].iloc[-1] + data_y["sigma"].iloc[-1])
                            / 2
                            * 10**i
                        )
                    else:
                        pos = theo_y["theory"].iloc[-1] * 10**i

                    ticks.append(pos)
                    tick_labels.append(label_rapidity)

            if chi2_label and "chi2" in theo_y.columns:
                if data_y is not None:
                    pos = (
                        np.maximum(
                            theo_y["theory"].to_numpy(),
                            (data_y["sigma"] + data_y["unc_tot"]).to_numpy(),
                        )
                        * 10**i
                    )
                else:
                    pos = theo_y["theory"] * 10**i

                kwargs = {
                    "xytext": (0, 0.25),
                    "textcoords": "offset fontsize",
                    "ha": "center",
                } | kwargs_annotate_chi2

                for chi2_i, pos_i, pT_i in zip(
                    theo_y["chi2"], pos, (data_y["pT_min"] + data_y["pT_max"]) / 2
                ):
                    ax.annotate(f"{chi2_i:.1f}", (pT_i, pos_i), **kwargs)

            kwargs_index += 1

        if bin_label == "colorbar":
            colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
            cmap = m_colors.LinearSegmentedColormap.from_list(
                "cmap", colors, N=len(colors)
            )
            norm = m_colors.BoundaryNorm(
                np.append(
                    theory["y_min"].min(), theory["y_max"].sort_values().unique()
                ),
                ncolors=len(gb),
            )
            c = plt.colorbar(mappable=m_cm.ScalarMappable(norm, cmap), ax=ax)

            for i in range(len(gb)):
                c.ax.annotate(
                    f"$\\times 10^{i}$",
                    (0.5, i / 8 + 1 / 16),
                    xycoords="axes fraction",
                    xytext=(0.15, 0.0),
                    textcoords="offset fontsize",
                    rotation="vertical",
                    rotation_mode="anchor",
                    ha="center",
                    va="center",
                )
            if xlabel is not None:
                if isinstance(xlabel, dict):
                    c.set_label(xlabel["y_min"])
                else:
                    c.set_label("y")

    if xlabel is not None:
        if isinstance(xlabel, str):
            if xlabel == "fallback":
                ax.set_xlabel("$p_{\\rm T}$", **kwargs_xlabel)
            else:
                ax.set_xlabel(xlabel, **kwargs_xlabel)
        elif isinstance(xlabel, dict):
            ax.set_xlabel(xlabel["pT_min"], **kwargs_xlabel)
        else:
            raise ValueError(f"xlabel must be str or dict but given was {type(xlabel)}")

    if ylabel is not None:
        if isinstance(ylabel, str):
            if ylabel == "fallback":
                ax.set_ylabel(
                    "$\\dfrac{\\mathrm{d}^2 \\sigma}{\\mathrm{d}p_{\\rm T}\\,\\mathrm{d}y}$",
                    **kwargs_ylabel,
                )
            else:
                ax.set_ylabel(ylabel, **kwargs_ylabel)
        else:
            raise ValueError(f"ylabel must be str or dict but given was {type(ylabel)}")

    if title is not None:
        ax.set_title(title, fontsize="medium")  # TODO: kwargs for the title

    ax.set_yscale("log")
    ax.grid()

    if legend or bin_label is not None and bin_label == "legend":
        leg = ax.legend(handles=legend_handles, **kwargs_legend)

    plt.draw()

    if chi2_legend and theory is not None and "chi2" in theory.columns:
        bbox = leg.get_window_extent()
        leg2 = AvoidingLegend(
            ax,
            handles=[Patch(), Patch(), Patch()],
            labels=[
                f"$N_{{\\text{{points}}}} = {len(theory)}$",
                f"$\\chi^2_{{\\text{{total}}}} = {theory['chi2'].sum():.3f}$",
                f"$\\chi^2_{{\\text{{total}}}}\\,/\\, N_{{\\text{{points}}}} = {theory['chi2'].sum() / len(theory):.3f}$",
            ],
            avoid=[bbox],
            labelspacing=0,
            handlelength=0,
            handleheight=0,
            handletextpad=0,
        )
        ax.add_artist(leg2)

    # for some reason the ticks don't work if we set them at an earlier stage
    if bin_label == "ticks" and ticks and tick_labels:
        ax2 = ax.twinx()
        ax2.set_yticks(ticks=ticks, labels=tick_labels)
        ax2.set_ybound(*ax.get_ybound())
        ax2.set_yscale(ax.get_yscale())
        ax2.set_yticks(ticks=ticks, labels=tick_labels)
