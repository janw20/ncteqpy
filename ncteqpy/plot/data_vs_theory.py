from __future__ import annotations

from itertools import zip_longest

import matplotlib.artist as martist
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas.core.groupby import DataFrameGroupBy
from typing_extensions import Any, Literal, Sequence, cast

import ncteqpy.labels as nc_labels
from ncteqpy.plot.util import AdditionalLegend
from ncteqpy.util import update_kwargs


def plot(
    type_experiment: str,
    ax: plt.Axes | Sequence[plt.Axes],
    points: pd.DataFrame,
    x_variable: str | list[str] | Literal["fallback"] | None = "fallback",
    xlabel: str | dict[str, str] | Literal["fallback"] | None = "fallback",
    ylabel: str | Literal["fallback"] | None = "fallback",
    xscale: str | None = None,
    yscale: str | None = None,
    title: str | None = None,
    legend: bool = True,
    curve_label: (
        Literal[
            "annotate above",
            "annotate right",
            "ticks",
            "legend",
        ]
        | None
    ) = "ticks",
    subplot_label: Literal["legend"] | None = None,
    subplot_label_format: str | None = None,
    chi2_annotation: bool = True,
    chi2_legend: bool = True,
    curve_groupby: str | list[str] | Literal["fallback"] | None = "fallback",
    apply_normalization: bool = True,
    theory_min_width: float = 0.06,
    plot_pdf_uncertainty: bool = True,
    pdf_uncertainty_convention: Literal["sym", "asym"] = "asym",
    y_offset_add: float | None = None,
    y_offset_mul: float | None = None,
    kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_theory_unc: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_xlabel: dict[str, Any] = {},
    kwargs_ylabel: dict[str, Any] = {},
    kwargs_title: dict[str, Any] = {},
    kwargs_legend: dict[str, Any] = {},
    kwargs_legend_chi2: dict[str, Any] = {},
    kwargs_legend_curves: dict[str, Any] = {},
    kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_annotate_chi2: dict[str, Any] = {},
    kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
    **kwargs: Any,
) -> None:
    """Plot data vs. theory for each process. Passing "fallback" for parameters that allow it sets them automatically.

    Parameters
    ----------
    type_experiment : str
        The process the points belong to.
    ax : plt.Axes | Sequence[plt.Axes]
        Axes to plot on. If more than one axes are given, a ratio plot is drawn on the second one.
    points : pd.DataFrame
        The points to be plotted. Needs to have the columns `x_variable`, `"data"` (to plot the data points) and `"theory"` or `"theory_with_normalization"` to plot the theory curves (normalization-uncorrected or corrected, respectively).
    x_variable : str | list[str] | None, optional
        The kinematic variable to put on the x axis, by default "fallback". Plots a binned distribution if a list is passed.
    xlabel : str | dict[str, str] | Literal["fallback"] | None, optional
        x label of the plot, by default None.
    ylabel : str | Literal["fallback"] | None, optional
        y label of the plot, by default None.
    xscale : str | None, optional
        x scale of the plot, by default no changing of x scale.
    yscale : str | None, optional
        y scale of the plot, by default no changing of y scale.
    title : str | None, optional
        Title of the plot, by default None.
    legend : bool, optional
        If a legend annotating data & theory is shown, by default True.
    curve_label : Literal["annotate above", "annotate right", "ticks", "legend"] | None, optional
        Where the curve labels are shown, by default "ticks". If None, no labels are shown.
    subplot_label : Literal["legend"] | None, optional  # FIXME
        Where to label the subplot, by default None.
    subplot_label_format : str | None, optional
        Format of the subplot label, by default None.
    chi2_annotation : bool, optional
        If the χ²/point value is annotated, by default True.
    chi2_legend : bool, optional
        If a legend with the total χ² is shown, by default True.
    curve_groupby : str | list[str] | Literal["fallback"] | None, optional
        Variable(s) to group the curves by.
    apply_normalization : bool, optional
        If the normalization-corrected theory is plotted, by default True.
    theory_min_width : float, optional
        Width of the theory curve (in units of axes fraction) if there is only one point, by default 0.06.
    plot_pdf_uncertainty : bool, optional
        If the PDF uncertainty is plotted around the theory curve, by default True.
    pdf_uncertainty_convention : Literal["sym", "asym"], optional
        If the PDF uncertainties should be symmetric ("sym") or asymmetric ("asym"), by default "asym".
    y_offset_add : float | None, optional
        Additive offset by which the curves are separated, by default 0 (no offset).
    y_offset_mul : float | None, optional
        Multiplicative offset by which the curves are separated, by default 0 (no offset). The factor between each curve is `10**y_offset_mul`.
    kwargs_data : dict[str, Any] | list[dict[str, Any] | None], optional
        Keyword arguments to pass to `plt.Axes.plot` for plotting the data.
    kwargs_theory : dict[str, Any] | list[dict[str, Any]  |  None], optional
        Keyword arguments to pass to `plt.Axes.plot` for plotting the theory.
    kwargs_theory_unc : dict[str, Any] | list[dict[str, Any]  |  None], optional
        Keyword arguments to pass to `plt.Axes.fill_between` for plotting the PDF uncertainty.
    kwargs_xlabel : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.set_xlabel`.
    kwargs_ylabel : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.set_ylabel`.
    kwargs_title : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.set_title`.
    kwargs_legend : dict[str, Any], optional
        Keyword arguments to pass to `AdditionalLegend` for annotating data & theory.
    kwargs_legend_chi2 : dict[str, Any], optional
        Keyword arguments to pass to `AdditionalLegend` for annotating the total χ².
    kwargs_legend_curves : dict[str, Any], optional
        Keyword arguments to pass to `AdditionalLegend` for annotating the curve labels.
    kwargs_ticks_curves : dict[str, Any] | list[dict[str, Any]  |  None], optional
        Keyword arguments to pass to `plt.Axes.set_yticks` for annotating the curve labels.
    kwargs_annotate_chi2 : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.annotate` for annotating the χ²/point values.
    kwargs_annotate_curves : dict[str, Any] | list[dict[str, Any] | None], optional
        Keyword arguments to pass to `plt.Axes.annotate` for annotating the curve labels.
    """

    match type_experiment:
        case "DIS" | "DY":
            if x_variable == "fallback":
                x_variable = "x"

            if y_offset_add is None:
                y_offset_add = 0.5

            if y_offset_mul is None:
                y_offset_mul = 0

        case "DISNEU":
            if x_variable == "fallback":
                x_variable = "Q2"
                if xscale is None:
                    xscale = "log"

            if curve_groupby == "fallback":
                curve_groupby = "x"

            if y_offset_add is None:
                y_offset_add = 0.5

            if y_offset_mul is None:
                y_offset_mul = 0

        case "DISDIMU":
            if x_variable == "fallback":
                x_variable = "x"

            if curve_groupby == "fallback":
                curve_groupby = "y"

            if y_offset_add is None:
                y_offset_add = 0.5

            if y_offset_mul is None:
                y_offset_mul = 0

        case "HQ" | "OPENHEAVY" | "QUARKONIA":
            if x_variable == "fallback":
                x_variable = ["pT_min", "pT_max"]

            if curve_groupby == "fallback":
                curve_groupby = ["y_min", "y_max"]

            if yscale is None:
                yscale = "log"

            if y_offset_add is None:
                y_offset_add = 0

            if y_offset_mul is None:
                y_offset_mul = 1

        case "SIH":
            if x_variable == "fallback":
                x_variable = "pT"

            if yscale is None:
                yscale = "log"

            if y_offset_add is None:
                y_offset_add = 0

            if y_offset_mul is None:
                y_offset_mul = 1

        case "WZPROD":
            if x_variable == "fallback":
                x_variable = "eta"

            if y_offset_add is None:
                y_offset_add = 0

            if y_offset_mul is None:
                y_offset_mul = 1

    assert x_variable is not None

    if curve_groupby == "fallback":
        curve_groupby = None

    if y_offset_add is None:
        y_offset_add = 0

    if y_offset_mul is None:
        y_offset_mul = 0

    plot_common(
        ax=ax,
        points=points,
        x_variable=x_variable,
        xlabel=xlabel,
        ylabel=ylabel,
        xscale=xscale,
        yscale=yscale,
        title=title,
        legend=legend,
        curve_label=curve_label,
        subplot_label=subplot_label,
        subplot_label_format=subplot_label_format,
        chi2_annotation=chi2_annotation,
        chi2_legend=chi2_legend,
        curve_groupby=curve_groupby,
        apply_normalization=apply_normalization,
        theory_min_width=theory_min_width,
        plot_pdf_uncertainty=plot_pdf_uncertainty,
        pdf_uncertainty_convention=pdf_uncertainty_convention,
        y_offset_add=y_offset_add,
        y_offset_mul=y_offset_mul,
        kwargs_data=kwargs_data,
        kwargs_theory=kwargs_theory,
        kwargs_theory_unc=kwargs_theory_unc,
        kwargs_xlabel=kwargs_xlabel,
        kwargs_ylabel=kwargs_ylabel,
        kwargs_title=kwargs_title,
        kwargs_legend=kwargs_legend,
        kwargs_legend_chi2=kwargs_legend_chi2,
        kwargs_legend_curves=kwargs_legend_curves,
        kwargs_ticks_curves=kwargs_ticks_curves,
        kwargs_annotate_chi2=kwargs_annotate_chi2,
        kwargs_annotate_curves=kwargs_annotate_curves,
        **kwargs,
    )


def plot_common(
    ax: plt.Axes | Sequence[plt.Axes],
    points: pd.DataFrame,
    x_variable: str | list[str],
    xlabel: str | None = None,
    ylabel: str | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    title: str | None = None,
    legend: bool = True,
    curve_label: (
        Literal[
            "annotate above",
            "annotate right",
            "ticks",
            "legend",
        ]
        | None
    ) = "ticks",
    subplot_label: Literal["legend"] | None = None,
    subplot_label_format: str | None = None,
    chi2_annotation: bool = True,
    chi2_legend: bool = True,
    curve_groupby: str | list[str] | None = None,
    apply_normalization: bool = True,
    theory_min_width: float = 0.06,
    plot_pdf_uncertainty: bool = True,
    pdf_uncertainty_convention: Literal["sym", "asym"] = "asym",
    y_offset_add: float = 0,
    y_offset_mul: float = 0,
    kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_theory_unc: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_xlabel: dict[str, Any] = {},
    kwargs_ylabel: dict[str, Any] = {},
    kwargs_title: dict[str, Any] = {},
    kwargs_legend: dict[str, Any] = {},
    kwargs_legend_chi2: dict[str, Any] = {},
    kwargs_legend_curves: dict[str, Any] = {},
    kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_annotate_chi2: dict[str, Any] = {},
    kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
) -> None:
    """Plot data vs. theory in a way that is not experiment-specific.

    Parameters
    ----------
    ax : plt.Axes | Sequence[plt.Axes]
        Axes to plot on. If more than one axes are given, a ratio plot is drawn on the second one.
    points : pd.DataFrame
        The points to be plotted. Needs to have the columns `x_variable`, `"data"` (to plot the data points) and `"theory"` or `"theory_with_normalization"` to plot the theory curves (normalization-uncorrected or corrected, respectively).
    x_variable : str | list[str]
        The kinematic variable to put on the x axis. Plots a binned distribution if a list is passed.
    xlabel : str | None, optional
        x label of the plot, by default None.
    ylabel : str | None, optional
        y label of the plot, by default None.
    xscale : str | None, optional
        x scale of the plot, by default no changing of x scale.
    yscale : str | None, optional
        y scale of the plot, by default no changing of y scale.
    title : str | None, optional
        Title of the plot, by default None.
    legend : bool, optional
        If a legend annotating data & theory is shown, by default True.
    curve_label : Literal["annotate above", "annotate right", "ticks", "legend"] | None, optional
        Where the curve labels are shown, by default "ticks". If None, no labels are shown.
    subplot_label : Literal["legend"] | None, optional  # FIXME
        Where to label the subplot, by default None.
    subplot_label_format : str | None, optional
        Format of the subplot label, by default None.
    chi2_annotation : bool, optional
        If the χ²/point value is annotated, by default True.
    chi2_legend : bool, optional
        If a legend with the total χ² is shown, by default True.
    curve_groupby : str | list[str] | None, optional
        Variable(s) to group the curves by, by default no grouping.
    apply_normalization : bool, optional
        If the normalization-corrected theory is plotted, by default True.
    theory_min_width : float, optional
        Width of the theory curve (in units of axes fraction) if there is only one point, by default 0.06.
    plot_pdf_uncertainty : bool, optional
        If the PDF uncertainty is plotted around the theory curve, by default True.
    pdf_uncertainty_convention : Literal["sym", "asym"], optional
        If the PDF uncertainties should be symmetric ("sym") or asymmetric ("asym"), by default "asym".
    y_offset_add : float, optional
        Additive offset by which the curves are separated, by default 0 (no offset).
    y_offset_mul : float, optional
        Multiplicative offset by which the curves are separated, by default 0 (no offset). The factor between each curve is `10**y_offset_mul`.
    kwargs_data : dict[str, Any] | list[dict[str, Any] | None], optional
        Keyword arguments to pass to `plt.Axes.plot` for plotting the data.
    kwargs_theory : dict[str, Any] | list[dict[str, Any]  |  None], optional
        Keyword arguments to pass to `plt.Axes.plot` for plotting the theory.
    kwargs_theory_unc : dict[str, Any] | list[dict[str, Any]  |  None], optional
        Keyword arguments to pass to `plt.Axes.fill_between` for plotting the PDF uncertainty.
    kwargs_xlabel : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.set_xlabel`.
    kwargs_ylabel : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.set_ylabel`.
    kwargs_title : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.set_title`.
    kwargs_legend : dict[str, Any], optional
        Keyword arguments to pass to `AdditionalLegend` for annotating data & theory.
    kwargs_legend_chi2 : dict[str, Any], optional
        Keyword arguments to pass to `AdditionalLegend` for annotating the total χ².
    kwargs_legend_curves : dict[str, Any], optional
        Keyword arguments to pass to `AdditionalLegend` for annotating the curve labels.
    kwargs_ticks_curves : dict[str, Any] | list[dict[str, Any]  |  None], optional
        Keyword arguments to pass to `plt.Axes.set_yticks` for annotating the curve labels.
    kwargs_annotate_chi2 : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.annotate` for annotating the χ²/point values.
    kwargs_annotate_curves : dict[str, Any] | list[dict[str, Any] | None], optional
        Keyword arguments to pass to `plt.Axes.annotate` for annotating the curve labels.
    """

    if isinstance(ax, plt.Axes):
        ax = [ax]

    theory_column = "theory_with_normalization" if apply_normalization else "theory"

    # Data & Theory legend
    legend_handles = []
    legend_labels = []

    # bin labels for legend or ticks
    legend_curve_handles = []

    if curve_groupby is not None:
        ascending = curve_groupby != "x"  # TODO: replace with PointsGroupBy
        points = points.sort_values(curve_groupby, ascending=ascending)

    curve_gb = (
        points.groupby(curve_groupby, sort=False) if curve_groupby is not None else None
    )

    # enumerate groupby if curve_groupby is given, else just enumerate one tuple (i.e. no iteration)
    iter_curves_data = (
        enumerate(curve_gb) if curve_gb is not None else enumerate([(0.0, points)])
    )
    for i, (label_i, points_i) in iter_curves_data:
        points_i = points_i.sort_values(x_variable)

        if "theory" in points:
            l = _plot_theory(
                ax,
                points_i,
                x_col=x_variable,
                y_col=theory_column,
                index_curve=i,
                y_offset_add=y_offset_add,
                y_offset_mul=y_offset_mul,
                theory_min_width=theory_min_width,
                plot_pdf_uncertainty=plot_pdf_uncertainty,
                pdf_uncertainty_convention=pdf_uncertainty_convention,
                kwargs_theory=kwargs_theory,
                kwargs_theory_unc=kwargs_theory_unc,
            )

            if l is not None and i == 0:
                legend_handles.append(l)
                legend_labels.append("Theory")

            legend_curve_handles.append(l)
        else:
            l = None

        kwargs_data_default = (
            {"markerfacecolor": l.get_color()} if l is not None else {}
        )
        kwargs_data_updated = update_kwargs(kwargs_data_default, kwargs_data, i)

        e = _plot_data(
            ax=ax,
            points=points_i,
            x_col=x_variable,
            index_curve=i,
            y_offset_add=y_offset_add,
            y_offset_mul=y_offset_mul,
            kwargs_data=kwargs_data_updated,
        )

        if e is not None and i == 0:
            legend_handles.append(e)
            legend_labels.append("Data")

        if chi2_annotation:
            _annotate_chi2(
                ax=ax[0],
                points=points_i,
                x_col=x_variable,
                y_col=theory_column,
                index_curve=i,
                y_offset_add=y_offset_add,
                y_offset_mul=y_offset_mul,
                kwargs_annotate_chi2=kwargs_annotate_chi2,
            )

    if yscale is not None:
        ax[0].set_yscale(yscale)  # pyright: ignore[reportArgumentType]

    if curve_gb is not None:
        _add_curve_labels(
            ax[0],
            curve_label_position=curve_label,
            curve_groupby=curve_gb,
            x_col=x_variable,
            y_col=theory_column,
            # curve_labels=curve_labels,
            y_offset_add=y_offset_add,
            y_offset_mul=y_offset_mul,
            legend_curve_handles=legend_curve_handles,
            kwargs_annotate_curves=kwargs_annotate_curves,
            kwargs_legend_curves=kwargs_legend_curves,
            kwargs_tick_curves=kwargs_ticks_curves,
        )

    _set_labels(
        ax=ax,
        x_fallback=f"${nc_labels.kinvars_py_to_tex[x_variable if isinstance(x_variable, str) else x_variable[0]]}$",
        y_fallback="",
        x_label=xlabel,
        kwargs_xlabel=kwargs_xlabel,
        y_label=ylabel,
        kwargs_ylabel=kwargs_ylabel,
        title=title,
        kwargs_title=kwargs_title,
    )

    if legend:
        kwargs_legend_default = {
            "order": 2,
            "parent": ax[0],
            "handles": legend_handles,
            "labels": ["Data", "Theory"],
        }
        kwargs_legend_updated = update_kwargs(kwargs_legend_default, kwargs_legend)

        legend_data_theory = AdditionalLegend(**kwargs_legend_updated)
        ax[0].add_artist(legend_data_theory)

    if chi2_legend and "chi2" in points:
        _add_chi2_legend(
            order=0, ax=ax[0], points=points, kwargs_legend_chi2=kwargs_legend_chi2
        )

    if subplot_label == "legend":
        if subplot_label_format is None:
            raise ValueError(
                "subplot_label_format must be given if subplot_label is 'legend'"
            )

        A_symbols = {}

        if "data" in points:
            if points.iloc[0][["Z1", "A1"]].notna().all():
                A_symbols["A1_sym"] = nc_labels.nucleus_to_latex(
                    Z=points.iloc[0]["Z1"], A=points.iloc[0]["A1"], superscript=True
                )
            if points.iloc[0][["Z2", "A2"]].notna().all():
                A_symbols["A2_sym"] = nc_labels.nucleus_to_latex(
                    Z=points.iloc[0]["Z2"], A=points.iloc[0]["A2"], superscript=True
                )

            subplot_label_str = subplot_label_format.format(
                **points.iloc[0].to_dict(), **A_symbols
            )
        else:
            raise ValueError(
                "Either data or theory must be given if subplot_label is 'legend'"
            )

        leg3 = AdditionalLegend(
            1,
            ax,
            handles=[Patch()],
            labels=[subplot_label_str],
            labelspacing=0,
            handlelength=0,
            handleheight=0,
            handletextpad=0,
        )
        ax[0].add_artist(leg3)

    if xscale is not None:
        ax[0].set_xscale(xscale)  # pyright: ignore[reportArgumentType]
        ax[0].relim()  # otherwise axes autoscaling does not work for some reason
        if len(ax) > 1:
            ax[1].set_xscale(xscale)  # pyright: ignore[reportArgumentType]
            ax[1].relim()


def _plot_data(
    ax: plt.Axes | Sequence[plt.Axes],
    points: pd.DataFrame,
    x_col: str | list[str],
    index_curve: int = 0,
    num_curves: int = 1,
    y_offset_add: float = 0,
    y_offset_mul: float = 0,
    ratio_offset: bool = True,
    kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
) -> ErrorbarContainer | None:
    if isinstance(ax, plt.Axes):
        ax = [ax]

    if "data" in points:
        offset_factor = 10 ** (y_offset_mul * index_curve)
        offset_summand = y_offset_add * index_curve

        kwargs_data_default: dict[str, Any] = {
            "capsize": 1.7,
            "marker": ".",
            "markersize": 7,
            "ls": "",
            "color": "black",
        }

        if isinstance(x_col, list):
            xerr = (points[x_col[1]] - points[x_col[0]]) / 2
            kwargs_data_default["xerr"] = (xerr,)
        else:
            xerr = None

        if "unc_tot" in points:
            kwargs_data_default["yerr"] = points["unc_tot"] * offset_factor

        kwargs_data_updated = update_kwargs(
            kwargs_data_default, kwargs_data, index_curve
        )

        x = points[x_col].mean(axis=1) if isinstance(x_col, list) else points[x_col]

        e = ax[0].errorbar(
            x,
            points["data"] * offset_factor + offset_summand,
            **kwargs_data_updated,
        )

        e.lines[0].set_markeredgewidth(0.4)

        if len(ax) >= 2:
            if "unc_tot" in points:
                kwargs_data_updated["yerr"] = points["unc_tot"] / points["data"]

            # TODO: make ratio_offset work if x_col is not a list, i.e. when not plotting a binned distribution
            if xerr is not None:
                x_offsets = (
                    1.9 * xerr * (1 / (num_curves + 1) * (index_curve + 0.5) - 0.5)
                )
                x += x_offsets

            kwargs_data_updated["markerfacecolor"] = plt.rcParams[
                "axes.prop_cycle"
            ].by_key()["color"][index_curve]
            kwargs_data_updated["markeredgecolor"] = kwargs_data_updated[
                "markerfacecolor"
            ]
            kwargs_data_updated.pop("xerr")
            ax[1].errorbar(
                x,
                points["data"] / points["data"],
                **kwargs_data_updated,
            )

        return e
    else:
        return None


def _plot_theory(
    ax: plt.Axes | Sequence[plt.Axes],
    points: pd.DataFrame,
    x_col: str | list[str],
    y_col: str = "theory",
    index_curve: int = 0,
    y_offset_add: float = 0,
    y_offset_mul: float = 0,
    theory_min_width: float = 0.06,
    plot_pdf_uncertainty: bool = True,
    pdf_uncertainty_convention: Literal["sym", "asym"] = "asym",
    kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_theory_unc: dict[str, Any] | list[dict[str, Any] | None] = {},
) -> Line2D | None:
    if isinstance(ax, plt.Axes):
        ax = [ax]

    if isinstance(x_col, list) and len(x_col) == 1:
        x_col = x_col[0]

    plot_binned = isinstance(x_col, list)

    if y_col in points:
        # exponent_offset = 10**index_curve if abs_offset else 1
        offset_factor = 10 ** (y_offset_mul * index_curve)
        offset_summand = y_offset_add * index_curve

        kwargs_theory_default = {}

        # for nan values in points[y_col] we fall back to "theory"
        y = points[y_col].fillna(points["theory"], inplace=False).to_numpy()

        if plot_binned:
            x = np.append(points[x_col[0]].iloc[0], points[x_col[1]])
            y = np.append(y, y[-1])
            kwargs_theory_default["drawstyle"] = "steps-post"
        else:
            x = points[x_col].to_numpy()

        # add some padding if otherwise the point would not be visible
        if not plot_binned and x.size == 1:
            transforms = [
                (
                    mtransforms.blended_transform_factory(
                        # fmt: off
                        mtransforms.ScaledTranslation(
                            x[0], 0, ax_i.transScale + ax_i.transLimits  # pyright: ignore[reportAttributeAccessIssue]
                        )
                        # fmt: on
                        + ax_i.transAxes,
                        ax_i.transData,
                    )
                )
                for ax_i in ax
            ]
            x = np.array([-theory_min_width / 2, theory_min_width / 2])
            y = y.repeat(2)
        else:
            transforms = None

        if transforms is not None:
            kwargs_theory_default["transform"] = transforms[0]  # FIXME for ratio plot

        kwargs_theory_updated = update_kwargs(
            kwargs_theory_default, kwargs_theory, index_curve
        )

        l = ax[0].plot(
            x,
            y * offset_factor + offset_summand,
            **kwargs_theory_updated,
        )

        plot_ratio = len(ax) >= 2 and "data" in points

        if plot_ratio:

            y_denom = (
                np.append(points["data"], points["data"].iloc[-1])
                if isinstance(x_col, list)
                else points["data"].to_numpy()
            )

            ax[1].plot(
                x,
                y / y_denom,
                **kwargs_theory_updated,
            )

        if plot_pdf_uncertainty:
            kwargs_theory_unc_default = {
                "alpha": 0.5,
                "facecolor": l[0].get_color(),
                "edgecolor": None,
                "lw": 0,
            }

            if transforms is not None:
                kwargs_theory_unc_default["transform"] = transforms[0]

            if plot_binned:
                kwargs_theory_unc_default["step"] = "post"

            kwargs_theory_unc_updated = update_kwargs(
                kwargs_theory_unc_default,
                kwargs_theory_unc,
                index_curve,
            )

            if pdf_uncertainty_convention == "sym" and "theory_pdf_unc_sym" in points:
                y_pdf_unc_lower = points["theory_pdf_unc_sym"].to_numpy()
                y_pdf_unc_upper = y_pdf_unc_lower
            elif (
                pdf_uncertainty_convention == "asym"
                and "theory_pdf_unc_asym_lower" in points
                and "theory_pdf_unc_asym_upper" in points
            ):
                y_pdf_unc_lower = points["theory_pdf_unc_asym_lower"].to_numpy()
                y_pdf_unc_upper = points["theory_pdf_unc_asym_upper"].to_numpy()
            else:
                y_pdf_unc_upper = np.ones(points[y_col].size) * np.nan
                y_pdf_unc_lower = y_pdf_unc_upper

            if isinstance(x_col, list):
                y_pdf_unc_lower = np.append(y_pdf_unc_lower, y_pdf_unc_lower[-1])
                y_pdf_unc_upper = np.append(y_pdf_unc_upper, y_pdf_unc_upper[-1])

            # fmt: off
            ax[0].fill_between(
                x,
                (y + y_pdf_unc_lower) * offset_factor + offset_summand,  # pyright: ignore[reportArgumentType]
                (y - y_pdf_unc_lower) * offset_factor + offset_summand,  # pyright: ignore[reportArgumentType]
                **kwargs_theory_unc_updated,
            )
            # fmt: on

            if plot_ratio:
                # fmt: off
                ax[1].fill_between(
                    x,
                    (y + y_pdf_unc_lower) / y_denom, # pyright: ignore[reportArgumentType,reportPossiblyUnboundVariable]
                    (y - y_pdf_unc_lower) / y_denom, # pyright: ignore[reportArgumentType,reportPossiblyUnboundVariable]
                    **kwargs_theory_unc_updated,
                )
                # fmt: on

        return l[0]


def _annotate_chi2(
    ax: plt.Axes,
    points: pd.DataFrame,
    x_col: str | list[str],
    y_col: str = "theory",
    index_curve: int = 0,
    y_offset_add: float = 0,
    y_offset_mul: float = 0,
    pdf_uncertainty_convention: Literal["sym", "asym"] = "asym",
    kwargs_annotate_chi2: dict[str, Any] = {},
) -> None:
    positions = []

    if y_col in points:
        pos_theory = points[y_col]

        # if pdf_uncertainty_convention == "sym" and f"{y_col}_pdf_unc_sym" in points:
        #     pos_theory = pos_theory.add(points[f"{y_col}_pdf_unc_sym"], fill_value=0)
        # elif (
        #     pdf_uncertainty_convention == "asym"
        #     and f"{y_col}_pdf_unc_asym_upper" in points
        # ):
        #     pos_theory = pos_theory.add(
        #         points[f"{y_col}_pdf_unc_asym_upper"], fill_value=0
        #     )

        positions.append(pos_theory)

    if "data" in points:
        pos_data = points["data"]

        if "unc_tot" in points:
            pos_data = pos_data.add(points["unc_tot"], fill_value=0)

        positions.append(pos_data)

    if not positions:
        positions.append(np.zeros(points["chi2"].size))

    kwargs_annotate_default = {
        "xytext": (0, 0.3),
        "textcoords": "offset fontsize",
        "ha": "center",
        "fontsize": "x-small",
    }
    kwargs_annotate_updated = update_kwargs(
        kwargs_annotate_default, kwargs_annotate_chi2
    )

    x = points[x_col].mean(axis=1) if isinstance(x_col, list) else points[x_col]

    offset_factor = 10 ** (y_offset_mul * index_curve)
    offset_summand = y_offset_add * index_curve

    for chi2_i, pos_i, x_i in zip(points["chi2"], np.nanmax(positions, axis=0), x):
        ax.annotate(
            f"{chi2_i:.1f}",
            (x_i, pos_i * offset_factor + offset_summand),
            **kwargs_annotate_updated,
        )


def _add_curve_labels(
    ax: plt.Axes,
    curve_label_position: (
        Literal["ticks", "legend", "annotate above", "annotate right"] | None
    ),
    curve_groupby: DataFrameGroupBy,
    x_col: str | list[str],
    y_col: str,
    curve_labels: list[str] = [],
    y_offset_add: float = 0,
    y_offset_mul: float = 0,
    legend_curve_handles: list[martist.Artist] = [],
    kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_legend_curves: dict[str, Any] = {},
    kwargs_tick_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
) -> None:
    curve_labels_fmt = []
    for i, (value, label) in enumerate(
        zip_longest(curve_groupby.groups.keys(), curve_labels, fillvalue=None)
    ):
        curve_labels_fmt.append(
            _format_curve_label(
                variables=curve_groupby.keys,  # pyright: ignore[reportArgumentType]
                values=value,  # pyright: ignore[reportArgumentType]
                variables_labels=label,
                index_group=i,
                y_offset_add=y_offset_add,
                y_offset_mul=y_offset_mul,
            )
        )

    if curve_label_position == "legend":
        kwargs_legend_bins_default = {
            "order": 0,
            "parent": ax,
            "handles": legend_curve_handles,
            "labels": curve_labels_fmt,
            "fontsize": "small",
        }
        kwargs_legend_bins_updated = update_kwargs(
            kwargs_legend_bins_default, kwargs_legend_curves
        )

        legend_curves = AdditionalLegend(**kwargs_legend_bins_updated)
        ax.add_artist(legend_curves)
    else:
        for i, (_, points_i) in enumerate(curve_groupby):
            if curve_label_position in ("annotate above", "annotate right"):
                if curve_label_position == "annotate above":
                    kwargs_annotate_curves_default = {
                        "xytext": (0, 0.3),
                        "textcoords": "offset fontsize",
                        "ha": "right",
                        "fontsize": "small",
                    }
                else:
                    kwargs_annotate_curves_default = {
                        "xytext": (0.3, 0),
                        "textcoords": "offset fontsize",
                        "ha": "left",
                        "va": "center",
                        "fontsize": "small",
                    }

                kwargs_annotate_curves_updated = update_kwargs(
                    kwargs_annotate_curves_default, kwargs_annotate_curves
                )

                y_offset_factor = 10 ** (i * y_offset_mul)
                y_offset_summand = i * y_offset_add

                ax.annotate(
                    curve_labels_fmt[i],
                    (
                        points_i[x_col].iloc[-1].max(),
                        points_i[y_col].iloc[-1] * y_offset_factor + y_offset_summand,
                    ),
                    **kwargs_annotate_curves_updated,
                )

            elif curve_label_position == "ticks":
                ax_ticks = ax.secondary_yaxis("right")
                # fmt: off
                ax_ticks.spines["right"].set_visible(False)  # pyright: ignore[reportAttributeAccessIssue]
                # fmt: on
                ax_ticks.tick_params("y", which="minor", right=False, labelright=False)
                ax_ticks.tick_params("y", which="major", right=True, labelright=True)

                kwargs_tick_curves_default = {"fontsize": "small"}
                kwargs_tick_curves_updated = update_kwargs(
                    kwargs_tick_curves_default, kwargs_tick_curves, i
                )

                y_offset_factor = 10 ** (i * y_offset_mul)
                y_offset_summand = i * y_offset_add

                ax_ticks.set_yticks(
                    [points_i[y_col].iloc[-1] * y_offset_factor + y_offset_summand],
                    [curve_labels_fmt[i]],
                    **kwargs_tick_curves_updated,
                )


def _format_curve_label(
    variables: str | list[str],
    values: float | Sequence[float],
    variables_labels: str | list[str | None] | None = None,
    units: str | list[str | None] | None = None,
    index_group: int = 0,
    y_offset_add: float = 0,
    y_offset_mul: float = 0,
) -> str:
    if not isinstance(variables, list):
        variables = [variables]

    if not isinstance(values, Sequence):
        values = [values]

    if units is None:
        units = cast(list[str | None], len(variables) * [None])

    if not isinstance(units, list):
        units = [units]

    if variables_labels is None:
        variables_labels = cast(list[str | None], len(variables) * [None])

    if not isinstance(variables_labels, list):
        variables_labels = [variables_labels]

    if len(values) != len(variables):
        raise ValueError("Please pass as many values as variables")

    if len(units) != len(variables):
        raise ValueError("Please pass as many units as variables")

    if len(variables_labels) != len(variables):
        raise ValueError("Please pass as many labels as variables")

    values_fmt: list[str] = []
    for v, u in zip(values, units):
        value_fmt = f"{v:.3g}"

        if u is not None and u != "":
            value_fmt += rf"\,\mathrm{{{u}}}"

        values_fmt.append(value_fmt)

    # format offsets
    offset_labels = []

    if y_offset_mul != 0:
        offset_labels.append(rf"$\times 10^{{{index_group * y_offset_mul}}}$")

    if y_offset_add != 0:
        offset_labels.append(rf"$+ {index_group * y_offset_add}$")

    offset_label = f"  ({', '.join(offset_labels)})" if offset_labels else ""

    variables_fmt = []

    for variable, value, label in zip(variables, values_fmt, variables_labels):
        # find binned variables
        rp = variable.rpartition("_")

        if rp[2] == "max" and rp[0] + "_min" in variables:
            continue

        variable_label = (
            label if label is not None else nc_labels.kinvars_py_to_tex[variable]
        )

        if rp[2] == "min" and rp[0] + "_max" in variables:
            i_max = variables.index(rp[0] + "_max")

            variables_fmt.append(f"${value} < {variable_label} < {values_fmt[i_max]}$")
        else:
            variables_fmt.append(f"${variable_label} = {value}$")

    return ",  ".join(variables_fmt) + offset_label


def _set_labels(
    ax: plt.Axes | Sequence[plt.Axes],
    x_fallback: str,
    y_fallback: str,
    y_ratio_fallback: str = r"\dfrac{\rm Theory}{\rm Data}",
    x_label: Literal["fallback"] | str | None = "fallback",
    y_label: Literal["fallback"] | str | None = "fallback",
    y_ratio_label: Literal["fallback"] | str | None = "fallback",
    title: str | None = None,
    kwargs_xlabel: dict[str, Any] = {},
    kwargs_ylabel: dict[str, Any] = {},
    kwargs_ylabel_ratio: dict[str, Any] = {},
    kwargs_title: dict[str, Any] = {},
) -> None:
    if isinstance(ax, plt.Axes):
        ax = [ax]

    if x_label is not None:
        if x_label == "fallback":
            x_label = x_fallback

        ax[-1].set_xlabel(x_label, **kwargs_xlabel)

    if y_label is not None:
        if y_label == "fallback":
            y_label = y_fallback

        ax[0].set_ylabel(y_label, **kwargs_ylabel)

    if y_ratio_label is not None and len(ax) > 1:
        if y_ratio_label == "fallback":
            y_ratio_label = y_ratio_fallback

        ax[1].set_ylabel(y_ratio_fallback, **kwargs_ylabel_ratio)

    if title is not None:
        kwargs_title_default = {"fontsize": "medium"}
        kwargs_title_updated = update_kwargs(kwargs_title_default, kwargs_title)

        ax[0].set_title(title, **kwargs_title_updated)


def _add_chi2_legend(
    order: int,
    ax: plt.Axes,
    points: pd.DataFrame,
    kwargs_legend_chi2: dict[str, Any] = {},
) -> AdditionalLegend:
    kwargs_default = {
        "order": order,
        "parent": ax,
        "handles": [Patch(), Patch(), Patch()],
        "labels": [
            f"$N_{{\\text{{points}}}} = {len(points)}$",
            f"$\\chi^2_{{\\text{{total}}}} = {points['chi2'].sum():.3f}$",
            f"$\\chi^2_{{\\text{{total}}}}\\,/\\, N_{{\\text{{points}}}} = {points['chi2'].sum() / len(points):.3f}$",
        ],
        "labelspacing": 0,
        "handlelength": 0,
        "handleheight": 0,
        "handletextpad": 0,
    }
    kwargs = update_kwargs(kwargs_default, kwargs_legend_chi2)

    legend = AdditionalLegend(**kwargs)
    ax.add_artist(legend)

    return legend
