from __future__ import annotations

from typing import Any, Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ncteqpy.data_groupby import DatasetsGroupBy
from ncteqpy.plot.grid import AxesGrid
from ncteqpy.util import update_kwargs


def plot_chi2_data_breakdown(
    ax: plt.Axes,
    chi2: pd.Series[float],
    per_point: bool = True,
    bar_orientation: Literal["horizontal", "vertical"] = "vertical",
    num_points: pd.Series[int] | None = None,
    chi2_line_1: bool = True,
    chi2_drop_0: bool = True,
    bar_groupby: DatasetsGroupBy | None = None,
    bar_order_groupby: str | list[str] | None = None,  # TODO?
    bar_props_groupby: DatasetsGroupBy | None = None,
    kwargs_bar: dict[str, Any] = {},
    kwargs_bar_label: dict[str, Any] = {},
    kwargs_chi2_line_1: dict[str, Any] = {},
    kwargs_legend: dict[str, Any] = {},
) -> None:
    """Plot histogram of χ² vs. (grouped) datasets.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    chi2 : pd.Series[float]
        `Series` that maps data set ID to χ².
    per_point : bool, optional
        If χ² per point should be plotted, by default True.
    bar_orientation : Literal["horizontal", "vertical"], optional
        Direction in which the bars are oriented, by default "vertical".
    num_points : pd.Series[int] | None, optional
        `Series` that maps data set ID to number of points in that data set, by default None. Must be passed if `per_point` is set to `True`.
    chi2_line_1 : bool, optional
        If a line should be plotted at χ²/point = 1, by default True. Does nothing if `per_point` is `False`.
    chi2_drop_0 : bool, optional
        If data sets with χ² = 0 should be ignored when plotting (i.e., data sets that did not survive the cuts), by default True.
    bar_groupby : DatasetsGroupBy | None, optional
        How to group the bars, by default no grouping. One bar per group is plotted.
    bar_order_groupby : str | list[str] | None, optional
        Not implemented yet
    bar_props_groupby : DatasetsGroupBy | None, optional
        How to group the properties (color etc.) of each bar, by default no grouping, i.e., all bars get the same properties.
    kwargs_bar : dict[str, Any], optional
        Keyword arguments passed to `plt.Axes.bar` or `plt.Axes.barh`.
    kwargs_bar_label : dict[str, Any], optional
        Keyword arguments passed to `plt.Axes.bar_label`.
    kwargs_chi2_line_1 : dict[str, Any], optional
        Keyword arguments passed to `plt.Axes.axhline` or `plt.Axes.axvline` for the line at χ²/point = 1.
    kwargs_legend : dict[str, Any], optional
        Keyword arguments passed to `plt.Axes.legend` for the legend set by `bar_props_groupby`.
    """

    if per_point and num_points is None:
        raise ValueError("Please pass num_points when passing per_point=True")

    bar_grouper = bar_groupby.grouper if bar_groupby is not None else "id_dataset"

    # construct dataframe with the bar_groupby values as columns (transpose before and after groupby to group the columns)
    chi2_grouped = (
        chi2.sort_index(
            key=(
                bar_groupby.sort_key if bar_groupby is not None else None
            ),  # pyright: ignore [reportArgumentType]
        )
        .groupby(bar_grouper, sort=False, dropna=False)
        .sum()
        .copy()
    )

    # drop experiments that didn't survive the cuts
    if chi2_drop_0:
        chi2_grouped = chi2_grouped[chi2_grouped > 0.0]

    num_points_grouped: pd.Series[int] | None = (
        num_points.sort_index(  # pyright: ignore [reportCallIssue]
            key=(
                bar_groupby.sort_key  # pyright: ignore [reportArgumentType]
                if bar_groupby is not None
                else None
            ),
        )
        .groupby(bar_grouper, sort=False, dropna=False)
        .sum()
        .reindex_like(chi2_grouped)  # pyright: ignore [reportArgumentType]
        if num_points is not None
        else None
    )

    num_groups: int = chi2_grouped.shape[0]
    width: float = 0.8

    labels: pd.Series[str] | pd.Index = (
        bar_groupby.labels.loc[chi2_grouped.index]
        if bar_groupby is not None
        else chi2_grouped.index
    )

    chi2_label = (
        r"$\chi^2_{\text{total}}"
        + (r"\, / \, N_{\text{points}}" if per_point else "")
        + "$"
    )

    if bar_orientation == "vertical":
        ax_hvline = ax.axhline
        ax_barhv = ax.bar
        kwargs_bar_default: dict[str, Any] = {
            "width": width,
        }
    elif bar_orientation == "horizontal":
        ax_hvline = ax.axvline
        ax_barhv = ax.barh
        kwargs_bar_default: dict[str, Any] = {
            "height": width,
        }
    else:
        raise ValueError("bar_orientation must be either 'vertical' or 'horizontal'")

    if per_point and chi2_line_1:
        kwargs_chi2_line_1_default = {
            "color": "black",
            "ls": (0, (5, 7)),
            "lw": 0.8,
            "zorder": 0.5,
        }
        kwargs_chi2_line_1_updated = update_kwargs(
            kwargs_chi2_line_1_default, kwargs_chi2_line_1
        )
        ax_hvline(1.0, **kwargs_chi2_line_1_updated)

    bar_props = {}

    if bar_props_groupby is not None:

        bar_props = bar_props_groupby.get_props(
            chi2_grouped.index.name,
            chi2_grouped.index,  # pyright: ignore [reportArgumentType]
        )

        chi2_grouped_props = (
            chi2.sort_index(
                key=bar_props_groupby.sort_key,  # pyright: ignore [reportArgumentType]
            )
            .groupby(bar_props_groupby.grouper, sort=False, dropna=False)
            .sum()
            .copy()
        )

        num_points_grouped_props = (
            num_points.sort_index(  # pyright: ignore [reportCallIssue]
                key=bar_props_groupby.sort_key,  # pyright: ignore [reportArgumentType]
            )
            .groupby(bar_props_groupby.grouper, sort=False, dropna=False)
            .sum()
            .reindex_like(chi2_grouped_props)  # pyright: ignore [reportArgumentType]
            .copy()
            if num_points is not None
            else None
        )

        kwargs_legend_default = {}
        kwargs_legend_updated = update_kwargs(kwargs_legend_default, kwargs_legend)
        ax.legend(
            [mpatches.Patch(**v) for _, v in bar_props_groupby.props.loc[chi2_grouped.index].iterrows()],
            bar_props_groupby.labels[chi2_grouped.index]
            + " ("
            + (
                chi2_grouped_props
                / (
                    num_points_grouped_props
                    if per_point and num_points_grouped_props is not None
                    else 1
                )
            ).apply("{:.2f}".format)
            + ")",
            **kwargs_legend_updated,
        )

    kwargs_bar_default.update(**bar_props)
    kwargs_bar_updated = update_kwargs(kwargs_bar_default, kwargs_bar)

    bar_locs = np.arange(num_groups)

    if bar_orientation == "horizontal":
        bar_locs = bar_locs[::-1]

    bar = ax_barhv(
        bar_locs,
        (
            chi2_grouped / num_points_grouped
            if per_point and num_points_grouped is not None
            else chi2_grouped
        ),
        **kwargs_bar_updated,
    )

    # bar_labels = num_points_grouped

    # if bar_orientation == "horizontal" and bar_labels is not None:
    #     bar_labels = bar_labels.iloc[::-1]

    num_points_labels = (
        num_points_grouped.apply("{:.0f}".format)
        if num_points_grouped is not None
        else None
    )

    kwargs_bar_label_default: dict[str, Any] = {
        "labels": num_points_labels,
        "padding": 3,
        "fontsize": "small",
        "bbox": dict(
            facecolor=(1, 1, 1),
            alpha=0.8,
            lw=0,
            boxstyle="round,pad=0.1,rounding_size=0.2",
        ),
    }
    kwargs_bar_label_updated = update_kwargs(kwargs_bar_label_default, kwargs_bar_label)

    ax.bar_label(bar, **kwargs_bar_label_updated)

    if bar_orientation == "vertical":

        ax.set_xlim(-0.8, num_groups - 0.2)
        ax.set_xticks(np.arange(num_groups), labels)

        ax.tick_params(axis="x", rotation=90)

        ax.set_ylabel(chi2_label)
    elif bar_orientation == "horizontal":
        ax.set_ylim(-0.8, num_groups - 0.2)
        ax.set_yticks(np.arange(num_groups)[::-1], labels)

        ax.set_xlabel(chi2_label)


def plot_chi2_histogram(
    chi2: pd.DataFrame,
    bin_width: float | None = None,
    subplot_groupby: DatasetsGroupBy | None = None,
    kwargs_subplots: dict[str, Any] | None = None,
    kwargs_histogram: dict[str, Any] | list[dict[str, Any] | None] | None = None,
) -> AxesGrid:
    """Plots a histogram of the χ² values of the data points.

    Parameters
    ----------
    chi2 : pd.DataFrame
        The χ² values.
    bin_width : float | None, optional
        Width of a bin, by default chosen by `np.histogram`.
    subplot_groupby : DatasetsGroupBy | None, optional
        How to group χ² values that are shown in distributions on different subplots, by default None.
    kwargs_subplots : dict[str, Any] | None, optional
        Keyword arguments passed to `plt.subplots` through `AxesGrid`.
    kwargs_histogram : dict[str, Any] | list[dict[str, Any]  |  None] | None, optional
        Keyword arguments passed to `plt.Axes.hist`.

    Returns
    -------
    AxesGrid
        `AxesGrid` that holds the subplot(s).
    """
    num_axes = subplot_groupby.groupby.ngroups if subplot_groupby is not None else 1

    kwargs_subplots_default = {"sharex": True, "sharey": True}
    kwargs = update_kwargs(kwargs_subplots_default, kwargs_subplots)

    ax_grid = AxesGrid(num_axes, **kwargs)

    if bin_width is not None:
        lim_neg: float = bin_width * np.floor(chi2["chi2"].min() / bin_width)
        lim_pos: float = bin_width * np.ceil(chi2["chi2"].max() / bin_width)

        lim = np.max([np.abs(lim_neg), lim_pos])

        bins = np.arange(-lim, lim + bin_width, bin_width)
    else:
        lim = np.max(np.abs([chi2["chi2"].min(), chi2["chi2"].max()]))
        bins = "auto"

    def chi2_k1_func(x):
        return 1 / np.sqrt(2 * np.pi * x) * np.exp(-x / 2)

    chi2_k1_x = np.logspace(-5, np.log10(lim), 200, endpoint=True)
    chi2_k1_y = chi2_k1_func(chi2_k1_x)

    if subplot_groupby is not None:
        chi2_groupby = (
            chi2.sort_values(
                by="id_dataset",
                key=subplot_groupby.sort_key,
            )
            .set_index("id_dataset")
            .groupby(subplot_groupby.grouper, sort=False, dropna=False)
        )
    else:
        chi2_groupby = [(None, chi2)]

    for i, (ax, (_, chi2_i)) in enumerate(zip(ax_grid.ax_real, chi2_groupby)):
        ax: plt.Axes

        # np.histogram cannot deal with NaN
        chi2_i = chi2_i[chi2_i["chi2"].notna()]

        hist, bin_edges = np.histogram(chi2_i["chi2"], bins=bins)
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # normalize area to 1 (/= does not work here)
        hist = hist / np.sum(hist * bin_widths)

        kwargs_histogram_default = {
            "x": bin_edges[:-1],
            "bins": bin_edges,
            "weights": hist,
        }
        kwargs = update_kwargs(kwargs_histogram_default, kwargs_histogram, i)
        ax.hist(**kwargs)

        ax.plot(chi2_k1_x, chi2_k1_y, scaley=False)

    if subplot_groupby is not None:
        ax_grid.set_labels(
            subplot_groupby.labels  # pyright: ignore [reportArgumentType]
        )

    ax_grid.set_xlabel(r"$\chi^2$")
    ax_grid.set_ylabel("Probability")

    return ax_grid


def plot_S_E_histogram(
    S_E: pd.Series[float],
    bin_width: float | None = None,
    gaussian: bool = True,
    gaussian_fit: bool = True,
    subplot_groupby: DatasetsGroupBy | None = None,
    kwargs_subplots: dict[str, Any] | None = None,
    kwargs_histogram: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    kwargs_gaussian: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    kwargs_fit: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    kwargs_gaussian_fit: dict[str, Any] | list[dict[str, Any] | None] | None = None,
) -> AxesGrid:
    """Plots the `S_E` distribution (see arXiv:1905.06957 eq. 157).

    Parameters
    ----------
    S_E : pd.Series[float]
        `Series` that maps data set ID to `S_E`.
    bin_width : float | None, optional
        Width of a bin, by default chosen by `np.histogram`.
    subplot_groupby : DatasetsGroupBy | None, optional
        How to group S_E values that are shown in distributions on different subplots, by default None.
    kwargs_subplots : dict[str, Any] | None, optional
        Keyword arguments passed to `plt.subplots` through `AxesGrid`.
    kwargs_histogram : dict[str, Any] | list[dict[str, Any]  |  None] | None, optional
        Keyword arguments passed to `plt.Axes.hist`.
    kwargs_gaussian : dict[str, Any] | list[dict[str, Any]  |  None] | None, optional
        Keyword arguments passed to `plt.Axes.plot` for plotting the standard gaussian.
    kwargs_fit : dict[str, Any] | list[dict[str, Any]  |  None] | None, optional
        Keyword arguments passed to `scipy.optimize.curve_fit` for fitting the gaussian.
    kwargs_gaussian_fit : dict[str, Any] | list[dict[str, Any]  |  None] | None, optional
        Keyword arguments passed to `plt.Axes.plot` for plotting the fitted gaussian.

    Returns
    -------
    AxesGrid
        `AxesGrid` that holds the subplot(s)
    """
    num_axes = subplot_groupby.groupby.ngroups if subplot_groupby is not None else 1

    kwargs_subplots_default = {"sharex": True, "sharey": True}
    kwargs = update_kwargs(kwargs_subplots_default, kwargs_subplots)

    ax_grid = AxesGrid(num_axes, **kwargs)

    if bin_width is not None:
        lim_neg: float = bin_width * np.floor(S_E.min() / bin_width)
        lim_pos: float = bin_width * np.ceil(S_E.max() / bin_width)

        lim = np.max([np.abs(lim_neg), lim_pos])

        bins = np.arange(-lim, lim + bin_width, bin_width)
    else:
        lim = np.max(np.abs([S_E.min(), S_E.max()]))
        bins = "auto"

    def gaussian_func(x, a, mu, sigma):
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    gaussian_x = np.linspace(-lim, lim, 200, endpoint=True)
    gaussian_y = gaussian_func(gaussian_x, 1 / np.sqrt(2 * np.pi), 0, 1)

    if subplot_groupby is not None:
        S_E_groupby = S_E.sort_index(
            key=subplot_groupby.sort_key,  # pyright: ignore [reportArgumentType]
        ).groupby(subplot_groupby.grouper, sort=False, dropna=False)
    else:
        S_E_groupby = [(None, S_E)]

    for i, (ax, (_, S_E_i)) in enumerate(zip(ax_grid.ax_real, S_E_groupby)):
        ax: plt.Axes

        # np.histogram cannot deal with NaN
        S_E_i = S_E_i.dropna()

        hist, bin_edges = np.histogram(S_E_i, bins=bins)
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

        # normalize area to 1 (/= does not work here)
        hist = hist / np.sum(hist * bin_widths)

        kwargs_histogram_default = {
            "x": bin_edges[:-1],
            "bins": bin_edges,
            "weights": hist,
        }
        kwargs_histogram_updated = update_kwargs(
            kwargs_histogram_default, kwargs_histogram
        )
        ax.hist(**kwargs_histogram_updated)

        kwargs_gaussian_default = {}
        kwargs_gaussian_updated = update_kwargs(
            kwargs_gaussian_default, kwargs_gaussian, i
        )
        ax.plot(gaussian_x, gaussian_y, **kwargs_gaussian_updated)

        kwargs_fit_default = {
            "f": gaussian_func,
            "xdata": bin_mids,
            "ydata": hist,
            "p0": [1 / np.sqrt(2 * np.pi), 0, 1],
        }
        kwargs_fit_updated = update_kwargs(kwargs_fit_default, kwargs_fit, i)
        popt, _ = curve_fit(**kwargs_fit_updated)

        kwargs_gaussian_fit_default = {}
        kwargs_gaussian_fit_updated = update_kwargs(
            kwargs_gaussian_fit_default, kwargs_gaussian_fit
        )
        ax.plot(
            gaussian_x, gaussian_func(gaussian_x, *popt), **kwargs_gaussian_fit_updated
        )

    if subplot_groupby is not None:
        ax_grid.set_labels(
            subplot_groupby.labels  # pyright: ignore [reportArgumentType]
        )

    ax_grid.set_xlabel("$S_E$")
    ax_grid.set_ylabel("Probability")

    return ax_grid
