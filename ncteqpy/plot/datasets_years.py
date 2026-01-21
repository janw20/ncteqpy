from __future__ import annotations

from calendar import day_abbr
from typing import Any, Literal, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ncteqpy.data_groupby import DatasetsGroupBy
from ncteqpy.plot.grid import AxesGrid
from ncteqpy.plot.util import AdditionalLegend
from ncteqpy.util import update_kwargs, forward_fill_dict
import sympy as sp
import numpy.typing as npt


def plot_datasets_timeline(
    datasets_index: pd.DataFrame,
    data: Literal["all", "fitted"] = "all",
    bar_groupby: DatasetsGroupBy | None = None,
    subplot_groupby: DatasetsGroupBy | None = None,
    # bar_order_groupby: str | list[str] | None = None,  # TODO?
    # bar_props_groupby: DatasetsGroupBy | None = None,
    # bar_labels: Literal["num_points", "chi2"] = "num_points",
    bar_orientation: Literal["vertical", "horizontal"] = "vertical",
    share_y: bool = False,
    width: float = 0.8,
    kwargs_bar: dict[str, Any] | List[dict[str, Any]] = {},
    kwargs_bar_cum: dict[str, Any] | List[dict[str, Any]] = {},
    kwargs_bar_label: dict[str, Any] = {},
    kwargs_legend: dict[str, Any] = {},
    kwargs_subplots: dict[str, Any] | List[dict[str, Any]] = {},
    kwargs_annotation: dict[str, Any] | List[dict[str, Any]] = {},
) -> AxesGrid:
    """Plot histogram of number of data points vs. years.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    """
    num_axes = subplot_groupby.groupby.ngroups if subplot_groupby is not None else 1

    kwargs_subplots_default = {"sharex": True, "sharey": share_y}
    kwargs = update_kwargs(kwargs_subplots_default, kwargs_subplots)

    ax_grid = AxesGrid(num_axes, **kwargs)

    if subplot_groupby is not None:
        data_groupby = (
            datasets_index.sort_values(
                by="id_dataset",
                key=subplot_groupby.sort_key,
            )
            .set_index("id_dataset")
            .groupby(subplot_groupby.grouper, sort=False, dropna=False)
        )
    else:
        data_groupby = [(None, datasets_index)]

    for i, (ax, (_, data_i)) in enumerate(zip(ax_grid.ax_real, data_groupby)):
        ax: plt.Axes

        if bar_orientation == "vertical":
            ax_hvline = ax.axhline
            ax_barhv = ax.bar
            ax_barhv_cum = ax.bar
            kwargs_bar_default: dict[str, Any] = {
                "width": width,
            }
        elif bar_orientation == "horizontal":
            ax_hvline = ax.axvline
            ax_barhv = ax.barh
            ax_barhv_cum = ax.barh
            kwargs_bar_default: dict[str, Any] = {
                "height": width,
            }
        else:
            raise ValueError(
                "bar_orientation must be either 'vertical' or 'horizontal'"
            )

        if bar_groupby is not None:
            data_i_groupby = (
                data_i.sort_values(
                    by="id_dataset",
                    key=bar_groupby.sort_key,
                )
                .set_index("id_dataset")
                .groupby(bar_groupby.grouper, sort=False, dropna=False)
            )
        else:
            data_i_groupby = [(None, data_i)]

        for j, (_, data_bar_j) in enumerate(data_i_groupby):
            l = len(data_i_groupby)
            num_points = {}
            num_points_cum = {}

            for y in np.sort(np.unique(data_bar_j["year"].to_numpy())):
                if data == "all":
                    num_points[y] = np.sum(
                        data_bar_j.query(f"year =={y}")["num_points"].to_numpy()
                    )
                elif data == "fitted":
                    num_points[y] = np.sum(
                        data_bar_j.query(f"year =={y}")[
                            "num_points_after_cuts"
                        ].to_numpy()
                    )
                num_points_cum[y] = np.sum(list(num_points.values()))

            num_points_cum[2026] = list(num_points_cum.values())[-1]
            num_points_cum_filled = forward_fill_dict(num_points_cum)

            bar_locs = np.array(list(num_points.keys()))
            bar_locs_cum = np.array(list(num_points_cum_filled.keys()))

            kwargs_bar_default = {"width": width / l, "color": "black"}
            kwargs_bar_cum_default = {"width": width / l, "alpha": 0.3, "color": "grey"}

            if l == 1:
                kwargs_bar_updated = update_kwargs(kwargs_bar_default, kwargs_bar, i)
                kwargs_bar_cum_updated = update_kwargs(
                    kwargs_bar_cum_default, kwargs_bar_cum, i
                )
            else:
                kwargs_bar_updated = update_kwargs(kwargs_bar_default, kwargs_bar, j)
                kwargs_bar_cum_updated = update_kwargs(
                    kwargs_bar_cum_default, kwargs_bar_cum, j
                )

            bar = ax_barhv(
                bar_locs - width / 2 + j * width / l,
                num_points.values(),
                **kwargs_bar_updated,
            )
            bar_cum = ax_barhv_cum(
                bar_locs_cum - width / 2 + j * width / l,
                num_points_cum_filled.values(),
                **kwargs_bar_cum_updated,
            )
            kwargs_annotation_default = {
                "text": rf"$N_\text{{data}}=${list(num_points_cum.values())[-1]}",
                "xy": (0.15, 0.85 - j * 0.1),
                "xycoords": ax.transAxes,
                "color": kwargs_bar_updated["color"],
                "fontsize": 15,
            }
            if l == 1:
                kwargs_annotation_updated = update_kwargs(
                    kwargs_annotation_default, kwargs_annotation, i
                )
            else:
                kwargs_annotation_updated = update_kwargs(
                    kwargs_annotation_default, kwargs_annotation, j
                )
            ax.annotate(**kwargs_annotation_updated)