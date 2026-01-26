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
    subbar_groupby: DatasetsGroupBy | None = None,
    subplot_groupby: DatasetsGroupBy | None = None,
    # bar_order_groupby: str | list[str] | None = None,  # TODO?
    # bar_props_groupby: DatasetsGroupBy | None = None,
    bar_labels: Literal["num_points", "year"] = "num_points" ,
    bar_orientation: Literal["vertical", "horizontal"] = "vertical",
    share_y: bool = False,
    width: float = 0.8,
    annotate_bars: bool=True,
    annotate_total_number: bool = True,
    legend:bool=True,
    cumulated: Literal["separated", "unified"]= "unified",
    kwargs_subbar: dict[str, Any] | List[dict[str, Any]] = {},
    kwargs_bar_cum: dict[str, Any] | List[dict[str, Any]] = {},
    kwargs_bar_label: dict[str, Any] = {},
    kwargs_legend: dict[str, Any] = {},
    kwargs_subplots: dict[str, Any] | List[dict[str, Any]] = {},
    kwargs_annotation_bars: dict[str, Any] | List[dict[str, Any]] = {},
    kwargs_annotation_total_number: dict[str, Any] | List[dict[str, Any]] = {},
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

    years_with_data=np.sort(np.unique(datasets_index["year"].to_numpy()))

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

    labels_subplot=subplot_groupby.labels.to_numpy() if subplot_groupby is not None else [""]

    for i, (ax, label_i, (_, data_i)) in enumerate(zip(ax_grid.ax_real, labels_subplot, data_groupby)):
        num_points_cum_all={}
        num_points_all={}
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

        if subbar_groupby is not None:
            data_i_groupby = (
                data_i.sort_values(
                    by="id_dataset",
                    key=subbar_groupby.sort_key,
                )
                .set_index("id_dataset")
                .groupby(subbar_groupby.grouper, sort=False, dropna=False)
            )
        else:
            data_i_groupby = [(None, data_i)]
        

        l_j=len(data_i_groupby)
        labels_subbar=subbar_groupby.labels.to_numpy() if subbar_groupby is not None else [""]

        for j, (label_j,  (_, data_subbar_j)) in enumerate(zip(labels_subbar ,data_i_groupby)):
            l = len(data_i_groupby)
            num_points = {}
            num_points_cum = {}

            for y in np.sort(np.unique(data_subbar_j["year"].to_numpy())):
                if data == "all":
                    num_points[y] = np.sum(
                        data_subbar_j.query(f"year =={y}")["num_points"].to_numpy()
                    )
                elif data == "fitted":
                    num_points[y] = np.sum(
                        data_subbar_j.query(f"year =={y}")[
                            "num_points_after_cuts"
                        ].to_numpy()
                    )
                num_points_cum[y] = np.sum(list(num_points.values()))
                if y in num_points_all.keys():
                    num_points_all[y]+=num_points[y]
                else:
                    num_points_all[y]=num_points[y]

            bar_locs = np.array(list(num_points.keys()))

            kwargs_subbar_default = {"width": width / l, "color": "black", "label": f"{label_i} {label_j}" }

            if l == 1:
                kwargs_subbar_updated = update_kwargs(kwargs_subbar_default, kwargs_subbar, i)

            else:
                kwargs_subbar_updated = update_kwargs(kwargs_subbar_default, kwargs_subbar, j)


            bar = ax_barhv(
                bar_locs - width / 2 + j * width / l,
                num_points.values(),
                **kwargs_subbar_updated,
            )
            num_points_cum[2026] = list(num_points_cum.values())[-1]
            num_points_cum_filled = forward_fill_dict(num_points_cum)
            if cumulated=="separated":

                bar_locs_cum = np.array(list(num_points_cum_filled.keys()))

                kwargs_bar_cum_default = {"width": width / l, "alpha": 0.3, "color": "grey"}

                if l == 1:
                    kwargs_bar_cum_updated = update_kwargs(
                        kwargs_bar_cum_default, kwargs_bar_cum, i
                    )
                else:
                    kwargs_bar_cum_updated = update_kwargs(
                        kwargs_bar_cum_default, kwargs_bar_cum, j
                    )
                bar_cum = ax_barhv_cum(
                    bar_locs_cum - width / 2 + j * width / l,
                    num_points_cum_filled.values(),
                    **kwargs_bar_cum_updated,
                )
            else: 
                for y in list(num_points_cum_filled.keys()):
                    if y not in num_points_cum_all.keys():
                        num_points_cum_all[y]=num_points_cum_filled[int(y)]
                    else:
                        num_points_cum_all[y]+=num_points_cum_filled[int(y)]

        if cumulated=="unified":

            num_points_cum_filled_all=forward_fill_dict(num_points_cum_all)
            bar_locs_cum = np.array(list(num_points_cum_filled_all.keys()))
            kwargs_bar_cum_default = {"width": width, "alpha": 0.3, "color": "grey"}

            kwargs_bar_cum_updated = update_kwargs(
                    kwargs_bar_cum_default, kwargs_bar_cum, i
                )
            bar_cum = ax_barhv_cum(
                bar_locs_cum - width / 2,
                num_points_cum_filled_all.values(),
                **kwargs_bar_cum_updated,
            )
        if annotate_bars:
            for j,y in enumerate(list(num_points_all.keys())):
                kwargs_annotation_bars_default_f = {
                    "text": rf"{num_points_all[y]}" if bar_labels=="num_points" else bar_labels=="year" f"{y}",
                    "xy": (y-width/2, float(num_points_all[y])) if l_j==1 or cumulated=="unified" else (y, float(num_points_all[y])),
                    "color": "black",
                    "fontsize": 12,
                    "rotation": 0,
                    "ha": "center",
                    "va": "center",
                }
                kwargs_annotation_bars_updated = update_kwargs(
                    kwargs_annotation_bars_default_f, kwargs_annotation_bars, j
                )
                ax.annotate(**kwargs_annotation_bars_updated)
        if annotate_total_number:
            kwargs_annotation_total_number_default = {
                "text": rf"$N_\text{{data}}=${list(num_points_cum_filled.values())[-1]}",
                "xy": (0.15, 0.85),
                "xycoords": ax.transAxes,
                "color": "black",
                "fontsize": 15,
            }
            kwargs_annotation_total_number_updated = update_kwargs(
                kwargs_annotation_total_number_default, kwargs_annotation_total_number, i
            )
            ax.annotate(**kwargs_annotation_total_number_updated)
            ax.annotate(**kwargs_annotation_total_number_updated)
        if legend:
            ax.legend()