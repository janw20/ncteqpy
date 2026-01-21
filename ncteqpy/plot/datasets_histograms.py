from __future__ import annotations

from typing import Any, Literal, List, Tuple

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


def plot_datasets_histogram(
    datasets_index: pd.DataFrame,
    x_var:str="A_heavier",
    order: List[Any] | Literal["ascending", "descending"] | None = None,
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
    kwargs_xlabel:  dict[str, Any] | List[dict[str, Any]] = {},
    symlog_y:bool=True,
    linthresh: float= 500,
    x_tick_labels: List[str] | None = None, 
    annotate_bars: bool=True,
    legend:bool=True,
    yticks: Tuple[List[float], List[Any]] | None = None


) -> AxesGrid:
    """Plot histogram of number of data points vs. years.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    """

    if x_var not in datasets_index.columns:
        raise ValueError(f"{x_var} is not a column in datasets_index")
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
        print(f"{data_i[f"{x_var}"].sort_values().to_numpy()[5]}, type: {type(data_i[f"{x_var}"].sort_values().to_numpy()[5])}")
        if all(isinstance(v, float) for v in data_i[f"{x_var}"].sort_values().to_numpy()) or all(isinstance(v, int) for v in data_i[f"{x_var}"].sort_values().to_numpy()):
            x = np.unique(data_i[f"{x_var}"].sort_values().to_numpy())
            number=True
        else:
            if order == "descending":
                x = np.unique(data_i[f"{x_var}"].sort_values().to_numpy())[::-1]
            else:
                x = np.unique(data_i[f"{x_var}"].sort_values().to_numpy())
            number=False
        bar_locs_all = {x_f: i for i,x_f in enumerate(x)}  


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

        num_points = {}

        for j, (_, data_bar_j) in enumerate(data_i_groupby):
            
            bar_locs_j=[]

            l = len(data_i_groupby)
            kwargs_bar_default = {"width": width / l, "color": "black", "label": bar_groupby.labels.to_numpy()[j]}
            kwargs_bar_updated = update_kwargs(kwargs_bar_default, kwargs_bar, j)

            num_points[j] = {}


            for x_f in x:
                if data == "all":
                    if number:
                        num_points[j][x_f] = np.sum(
                            data_bar_j.query(rf"{x_var}=={x_f}")[
                                "num_points_after_cuts"
                            ].to_numpy()
                        )
                    else:
                        num_points[j][x_f] = np.sum(
                            data_bar_j.query(rf"{x_var}=='{x_f}'")[
                                "num_points_after_cuts"
                            ].to_numpy()
                        )   
                elif data == "fitted":
                    if number:
                        num_points[j][x_f] = np.sum(
                            data_bar_j.query(rf"{x_var}=={x_f}")[
                                "num_points_after_cuts"
                            ].to_numpy()
                        )
                    else:
                        num_points[j][x_f] = np.sum(
                            data_bar_j.query(rf"{x_var}=='{x_f}'")[
                                "num_points_after_cuts"
                            ].to_numpy()
                        )                       

                bar_locs_j.append(bar_locs_all[x_f])

            bar = ax_barhv(
                np.array(bar_locs_j) - width / 2 + j * width / l if l%2==0 else np.array(bar_locs_j) - width / 2 +  (j+1/2) * width / l,
                num_points[j].values(),
                **kwargs_bar_updated,
            )
        
        num_points_cum={}
        for x_f in x:
            num_points_cum[x_f]=np.sum([num_points[j][x_f] for j in range(len(data_i_groupby)) if x_f in num_points[j].keys()])


        kwargs_bar_cum_default = {"width": width, "alpha": 0.3, "color": "grey", "zorder":-1}
        
        kwargs_bar_cum_updated = update_kwargs(
            kwargs_bar_cum_default, kwargs_bar_cum, i
        )
        bar_cum = ax_barhv_cum(
            bar_locs_all.values(),
            num_points_cum.values(),
            **kwargs_bar_cum_updated,
            )
        if annotate_bars:
            for f, x_f in enumerate(x):
                kwargs_annotation_default_f = {
                    "text": rf"{num_points_cum[x_f]}",
                    "xy": (float(bar_locs_all[x_f])-width/2, float(num_points_cum[x_f])),
                    "color": "black",
                    "fontsize": 15,
                }
                kwargs_annotation_updated = update_kwargs(
                    kwargs_annotation_default_f, kwargs_annotation, f
                )
                ax.annotate(**kwargs_annotation_updated)
        else:
            kwargs_annotation_default = {
                "text": rf"$N_\text{{data}}=${np.sum(list(num_points_cum.values()))}",
                "xy": (0.15, 0.85),
                "xycoords": ax.transAxes,
                "color": "black",
                "fontsize": 15,
            }

            kwargs_annotation_updated = update_kwargs(
                kwargs_annotation_default, kwargs_annotation, i
            )
            ax.annotate(**kwargs_annotation_updated)

        if symlog_y:
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.hlines(y=linthresh, xmin=bar_locs_all[x[0]]-width/2, xmax=bar_locs_all[x[-1]]+width/2, linestyle="--", color="grey")

        if x_tick_labels:
            ax.set_xticks(np.array(list(bar_locs_all.values())), labels=x_tick_labels)

        if yticks: 
            ax.set_yticks(yticks[0], yticks[1])
        if legend:
            ax.legend()
        
        kwargs_xlabel_default={"xlabel": x_var}
        kwargs_xlabel_updated = update_kwargs(
                kwargs_xlabel_default, kwargs_xlabel, i
        )

        ax.set_xlabel(**kwargs_xlabel_updated)