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
    order: List[Any] | Literal["ascending", "descending"] | None = None,
    data: Literal["all", "fitted"] = "all",
    bar_groupby: DatasetsGroupBy | None = None,
    subbar_groupby: DatasetsGroupBy | None = None,
    subplot_groupby: DatasetsGroupBy | None = None,
    # bar_order_groupby: str | list[str] | None = None,  # TODO?
    # bar_props_groupby: DatasetsGroupBy | None = None,
    bar_labels: Literal["num_points", "x"] = "num_points",
    bar_orientation: Literal["vertical", "horizontal"] = "vertical",
    share_y: bool = False,
    width: float = 0.8,
    symlog_y:bool=True,
    linthresh: float= 500,
    x_tick_labels: List[str] | None = None, 
    annotate_bars: bool=True,
    annotate_total_number: bool = True,
    legend:bool=True,
    yticks: Tuple[List[float], List[Any]] | None = None,
    kwargs_bar: dict[str, Any] | List[dict[str, Any]] | List[List[dict[str, Any]] ] = {},
    kwargs_subbar: dict[str, Any] | List[dict[str, Any]] = {},
    kwargs_legend: dict[str, Any] = {},
    kwargs_subplots: dict[str, Any] | List[dict[str, Any]] = {},
    kwargs_annotation_bars: dict[str, Any] | List[dict[str, Any]] = {},
    kwargs_annotation_total_number: dict[str, Any] = {},
    kwargs_xlabel:  dict[str, Any] | List[dict[str, Any]] = {},

) -> AxesGrid:
    """Plot histogram of number of data points vs. years.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    """
    x_var=(bar_groupby.keys if not isinstance(bar_groupby.keys, list) else bar_groupby.keys[0]) if bar_groupby is not None else 0
    if x_var not in datasets_index.columns and bar_groupby is not None:
        raise ValueError(f"{x_var} is not a column in datasets_index")
    num_axes = subplot_groupby.groupby.ngroups if subplot_groupby is not None else 1

    kwargs_subplots_default = {"sharex": True, "sharey": share_y}
    kwargs = update_kwargs(kwargs_subplots_default, kwargs_subplots)

    ax_grid = AxesGrid(num_axes, **kwargs)

    if subplot_groupby is not None:
        data_groupby_subplot = (
            datasets_index.sort_values(
                by="id_dataset",
                key=subplot_groupby.sort_key,
            )
            .set_index("id_dataset")
            .groupby(subplot_groupby.grouper, sort=False, dropna=False)
        )
    else:
        data_groupby_subplot = [(None, datasets_index)]

    for i, (ax, (_, data_i)) in enumerate(zip(ax_grid.ax_real, data_groupby_subplot)):
        ax: plt.Axes
        if bar_groupby is not None:
            if all(isinstance(v, float) for v in data_i[f"{x_var}"].sort_values().to_numpy()) or all(isinstance(v, int) for v in data_i[f"{x_var}"].sort_values().to_numpy()):
                x = np.unique(data_i[f"{x_var}"].sort_values().to_numpy())
                number=True
            else:
                x = np.unique(data_i[f"{x_var}"].sort_values().to_numpy())
                number=False
        else: 
            x=[0]
            number=True
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
            data_i_groupby_bar = (
                data_i.sort_values(
                    by="id_dataset",
                    key=bar_groupby.sort_key,
                )
                .set_index("id_dataset")
                .groupby(bar_groupby.grouper, sort=False, dropna=False)
            )
        else:
            data_i_groupby_bar = [(None, data_i)]

        l_j = len(bar_groupby.labels) if bar_groupby != None else 0

        l_k = len(subbar_groupby.labels) if subbar_groupby != None else 0

        num_points={}
        labelsj= bar_groupby.labels if bar_groupby !=None else ["0"]
        labelsk= subbar_groupby.labels if subbar_groupby != None else ["0"]

        for j, (_, data_bar_j) in enumerate(data_i_groupby_bar):
            
            label_j=data_bar_j[f"{bar_groupby.keys}"].iloc[0] if bar_groupby != None else 0
            num_points[f"{label_j}"]={} 

            if subbar_groupby is not None:
                if bar_groupby is not None:
                    data_j_groupby_subbar = (
                        data_bar_j.sort_values(
                            by="id_dataset",
                            key=subbar_groupby.sort_key,
                        )
                        .groupby(subbar_groupby.grouper, sort=False, dropna=False)
                    )
                else: 
                    data_j_groupby_subbar = (
                        data_bar_j.sort_values(
                        by="id_dataset",
                        key=subbar_groupby.sort_key,
                        )
                        .set_index("id_dataset")
                        .groupby(subbar_groupby.grouper, sort=False, dropna=False)
                )
                for k in subbar_groupby.labels:
                    num_points[f"{label_j}"][f"{k}"]=0
            else:
                data_j_groupby_subbar = [(None, data_i)]
                num_points[f"{label_j}"]["0"]=0

            


            for k, (_, data_subbar_k) in enumerate(data_j_groupby_subbar):
                label_k=data_subbar_k[f"{subbar_groupby.keys}"].iloc[0] if subbar_groupby != None else 0
                if bar_groupby is not None:
                    if data == "all":
                        if number:
                            num_points[f"{label_j}"][f"{label_k}"] += np.sum(
                                data_subbar_k.query(rf"{x_var}=={label_j}")[
                                "num_points"
                            ].to_numpy()
                        )
                        else:
                            num_points[f"{label_j}"][f"{label_k}"]+= np.sum(
                                data_subbar_k.query(rf"{x_var}=='{label_j}'")[
                                    "num_points"
                                ].to_numpy()
                        )   
                    elif data == "fitted":
                        if number:
                            num_points[f"{label_j}"][f"{label_k}"] += np.sum(
                                data_subbar_k.query(rf"{x_var}=={label_j}")[
                                    "num_points_after_cuts"
                                ].to_numpy() 
                            )
                        else:
                            num_points[f"{label_j}"][f"{label_k}"] += np.sum(
                                data_subbar_k.query(rf"{x_var}=='{label_j}'")[
                                    "num_points_after_cuts"
                                ].to_numpy() 
                            )
                else:
                    if data == "all":
                        num_points[f"{label_j}"][f"{label_k}"] += np.sum(
                            data_subbar_k[
                            "num_points"
                        ].to_numpy())
                    elif data == "fitted":
                        num_points[f"{label_j}"][f"{label_k}"] += np.sum(
                            data_subbar_k[
                                "num_points_after_cuts"
                            ].to_numpy()) 

        num_points_cum={}
        if subbar_groupby != None: 
            if bar_groupby!=None:
                for k, labelk in enumerate(subbar_groupby.labels):
                    kwargs_subbar_default = {"width": width / l_k, "color": "black", "label": labelk if subbar_groupby else ""}
                    kwargs_subbar_updated = update_kwargs(kwargs_subbar_default, kwargs_subbar, k)
                    bar = ax_barhv(
                        np.array(list(bar_locs_all.values())) - width / 2 + k * width / l_k if l_k%2==0 else np.array(list(bar_locs_all.values())) - width / 2 +  (k+1/2) * width / l_k,
                        [num_points[f"{label_j}"][labelk] for label_j in bar_groupby.labels],
                        **kwargs_subbar_updated,
                    )
                for label_j in bar_groupby.labels:
                    num_points_cum[f"{label_j}"]=np.sum([num_points[f"{label_j}"][k] for k in subbar_groupby.labels]) 
            else:
                for k, labelk in enumerate(subbar_groupby.labels):
                    kwargs_subbar_default = {"width": width / l_k, "color": "black", "label": labelk if subbar_groupby else ""}
                    kwargs_subbar_updated = update_kwargs(kwargs_subbar_default, kwargs_subbar, k)
                    bar = ax_barhv(
                        np.array(list(bar_locs_all.values())) - width / 2 + k * width / l_k if l_k%2==0 else np.array(list(bar_locs_all.values())) - width / 2 +  (k+1/2) * width / l_k,
                        [num_points["0"][labelk]],
                        **kwargs_subbar_updated,
                    )
                num_points_cum["0"]=np.sum([num_points["0"][k] for k in subbar_groupby.labels])
        else:
            if bar_groupby!=None:
                for label_j in bar_groupby.labels:
                    num_points_cum[f"{label_j}"]=np.sum([num_points[f"{label_j}"]["0"]])    
            else: 
                num_points_cum["0"]=np.sum([num_points["0"]["0"]])           
        if subbar_groupby != None:
            kwargs_bar_default = {"width": width, "alpha": 0.3, "color": "grey", "zorder":-1}
            kwargs_bar_updated = update_kwargs(
                kwargs_bar_default, kwargs_bar, i
            )
            bar_cum = ax_barhv_cum(
                list(bar_locs_all.values()),
                list(num_points_cum.values()),
                **kwargs_bar_updated,
            )
        else:
            kwargs_bar_default = {"width": width, "alpha": 0.9}

            for  j, labelj in enumerate(labelsj):     
                kwargs_bar_updated = update_kwargs(
                    kwargs_bar_default, kwargs_bar[i], j
                )

                bar_cum = ax_barhv_cum(
                    list(bar_locs_all.values())[j],
                    num_points_cum[f"{labelj}"],
                    **kwargs_bar_updated,
                )

   
        if annotate_bars:
            for j,labelj in enumerate(labelsj):
                kwargs_annotation_bars_default_f = {
                    "text": rf"{num_points_cum[f"{labelj}"]}" if bar_labels=="num_points" else f"{labelj}",
                    "xy": (float(list(bar_locs_all.values())[j])-width/2, float(num_points_cum[f"{labelj}"])),
                    "color": "black",
                    "fontsize": 15,
                }
                kwargs_annotation_bars_updated = update_kwargs(
                    kwargs_annotation_bars_default_f, kwargs_annotation_bars, j
                )
                ax.annotate(**kwargs_annotation_bars_updated)
        if annotate_total_number:
            kwargs_annotation_total_number_default = {
                "text": rf"$N_\text{{data}}=${np.sum(list(num_points_cum.values()))}",
                "xy": (0.85, 0.85),
                "xycoords": ax.transAxes,
                "color": "black",
                "fontsize": 15,
            }

            kwargs_annotation_total_number_updated = update_kwargs(
                kwargs_annotation_total_number_default, kwargs_annotation_total_number, i
            )
            ax.annotate(**kwargs_annotation_total_number_updated)

        if symlog_y:
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.hlines(y=linthresh, xmin=bar_locs_all[x[0]]-width/2, xmax=bar_locs_all[x[-1]]+width/2, linestyle="--", color="grey")

        if x_tick_labels:
            ax.set_xticks(np.array(list(bar_locs_all.values())), labels=x_tick_labels)

        if yticks: 
            ax.set_yticks(yticks[0], yticks[1])
        if legend:
            ax.legend(**kwargs_legend)
        
        kwargs_xlabel_default={"xlabel": x_var}
        kwargs_xlabel_updated = update_kwargs(
                kwargs_xlabel_default, kwargs_xlabel, i
        )

        ax.set_xlabel(**kwargs_xlabel_updated)