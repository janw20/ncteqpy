from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing_extensions import Any, Hashable, Sequence

from ncteqpy._typing import SequenceNotStr
from ncteqpy.data_groupby import DatasetsGroupBy
from ncteqpy.labels import parameters_cj15_py_to_tex
from ncteqpy.util import update_kwargs


def plot_scan_1d(
    ax: plt.Axes | Sequence[plt.Axes],
    minimum: pd.DataFrame,  # TODO: should be pd.Series
    profile_params: pd.DataFrame,
    profile_chi2_total: pd.DataFrame | None = None,
    profile_chi2_per_data: pd.DataFrame | None = None,
    parameter: str | Sequence[str] | None = None,
    profile_groupby: DatasetsGroupBy | None = None,
    highlight_groups: Hashable | SequenceNotStr[Hashable] | None = None,
    highlight_important_groups: int | None = None,
    legend: bool = True,
    kwargs_chi2_profile_total: dict[str, Any] | None = None,
    kwargs_chi2_minimum: dict[str, Any] | None = None,
    kwargs_chi2_profiles: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    kwargs_chi2_profiles_not_highlighted: (
        dict[str, Any] | list[dict[str, Any] | None] | None
    ) = None,
    kwargs_legend: dict[str, Any] | None = None,
) -> None:
    """Plot 1D parameter scan(s).

    Parameters
    ----------
    ax : plt.Axes | Sequence[plt.Axes]
        The axes to plot on, needs to be the same length as `parameter`.
    minimum : pd.DataFrame
        The minimum of the χ² function with a column for each parameter and one "chi2" column, which each have one row for the parameter and χ² values in the minimum.
    profile_params: pd.DataFrame
        The values of the profiled parameters with the parameter names as columns and the profiled points as rows.
    profile_chi2_total : pd.DataFrame
        The profile of the total χ² function with a column for each parameter and the profiled points as rows.
    profile_chi2_total : pd.DataFrame
        The profile of the χ² function of each data set with the parameter names and data set IDs as columns and the profiled points as rows.
    parameter : str | Sequence[str] | None, optional
        The parameter(s) whose scan(s) to plot, by default None, meaning all scanned parameters.
    profile_groupby : DatasetsGroupBy | None, optional
        How to group the profiles, by default None, meaning no grouping. Profiles of data sets in the same group are summed.
    highlight_groups : Hashable | SequenceNotStr[Hashable] | None, optional
        Keys of the groups that are highlighted, by default None, meaning all groups are highlighted. By default, groups not highlighted are grayed out.
    highlight_important_groups : int | None, optional
        How many groups with the largest Δχ² to highlight, by default None, meaning all groups are highlighted. The "importance" of a group is determined by the maximum of the absolute Δχ² of the profile.
    legend : bool, optional
        If a legend of the group labels is shown, by default True.
    kwargs_chi2_profile_total : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.plot` for the profile of the total Δχ².
    kwargs_chi2_profiles : dict[str, Any] | list[dict[str, Any] | None], optional
        Keyword arguments to pass to `plt.Axes.plot` for the grouped Δχ² profiles. In case `highlight_groups` or `highlight_important_groups` is passed, these are the highlighted profiles.
    kwargs_chi2_profiles_not_highlighted : dict[str, Any] | list[dict[str, Any] | None], optional
        Keyword arguments to pass to `plt.Axes.plot` for the grouped, non-highlighted Δχ² profiles.
    kwargs_chi2_minimum : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.plot` for the parameter minimum point shown at Δχ² = 0.
    kwargs_legend : dict[str, Any], optional
        Keyword arguments to pass to `plt.Axes.legend` for the legend of the group labels.

    Returns
    -------
    AxesGrid
        The `AxesGrid` that is created when `ax` is None.
    """

    if isinstance(ax, plt.Axes):
        ax = [ax]

    if parameter is None:
        parameter = profile_params.columns.to_list()
    elif isinstance(parameter, str):
        parameter = [parameter]

    if profile_chi2_total is None and profile_chi2_per_data is None:
        raise ValueError(
            "Either `profile_chi2` or `profile_chi2_groups` must be provided."
        )

    if len(ax) != len(parameter):
        raise ValueError(
            "The number of axes must be equal to the number of parameters."
        )

    if len(ax) != len(parameter):
        raise ValueError(
            "The number of axes must be equal to the number of parameters."
        )

    if profile_chi2_per_data is not None:
        if profile_groupby is not None:
            profile_chi2_grouped = (
                profile_chi2_per_data.T.sort_index(
                    level=1,
                    key=(
                        profile_groupby.sort_key
                        if profile_groupby is not None
                        else None
                    ),  # pyright: ignore [reportArgumentType]
                )
                .groupby(
                    lambda x: (x[0], profile_groupby.grouper[x[1]]),
                    sort=False,
                    dropna=False,
                )
                .sum()
                .copy()
            )

            profile_chi2_grouped.index = pd.MultiIndex.from_tuples(
                profile_chi2_grouped.index,
                names=["parameter", profile_groupby.grouper.name],
            )

            profile_chi2_grouped = profile_chi2_grouped.T
        else:
            profile_chi2_grouped = profile_chi2_per_data.copy()
    else:
        profile_chi2_grouped = None

    for p_i, ax_i in zip(parameter, ax):
        if profile_chi2_total is not None:

            kwargs_chi2_total_default = {
                "color": "black",
                "label": "Total",
                "marker": ".",
                "zorder": 4,
            }
            kwargs_chi2_total_updated = update_kwargs(
                kwargs_chi2_total_default, kwargs_chi2_profile_total
            )

            ax_i.plot(
                profile_params[p_i],
                profile_chi2_total[p_i] - minimum["chi2"].iloc[0],
                **kwargs_chi2_total_updated,
            )

        kwargs_chi2_minimum_default = {"marker": "*", "color": "black", "zorder": 4}
        kwargs_chi2_minimum_updated = update_kwargs(
            kwargs_chi2_minimum_default, kwargs_chi2_minimum
        )

        ax_i.plot(minimum[p_i], 0, **kwargs_chi2_minimum_updated)

        if profile_chi2_grouped is not None:
            # group -> chi2 value at the minimum parameter
            chi2_min = pd.Series(
                [
                    np.interp(minimum[p_i], profile_params[p_i], profile_chi2_grouped[p_i, g])[0]  # pyright: ignore[reportIndexIssue]  # fmt: skip
                    for g in profile_chi2_grouped[p_i]
                ],
                index=profile_chi2_grouped[p_i].columns,
            )

            important: pd.Index = (
                (
                    (profile_chi2_grouped[p_i] - chi2_min)
                    .abs()
                    .max(axis=0)
                    .sort_values(ascending=False)
                    .iloc[:highlight_important_groups]
                    .index
                )
                if highlight_important_groups is not None
                else pd.Index([])
            )

            if isinstance(highlight_groups, str):
                highlight_groups = [highlight_groups]

            for j, g in enumerate(profile_chi2_grouped[p_i]):
                if (
                    highlight_groups is not None
                    or highlight_important_groups is not None
                ):
                    if (
                        highlight_groups is not None
                        and g in highlight_groups
                        or g in important
                    ):
                        if profile_groupby is not None:
                            label = profile_groupby.labels[g]
                            props = (
                                profile_groupby.get_props(
                                    by=profile_groupby.keys, of=[g]
                                )
                                .iloc[0]
                                .to_dict()
                            )
                        else:
                            label = g
                            props = {}

                        zorder = 3
                    else:
                        label = None
                        kwargs_chi2_profiles_not_highlighted_default = {"color": "gray"}
                        props = update_kwargs(
                            kwargs_chi2_profiles_not_highlighted_default,
                            kwargs_chi2_profiles_not_highlighted,
                        )
                        zorder = None
                else:
                    if profile_groupby is not None:
                        label = profile_groupby.labels[g]
                        props = (
                            profile_groupby.get_props(by=profile_groupby.keys, of=[g])
                            .iloc[0]
                            .to_dict()
                        )
                    else:
                        label = g
                        props = {}

                    zorder = None

                kwargs_chi2_profiles_default = {
                    "marker": ".",
                    "label": label,
                    "zorder": zorder,
                } | props
                kwargs_chi2_profiles_updated = update_kwargs(
                    kwargs_chi2_profiles_default, kwargs_chi2_profiles, j
                )

                ax_i.plot(
                    profile_params[p_i],
                    profile_chi2_grouped[p_i, g] - chi2_min[g],
                    **kwargs_chi2_profiles_updated,
                )

        ax_i.set_xlim(profile_params[p_i].min(), profile_params[p_i].max())

        ax_i.set(
            xlabel=f"${parameters_cj15_py_to_tex[p_i]}$", ylabel=r"$\Delta \chi^2$"
        )
        ax_i.grid()

        if legend:
            kwargs_legend_default = {}
            kwargs_legend_updated = update_kwargs(kwargs_legend_default, kwargs_legend)
            ax_i.legend(**kwargs_legend_updated)
