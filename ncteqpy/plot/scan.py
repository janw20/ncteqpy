from __future__ import annotations

from math import sqrt
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ncteqpy.labels import parameters_cj15_py_to_tex


def plot_scan_1d(
    ax: plt.Axes | Sequence[plt.Axes],
    profile_params: pd.DataFrame,
    minimum: pd.DataFrame,
    profile_chi2: pd.DataFrame | None = None,
    parameter: str | Sequence[str] | None = None,
    profile_chi2_groups: pd.DataFrame | None = None,
    groups_labels: dict[str, str] | None = None,
    dof: int | None = None,
    **kwargs: Any,
) -> None:

    if isinstance(ax, plt.Axes):
        ax = [ax]

    if parameter is None:
        parameter = list(profile_params.columns)
    elif isinstance(parameter, str):
        parameter = [parameter]

    if profile_chi2 is None and profile_chi2_groups is None:
        raise ValueError(
            "Either `profile_chi2` or `profile_chi2_groups` must be provided."
        )

    if len(ax) != len(parameter):
        raise ValueError(
            "The number of axes must be equal to the number of parameters."
        )

    for i, p in enumerate(parameter):

        if profile_chi2 is not None:
            ax[i].plot(
                profile_params[p],
                profile_chi2[p] - minimum["chi2"].iloc[0],
                color="black",
                label="Total",
                marker=".",
                zorder=3,
            )

        ax[i].plot(minimum[p], 0, "*", color="black", zorder=3)

        if profile_chi2_groups is not None:

            for g in profile_chi2_groups[p]:
                # chi2 value at the minimum parameter
                chi2_min = np.interp(
                    minimum[p], profile_params[p], profile_chi2_groups[p, g]
                )

                ax[i].plot(
                    profile_params[p],
                    profile_chi2_groups[p, g] - chi2_min,
                    marker=".",
                    label=groups_labels[g] if groups_labels is not None else None,
                )  # TODO: option to display data IDs?

        if dof is not None:
            ax2 = ax[i].secondary_yaxis(
                "right", functions=(lambda x: x / dof, lambda x: x * dof)
            )
            ax2.set_ylabel(r"$\Delta \chi^2 \: / \: N_\text{d.o.f.}$")

        ax[i].set(xlabel=f"${parameters_cj15_py_to_tex[p]}$", ylabel=r"$\Delta \chi^2$")
        ax[i].grid()

        ax[i].legend()


def plot_scan_2d(
    ax: plt.Axes | Sequence[plt.Axes],
    profile_params: pd.DataFrame,
    minimum: pd.DataFrame,
    profile_chi2: pd.DataFrame,
    parameters: tuple[str, str] | list[tuple[str, str]] | None = None,
    tolerance: float | None = None,
    **kwargs: Any,
) -> None:

    if parameters is None:
        parameters = list(
            zip(
                profile_params.columns.get_level_values(0),
                profile_params.columns.get_level_values(1),
            )
        )
    elif isinstance(parameters, tuple):
        parameters = [parameters]

    if isinstance(ax, plt.Axes):
        ax = [ax]
    elif len(ax) != len(parameters):
        raise ValueError(
            "The number of axes must be equal to the number of parameter pairs."
        )

    if len(profile_params) != len(profile_chi2):
        raise ValueError(
            "The number of parameter points must be equal to the number of chi2 values."
        )

    n = int(sqrt(len(profile_params)))
    if n**2 != len(profile_params):
        raise ValueError("Only scans on square grids are supported.")

    for p, ax_i in zip(parameters, ax):

        ax_i.set_adjustable("box")
        ax_i.set_box_aspect(1)

        image = ax_i.imshow(
            np.reshape(profile_chi2[p] - minimum["chi2"].iloc[0], (n, n)),
            extent=(
                profile_params[*p, 0].min(),
                profile_params[*p, 0].max(),
                profile_params[*p, 1].min(),
                profile_params[*p, 1].max(),
            ),
            interpolation="bicubic",
            origin="lower",
            aspect="auto",
        )

        cb = ax_i.figure.colorbar(image, ax=ax_i)
        cb.set_label(r"$\Delta \chi^2$")

        if tolerance is not None:
            c = ax_i.contour(
                np.reshape(profile_params[*p, 0], (n, n)),
                np.reshape(profile_params[*p, 1], (n, n)),
                np.reshape(profile_chi2[p] - minimum["chi2"].iloc[0], (n, n)),
                levels=[tolerance / 2, tolerance, 2 * tolerance],
                colors="black",
                **kwargs,
            )
            ax_i.clabel(c, c.levels)  # pyright: ignore[reportAttributeAccessIssue]

        ax_i.plot(minimum[p[0]], minimum[p[1]], "*", color="black")

        ax_i.set(
            xlabel=f"${parameters_cj15_py_to_tex[p[0]]}$",
            ylabel=f"${parameters_cj15_py_to_tex[p[1]]}$",
        )
        ax_i.grid()
