from __future__ import annotations

from typing import Any, Sequence, cast
import pandas as pd
import matplotlib.pyplot as plt

from ncteqpy.labels import parameters_cj15_py_to_tex


def plot_scan_1d(
    ax: plt.Axes | Sequence[plt.Axes],
    profile_params: pd.DataFrame,
    profile_chi2: pd.DataFrame | None = None,
    parameter: str | Sequence[str] | None = None,
    minimum: pd.DataFrame | None = None,
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
            chi2_mid = cast(float, profile_chi2[p].iloc[len(profile_chi2[p]) // 2])
            ax[i].plot(
                profile_params[p],
                profile_chi2[p] - chi2_mid,
                color="black",
                label="Total",
                zorder=3,
            )

        if profile_chi2_groups is not None:

            for g in profile_chi2_groups[p]:
                chi2_mid = cast(
                    float,
                    profile_chi2_groups[p, g].iloc[len(profile_chi2_groups[p, g]) // 2],
                )
                ax[i].plot(
                    profile_params[p],
                    profile_chi2_groups[p, g] - chi2_mid,
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
