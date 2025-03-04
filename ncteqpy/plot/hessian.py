from __future__ import annotations
from typing import Sequence, cast

from matplotlib.patches import FancyArrowPatch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy.typing as npt


def plot_eigenvectors(
    ax: plt.Axes,
    eigenvectors: pd.DataFrame,
    parameter_indices: tuple[int, int],
    eigenvector_indices: int | Sequence[int] | None = None,
    minimum: pd.Series | None = None,
) -> None:

    for id_eigenvector, direction in eigenvectors.columns:
        ev = cast(
            npt.NDArray[np.float64],
            eigenvectors.loc[
                list(parameter_indices), (id_eigenvector, direction)
            ].to_numpy(),  # pyright: ignore[reportCallIssue,reportArgumentType]
        )
        min = cast(
            npt.NDArray[np.float64],
            (
                minimum.iloc[
                    list(parameter_indices)
                ].to_numpy()  # pyright: ignore[reportCallIssue,reportArgumentType]
                if minimum is not None
                else np.array([0, 0])
            ),
        )
        # ax.plot(ev[0], ev[1], ls="", marker=".")
        ha = "left" if ev[0] > 0 else "right"
        va = "bottom" if ev[1] > 0 else "top"
        print(ev + min)
        a = ax.annotate(
            text=f"${id_eigenvector}^{direction}$",
            xy=ev + min,  # pyright: ignore[reportArgumentType]
            horizontalalignment=ha,
            verticalalignment=va,
            fontsize="small",
            bbox=dict(
                boxstyle="round,pad=0.1", fc="white", ec="black", lw=0.5, alpha=0.5
            ),
        )
        ax.add_artist(
            FancyArrowPatch(
                min,  # pyright: ignore[reportArgumentType]
                ev + min,  # pyright: ignore[reportArgumentType]
                arrowstyle="->",
                mutation_scale=a.get_size(),  # pyright: ignore[reportAttributeAccessIssue]
                shrinkA=0,
                shrinkB=0,
            )
        )
        ax.update_datalim([min, ev + min])
        ax.autoscale_view()

    # ax._request_autoscale_view()
