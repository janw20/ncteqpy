from __future__ import annotations

from math import ceil
from typing import Any, Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def subplots(n: int, **kwargs: Any) -> tuple[Figure, Axes | npt.NDArray[Axes | None]]:  # type: ignore[type-var]
    """Creates `n` subplots in a grid. If `n > nrows * ncols`, the grid entries on the bottom right don't contain `Axes`.

    Parameters
    ----------
    n : int
        Number of subplots to create
    kwargs : Any
        Keyword arguments passed to `plt.subplots`

    Returns
    -------
    tuple[Figure, Axes | npt.NDArray[Axes | None]]
        Returns the result of the `plt.subplots` call, except that missing grid entries are filled with `None`

    Raises
    ------
    ValueError
        If `n > kwargs[\"nrows\"] * kwargs[\"ncols\"]`
    """

    # the nrows and ncols kwargs
    kwargs_naxes = {}

    # if both nrows and ncols are given, we just have to check if the number of subplots is compatible with the values grouped by subplot_groupby
    if "nrows" in kwargs and "ncols" in kwargs:
        if kwargs["nrows"] * kwargs["ncols"] < n:
            raise ValueError(
                f"nrows * ncols must be greater than or equal {n}, the number of requested subplots"
            )
    # if only nrows is given, we determine ncols automatically
    elif "nrows" in kwargs:
        kwargs_naxes["nrows"] = kwargs["nrows"]
        kwargs_naxes["ncols"] = ceil(n / kwargs_naxes["nrows"])
    # same for ncols
    elif "ncols" in kwargs:
        kwargs_naxes["ncols"] = kwargs["ncols"]
        kwargs_naxes["nrows"] = ceil(n / kwargs_naxes["ncols"])
    # if none of them are given, we try to make the figure as square as possible. ncols is always rounded down since usually the width of a subplot should be larger than the height
    else:
        kwargs_naxes["ncols"] = int(np.sqrt(n))
        kwargs_naxes["nrows"] = ceil(n / kwargs_naxes["ncols"])

    # how many axes we have to remove in the end
    surplus_axes = kwargs_naxes["nrows"] * kwargs_naxes["ncols"] - n

    fig, ax = plt.subplots(**(kwargs | kwargs_naxes))

    # remove the superfluous axes in the last row
    if surplus_axes > 0:
        for ax in cast(Iterable[Axes], ax[-1, -surplus_axes:]):
            ax.remove()

        ax[-1, :-surplus_axes] = None

    return fig, ax
