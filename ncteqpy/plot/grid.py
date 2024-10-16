from __future__ import annotations

from math import ceil
from typing import Any, Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend


class AxesGrid:
    """Class to arrange subplots in a grid, without the need for the number of subplots to be `n_rows` * `n_cols`. The remainder of space in the bottom right corner is left blank."""

    _fig: Figure
    _ax: npt.NDArray[Axes | None]

    _n_rows: int
    _n_cols: int
    _n_real: int
    _n_none: int

    _ax_real: npt.NDArray[Axes] | None = None
    _indices_real: npt.NDArray[tuple[int, int]] | None = None
    _indices_none: npt.NDArray[tuple[int, int]] | None = None

    def __init__(self, n_real: int, **kwargs: Any) -> None:
        """Creates `n` subplots in a grid. If `n > nrows * ncols`, the grid entries on the bottom right don't contain `Axes`.

        Parameters
        ----------
        n_real : int
            Actual number of subplot Axes to create
        kwargs : Any
            Keyword arguments passed to `plt.subplots`

        Raises
        ------
        ValueError
            If `n > kwargs[\"nrows\"] * kwargs[\"ncols\"]`
        """

        # the nrows and ncols kwargs
        kwargs_naxes = {}

        # if both nrows and ncols are given, we just have to check if the number of subplots is compatible with the values grouped by subplot_groupby
        if "nrows" in kwargs and "ncols" in kwargs:
            if kwargs["nrows"] * kwargs["ncols"] < n_real:
                raise ValueError(
                    f"nrows * ncols must be greater than or equal {n_real}, the number of requested subplots"
                )
        # if only nrows is given, we determine ncols automatically
        elif "nrows" in kwargs:
            kwargs_naxes["nrows"] = kwargs["nrows"]
            kwargs_naxes["ncols"] = ceil(n_real / kwargs_naxes["nrows"])
        # same for ncols
        elif "ncols" in kwargs:
            kwargs_naxes["ncols"] = kwargs["ncols"]
            kwargs_naxes["nrows"] = ceil(n_real / kwargs_naxes["ncols"])
        # if none of them are given, we try to make the figure as square as possible. ncols is always rounded down since usually the width of a subplot should be larger than the height
        else:
            kwargs_naxes["ncols"] = int(np.sqrt(n_real))
            kwargs_naxes["nrows"] = ceil(n_real / kwargs_naxes["ncols"])

        # how many axes we have to remove in the end
        n_none = kwargs_naxes["nrows"] * kwargs_naxes["ncols"] - n_real

        fig, ax = cast(tuple[Figure, Axes | npt.NDArray[Axes | None]], plt.subplots(**(kwargs | kwargs_naxes)))  # type: ignore[type-var]

        if isinstance(ax, np.ndarray):
            # remove the superfluous axes in the last row
            if n_none > 0:
                for ax_i in cast(Iterable[Axes], ax[-1, -n_none:]):
                    ax_i.remove()

                ax[-1, -n_none:] = None

        self._fig = fig
        self._ax = np.atleast_2d(ax)

        self._n_real = n_real
        self._n_none = n_none
        self._n_rows, self._n_cols = ax.shape if isinstance(ax, np.ndarray) else (1, 1)

    @property
    def fig(self) -> Figure:
        """The figure containing the subplots"""
        return self._fig

    @property
    def ax(self) -> npt.NDArray[Axes | None]:
        """numpy.array containing the Axes or None"""
        return self._ax

    @property
    def n_rows(self) -> int:
        """Number of rows in the grid"""
        return self._n_rows

    @property
    def n_cols(self) -> int:
        """Number of columns in the grid"""
        return self._n_cols

    @property
    def n_real(self) -> int:
        """Number of actual (not left blank) axes"""
        return self._n_real

    @property
    def n_none(self) -> int:
        return self._n_none

    @property
    def ax_real(self) -> npt.NDArray[Axes]:
        """numpy.array containing the actual (not left blank) axes"""
        if self._ax_real is None:
            self._ax_real = np.array([ax for ax in self.ax if ax is not None])

        return self._ax_real

    @property
    def indices_real(self) -> npt.NDArray[tuple[int, int]]:
        """numpy.array containing the indices of the actual (not left blank) axes"""
        if self._indices_real is None:
            self._indices_real = np.array(
                [(i, j) for (i, j), a in np.ndenumerate(self.ax) if a is not None]
            )

        return self._indices_real

    @property
    def indices_none(self) -> npt.NDArray[tuple[int, int]]:
        """numpy.array containing the indices of the axes that are left blank"""
        if self._indices_none is None:
            self._indices_none = np.array(
                [(i, j) for (i, j), a in np.ndenumerate(self.ax) if a is None]
            )

        return self._indices_none

    def set_xlabel(self, **kwargs) -> None:
        """Set the xlabel of the actual (not left blank) axes that are at the bottom of each column

        Parameters
        ----------
        kwargs
            Keyword arguments passed to `matplotlib.axes.Axes.set_xlabel`
        """
        for (i, j), ax_ij in np.ndenumerate(self.ax_real):
            if i == self.n_rows - 1:
                if ax_ij is not None:
                    ax_ij.set_xlabel(**kwargs)
                elif j != 0:
                    self.ax[i - 1, j].set_xlabel(**kwargs)

    def set_ylabel(self, **kwargs) -> None:
        """Set the ylabel of the actual (not left blank) axes that are the first in each row

        Parameters
        ----------
        kwargs
            Keyword arguments passed to `matplotlib.axes.Axes.set_ylabel`
        """
        for ax_i in self.ax_real[:, 0]:
            ax_i.set_ylabel(**kwargs)

    # def set_legend(self, legend: Legend) -> None:
    #     self.fig.add_artist(legend)

    def prune_labels(self) -> None:
        """Set xlabels and ylabels to nothing if they are not at the bottom are to the left in the girid, respectively"""
        for (i, j), ax_ij in np.ndenumerate(self.ax_real):
            # for the axes not in the last row we delete the xlabel
            if i != self.n_rows - 1:
                # except if there is not ax below ax_ij, then we don't delete the xlabel
                if self.ax[i + 1, j] is not None:
                    ax_ij.set_xlabel("")
                    if j != 0:
                        cast(Axes, self.ax[i + 1, j]).tick_params("x", which="both", labelbottom=True)

            # for the axes not in the first column we delete the ylabel
            if j != 0:
                if ax_ij is not None:
                    ax_ij.set_ylabel("")

    # TODO
    def tight_layout(self) -> None:
        """Tighten the layout of the figure"""
        self.fig.tight_layout(w_pad=0.5, h_pad=-2.5)
