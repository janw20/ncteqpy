from __future__ import annotations

from math import ceil
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ncteqpy.plot.util import AdditionalLegend
from ncteqpy.util import update_kwargs


class AxesGrid:
    """Class to arrange subplots in a grid, without the need for the number of subplots to be `n_rows` * `n_cols`. The remainder of space in the bottom right corner is left blank."""

    _fig: Figure
    _ax: npt.NDArray[Axes | None]
    _ax_all: npt.NDArray[Axes]
    _ax_real: npt.NDArray[Axes]

    _ax_left: npt.NDArray[Axes]
    _ax_right: npt.NDArray[Axes]
    _ax_bottom: npt.NDArray[Axes]
    _ax_top: npt.NDArray[Axes]
    _ax_inner: npt.NDArray[Axes]

    _n_rows: int
    _n_cols: int
    _n_real: int
    _n_none: int

    _sharex: bool
    _sharey: bool

    _indices_real: npt.NDArray[tuple[int, int]] | None = None
    _indices_none: npt.NDArray[tuple[int, int]] | None = None

    def __init__(
        self, n_real: int, sharex: bool = False, sharey: bool = False, **kwargs: Any
    ) -> None:
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

        if n_real <= 0:
            n_real = 1

        # the nrows and ncols kwargs
        kwargs_naxes = {}

        # if both nrows and ncols are given, we just have to check if the number of subplots is compatible with the values grouped by subplot_groupby
        if "nrows" in kwargs and "ncols" in kwargs:
            if kwargs["nrows"] * kwargs["ncols"] < n_real:
                raise ValueError(
                    f"nrows * ncols must be greater than or equal {n_real}, the number of requested subplots"
                )
            kwargs_naxes["nrows"] = kwargs["nrows"]
            kwargs_naxes["ncols"] = kwargs["ncols"]
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

        kwargs_default = {"layout": "compressed"}

        fig, ax = cast(tuple[Figure, Axes | npt.NDArray[Axes | None]], plt.subplots(sharex=sharex, sharey=sharey, **(kwargs_default | kwargs | kwargs_naxes)))  # type: ignore[type-var]

        self._ax_all = np.atleast_2d(ax)

        self._fig = fig
        self._ax = np.atleast_2d(np.array(ax, copy=True))
        self._ax_real = self._ax.flat[:-n_none] if n_none > 0 else self._ax[0]

        if isinstance(ax, np.ndarray):
            # remove the superfluous axes in the last row
            if n_none > 0:
                self._ax[-1, -n_none:] = None

        self._n_real = n_real
        self._n_none = n_none
        self._n_rows, self._n_cols = self._ax.shape

        mask_left = np.zeros_like(self.ax_real, dtype=bool)
        mask_left[:: self.n_cols] = True
        mask_bottom = np.zeros_like(self.ax_real, dtype=bool)
        mask_bottom[-self.n_cols :] = True

        self._ax_left = self._ax_real[mask_left]
        self._ax_right = self._ax_real[~mask_left]
        self._ax_bottom = self._ax_real[mask_bottom]
        self._ax_top = self._ax_real[~mask_bottom]

        self._ax_outer = self._ax_real[mask_left | mask_bottom]
        self._ax_inner = self._ax_real[~mask_left & ~mask_bottom]

        self._sharex = sharex
        self._sharey = sharey

        if sharex:
            for ax_i in self.ax[:-1, :].flat:
                ax_i: plt.Axes
                ax_i.tick_params("x", which="both", labelbottom=False)
                ax_i.set_xlabel("")

        if sharey:
            for ax_i in self.ax_right:
                ax_i.tick_params("y", which="both", labelleft=False)
                ax_i.set_ylabel("")

        def set_missing_labels(event) -> None:

            self.fig.set_layout_engine("none")

            for i, j in self.indices_none:
                self._ax_all[i, j].set_visible(False)

            if self.n_none > 0:
                for ax_i in self.ax_bottom:
                    ax_i: plt.Axes
                    ax_i.tick_params("x", which="both", labelbottom=True)
                    ax_i.set_xlabel(self.ax_bottom[-1].get_xlabel())

        self.fig.canvas.mpl_connect("draw_event", set_missing_labels)

    @property
    def fig(self) -> Figure:
        """The figure containing the subplots"""
        return self._fig

    @property
    def ax(self) -> npt.NDArray[Axes | None]:
        """numpy.array containing the Axes or None"""
        return self._ax

    @property
    def ax_left(self) -> npt.NDArray[Axes]:
        """numpy.array containing the leftmost axes"""
        return self._ax_left

    @property
    def ax_right(self) -> npt.NDArray[Axes]:
        """numpy.array containing all axes except the leftmost"""
        return self._ax_right

    @property
    def ax_bottom(self) -> npt.NDArray[Axes]:
        """numpy.array containing the bottommost axes"""
        return self._ax_bottom

    @property
    def ax_top(self) -> npt.NDArray[Axes]:
        """numpy.array containing all axes except the bottommost"""
        return self._ax_top

    @property
    def ax_outer(self) -> npt.NDArray[Axes]:
        """numpy.array containing the leftmost and bottommost axes"""
        return self._ax_outer

    @property
    def ax_inner(self) -> npt.NDArray[Axes]:
        """numpy.array containing all axes except the leftmost and bottommost"""
        return self._ax_inner

    @property
    def sharex(self) -> bool:
        """True if xlabels and xticks are shared"""
        return self._sharex

    @property
    def sharey(self) -> bool:
        """True if ylabels and yticks are shared"""
        return self._sharey

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
            self._ax_real = np.array([ax for ax in self.ax.flat if ax is not None])

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

    def set_xlabel(self, xlabel: str, **kwargs: Any) -> None:
        """Set the xlabel of the actual (not left blank) axes that are at the bottom of each column

        Parameters
        ----------
        kwargs
            Keyword arguments passed to `matplotlib.axes.Axes.set_xlabel`
        """

        # other labels are set in the matplotlib draw_event that was connected in __init__
        for ax_i in self._ax_all[-1:].flat:
            ax_i: plt.Axes
            ax_i.set_xlabel(xlabel, **kwargs)

    def set_ylabel(self, ylabel: str, **kwargs: Any) -> None:
        """Set the ylabel of the actual (not left blank) axes that are the first in each row

        Parameters
        ----------
        kwargs
            Keyword arguments passed to `matplotlib.axes.Axes.set_ylabel`
        """

        for ax_i in self.ax_left if self.sharey else self.ax_real:
            ax_i: plt.Axes

            ax_i.set_ylabel(ylabel, **kwargs)

    def set(self, **kwargs) -> None:
        """Set the properties of the actual (not left blank) axes

        Parameters
        ----------
        kwargs
            Keyword arguments passed to `matplotlib.axes.Axes.set`
        """
        for ax_i in self.ax_real:
            ax_i.set(**kwargs)

    def set_labels(
        self,
        labels: list[str],
        kwargs_legend: dict[str, Any] | list[dict[str, Any] | None] = {},
    ) -> None:
        for i, (ax, label) in enumerate(zip(self.ax_real, labels)):
            ax: plt.Axes

            kwargs_default = dict(
                order=0,
                parent=ax,
                handles=[Patch()],
                labels=[label],
                labelspacing=0,
                handlelength=0,
                handleheight=0,
                handletextpad=0,
                fontsize="small",
            )
            kwargs_updated = update_kwargs(kwargs_default, kwargs_legend, i)

            leg2 = AdditionalLegend(**kwargs_updated)
            ax.add_artist(leg2)

    def prune_labels(self) -> None:
        if self.sharex:
            for ax_i in self._ax_all[:-1].flat:
                ax_i: plt.Axes
                ax_i.set_xlabel("")

        if self.sharey:
            for ax_i in self.ax_right:
                ax_i: plt.Axes
                ax_i.set_ylabel("")
