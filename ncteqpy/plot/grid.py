from __future__ import annotations

from math import ceil
from typing import Any, cast
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.artist import Artist

from ncteqpy.plot.util import AdditionalLegend
from ncteqpy.util import update_kwargs

SubplotPos = Literal["upper right", "upper left", "lower left", "lower right"] | int


class AxesGrid:
    """Class to arrange subplots in a grid, without the need for the number of subplots to be `n_rows` * `n_cols`. The remainder of space in the bottom right corner is left blank."""

    _fig: Figure
    _ax: npt.NDArray[Axes | None]
    _ax_all: npt.NDArray[Axes]
    _ax_real: npt.NDArray[Axes]
    _ax_unit_real: npt.NDArray[Axes]

    _ax_left: npt.NDArray[Axes]
    _ax_right: npt.NDArray[Axes]
    _ax_bottom: npt.NDArray[Axes]
    _ax_top: npt.NDArray[Axes]
    _ax_inner: npt.NDArray[Axes]

    _n_rows: int
    _n_cols: int
    _n_real: int
    _n_none: int

    _unit_shape: tuple[int, int]
    _n_unit_rows: int
    _n_unit_cols: int
    _n_unit_real: int
    _n_unit_none: int

    _sharex: bool
    _sharey: bool

    _indices_real: npt.NDArray[tuple[int, int]] | None = None
    _indices_none: npt.NDArray[tuple[int, int]] | None = None

    def __init__(
        self,
        n_real: int,
        n_cols: int | None = None,
        n_rows: int | None = None,
        sharex: bool = False,
        sharey: bool = False,
        ax_size: tuple[float, float] = plt.rcParams["figure.figsize"],
        unit_shape: tuple[int, int] = (1, 1),
        unit_width_ratios: list[float] | None = None,
        unit_height_ratios: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates `n_real` subplots in a grid. If `n_real > nrows * ncols`, the grid entries on the bottom right don't contain `Axes`.

        Parameters
        ----------
        n_real : int
            Actual number of subplot Axes to create.
        ax_size : tuple[float, float], optional
            Size of one subplot, by default `plt.rcParams["figure.figsize"]`.
        unit_shape : tuple[int, int], optional
            Axes subarrays of this shape are left blank in the last row.
        unit_width_ratios : float | list[float], optional
            Analogous to the `width_ratios` argument of `plt.subplots`, but only for one unit. If `unit_width_ratios` is None, all Axes in one unit get the same width. `len(unit_width_ratios)` must be equal to `unit_shape[1]`.
        unit_height_ratios : list[float] | None, optional
            Analogous to the `height_ratios` argument of `plt.subplots`, but only for one unit. If `unit_height_ratios` is None, all Axes in one unit get the same height. `len(unit_height_ratios)` must be equal to `unit_shape[0]`.
        kwargs : Any
            Keyword arguments passed to `plt.subplots`

        Raises
        ------
        ValueError
            If `n > kwargs[\"nrows\"] * kwargs[\"ncols\"]`
        """

        if n_real <= 0:
            n_real = 1

        if n_rows is None and "nrows" in kwargs:
            n_rows = kwargs["nrows"]
        if n_cols is None and "ncols" in kwargs:
            n_cols = kwargs["ncols"]

        if n_rows is not None and n_rows % unit_shape[0] != 0:
            raise ValueError("n_rows must be divisible by unit_shape[1]")
        if n_cols is not None and n_cols % unit_shape[1] != 0:
            raise ValueError("n_cols must be divisible by unit_shape[0]")
        if n_real % (unit_shape[0] * unit_shape[1]) != 0:
            raise ValueError(
                "n_real must be divisible by unit_shape[0] * unit_shape[1]"
            )

        if (
            isinstance(unit_height_ratios, list)
            and not len(unit_height_ratios) == unit_shape[0]
        ):
            raise ValueError(f"len(unit_height_ratios) must be {unit_shape[0]}")
        if (
            isinstance(unit_width_ratios, list)
            and not len(unit_width_ratios) == unit_shape[1]
        ):
            raise ValueError(f"len(unit_width_ratios) must be {unit_shape[1]}")

        n_unit_rows = n_rows // unit_shape[0] if n_rows is not None else None
        n_unit_cols = n_cols // unit_shape[1] if n_cols is not None else None
        n_unit_real = n_real // (unit_shape[0] * unit_shape[1])

        # if both nrows and ncols are given, we just have to check if the number of subplots is compatible with the values grouped by subplot_groupby
        if n_unit_rows is not None and n_unit_cols is not None:
            if n_unit_rows * n_unit_cols < n_unit_real:
                raise ValueError(
                    f"n_rows * n_cols must be greater than or equal to n_real"
                )
        # if only nrows is given, we determine ncols automatically
        elif n_unit_rows is not None:
            n_unit_cols = ceil(n_unit_real / n_unit_rows)
        # same for ncols
        elif n_unit_cols is not None:
            n_unit_rows = ceil(n_unit_real / n_unit_cols)
        # if none of them are given, we try to make the figure as square as possible. ncols is always rounded down since usually the width of a subplot should be larger than the height
        else:
            n_unit_cols = int(round(np.sqrt(n_unit_real)))
            n_unit_rows = ceil(n_unit_real / n_unit_cols)

        n_rows = n_unit_rows * unit_shape[0]
        n_cols = n_unit_cols * unit_shape[1]

        # how many axes we have to remove in the end
        n_unit_none = n_unit_rows * n_unit_cols - n_unit_real
        n_none = n_unit_none * unit_shape[0] * unit_shape[1]

        kwargs_default = {
            "layout": "compressed",
            "figsize": (
                ax_size[0] * n_unit_cols,
                ax_size[1] * n_unit_rows,
            ),
            "nrows": n_rows,
            "ncols": n_cols,
        }

        if unit_height_ratios is not None:
            if len(unit_height_ratios) != unit_shape[0]:
                raise ValueError(f"len(unit_height_ratios) must be {unit_shape[0]}")

            kwargs_default["height_ratios"] = n_unit_rows * list(unit_height_ratios)

        if unit_width_ratios is not None:
            if len(unit_width_ratios) != unit_shape[1]:
                raise ValueError(f"len(unit_width_ratios) must be {unit_shape[1]}")

            kwargs_default["width_ratios"] = n_unit_cols * list(unit_width_ratios)

        fig, ax = cast(
            tuple[Figure, Axes | npt.NDArray[Axes | None]],
            plt.subplots(sharex=sharex, sharey=sharey, **(kwargs_default | kwargs)),
        )  # pyright: ignore[reportInvalidTypeForm]

        self._ax_all = np.atleast_2d(ax)

        self._fig = fig
        self._ax = np.atleast_2d(np.array(ax, copy=True))
        
        # self._ax_unit_real = self._ax_real.reshape((n_unit_real, unit_shape[0], unit_shape[1]))
        self._ax_unit_real = np.array([self._ax[
            i // n_unit_cols * unit_shape[0]:(i // n_unit_cols + 1) * unit_shape[0],
            i % n_unit_cols * unit_shape[1]:(i % n_unit_cols + 1) * unit_shape[1]
        ] for i in range(n_unit_real)])

        if isinstance(ax, np.ndarray):
            # remove the superfluous axes in the lower left corner
            if n_none > 0:
                self._ax[-unit_shape[0] :, -n_unit_none * unit_shape[1] :] = None

        self._ax_real = self._ax[self._ax != None].flatten()

        self._n_real = n_real
        self._n_none = n_none
        self._n_rows = n_rows
        self._n_cols = n_cols

        self._unit_shape = unit_shape
        self._n_unit_real = n_unit_real
        self._n_unit_none = n_unit_none
        self._n_unit_rows = n_unit_rows
        self._n_unit_cols = n_unit_cols

        # TFFF
        # TFFF
        # TF
        # TF
        mask_left = np.zeros_like(self.ax_real, dtype=bool)
        mask_left[:: self.n_cols] = True

        # FFFF
        # FFTT
        # FF
        # TT
        mask_bottom = np.zeros_like(self.ax_real, dtype=bool)
        num_axes_last_row = n_cols - n_unit_none * unit_shape[1]
        mask_bottom[-num_axes_last_row:] = True
        mask_bottom[
            -num_axes_last_row * unit_shape[0]
            - n_unit_none * unit_shape[1] : -num_axes_last_row * unit_shape[0]
        ] = True

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
                if ax_i is not None:
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
    def unit_shape(self) -> tuple[int, int]:
        """Shape of a unit"""
        return self._unit_shape

    @property
    def n_unit_rows(self) -> int:
        """Number of unit rows in the grid"""
        return self._n_unit_rows

    @property
    def n_unit_cols(self) -> int:
        """Number of unit columns in the grid"""
        return self._n_unit_cols

    @property
    def n_unit_real(self) -> int:
        """Number of actual (not left blank) units"""
        return self._n_unit_real

    @property
    def n_unit_none(self) -> int:
        return self._n_unit_none

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
        """numpy.array of shape (n_real,) containing the actual (not left blank) axes"""
        return self._ax_real
    
    @property
    def ax_unit_real(self) -> npt.NDArray[Axes]:
        """numpy.array of shape (n_unit_real, unit_shape[0], unit_shape[1]) containing the actual (not left blank) axes"""
        return self._ax_unit_real

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

    def locate_ax(self, pos: SubplotPos) -> plt.Axes:
        if isinstance(pos, int):
            return self.ax_real[pos]
        elif pos == "upper right":
            return self.ax_real[self.n_cols - 1]
        elif pos == "upper left":
            return self.ax_real[0]
        elif pos == "lower left":
            return self.ax_bottom[-1]
        elif pos == "lower right":
            return self.ax_left[-1]
        
    # def locate_ax(self, pos: SubplotPos) -> plt.Axes:
    #     if isinstance(pos, int):
    #         return self.ax_real[pos]
    #     elif pos == "upper right":
    #         return self.ax_real[self.n_cols - 1]
    #     elif pos == "upper left":
    #         return self.ax_real[0]
    #     elif pos == "lower left":
    #         return self.ax_bottom[-1]
    #     elif pos == "lower right":
    #         return self.ax_left[-1]

    def add_artist(self, artist: Artist, pos: SubplotPos | None = None) -> None:
        if pos is not None:
            self.locate_ax(pos).add_artist(artist)

    def prune_labels(self) -> None:
        if self.sharex:
            for ax_i in self._ax_all[:-1].flat:
                ax_i: plt.Axes
                ax_i.set_xlabel("")

        if self.sharey:
            for ax_i in self.ax_right:
                ax_i: plt.Axes
                ax_i.set_ylabel("")
