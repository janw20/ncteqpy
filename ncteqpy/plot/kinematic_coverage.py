from __future__ import annotations

from itertools import cycle

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.typing as npt
import pandas as pd
import sympy as sp
from typing_extensions import Any, Hashable, cast

from ncteqpy.data_groupby import DatasetsGroupBy
from ncteqpy.util import update_kwargs


def plot_kinematic_coverage(
    ax: plt.Axes,
    points: pd.DataFrame,
    points_before_cuts: pd.DataFrame | None = None,
    groupby: DatasetsGroupBy | None = None,
    kinematic_variables: tuple[str, str] = ("x", "Q2"),
    cuts: list[tuple[float | sp.Rel | sp.Expr, npt.NDArray[np.floating]]] | None = None,
    cuts_labels: list[tuple[float, str] | None] | None = None,
    cuts_labels_offset: float | list[float | None] | None = None,
    kwargs_points: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    kwargs_points_before_cuts: (
        dict[str, Any] | list[dict[str, Any] | None] | None
    ) = None,
    kwargs_cuts: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    kwargs_cuts_labels: dict[str, Any] | list[dict[str, Any] | None] | None = None,
) -> None:
    """Plots the data points in the plane of 2 kinematic variables (by default x and Q²).

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    points : pd.DataFrame
        The data points to plot, by default, in color.
    points_before_cuts : pd.DataFrame | None, optional
        The data points to plot, by default, in gray.
    groupby : str | pd.Series, optional
        Column of `points` or `points_before_cuts` to group by, by default "id_dataset"
    kinematic_variables : tuple[str, str], optional
        Kinematic variables to display on the x and y axis, by default ("x", "Q2"). Must be in the columns of `points` and `points_before_cuts`.
    cuts : list[tuple[float | sp.Rel | sp.Expr, npt.NDArray[np.floating]]] | None, optional
        Cuts to display as curves, by default None. A cut is given as a tuple with first element a float (for a constant cut on the y axis) or a sympy expression that has at most one free variable which represents the x axis values, and second element a numpy array that gives the x axis values of the curve. For example, to display a W² cut on the (x, Q²) plane, pass
        ```
            cuts=[(nc.Q2_dis.subs({nc.W2: 1.7**2}), np.logspace(-0.7, -1e-3, 200))]
        ```
        (where `ncteqpy` is imported as `nc` and `numpy` is imported as `np`).
    cuts_labels : list[tuple[float, str]  |  None] | None, optional
        Labels to annotate the cuts by, by default None. These must be in the same ordering as `cuts`.
    cuts_labels_offset : float | list[float  |  None] | None, optional
        Offset in units of font size to shift the label orthogonally away from the curve representing a cut, by default None. Must be in the same order as `cuts` and `cuts_labels`.
    kwargs_points : dict[str, Any] | list[dict[str, Any] | None] | None, optional
        Keyword arguments to adjust plotting the points, passed to `ax.plot`, by default None.
    kwargs_points_before_cuts : dict[str, Any] | list[dict[str, Any] | None] | None, optional
        Keyword arguments to adjust plotting the points before cuts, passed to `ax.plot`, by default None.
    kwargs_cuts : dict[str, Any] | list[dict[str, Any] | None] | None, optional
        Keyword arguments to adjust plotting the cuts, passed to `ax.plot`, by default None. If a `list` is passed, it must be in the same order as `cuts`.
    kwargs_cuts_labels : dict[str, Any] | list[dict[str, Any] | None] | None, optional
        Keyword arguments to adjust plotting the labels of the cuts, passed to `ax.annotate`, by default None. If a `list` is passed, it must be in the same order as `cuts_labels`.
    """

    markers = cycle(["o", "v", "p", "^", "P", ">", "*", "<", "X", "D", "h"])

    grouper = groupby.grouper if groupby is not None else "id_dataset"
    sort_key = groupby.sort_key if groupby is not None else None

    points_gb = (
        points.set_index("id_dataset")
        .sort_index(key=sort_key)  # pyright: ignore[reportArgumentType]
        .groupby(grouper, sort=False, dropna=False)
    )
    points_before_cuts_gb = (
        (
            points_before_cuts.set_index("id_dataset")
            .sort_index(key=sort_key)  # pyright: ignore[reportArgumentType]
            .groupby(grouper, sort=False, dropna=False)
        )
        if points_before_cuts is not None
        else None
    )

    for i, (label_i, p_i) in enumerate(points_gb):
        marker = next(markers)

        if (
            points_before_cuts_gb is not None
            and label_i in points_before_cuts_gb.groups
        ):
            kwargs_default = {
                "marker": marker,
                "markersize": 3,
                "ls": "",
            }
            if groupby is not None:
                kwargs_default = update_kwargs(
                    kwargs_default,
                    groupby.get_props(
                        by=cast(Hashable | list[Hashable], groupby.keys), of=[label_i]
                    )
                    .iloc[0]
                    .to_dict(),
                )
            kwargs_default = update_kwargs(
                kwargs_default, {"color": "lightgray", "zorder": 1.1}
            )

            kwargs = update_kwargs(kwargs_default, kwargs_points_before_cuts, i)

            points_before_cuts_i = points_before_cuts_gb.get_group(label_i)

            ax.plot(
                points_before_cuts_i[kinematic_variables[0]],
                points_before_cuts_i[kinematic_variables[1]],
                **kwargs,
            )

        kwargs_default = {
            "marker": marker,
            "markersize": 3,
            "ls": "",
            "zorder": 1.2,
            "label": groupby.labels[label_i] if groupby is not None else str(label_i),
        }
        if groupby is not None:
            kwargs_default = update_kwargs(
                kwargs_default,
                groupby.get_props(
                    by=cast(Hashable | list[Hashable], groupby.keys),
                    of=[label_i],
                )
                .iloc[0]
                .to_dict(),
            )

        kwargs = update_kwargs(kwargs_default, kwargs_points, i)

        ax.plot(p_i[kinematic_variables[0]], p_i[kinematic_variables[1]], **kwargs)

    if cuts is not None:
        for i, cut in enumerate(cuts):
            cut_expr, cut_x = cut
            if (
                isinstance(cut_expr, (sp.Rel, sp.Expr))
                and len(cut_expr.free_symbols) != 0
            ):
                cut_y = sp.lambdify(tuple(cut_expr.free_symbols), cut_expr)(cut_x)
            else:
                # fmt: off
                cut_y = float(cut_expr) * np.ones_like(cut_x)  # pyright: ignore[reportArgumentType]  # if the sympy expression does not have free symbols, it can be converted to float
                # fmt: off

            kwargs_default = {
                "color": "black",
                "ls": (0, (5, 7)),
                "lw": 0.8,
                "zorder": 1.3,
                "scalex": False,
                "scaley": False,
            }
            kwargs = update_kwargs(kwargs_default, kwargs_cuts, i)
            ax.plot(
                cut_x,
                cut_y,
                **kwargs,
            )

            if cuts_labels is not None and i < len(cuts_labels):
                cut_label = cuts_labels[i]
                if cut_label is not None:
                    label_x = cut_label[0]
                    label_y = np.interp(label_x, cut_x, cut_y)

                    i_x = np.searchsorted(cut_x, label_x, side="left")
                    delta_x = cut_x[i_x] - cut_x[i_x - 1]
                    delta_y = cut_y[i_x] - cut_y[i_x - 1]

                    kwargs_default = {
                        "text": cut_label[1],
                        "xy": (label_x, label_y),
                        "xytext": (label_x, label_y),
                        "ha": "center",
                        "va": "top",
                        "rotation": np.rad2deg(np.atan2(delta_y, delta_x)),
                        "rotation_mode": "anchor",
                        "transform_rotates_text": True,
                        "zorder": 1.4,
                        "fontsize": "x-small",
                        "bbox": dict(
                            facecolor=(1, 1, 1),
                            alpha=0.5,
                            edgecolor=(0.8, 0.8, 0.8),
                            lw=0,
                            boxstyle="round,pad=0,rounding_size=0.2",
                        ),
                    }

                    kwargs = update_kwargs(kwargs_default, kwargs_cuts_labels, i)

                    # add "textcoords" after update_kwargs because we need the updated font size
                    if "textcoords" not in kwargs and cuts_labels_offset is not None:
                        offset = (
                            cuts_labels_offset[i]
                            if isinstance(cuts_labels_offset, list)
                            else cuts_labels_offset
                        )

                        if offset is not None:
                            if kwargs["va"] == "bottom":
                                offset *= -1

                            # transform delta (the tangent) to display coordinates (dots)
                            delta_tr = ax.transData.transform(
                                np.array([delta_x, delta_y])
                            )
                            # rotate by -90 degrees for orthogonal offset (must be done in display coordinates)
                            delta_tr_rotated = np.array([delta_tr[1], -delta_tr[0]])
                            # normalize amd multiply with offset
                            delta_tr_norm = delta_tr_rotated / np.linalg.norm(
                                delta_tr_rotated
                            )
                            delta_tr_offset = offset * delta_tr_norm

                            # get the font size (done like this to resolve "x-small" etc to a number)
                            t = mtext.Text()
                            t.set_fontsize(kwargs["fontsize"])
                            fontsize = t.get_fontsize()

                            # offset the label by translating in the transform (see https://matplotlib.org/stable/users/explain/artists/transforms_tutorial.html#using-offset-transforms-to-create-a-shadow-effect)
                            kwargs["textcoords"] = (
                                ax.transData
                                + mtransforms.ScaledTranslation(
                                    delta_tr_offset[0],
                                    delta_tr_offset[1],
                                    mtransforms.Affine2D().scale(
                                        fontsize * ax.figure.dpi / 72
                                    ),
                                )
                            )

                    ax.annotate(**kwargs)
