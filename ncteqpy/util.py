from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def format_unit(variable: str, unit: str) -> str:
    if unit == "":
        return variable
    else:
        return f"{variable} $\\,$[{unit}]"


def clamp(x: int, a: int, b: int) -> int:
    return max(a, min(b, x))


def pdf_uncertainty_sym(x: pd.Series[float]) -> float:
    """Calculate the PDF uncertainty of an observable in the symmetric convention.

    Parameters
    ----------
    x : pd.Series[float]
        Values of the observable evaluated with the central PDF and with the eigenvector PDFs ordered by negative and positive eigenvector directions alternating, i.e., `[central, negative, positive, negative, positive, ...]`.

    Returns
    -------
    float
        PDF Uncertainty of observable `x`
    """
    x_np = x.to_numpy()
    return 0.5 * np.sqrt(np.sum((x_np[2::2] - x_np[1::2]) ** 2))


def pdf_uncertainty_asym_lower(x: pd.Series[float]) -> float:
    """Calculate the lower PDF uncertainty of an observable in the asymmetric convention.

    Parameters
    ----------
    x : pd.Series[float]
        Values of the observable evaluated with the central PDF and with the eigenvector PDFs ordered by negative and positive eigenvector directions alternating, i.e., `[central, negative, positive, negative, positive, ...]`.

    Returns
    -------
    float
        PDF Uncertainty of observable `x`
    """
    x_np = x.to_numpy()
    return np.sqrt(
        np.sum(
            np.min(
                np.stack(
                    [
                        x_np[1::2] - x_np[0],
                        x_np[2::2] - x_np[0],
                        np.zeros_like(x_np[1::2]),
                    ]
                ),
                axis=0,
            )
            ** 2
        )
    )


def pdf_uncertainty_asym_upper(x: pd.Series[float]) -> float:
    """Calculate the lower PDF uncertainty of an observable in the asymmetric convention.

    Parameters
    ----------
    x : pd.Series[float]
        Values of the observable evaluated with the central PDF and with the eigenvector PDFs ordered by negative and positive eigenvector directions alternating, i.e., `[central, negative, positive, negative, positive, ...]`.

    Returns
    -------
    float
        PDF Uncertainty of observable `x`
    """
    x_np = x.to_numpy()
    return np.sqrt(
        np.sum(
            np.max(
                np.stack(
                    [
                        x_np[1::2] - x_np[0],
                        x_np[2::2] - x_np[0],
                        np.zeros_like(x_np[1::2]),
                    ]
                ),
                axis=0,
            )
            ** 2
        )
    )


def get_kwargs(
    kwargs: dict[str, Any] | list[dict[str, Any] | None] | None,
    i: int,
) -> dict[str, Any]:
    if kwargs is None:
        return {}
    elif isinstance(kwargs, dict):
        return kwargs
    elif isinstance(kwargs, list):
        kwargs_i = kwargs[i] if i < len(kwargs) else {}

        return kwargs_i if kwargs_i is not None else {}


def get_kwarg(
    key: str,
    kwargs: dict[str, Any],
    kwargs_user: dict[str, Any] | list[dict[str, Any] | None] | None,
    i: int | None = None,
) -> Any:
    """Return the keyword argument that updating `kwargs` with `kwargs_user` gives.

    Parameters
    ----------
    key : str
        Key to the keyword argument.
    kwargs : dict[str, Any]
        The keyword arguments to update.
    kwargs_user : dict[str, Any] | list[dict[str, Any]  |  None] | None
        The keyword arguments to update with.
    i : int | None, optional
        Index of `kwargs_user` to use if it is a list, by default None

    Returns
    -------
    Any
        The keyword argument accessed by `key`.

    Raises
    ------
    ValueError
        If `kwargs_user` is a list and `i` is not given
    """
    if isinstance(kwargs_user, dict):
        return kwargs_user.get(key, kwargs.get(key))
    elif isinstance(kwargs_user, list):
        if i is not None:
            kwargs_user_i = kwargs_user[i]
            if i < len(kwargs_user) and kwargs_user_i is not None:
                return kwargs_user_i[key]
            else:
                return kwargs.get(key)
        else:
            raise ValueError("i must be given if kwargs_user is list")
    elif kwargs_user is None:
        return kwargs.get(key)
    else:
        raise ValueError("kwargs_user must be dict or list")


def update_kwargs(
    kwargs: dict[str, Any],
    kwargs_user: dict[str, Any] | list[dict[str, Any] | None] | None,
    i: int | None = None,
) -> dict[str, Any]:
    """Update `kwargs` with `kwargs_user`. If `kwargs_user` is a list, `i` must be given to select the correct element.

    Parameters
    ----------
    kwargs : dict[str, Any]
        The keyword arguments to update.
    kwargs_user : dict[str, Any] | list[dict[str, Any]  |  None] | None
        The keyword arguments to update with.
    i : int | None, optional
        Index of `kwargs_user` to use if it is a list, by default None

    Returns
    -------
    dict[str, Any]
        The updated keyword arguments

    Raises
    ------
    ValueError
        If `kwargs_user` is a list and `i` is not given
    """
    if isinstance(kwargs_user, dict):
        return kwargs | kwargs_user
    elif isinstance(kwargs_user, list):
        if i is not None:
            if i < len(kwargs_user) and kwargs_user[i] is not None:
                return kwargs | kwargs_user[i]  # type: ignore[operator] # this is correct but mypy complains
            else:
                return kwargs
        else:
            raise ValueError("i must be given if kwargs_user is list")
    elif kwargs_user is None:
        return kwargs
    else:
        raise ValueError("kwargs_user must be dict or list")
