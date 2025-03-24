from __future__ import annotations
from typing import Any, overload


def format_unit(variable: str, unit: str) -> str:
    if unit == "":
        return variable
    else:
        return f"{variable} $\\,$[{unit}]"


def clamp(x: int, a: int, b: int) -> int:
    return max(a, min(b, x))


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
