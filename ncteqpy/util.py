from __future__ import annotations
from typing import Any


def format_unit(variable: str, unit: str) -> str:
    if unit == "":
        return variable
    else:
        return f"{variable} $\\,$[{unit}]"


def clamp(x: int, a: int, b: int) -> int:
    return max(a, min(b, x))


def update_kwargs(
    kwargs: dict[str, Any],
    kwargs_user: dict[str, Any] | list[dict[str, Any] | None],
    i: int | None = None,
) -> dict[str, Any]:
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
    else:
        raise ValueError("kwargs_user must be dict or list")
