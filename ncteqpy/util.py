from __future__ import annotations


def format_unit(variable: str, unit: str) -> str:
    if unit == "":
        return variable
    else:
        return f"{variable} $\\,$[{unit}]"


def clamp(x: int, a: int, b: int) -> int:
    return max(a, min(b, x))
