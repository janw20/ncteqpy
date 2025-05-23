from __future__ import annotations

from dataclasses import dataclass

from ncteqpy.cuts import Cuttable


@dataclass(frozen=True, eq=False, order=False, unsafe_hash=True)
class KinematicVariable(Cuttable):
    name: str


PT = KinematicVariable("pT")
PT_MIN = KinematicVariable("pT_min")
PT_MAX = KinematicVariable("pT_max")
Y = KinematicVariable("y")
Y_MIN = KinematicVariable("y_min")
Y_MAX = KinematicVariable("y_max")
X = KinematicVariable("x")
Q2 = KinematicVariable("Q2")
W2 = KinematicVariable("W2")
SQRT_S = KinematicVariable("sqrt_s")

label_to_kinvar: dict[str, KinematicVariable] = {
    "pT": PT,
    "pT_min": PT_MIN,
    "pT_max": PT_MAX,
    "y": Y,
    "y_min": Y_MIN,
    "y_max": Y_MAX,
    "x": X,
    "Q2": Q2,
    "W2": W2,
    "sqrt_s": SQRT_S,
}
