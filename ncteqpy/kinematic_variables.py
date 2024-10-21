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
YMIN = KinematicVariable("y_min")
YMAX = KinematicVariable("y_max")
X = KinematicVariable("x")
Q2 = KinematicVariable("Q2")
SQRT_S = KinematicVariable("sqrt_s")
