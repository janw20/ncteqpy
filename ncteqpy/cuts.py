from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import overload, override

import numpy as np
import numpy.typing as npt
import pandas as pd

from ncteqpy import data


@dataclass
class Cuts:
    """Class that represents a collection of cuts."""

    by_type_experiment: dict[str, Cut] = field(default_factory=dict)
    by_dataset_id: dict[int, Cut] = field(default_factory=dict)

    def get(
        self, type_experiment: str | None = None, id_dataset: int | None = None
    ) -> Cut:
        """Get the cuts corresponding to the given type_experiment and id_dataset

        Parameters
        ----------
        type_experiment : str, optional
            Type of experiment, by default None
        id_dataset : int, optional
            Dataset ID, by default None

        Returns
        -------
        Cut
            Cut corresponding to the given type_experiment and id_dataset
        """
        cuts = []
        if type_experiment is not None and type_experiment in self.by_type_experiment:
            cuts.append(self.by_type_experiment[type_experiment])

        if id_dataset is not None and id_dataset in self.by_dataset_id:
            cuts.append(self.by_dataset_id[id_dataset])

        if len(cuts) == 0:
            return Cut_True()
        elif len(cuts) == 1:
            return cuts[0]
        else:
            return cuts[0] & cuts[1]

    def accept(self, points: pd.DataFrame) -> pd.Series[bool]:
        """Determine which rows of a DataFrame pass the cuts

        Parameters
        ----------
        points : pd.DataFrame
            DataFrame containing the points to be tested. There must be columns `id_dataset`, `type_experiment` and one for each variable in the cuts

        Returns
        -------
        pd.Series[bool]
            Series of booleans indicating which rows pass the cuts
        """

        return points.apply(
                lambda row: self.get(
                    id_dataset=row["id_dataset"],
                    type_experiment=row["type_experiment"],
                ).accepts(row),
                # apply,
                axis=1,
            )


class Cuttable(ABC):
    """Abstract class that represents a variable to which a cut can be applied."""

    name: str
    """Name of the variable. This is used to decide if a cut accepts a dictionary or a DataFrame."""

    def __lt__(self, value: float) -> Cut_LessThan:
        return Cut_LessThan(self, value)

    def __le__(self, value: float) -> Cut_LessThanEqual:
        return Cut_LessThanEqual(self, value)

    def __gt__(self, value: float) -> Cut_GreaterThan:
        return Cut_GreaterThan(self, value)

    def __ge__(self, value: float) -> Cut_GreaterThanEqual:
        return Cut_GreaterThanEqual(self, value)

    def __eq__(self, value: float) -> Cut_Equal:  # type: ignore[override] # return type of __eq__ is bool normally
        return Cut_Equal(self, value)

    def __ne__(self, value: float) -> Cut_NotEqual:  # type: ignore[override] # return type of __ne__ is bool normally
        return Cut_NotEqual(self, value)

    @abstractmethod
    def __hash__(self) -> int: ...  # needed so we can make sets of Cuttables


class Cut(ABC):
    """Abstract class that represents a cut on a variable."""

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool:
        """Returns True if the value passes the cut.

        Parameters
        ----------
        value : float | dict[str, float]
            Value to be tested. If a float is passed, `value` is checked for every variable in the cut. If a dictionary is passed, the value corresponding to the variable name (`Cuttable.name`) is inserted

        Returns
        -------
        bool
            True if the value passes the cut, False otherwise
        """
        ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Determine which values pass the cut. This overload behaves as a vectorized version of calling the `accepts` method with a value of type float.

        Parameters
        ----------
        value : npt.NDArray[np.float64]
            Values to be tested.

        Returns
        -------
        npt.NDArray[np.bool_]
            Array of booleans indicating if the values pass the cut.
        """
        ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]:
        """Determine which rows pass the cut. `Cuttables` are checked against their corresponding columns.

        Parameters
        ----------
        value : pd.DataFrame
            Values to be tested.

        Returns
        -------
        pd.Series[bool]
            Series of booleans indicating which rows pass the cut.
        """

    @abstractmethod
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | pd.Series
            | dict[str, float]
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        """Check if `value` passes the cut.

        Parameters
        ----------
        value : numpy.ndarray[float] | pd.Series[bool] | pd.DataFrame
            Value(s) to be tested. If a float is passed, `value` is inserted for every variable in the cut. If a dictionary is passed, the value corresponding to the variable name (`Cuttable.name`) is inserted. Passing an array or a Series behaves as the vectorized version of passing a float, and passing a DataFrame behaves as the vectorized version of passing a dict.

        Returns
        -------
        bool | numpy.ndarray[bool] | pd.Series[bool]
            Indicates if the value(s) pass the cut. If `value` is a float, a bool is returned. If `value` is a numpy array, a numpy array of booleans is returned. If `value` is a Series or a DataFrame, a Series of booleans is returned.
        """
        ...

    @abstractmethod
    def variables(self) -> set[Cuttable]:
        """Returns as a set the variables appearing in the cut."""
        ...

    def __and__(self, other: Cut) -> Cut_And:
        return Cut_And(self, other)

    def __or__(self, other: Cut) -> Cut_Or:
        return Cut_Or(self, other)

    def __invert__(self) -> Cut_Not:
        return Cut_Not(self)

    def __xor__(self, other: Cut) -> Cut_Xor:
        return Cut_Xor(self, other)


@dataclass(frozen=True)
class Cut_RelOp(Cut, ABC):
    """Relational operation between a Cuttable and a value."""

    variable: Cuttable
    value: float

    @override
    def variables(self) -> set[Cuttable]:
        return {self.variable}


@dataclass(frozen=True)
class Cut_UnLogOp(Cut, ABC):
    """Unary logical operation on a Cut."""

    cut: Cut

    @override
    def variables(self) -> set[Cuttable]:
        return self.cut.variables()


@dataclass(frozen=True)
class Cut_BinLogOp(Cut, ABC):
    """Binary logical operation between two Cuts."""

    cut1: Cut
    cut2: Cut

    @override
    def variables(self) -> set[Cuttable]:
        return self.cut1.variables() | self.cut2.variables()


@dataclass(frozen=True)
class Cut_True(Cut):
    """Cut that always is True."""

    def __repr__(self) -> str:
        return "True"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        if isinstance(value, np.ndarray):
            return np.full(value.shape, True)
        elif isinstance(value, pd.DataFrame):
            return pd.Series(value.shape[0] * [True])
        else:
            return True

    @override
    def variables(self) -> set[Cuttable]:
        return set()


@dataclass(frozen=True)
class Cut_False(Cut):
    """Cut that always is False."""

    def __repr__(self) -> str:
        return "False"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        if isinstance(value, np.ndarray):
            return np.full(value.shape, False)
        elif isinstance(value, pd.DataFrame):
            return pd.Series(value.shape[0] * [False])
        else:
            return False


@dataclass(frozen=True)
class Cut_LessThan(Cut_RelOp):

    def __repr__(self) -> str:
        return f"{self.variable} < {self.value}"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool] | pd.DataFrame:
        if isinstance(value, (dict, pd.Series, pd.DataFrame)):
            if self.variable.name in value:
                return value[self.variable.name] < self.value
            else:
                return (
                    pd.Series(value.shape[0] * [True])
                    if isinstance(value, pd.DataFrame)
                    else True
                )
        else:
            return value < self.value


@dataclass(frozen=True)
class Cut_LessThanEqual(Cut_RelOp):

    def __repr__(self) -> str:
        return f"{self.variable} <= {self.value}"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        if isinstance(value, (dict, pd.Series, pd.DataFrame)):
            return value[self.variable.name] <= self.value
        else:
            return value <= self.value


@dataclass(frozen=True)
class Cut_GreaterThan(Cut_RelOp):

    def __repr__(self) -> str:
        return f"{self.variable} > {self.value}"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        if isinstance(value, (dict, pd.Series, pd.DataFrame)):
            return value[self.variable.name] > self.value
        else:
            return value > self.value


@dataclass(frozen=True)
class Cut_GreaterThanEqual(Cut_RelOp):

    def __repr__(self) -> str:
        return f"{self.variable} >= {self.value}"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        if isinstance(value, (dict, pd.Series, pd.DataFrame)):
            return value[self.variable.name] >= self.value
        else:
            return value >= self.value


@dataclass(frozen=True)
class Cut_Equal(Cut_RelOp):

    def __repr__(self) -> str:
        return f"{self.variable} == {self.value}"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        if isinstance(value, (dict, pd.Series, pd.DataFrame)):
            return value[self.variable.name] == self.value
        else:
            return value == self.value


@dataclass(frozen=True)
class Cut_NotEqual(Cut_RelOp):

    def __repr__(self) -> str:
        return f"{self.variable} != {self.value}"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        if isinstance(value, (dict, pd.Series, pd.DataFrame)):
            return value[self.variable.name] != self.value
        else:
            return value != self.value


@dataclass(frozen=True)
class Cut_And(Cut_BinLogOp):

    def __repr__(self) -> str:
        return f"({self.cut1}) & ({self.cut2})"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)):
            return self.cut1.accepts(value) & self.cut2.accepts(value)
        else:
            return self.cut1.accepts(value) and self.cut2.accepts(value)


@dataclass(frozen=True)
class Cut_Or(Cut_BinLogOp):

    def __repr__(self) -> str:
        return f"({self.cut1}) | ({self.cut2})"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)):
            return self.cut1.accepts(value) | self.cut2.accepts(value)
        else:
            return self.cut1.accepts(value) or self.cut2.accepts(value)


@dataclass(frozen=True)
class Cut_Xor(Cut_BinLogOp):

    def __repr__(self) -> str:
        return f"({self.cut1}) ^ ({self.cut2})"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        return self.cut1.accepts(value) != self.cut2.accepts(value)


@dataclass(frozen=True)
class Cut_Not(Cut_UnLogOp):

    def __repr__(self) -> str:
        return f"~({self.cut})"

    @overload
    def accepts(self, value: float | dict[str, float] | pd.Series[float]) -> bool: ...

    @overload
    def accepts(self, value: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]: ...

    @overload
    def accepts(self, value: pd.DataFrame) -> pd.Series[bool]: ...

    @override
    def accepts(
        self,
        value: (
            float
            | npt.NDArray[np.float64]
            | dict[str, float]
            | pd.Series
            | pd.DataFrame
        ),
    ) -> bool | npt.NDArray[np.bool_] | pd.Series[bool]:
        if isinstance(value, (np.ndarray, pd.DataFrame)):
            return ~self.cut.accepts(value)
        else:
            return not self.cut.accepts(value)
