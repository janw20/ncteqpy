from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence
from typing_extensions import Literal, overload

import pandas as pd
import sympy as sp

from ncteqpy.kinematic_variables import label_to_kinvar


@overload
def cut_accepts(
    cut: sp.Rel | sp.logic.boolalg.BooleanTrue, values: pd.Series
) -> bool: ...


@overload
def cut_accepts(
    cut: sp.Rel | sp.logic.boolalg.BooleanTrue, values: pd.DataFrame
) -> pd.Series[bool]: ...


def cut_accepts(
    cut: sp.Rel | sp.logic.boolalg.BooleanTrue, values: pd.Series[float] | pd.DataFrame
) -> bool | pd.Series[bool]:
    if isinstance(values, pd.DataFrame):
        return pd.Series(
            sp.lambdify(label_to_kinvar.values(), cut)(
                **values[list(label_to_kinvar.keys())]
            ),
            index=values.index,
        )
    else:
        return sp.lambdify(label_to_kinvar.values(), cut)(
            **values[list(label_to_kinvar.keys())]
        )


@dataclass
class Cuts:
    """Class that represents a collection of cuts."""

    by_type_experiment: dict[str, sp.Rel] = field(default_factory=dict)
    by_dataset_id: dict[int, sp.Rel] = field(default_factory=dict)

    def get(
        self, type_experiment: str | None = None, id_dataset: int | None = None
    ) -> sp.Rel | sp.logic.boolalg.BooleanTrue:
        """Get the cuts corresponding to the given `type_experiment` and `id_dataset`.

        Parameters
        ----------
        type_experiment : str, optional
            Type of experiment, by default None
        id_dataset : int, optional
            Dataset ID, by default None

        Returns
        -------
        sympy.Rel | sympy.logic.boolalg.BooleanTrue
            Relation corresponding to the cut of the given `type_experiment` and `id_dataset`, or `sympy.true` if neither of them have cuts.
        """
        cuts = []
        if type_experiment is not None and type_experiment in self.by_type_experiment:
            cuts.append(self.by_type_experiment[type_experiment])

        if id_dataset is not None and id_dataset in self.by_dataset_id:
            cuts.append(self.by_dataset_id[id_dataset])

        if len(cuts) == 0:
            return sp.true
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
            lambda row: cut_accepts(
                self.get(
                    id_dataset=row["id_dataset"],
                    type_experiment=row["type_experiment"],
                ),
                row,
            ),
            axis=1,
        )

    @overload
    def filter(
        self,
        include: Sequence[int] | None = None,
        exclude: Sequence[int] | None = None,
        *,
        inplace: Literal[False] = False,
    ) -> Cuts:
        """Apply filters to the data sets.

        Parameters
        ----------
        include : Sequence[int] | None, optional
            Data set IDs to include, by default None.
        exclude : Sequence[int] | None, optional
            Data set IDs to include, by default None, by default None.

        Returns
        -------
        Cuts
            Cuts with filtered data sets.
        """
        ...

    @overload
    def filter(
        self,
        include: Sequence[int] | None = None,
        exclude: Sequence[int] | None = None,
        *,
        inplace: Literal[True],
    ) -> None:
        """Apply filters to the data sets in place.

        Parameters
        ----------
        include : Sequence[int] | None, optional
            Data set IDs to include, by default None.
        exclude : Sequence[int] | None, optional
            Data set IDs to include, by default None, by default None.
        inplace : bool, optional
            True if this `Cuts` instance should be modified, False if a new instance should be returned, by default False.
        """
        ...

    def filter(
        self,
        include: Sequence[int] | None = None,
        exclude: Sequence[int] | None = None,
        *,
        inplace: bool = False,
    ) -> Cuts | None:
        """Apply filters to the data sets.

        Parameters
        ----------
        include : Sequence[int] | None, optional
            Data set IDs to include, by default None.
        exclude : Sequence[int] | None, optional
            Data set IDs to include, by default None, by default None.
        inplace : bool, optional
            True if this `Cuts` instance should be modified, False if a new instance should be returned, by default False.

        Returns
        -------
        Cuts | None
            Cuts with filtered data sets if `inplace` is True, otherwise None.
        """
        new_cuts = self if inplace else copy.deepcopy(self)

        if include is not None:
            include_set = set(include)
            new_cuts.by_dataset_id = {
                k: v for k, v in new_cuts.by_dataset_id.items() if k in include_set
            }

        if exclude is not None:
            exclude_set = set(exclude)
            new_cuts.by_dataset_id = {
                k: v for k, v in new_cuts.by_dataset_id.items() if k not in exclude_set
            }

        return new_cuts if not inplace else None
