from __future__ import annotations

from itertools import cycle

import matplotlib as mpl
import numpy as np
import pandas as pd
from pandas._typing import Scalar
from pandas.core.groupby import DataFrameGroupBy
from typing_extensions import Any, Callable, Hashable, Literal, Sequence, cast

from ncteqpy.labels import nucleus_to_latex


class DatasetsGroupBy:
    """Class to wrap a `pandas` `DataFrameGroupBy`, specialized for grouping data sets."""

    _keys: str | list[str]
    _datasets_index: pd.DataFrame
    _groupby: (
        DataFrameGroupBy[Scalar, Literal[True]]
        | DataFrameGroupBy[tuple[Scalar], Literal[True]]
    )
    _grouper: pd.Series[Any]
    _sorter: pd.Series[int] | None
    _sort_key: Callable[[pd.Series[Any]], pd.Series[int]] | None
    _label_format: str | None
    _labels: pd.Series[str]
    _props: pd.DataFrame | None = None
    _prop_cycle: cycle | None = None

    def __init__(
        self,
        datasets_index: pd.DataFrame,
        by: str | list[str],
        order: list[Hashable] | None = None,
        labels: dict[Hashable, str] | None = None,
        label_format: str | None = None,
        props: dict[Hashable, dict[str, Any]] | None = None,
    ) -> None:
        """Creates a `DataFrameGroupBy`. Instead of instantiating this class directly, you should call the `ncteqpy.Datasets.groupby` method.

        Parameters
        ----------
        datasets_index : pd.DataFrame
            Index of the datasets in the format given by `ncteqpy.Datasets.index`.
        by : str | list[str]
            Key(s) to group the data sets by, must be column labels of `datasets_index`, e.g., `"type_experiment"` or `["A_heavier", "Z_heavier"]`.
        order : list[Hashable] | None, optional
            Custom ordering of the group values, by default None. If `by` is a list, this must be a list of tuples.
        labels : dict[Hashable, str] | None, optional
            Manual relabeling of some or all group values, by default None. Must be given as a map from the group values to the new labels and takes precedence over `label_format`.
        label_format : str | None, optional
            Format string applied to all group values, by default None. Fields in the format string must be labeled by column names in `datasets_index`, or by `A1_sym`, `A2_sym`, `A_lighter_sym`, `A_heavier_sym`, which formats the element symbol of the respective nucleus.
        props : dict[Hashable, dict[str, Any]] | None, optional
            Matplotlib properties for some or all group values, by default None. Must be given as a map from the group values to a dictionary with matplotlib properties, e.g., the latter could be `{"color": "red"}`. If `props` is None, the property cycle in `matplotlib.rcParams["axes.prop_cycle"]` is used.
        """
        self._datasets_index = datasets_index
        self._keys = by

        if props is not None:
            self._props = pd.DataFrame(props).T
            self._props.index.set_names(self._keys, inplace=True)

        # map group keys to index of group key in `order`
        order_key = (
            cast(
                Callable[[pd.Series], pd.Series],
                pd.Series(np.arange(len(order)), index=order).get,
            )
            if order is not None
            else None
        )

        datasets_index_reindexed = datasets_index.set_index("id_dataset", drop=False)

        # map id_dataset to group keys
        if isinstance(by, str):
            self._grouper = datasets_index_reindexed[by].sort_values(key=order_key)
        else:
            self._grouper = (
                datasets_index_reindexed[by].apply(tuple, axis=1)
            ).sort_values(key=order_key)

        self._sorter = (
            pd.Series(
                np.arange(len(self._grouper)),
                index=self._grouper.index,
            )
            if order is not None
            else None
        )

        self._sort_key = (
            cast(
                Callable[[pd.Series], pd.Series],
                self._sorter.get,
            )
            if self._sorter is not None
            else None
        )

        self._groupby = datasets_index_reindexed.sort_index(
            level=self._keys, key=self._sort_key  # pyright: ignore[reportArgumentType]
        ).groupby(self._grouper, sort=False, dropna=False)

        labels_formatted: dict[Hashable, str] = {}

        if labels is None:
            if label_format is not None:
                for key, df in self._groupby:
                    A_symbols = {
                        "A1_sym": nucleus_to_latex(
                            A=df.iloc[0]["A1"], Z=df.iloc[0]["Z1"], superscript=True
                        ),
                        "A2_sym": nucleus_to_latex(
                            A=df.iloc[0]["A2"], Z=df.iloc[0]["Z2"], superscript=True
                        ),
                        "A_lighter_sym": nucleus_to_latex(
                            A=df.iloc[0]["A_lighter"],
                            Z=df.iloc[0]["Z_lighter"],
                            superscript=True,
                        ),
                        "A_heavier_sym": nucleus_to_latex(
                            A=df.iloc[0]["A_heavier"],
                            Z=df.iloc[0]["Z_heavier"],
                            superscript=True,
                        ),
                    }

                    labels_formatted[key] = label_format.format(
                        **df.iloc[0].to_dict(), **A_symbols
                    )

                self._labels = pd.Series(labels_formatted)

            else:
                group_keys = self._grouper.drop_duplicates()
                self._labels = pd.Series(
                    cast(Sequence[str], group_keys.astype(str).to_numpy()),
                    index=group_keys,
                )
        else:
            for i, (key, df) in enumerate(self._groupby):
                A_symbols = {
                    "A1_sym": nucleus_to_latex(
                        A=df.iloc[0]["A1"], Z=df.iloc[0]["Z1"], superscript=True
                    ),
                    "A2_sym": nucleus_to_latex(
                        A=df.iloc[0]["A2"], Z=df.iloc[0]["Z2"], superscript=True
                    ),
                    "A_lighter_sym": nucleus_to_latex(
                        A=df.iloc[0]["A_lighter"],
                        Z=df.iloc[0]["Z_lighter"],
                        superscript=True,
                    ),
                    "A_heavier_sym": nucleus_to_latex(
                        A=df.iloc[0]["A_heavier"],
                        Z=df.iloc[0]["Z_heavier"],
                        superscript=True,
                    ),
                }

                if isinstance(labels, list) and i < len(labels):
                    labels_formatted[key] = labels[i]
                elif isinstance(labels, dict) and key in labels:
                    labels_formatted[key] = labels[key]
                else:
                    if label_format is not None:
                        labels_formatted[key] = label_format.format(
                            **df.iloc[0].to_dict(), **A_symbols
                        )
                    else:
                        labels_formatted[key] = str(key)

            self._labels = pd.Series(labels_formatted)

    @property
    def keys(self) -> str | list[str]:
        """Keys that the datasets are grouped by."""
        return self._keys

    @property
    def datasets_index(self) -> pd.DataFrame:
        """Index of the datasets."""
        return self._datasets_index

    @property
    def grouper(self) -> pd.Series[Any]:
        """`pd.Series` mapping the group values to dataset IDs."""
        return self._grouper

    @property
    def groupby(
        self,
    ) -> (
        DataFrameGroupBy[Scalar, Literal[True]]
        | DataFrameGroupBy[tuple[Scalar], Literal[True]]
    ):
        """Dataset index grouped by the keys."""
        return self._groupby

    @property
    def labels(self) -> pd.Series[str]:
        """`Series` mapping each group value to its label."""
        return self._labels

    @property
    def props(self) -> pd.DataFrame:
        """Matplotlib properties for the group values. Index: group values, columns: property names"""
        if self._props is None:
            self._prop_cycle = cycle(mpl.rcParams["axes.prop_cycle"])

            self._props = pd.DataFrame(
                {k: next(self._prop_cycle) for k in self._groupby.groups.keys()}
            ).T
            self._props.index.set_names(self._keys, inplace=True)
        elif not self._props.index.isin(self._groupby.groups.keys()).all():
            self._prop_cycle = cycle(mpl.rcParams["axes.prop_cycle"])

            missing = {
                k: next(self._prop_cycle)
                for k in self._groupby.groups.keys()
                if not k in self._props.index
            }

            self._props = pd.concat([self._props, pd.DataFrame(missing).T])

        return self._props

    @property
    def sort_key(self) -> Callable[[pd.Series[Any]], pd.Series[int]] | None:
        """Key function to pass to `pandas.DataFrame.sort` or `pandas.Series.sort`"""
        return self._sort_key

    def get_props(
        self, by: Hashable | list[Hashable], of: Sequence[Hashable]
    ) -> pd.DataFrame:
        """Get the matplotlib properties of a group value according to a column in `self.datasets_index` that is not grouped by. Might be ambiguous if the mapping is not one-to-one.

        Parameters
        ----------
        by : Hashable | list[Hashable]
            Column names by which to get the props.
        of : Sequence[Hashable]
            Column values of which to get the props.

        Returns
        -------
        pd.DataFrame
            Property values.
        """
        map_to_group_keys: pd.Series | pd.DataFrame = self.datasets_index.set_index(
            by, drop=False
        ).loc[
            of, self.keys
        ]  # pyright: ignore[reportArgumentType,reportCallIssue]

        if not isinstance(by, list):
            by = [by]

        missing_values: list[Scalar] = []

        if isinstance(map_to_group_keys, pd.DataFrame):
            missing_index_tuples: list[tuple[Hashable, ...]] = []

            for _, row in map_to_group_keys.iterrows():
                index = tuple(row)

                if not index in self.props.index and not index in missing_index_tuples:
                    if self._prop_cycle is None:
                        self._prop_cycle = cycle(mpl.rcParams["axes.prop_cycle"])

                    missing_index_tuples.append(index)
                    missing_values.append(next(self._prop_cycle))

            assert isinstance(self.keys, list)

            missing_props_index = pd.MultiIndex.from_tuples(
                missing_index_tuples, names=self.keys
            )

        else:
            missing_index: list[Hashable] = []

            for index in map_to_group_keys:
                if not index in self.props.index and not index in missing_index:
                    if self._prop_cycle is None:
                        self._prop_cycle = cycle(mpl.rcParams["axes.prop_cycle"])

                    missing_index.append(index)
                    missing_values.append(next(self._prop_cycle))

            assert isinstance(self.keys, str)

            missing_props_index = pd.Index(missing_index, name=self.keys)

        self._props = pd.concat(
            [self.props, pd.DataFrame(missing_values, index=missing_props_index)]
        )

        group_keys_index = (
            pd.MultiIndex.from_frame(map_to_group_keys)
            if isinstance(map_to_group_keys, pd.DataFrame)
            else map_to_group_keys
        )

        return self.props.loc[group_keys_index]
