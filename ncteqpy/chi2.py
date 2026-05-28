from __future__ import annotations

import os
from collections import deque
from itertools import batched

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from pandas.core.indexes.frozen import FrozenList
from typing_extensions import (
    Any,
    Iterator,
    Literal,
    Sequence,
    TypedDict,
    cast,
    overload,
)

import ncteqpy.data as data
import ncteqpy.jaml as jaml
import ncteqpy.labels as labels
import ncteqpy.util as util
from ncteqpy._typing import SequenceNotStr
from ncteqpy.data_groupby import DatasetsGroupBy
from ncteqpy.plot import data_vs_theory
from ncteqpy.plot.chi2_histograms import (
    plot_chi2_data_breakdown,
    plot_chi2_histogram,
    plot_S_E_histogram,
)
from ncteqpy.plot.data_vs_theory import (
    DataVsTheoryType,
    _format_curve_label,
)
from ncteqpy.plot.grid import AxesGrid
from ncteqpy.plot.util import AdditionalLegend
from ncteqpy.tables.data_chi2 import ColumnType, table_data_chi2

LegendPos = (
    Literal["upper right", "upper left", "lower left", "lower right"] | int | None
)


# TODO: implement pickling for the other members
# TODO: implement some functionality to record the parsing time for each variable so they can be grouped together systematically
class Chi2(jaml.YAMLWrapper):

    _datasets: data.Datasets

    _parameters_names: list[str] | None = None
    _parameters_indices: dict[str, int] | None = None
    _parameters_last_values: npt.NDArray[np.float64] | None = None
    _parameters_input_values: npt.NDArray[np.float64] | None = None
    _parameters_values_at_min: npt.NDArray[np.float64] | None = None
    _last_value: float | None = None
    _last_value_with_penalty: float | None = None
    _last_value_per_data: pd.Series[float] | None = None
    _last_normalizations: pd.DataFrame | None = None
    _snapshots_parameters: pd.DataFrame | None = None
    _snapshots_values: npt.NDArray[np.float64] | None = None
    _snapshots_breakdown_points: pd.DataFrame | None = None
    _snapshots_breakdown_datasets: pd.DataFrame | None = None
    _snapshots_breakdown_nuisance: pd.DataFrame | None = None
    _snapshots_breakdown_normalizations: pd.DataFrame | None = None
    _num_points: pd.Series[int] | None = None

    _minimum_snapshot_index: int | None = None
    _minimum_value: float | None = None
    _minimum_parameters: pd.Series[float] | None = None
    _minimum_value_per_data: pd.Series[float] | None = None
    _minimum_value_per_data_with_penalty: pd.DataFrame | None = None
    # fmt: off
    _minimum_nuisance_parameters: pd.Series[npt.NDArray[np.float64]] | None = None  # pyright: ignore[reportInvalidTypeArguments]
    _minimum_normalizations: pd.DataFrame | None = None
    # fmt: on
    _minimum_points: pd.DataFrame | None = None
    _minimum_S_E: pd.Series[float] | None = None

    def __init__(
        self,
        paths: str | os.PathLike[str],
        datasets: data.Datasets,  # TODO: make optional
        cache_path: str | os.PathLike[str] = ".jaml_cache",
        retain_yaml: bool = False,
    ) -> None:
        super().__init__(paths, cache_path, retain_yaml)
        self._datasets = datasets

    # the traversing for these the members set here takes all about equal time, with negligible parsing time. Parsing them together is thus more efficient # TODO: not sure if snapshots_values should be here
    def _load_snapshots_without_breakdown_points(self) -> None:

        self._snapshots_parameters = cast(
            pd.DataFrame | None, self._unpickle("chi2_snapshots_parameters")
        )
        self._snapshots_values = cast(
            npt.NDArray[np.float64] | None, self._unpickle("chi2_snapshots_values")
        )
        self._snapshots_breakdown_datasets = cast(
            pd.DataFrame | None, self._unpickle("chi2_snapshots_breakdown_datasets")
        )
        self._snapshots_breakdown_nuisance = cast(
            pd.DataFrame | None, self._unpickle("chi2_snapshots_breakdown_nuisance")
        )
        self._snapshots_breakdown_normalizations = cast(
            pd.DataFrame | None,
            self._unpickle("chi2_snapshots_breakdown_normalizations"),
        )

        if (
            self._snapshots_parameters is None
            or self._snapshots_values is None
            or self._snapshots_breakdown_datasets is None
            or self._snapshots_breakdown_nuisance is None
            or self._snapshots_breakdown_normalizations is None
        ):
            pattern = jaml.Pattern(
                {
                    "Chi2Fcn": {
                        "Snapshots": [
                            {
                                "par": None,
                                "chi2Value": None,
                                "perDataBreakdown": None,
                                "nuisanceCorrBreakdown": None,
                                "penaltyBreakdown": None,
                            }
                        ]
                    }
                }
            )
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            snapshots = cast(
                list[dict[str, Any]], jaml.nested_get(yaml, ["Chi2Fcn", "Snapshots"])
            )

            self._snapshots_parameters = pd.DataFrame.from_records(
                [s["par"] for s in snapshots], columns=self.parameters_names
            )
            self._snapshots_parameters.index.name = "id_snapshot"
            self._snapshots_parameters.columns.name = "parameter"

            self._snapshots_values = np.array([s["chi2Value"] for s in snapshots])

            self._snapshots_breakdown_datasets = pd.DataFrame.from_records(
                [s["perDataBreakdown"] for s in snapshots]
            )
            self._snapshots_breakdown_datasets.columns.name = "id_dataset"
            self._snapshots_breakdown_datasets.index.name = "id_snapshot"

            if (
                "nuisanceCorrBreakdown" in snapshots[0]
                and snapshots[0]["nuisanceCorrBreakdown"] is not None
            ):
                self._snapshots_breakdown_nuisance = pd.DataFrame.from_records(
                    [
                        {
                            "id_snapshot": i,
                            "id_dataset": k,
                            "nuisance_parameters": np.array(v),
                        }
                        for i, s in enumerate(snapshots)
                        for k, v in s["nuisanceCorrBreakdown"].items()
                    ],
                    index=["id_snapshot", "id_dataset"],
                )
                self._snapshots_breakdown_nuisance["penalty"] = (
                    self._snapshots_breakdown_nuisance["nuisance_parameters"].apply(
                        lambda x: (x**2).sum()
                    )
                )

            if "penaltyBreakdown" in snapshots[0] and isinstance(
                snapshots[0]["penaltyBreakdown"], list
            ):
                self._snapshots_breakdown_normalizations = pd.concat(
                    [
                        self._get_normalizations_dataframe(s["penaltyBreakdown"])
                        for s in snapshots
                    ],
                    axis=0,
                    keys=range(len(snapshots)),
                    names=["id_snapshot"],
                )

            self._pickle(self._snapshots_parameters, "chi2_snapshots_parameters")
            self._pickle(self._snapshots_values, "chi2_snapshots_values")
            self._pickle(
                self._snapshots_breakdown_datasets,
                "chi2_snapshots_breakdown_datasets",
            )
            self._pickle(
                self._snapshots_breakdown_nuisance,
                "chi2_snapshots_breakdown_nuisance",
            )

    # parsing the perPointBreakdowns takes much longer than all the other fields, so we load these separately
    def _load_snapshots_breakdown_points(self) -> None:
        pickle_name = "chi2_snapshots_breakdown_points"

        self._snapshots_breakdown_points = cast(
            pd.DataFrame, self._unpickle(pickle_name)
        )

        if self._snapshots_breakdown_points is None:
            pattern = jaml.Pattern(
                {"Chi2Fcn": {"Snapshots": [{"perPointBreakdown": None}]}}
            )
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            snapshots = cast(
                list[dict[str, object]], jaml.nested_get(yaml, ["Chi2Fcn", "Snapshots"])
            )

            self._snapshots_breakdown_points = pd.DataFrame.from_records(
                [
                    {
                        "id_snapshot": i,
                        **ppb,
                        **cast(dict[str, object], ppb["KinVarVals"]),
                    }
                    for i, s in enumerate(snapshots)
                    for ppb in cast(list[dict[str, object]], s["perPointBreakdown"])
                ],
                exclude=["KinVarVals"],
            ).rename(
                columns=(
                    labels.chi2fcn_per_point_breakdown_yaml_to_py
                    | labels.kinvars_yaml_to_py
                )
            )

            int_cols = [
                "id_point",
                "id_dataset",
            ]  # no A and Z here since they are sometimes non-integer

            # not all experiments have id_bin
            if "id_bin" in self._snapshots_breakdown_points:
                int_cols.append("id_bin")

            self._snapshots_breakdown_points[int_cols] = (
                self._snapshots_breakdown_points[int_cols].astype("Int64", copy=False)
            )

            self._snapshots_breakdown_points.set_index(
                ["id_snapshot", "id_point"], inplace=True
            )

            # multiply the theory for which normalization is fitted with its corresponding factor
            mask = self._snapshots_breakdown_points["id_dataset"].isin(
                self.last_normalizations.index
            )
            # `theory_with_normalization_only` is NaN for data sets that are not normalization-corrected
            self._snapshots_breakdown_points.loc[
                mask, "theory_with_normalization_only"
            ] = (
                self._snapshots_breakdown_points.loc[mask, "theory"]
                * self.last_normalizations.loc[
                    self._snapshots_breakdown_points.loc[mask, "id_dataset"]
                ]["factor"].to_numpy()
            )

            # `theory_with_normalization` is filled with `theory` where `theory_with_normalization_only` is NaN
            self._snapshots_breakdown_points["theory_with_normalization"] = (
                self._snapshots_breakdown_points[
                    "theory_with_normalization_only"
                ].copy()
            )
            self._snapshots_breakdown_points.fillna(
                {
                    "theory_with_normalization": self._snapshots_breakdown_points[
                        "theory"
                    ]
                },
                inplace=True,
            )

            self._pickle(self._snapshots_breakdown_points, pickle_name)

    @property
    def datasets(self) -> data.Datasets:
        return self._datasets

    @property
    def parameters_names(self) -> list[str]:
        if self._parameters_names is None or self._yaml_changed():
            pattern = jaml.Pattern(
                {"Chi2Fcn": {"IndexOfInputParams": None, "ParamIndices": None}}
            )
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            index_of_input_params = cast(
                list[int], jaml.nested_get(yaml, ["Chi2Fcn", "IndexOfInputParams"])
            )
            param_indices = cast(
                dict[str, int], jaml.nested_get(yaml, ["Chi2Fcn", "ParamIndices"])
            )

            # order the parameter name to parameter index mapping by the index (dicts preserve ordering in Python 3.7+)
            self._parameters_indices = {
                p: i_p
                for i in index_of_input_params
                for p, i_p in param_indices.items()
                if i_p == i
            }
            self._parameters_names = list(self._parameters_indices.keys())

        return self._parameters_names

    @property
    def parameters_indices(self) -> dict[str, int]:
        if self._parameters_indices is None or self._yaml_changed():
            pattern = jaml.Pattern(
                {"Chi2Fcn": {"IndexOfInputParams": None, "ParamIndices": None}}
            )
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            index_of_input_params = cast(
                list[int], jaml.nested_get(yaml, ["Chi2Fcn", "IndexOfInputParams"])
            )
            param_indices = cast(
                dict[str, int], jaml.nested_get(yaml, ["Chi2Fcn", "ParamIndices"])
            )

            # order the parameter name to parameter index mapping by the index (dicts preserve ordering in Python 3.7+)
            self._parameters_indices = {
                p: i_p
                for i in index_of_input_params
                for p, i_p in param_indices.items()
                if i_p == i
            }
            self._parameters_names = list(self._parameters_indices.keys())

        return self._parameters_indices

    @property
    def parameters_last_values(self) -> npt.NDArray[np.float64]:
        if self._parameters_last_values is None or self._yaml_changed():
            pattern = jaml.Pattern({"Chi2Fcn": {"LastParams": None}})
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            self._parameters_last_values = np.array(
                cast(list[float], jaml.nested_get(yaml, ["Chi2Fcn", "LastParams"]))
            )

        return self._parameters_last_values

    @property
    def parameters_input_values(self) -> npt.NDArray[np.float64]:
        if self._parameters_input_values is None or self._yaml_changed():
            pattern = jaml.Pattern({"Chi2Fcn": {"InputParametrizationParams": None}})
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            input_parametrization_params = cast(
                dict[str, float],
                jaml.nested_get(yaml, ["Chi2Fcn", "InputParametrizationParams"]),
            )

            self._parameters_input_values = np.array(
                [input_parametrization_params[p] for p in self.parameters_names]
            )

        return self._parameters_input_values

    @property
    def parameters_values_at_min(self) -> npt.NDArray[np.float64]:
        if self._parameters_values_at_min is None or self._yaml_changed():
            pattern = jaml.Pattern(
                {"Chi2Fcn": {"InputParametrizationParamsAtMin": None}}
            )
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            input_parametrization_params_at_min = cast(
                dict[str, float],
                jaml.nested_get(yaml, ["Chi2Fcn", "InputParametrizationParamsAtMin"]),
            )

            self._parameters_values_at_min = np.array(
                [input_parametrization_params_at_min[p] for p in self.parameters_names]
            )

        return self._parameters_values_at_min

    @property
    def last_value(self) -> float:
        if self._last_value is None or self._yaml_changed():
            pattern = jaml.Pattern({"Chi2Fcn": {"LastValue": None}})
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            self._last_value = cast(
                float, jaml.nested_get(yaml, ["Chi2Fcn", "LastValue"])
            )

        return self._last_value

    @property
    def last_value_with_penalty(self) -> float:
        if self._last_value_with_penalty is None or self._yaml_changed():
            pattern = jaml.Pattern({"Chi2Fcn": {"LastValueWithPenalty": None}})
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            self._last_value_with_penalty = cast(
                float, jaml.nested_get(yaml, ["Chi2Fcn", "LastValueWithPenalty"])
            )

        return self._last_value_with_penalty

    @property
    def last_value_per_data(self) -> pd.Series[float]:
        if self._last_value_per_data is None or self._yaml_changed():
            pattern = jaml.Pattern({"Chi2Fcn": {"LastValuePerData": None}})
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            self._last_value_per_data = pd.Series(
                cast(
                    dict[int, float],
                    jaml.nested_get(yaml, ["Chi2Fcn", "LastValuePerData"]),
                )
            )
            self._last_value_per_data.index.name = "id_dataset"

        return self._last_value_per_data

    def _get_normalizations_dataframe(self, yaml: jaml.YAMLType) -> pd.DataFrame:
        if not isinstance(yaml, list):
            raise ValueError("yaml must be of type list")

        class NormInfoEntry(TypedDict):
            IDs: list[int]
            Scheme: int
            Value: float
            Penalty: float

        norm_records = []

        for norm_info in cast(list[NormInfoEntry], yaml):
            for id_dataset in norm_info["IDs"]:
                norm_records.append(
                    {
                        "id_dataset": id_dataset,
                        "id_dataset_group": FrozenList(norm_info["IDs"]),
                        "scheme": norm_info["Scheme"],
                        "factor": norm_info["Value"],
                        # the total penalty for the whole group
                        "penalty_group_total": norm_info["Penalty"],
                    }
                )

        normalizations = pd.DataFrame.from_records(norm_records, index="id_dataset")

        # It is ambiguous how to distribute the penalty between the datasets. Thus, we first implement the variant in which the penalty of each data set is the total penalty of the group divided by the number of data sets in the group, ...
        normalizations["penalty_dataset_even"] = normalizations[
            "penalty_group_total"
        ] / normalizations["id_dataset_group"].apply(len)

        # ... and second, the variant in which the total penalty of the group is distributed between each data set proportionally to its χ² without penalty.
        chi2_by_group = normalizations["id_dataset_group"].apply(
            lambda group: self.minimum_value_per_data[group].sum()
        )

        normalizations.loc[chi2_by_group.index, "penalty_dataset_proportional"] = (
            self.minimum_value_per_data.loc[chi2_by_group.index]
            / chi2_by_group
            * normalizations.loc[chi2_by_group.index, "penalty_group_total"]
        )

        # for data sets that did not survive the cuts, penalty_dataset_proportional was set to nan in the previous statement, so we set it to 0
        normalizations.loc[
            chi2_by_group[chi2_by_group == 0].index, "penalty_dataset_proportional"
        ] = 0

        return normalizations

    @property
    def last_normalizations(self) -> pd.DataFrame:
        if self._last_normalizations is None or self._yaml_changed():
            pattern = jaml.Pattern({"Chi2Fcn": {"NormInfo": None}})
            yaml = cast(dict[str, dict[str, jaml.YAMLType]], self._load_yaml(pattern))

            self._last_normalizations = self._get_normalizations_dataframe(
                yaml["Chi2Fcn"]["NormInfo"]
            )

        return self._last_normalizations

    @property
    def snapshots_parameters(self) -> pd.DataFrame:
        if self._snapshots_parameters is None or self._yaml_changed():
            self._load_snapshots_without_breakdown_points()

        assert self._snapshots_parameters is not None

        return self._snapshots_parameters

    @property
    def snapshots_values(self) -> npt.NDArray[np.float64]:
        if self._snapshots_values is None or self._yaml_changed():
            self._load_snapshots_without_breakdown_points()

        assert self._snapshots_values is not None

        return self._snapshots_values

    @property
    def snapshots_breakdown_datasets(self) -> pd.DataFrame:
        if self._snapshots_breakdown_datasets is None or self._yaml_changed():
            self._load_snapshots_without_breakdown_points()

        assert self._snapshots_breakdown_datasets is not None

        return self._snapshots_breakdown_datasets

    @property
    def snapshots_breakdown_nuisance(self) -> pd.DataFrame:
        if self._snapshots_breakdown_nuisance is None or self._yaml_changed():
            self._load_snapshots_without_breakdown_points()

        assert self._snapshots_breakdown_nuisance is not None

        return self._snapshots_breakdown_nuisance

    @property
    def snapshots_breakdown_normalizations(self) -> pd.DataFrame:
        if self._snapshots_breakdown_normalizations is None or self._yaml_changed():
            self._load_snapshots_without_breakdown_points()

        assert self._snapshots_breakdown_normalizations is not None

        return self._snapshots_breakdown_normalizations

    @property
    def snapshots_breakdown_points(self) -> pd.DataFrame:
        if self._snapshots_breakdown_points is None or self._yaml_changed():
            self._load_snapshots_breakdown_points()

        assert self._snapshots_breakdown_points is not None

        return self._snapshots_breakdown_points

    @property
    def num_points(self) -> pd.Series[int]:
        if self._num_points is None or self._yaml_changed():
            self._num_points = (
                self.snapshots_breakdown_points.loc[0]
                .groupby("id_dataset")["id_dataset"]
                .count()
            )
            self._num_points.name = None

        return self._num_points

    @property
    def minimum_snapshot_index(self) -> int:
        if self._minimum_snapshot_index is None or self._yaml_changed():
            self._minimum_snapshot_index = int(self.snapshots_values.argmin())

        return self._minimum_snapshot_index

    @property
    def minimum_value(self) -> float:
        if self._minimum_value is None or self._yaml_changed():
            self._minimum_value = float(
                self.snapshots_values[self.minimum_snapshot_index]
            )

        return self._minimum_value

    @property
    def minimum_parameters(self) -> pd.Series[float]:
        if self._minimum_parameters is None or self._yaml_changed():
            minimum_parameters = self.snapshots_parameters.loc[
                self.minimum_snapshot_index
            ]

            assert isinstance(minimum_parameters, pd.Series)

            self._minimum_parameters = minimum_parameters

        return self._minimum_parameters

    @property
    def minimum_value_per_data(self) -> pd.Series[float]:
        if self._minimum_value_per_data is None or self._yaml_changed():
            minimum_per_data = self.snapshots_breakdown_datasets.loc[
                self.minimum_snapshot_index
            ]

            assert isinstance(minimum_per_data, pd.Series)

            self._minimum_value_per_data = minimum_per_data

        return self._minimum_value_per_data

    @property
    def minimum_value_per_data_with_penalty(self) -> pd.DataFrame:
        if self._minimum_value_per_data_with_penalty is None or self._yaml_changed():
            minimum_with_penalty = pd.concat(
                2 * [self.minimum_value_per_data],
                axis="columns",
                keys=["even", "proportional"],
            )
            minimum_with_penalty.loc[:, "even"] = minimum_with_penalty.loc[
                :, "even"
            ].add(self.minimum_normalizations["penalty_dataset_even"], fill_value=0)
            minimum_with_penalty.loc[:, "proportional"] = minimum_with_penalty.loc[
                :, "proportional"
            ].add(
                self.minimum_normalizations["penalty_dataset_proportional"],
                fill_value=0,
            )

            self._minimum_value_per_data_with_penalty = minimum_with_penalty

        assert isinstance(self._minimum_value_per_data_with_penalty, pd.DataFrame)

        return self._minimum_value_per_data_with_penalty

    @property
    def minimum_nuisance_parameters(
        self,
    ) -> pd.Series[
        npt.NDArray[np.float64]  # pyright: ignore[reportInvalidTypeArguments]
    ]:
        if self._minimum_nuisance_parameters is None or self._yaml_changed():
            minimum_nuisance_parameters = self.snapshots_breakdown_nuisance.loc[
                self.minimum_snapshot_index
            ]

            assert isinstance(minimum_nuisance_parameters, pd.Series)

            self._minimum_nuisance_parameters = minimum_nuisance_parameters

        return self._minimum_nuisance_parameters

    @property
    def minimum_normalizations(
        self,
    ) -> pd.DataFrame:
        if self._minimum_normalizations is None or self._yaml_changed():
            minimum_normalizations = self.snapshots_breakdown_normalizations.loc[
                self.minimum_snapshot_index
            ]

            assert isinstance(minimum_normalizations, pd.DataFrame)

            self._minimum_normalizations = minimum_normalizations

        return self._minimum_normalizations

    @property
    def minimum_points(self) -> pd.DataFrame:
        if self._minimum_points is None or self._yaml_changed():
            points_snapshots = cast(
                pd.DataFrame,
                self.snapshots_breakdown_points.loc[self.minimum_snapshot_index].copy(),
            ).sort_values(
                "id_dataset"
            )  # FIXME figure out which snapshots to read

            datasets_index = self.datasets.index.set_index("id_dataset")
            datasets_points = self.datasets.points.set_index("id_dataset")

            points_list = []

            for id_dataset, points1_i in points_snapshots.groupby("id_dataset"):

                kinematic_variables = cast(
                    list[str], datasets_index.loc[id_dataset]["kinematic_variables"]
                )

                # drop nan columns when matching since we cannot match on them anyway
                points1_i_notna = points1_i.dropna(axis=1)

                points2_i = datasets_points.loc[[id_dataset]].reset_index("id_dataset")

                if not isinstance(kinematic_variables, list):
                    raise ValueError(
                        f"Expected list for kinematic_variables of data set with ID {id_dataset}, got {kinematic_variables}"
                    )

                match_cols = (
                    ["id_dataset"]
                    + points1_i_notna.columns[
                        points1_i_notna.columns.isin(kinematic_variables)
                    ].to_list()
                    + ["data"]
                )
                if "sqrt_s" in match_cols:
                    match_cols.remove("sqrt_s")

                cols = (
                    match_cols
                    + list(labels.uncertainties_yaml_to_py.values())
                    + ["unc_tot"]
                )

                points_list.append(
                    pd.merge(
                        points1_i.reset_index(),
                        points2_i[cols],
                        how="left",
                        on=match_cols,
                    )
                )
            self._minimum_points = (
                pd.concat(points_list).set_index("id_point").sort_index()
            )

            # compute PDF uncertainties for each point
            if self.snapshots_breakdown_points.shape[0] > 2 * len(
                self.parameters_names
            ):
                for col_in in (
                    "theory",
                    "theory_with_normalization_only",
                    "theory_with_normalization",
                ):
                    for col_out, func in (
                        ("pdf_unc_sym", util.pdf_uncertainty_sym),
                        ("pdf_unc_asym_lower", util.pdf_uncertainty_asym_lower),
                        ("pdf_unc_asym_upper", util.pdf_uncertainty_asym_upper),
                    ):
                        self._minimum_points[f"{col_in}_{col_out}"] = (
                            self.snapshots_breakdown_points.loc[
                                self.minimum_snapshot_index :
                            ]
                            .groupby("id_point")[col_in]
                            .apply(func)
                        )

        return self._minimum_points

    @property
    def minimum_S_E(self) -> pd.Series[float]:
        if self._minimum_S_E is None or self._yaml_changed():
            self._minimum_S_E = cast(
                pd.Series,
                np.sqrt(2 * self.minimum_value_per_data)
                - np.sqrt(2 * self.num_points - 1),
            )

        return self._minimum_S_E

    def table_data(
        self,
        columns: Sequence[ColumnType],
        id_dataset: Sequence[int] | None = None,
        type_experiment: str | SequenceNotStr[str] | None = None,
        column_types: SequenceNotStr[str] | None = None,
        format_columns: str | list[str | None] | dict[ColumnType, str] | None = None,
        format_total: str | list[str | None] | dict[ColumnType, str] | None = None,
        sort_by: str | SequenceNotStr[str] | None = None,
        sort_ascending: bool | Sequence[bool | None] | None = None,
        sparse_columns: Literal["all"] | Sequence[ColumnType] | None = "all",
        hlines: ColumnType | Sequence[ColumnType] | None = None,
        highlight: int | Sequence[int] | None = None,
        labels: dict[ColumnType, str] | None = None,
        title: str | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Generate a data set table as LaTeX `tabular` environment with a row for each data set.

        Parameters
        ----------
        columns : Sequence[ColumnType]
            The columns of the table.
        id_dataset : Sequence[int] | None, optional
            IDs of the data sets, by default None, i.e., all IDs in `datasets_index`.
        type_experiment : str | SequenceNotStr[str] | None, optional
            Experiment types whose IDs are included in the table, by default None, i.e., all experiment types. `type_experiment` only has an effect if `id_dataset` is None.
        column_types : SequenceNotStr[str] | None, optional
            LaTeX column type to be inserted into `\\begin{tabular}{...}`, e.g., `c` or `p{1cm}`, by default None. The default column type that is used is `l`.
        format_columns : str | list[str | None] | dict[ColumnType, str] | None, optional
            Format string for the column labels, by default None.
        format_total : str | SequenceNotStr[str | None] | Mapping[ColumnType, str | None] | None, optional
            Format string for the cells in the row for total values, by default None.
        sort_by : str | SequenceNotStr[str] | None, optional
            Column(s) by which the rows are sorted, by default None, i.e., no sorting.
        sort_ascending : bool | Sequence[bool | None] | None, optional
            Sort order of the column(s), by default None, i.e., ascending.
        sparse_columns : Literal["all"] | Sequence[ColumnType] | None, optional
            Columns in which duplicate adjacent values are to be wrapped with `\\multirow`, by default "all".
        hlines : ColumnType | Sequence[ColumnType] | None, optional
            Column(s) for which a horizontal line is inserted after each value, by default None, i.e., not horizontal lines are inserted. If a column is in `hlines` and `sparse_columns`, the line is inserted after the multi-row cell. The horizontal lines extend from the column(s) only to right. The actual command that is used is `\\cline`.
        highlight : int | Sequence[int] | None, optional
            Data set IDs which are to be bold-faced, by default None, i.e., no highlights.
        labels : dict[str, str] | None, optional
            Labels of the columns, by default None, i.e., default values are chosen for the labels.
        title : str | None, optional
            Title to be inserted in the first row of the table, by default None, i.e., no title.

        Returns
        -------
        table : pd.DataFrame
            The table in DataFrame format.
        table_latex : str
            The table as LaTeX code.
        """
        if id_dataset is None:
            if type_experiment is None:
                raise ValueError(
                    "Please provide either `id_dataset` or `type_experiment`"
                )
            else:
                id_dataset = cast(
                    Sequence[int],
                    self.datasets.index.query("type_experiment == @type_experiment")[
                        "id_dataset"
                    ].unique(),
                )  # actually npt.NDArray[np.int_]
        else:
            if isinstance(id_dataset, int):
                id_dataset = [id_dataset]

        return table_data_chi2(
            columns=columns,
            datasets_index=self.datasets.index,
            id_dataset=id_dataset,
            chi2=self.minimum_value_per_data,
            normalization=self.last_normalizations["factor"],
            num_points_after_cuts=self.num_points,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            sparse_columns=sparse_columns,
            hlines=hlines,
            highlight=highlight,
            title=title,
            labels=labels,
            column_types=column_types,
            format_columns=format_columns,
            format_total=format_total,
        )

    def plot_data_breakdown(
        self,
        ax: Axes,
        id_dataset: int | Sequence[int] | None = None,
        per_point: bool = True,
        bar_orientation: Literal["horizontal", "vertical"] = "horizontal",
        chi2_line_1: bool = True,
        chi2_drop_0: bool = True,
        bar_groupby: DatasetsGroupBy | None = None,
        bar_props_groupby: DatasetsGroupBy | None = None,
        bar_order_groupby: str | list[str] | None = None,
        bar_labels: Literal["num_points", "chi2"] = "num_points",
        add_norm_penalty: Literal["bar_top", "bar_bottom"] | None = "bar_top",
        norm_penalty_variant: Literal["even", "proportional"] = "proportional",
        kwargs_bar: dict[str, Any] = {},
        kwargs_bar_norm_penalty: dict[str, Any] = {},
        kwargs_bar_label: dict[str, Any] = {},
        kwargs_chi2_line_1: dict[str, Any] = {},
        kwargs_legend: dict[str, Any] = {},
    ) -> None:
        """Plot histogram of χ² vs. (grouped) datasets.

        Parameters
        ----------
        ax : plt.Axes
            The axes to plot on.
        id_dataset : int | Sequence[int] | None, optional
            IDs of the datasets to be plotted, by default None.
        per_point : bool, optional
            If χ² per point should be plotted, by default True.
        bar_orientation : Literal["horizontal", "vertical"], optional
            Direction in which the bars are oriented, by default "vertical".
        chi2_line_1 : bool, optional
            If a line should be plotted at χ²/point = 1, by default True. Does nothing if `per_point` is `False`.
        chi2_drop_0 : bool, optional
            If data sets with χ² = 0 should be ignored when plotting (i.e., data sets that did not survive the cuts), by default True.
        bar_groupby : DatasetsGroupBy | None, optional
            How to group the bars, by default no grouping. One bar per group is plotted.
        bar_props_groupby : DatasetsGroupBy | None, optional
            How to group the properties (color etc.) of each bar, by default no grouping, i.e., all bars get the same properties.
        bar_labels : Literal["num_points", "chi2"], optional
            If the bars should be labeled with the number of points ("num_points") or the χ² value ("chi2"), by default "num_points".
        add_norm_penalty : Literal["bar_top", "bar_bottom"] | None, optional
            If and where the normalization penalty is added to the bars, by default "bar_top". "bar_bottom" is not implemented yet.
        norm_penalty_variant : Literal["even", "proportional"], optional
            How to distribute the normalization penalty across data sets (this is ambiguous for data sets that share their normalization). "even" distributes the penalty evenly between data sets so that every data set gets the same penalty. "proportional" distributes the penalty between the data sets proportional to the χ² that does not yet include the penalty of each data set. The default is "proportional".
        kwargs_bar : dict[str, Any], optional
            Keyword arguments passed to `plt.Axes.bar` or `plt.Axes.barh` when plotting the bars with normalization penalty.
        kwargs_bar_norm_penalty : dict[str, Any], optional
            Keyword arguments passed to `plt.Axes.bar` or `plt.Axes.barh` when plotting the bars with normalization penalty.
        kwargs_bar : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.bar` or `plt.Axes.barh`.
        kwargs_bar_label : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.bar_label`.
        kwargs_chi2_line_1 : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.axhline` or `plt.Axes.axvline` for the line at χ²/point = 1.
        kwargs_legend : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.legend` for the legend set by `bar_props_groupby`.
        """

        if isinstance(id_dataset, int):
            id_dataset = [id_dataset]
        elif id_dataset is not None:
            id_dataset = list(id_dataset)

        # FIXME: this needs to be removed as soon as ncteqpp-2.0!44 is merged
        chi2 = (
            self.minimum_value_per_data.loc[
                self.minimum_value_per_data.index.isin(id_dataset)
            ].add(
                -self.last_normalizations.loc[
                    self.last_normalizations.index.isin(id_dataset),
                    "penalty_group_total",
                ],
                fill_value=0,
            )
            if id_dataset is not None
            else self.minimum_value_per_data.add(
                -self.last_normalizations["penalty_group_total"], fill_value=0
            )
        )

        penalty_col = f"penalty_dataset_{norm_penalty_variant}"

        chi2_with_penalty = (
            chi2.add(
                self.last_normalizations.loc[
                    self.last_normalizations.index.isin(id_dataset), penalty_col
                ],
                fill_value=0,
            )
            if id_dataset is not None
            else chi2.add(
                self.last_normalizations[penalty_col],
                fill_value=0,
            )
        )

        plot_chi2_data_breakdown(
            ax=ax,
            chi2=chi2,
            chi2_with_penalty=chi2_with_penalty,
            per_point=per_point,
            num_points=self.num_points,
            chi2_line_1=chi2_line_1,
            chi2_drop_0=chi2_drop_0,
            bar_orientation=bar_orientation,
            bar_groupby=bar_groupby,
            bar_props_groupby=bar_props_groupby,
            bar_order_groupby=bar_order_groupby,
            bar_labels=bar_labels,
            add_norm_penalty=add_norm_penalty,
            kwargs_bar=kwargs_bar,
            kwargs_bar_norm_penalty=kwargs_bar_norm_penalty,
            kwargs_bar_label=kwargs_bar_label,
            kwargs_chi2_line_1=kwargs_chi2_line_1,
            kwargs_legend=kwargs_legend,
        )

    def plot_chi2_histogram(
        self,
        bin_width: float | None = None,
        subplot_groupby: DatasetsGroupBy | None = None,
        kwargs_subplots: dict[str, Any] = {},
        kwargs_histogram: dict[str, Any] | list[dict[str, Any] | None] = {},
    ) -> AxesGrid:
        """Plots a histogram of the χ² values of the data points.

        Parameters
        ----------
        bin_width : float | None, optional
            Width of a bin, by default chosen by `np.histogram`.
        subplot_groupby : DatasetsGroupBy | None, optional
            How to group χ² values that are shown in distributions on different subplots, by default None.
        kwargs_subplots : dict[str, Any], optional
            Keyword arguments passed to `plt.subplots` through `AxesGrid`.
        kwargs_histogram : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments passed to `plt.Axes.hist`.

        Returns
        -------
        AxesGrid
            `AxesGrid` that holds the subplot(s).
        """

        return plot_chi2_histogram(
            chi2=self.minimum_points,
            bin_width=bin_width,
            subplot_groupby=subplot_groupby,
            kwargs_subplots=kwargs_subplots,
            kwargs_histogram=kwargs_histogram,
        )

    def plot_S_E_histogram(
        self,
        bin_width: float | None = None,
        subplot_groupby: DatasetsGroupBy | None = None,
        kwargs_subplots: dict[str, Any] = {},
        kwargs_histogram: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_gaussian: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_fit: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_gaussian_fit: dict[str, Any] | list[dict[str, Any] | None] = {},
    ) -> AxesGrid:
        """Plots the `S_E` distribution (see arXiv:1905.06957 eq. 157).

        Parameters
        ----------
        bin_width : float | None, optional
            Width of a bin, by default chosen by `np.histogram`.
        subplot_groupby : DatasetsGroupBy | None, optional
            How to group S_E values that are shown in distributions on different subplots, by default None.
        kwargs_subplots : dict[str, Any], optional
            Keyword arguments passed to `plt.subplots` through `AxesGrid`.
        kwargs_histogram : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments passed to `plt.Axes.hist`.
        kwargs_gaussian : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments passed to `plt.Axes.plot` for plotting the standard gaussian.
        kwargs_fit : dict[str, Any] | list[dict[str, Any]   None], optional
            Keyword arguments passed to `scipy.optimize.curve_fit` for fitting the gaussian.
        kwargs_gaussian_fit : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments passed to `plt.Axes.plot` for plotting the fitted gaussian.

        Returns
        -------
        AxesGrid
            `AxesGrid` that holds the subplot(s)
        """

        return plot_S_E_histogram(
            S_E=self.minimum_S_E,
            bin_width=bin_width,
            subplot_groupby=subplot_groupby,
            kwargs_subplots=kwargs_subplots,
            kwargs_histogram=kwargs_histogram,
            kwargs_gaussian=kwargs_gaussian,
            kwargs_fit=kwargs_fit,
            kwargs_gaussian_fit=kwargs_gaussian_fit,
        )

    # def plot_nuisance_histogram(
    #     self,
    #     bin_width: float | None = None,
    #     subplot_groupby: DatasetsGroupBy | None = None,
    #     kwargs_subplots: dict[str, Any] = {},
    #     kwargs_histogram: dict[str, Any] | list[dict[str, Any] | None] = {},
    #     kwargs_gaussian: dict[str, Any] | list[dict[str, Any] | None] = {},
    #     kwargs_fit: dict[str, Any] | list[dict[str, Any] | None] = {},
    #     kwargs_gaussian_fit: dict[str, Any] | list[dict[str, Any] | None] = {},
    # ) -> AxesGrid:

    @overload
    def plot_data_vs_theory(
        self,
        id_dataset: int | Sequence[int] | None = ...,
        type_experiment: str | None = ...,
        x_variable: str | list[str] | Literal["fallback"] | None = ...,
        xlabel: str | dict[str, str] | Literal["fallback"] | None = ...,
        ylabel: str | Literal["fallback"] | None = ...,
        xscale: str | None = ...,
        yscale: str | None = ...,
        title: str | None = ...,
        legend: LegendPos = ...,
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = ...,
        plot_types: DataVsTheoryType | Sequence[DataVsTheoryType] = ...,
        subplot_groupby: str | None = ...,
        subplot_label: Literal["legend"] | None = ...,
        subplot_label_format: str | None = ...,
        chi2_annotation: bool = ...,
        chi2_legend: LegendPos = ...,
        info_legend: LegendPos = ...,
        curve_groupby: str | list[str] | Literal["fallback"] | None = ...,
        apply_normalization: bool = ...,
        shift_correlated: Literal["data", "theory"] | None = ...,
        theory_min_width: float = ...,
        plot_pdf_uncertainty: bool = ...,
        pdf_uncertainty_convention: Literal["sym", "asym"] = ...,
        y_offset_add: float | None = ...,
        y_offset_mul: float | None = ...,
        kwargs_subplots: dict[str, Any] = ...,
        kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_theory_unc: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_xlabel: dict[str, Any] = ...,
        kwargs_ylabel: dict[str, Any] = ...,
        kwargs_title: dict[str, Any] = ...,
        kwargs_legend: dict[str, Any] = ...,
        kwargs_legend_chi2: dict[str, Any] = ...,
        kwargs_legend_info: dict[str, Any] = ...,
        kwargs_legend_curves: dict[str, Any] = ...,
        kwargs_legend_subplots: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_annotate_chi2: dict[str, Any] = ...,
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = ...,
        *,
        ax: None = ...,
        iterate: Literal[False] = ...,
        **kwargs: Any,
    ) -> AxesGrid | list[AxesGrid]:
        """Plot data vs. theory (one data set per figure).

        Parameters
        ----------
        id_dataset : int | Sequence[int] | None, optional
            Data sets for which to plot the data vs. theory, by default None. Must be of the same type_experiment. If None, `type_experiment` must be given.
        type_experiment : str | None, optional
            Process for which to plot the data vs. theory, by default None. If None, `id_dataset` must be given.
        x_variable : str | list[str] | None, optional
            The kinematic variable to put on the x axis, by default "fallback". Plots a binned distribution if a list is passed.
        xlabel : str | dict[str, str] | Literal["fallback"] | None, optional
            x label of the plot, by default None.
        ylabel : str | Literal["fallback"] | None, optional
            y label of the plot, by default None.
        xscale : str | None, optional
            x scale of the plot, by default no changing of x scale.
        yscale : str | None, optional
            y scale of the plot, by default no changing of y scale.
        title : str | None, optional
            Title of the plot, by default None.
        legend : bool, optional
            If a legend annotating data & theory is shown, by default True.
        curve_label : Literal["annotate above", "annotate right", "ticks", "legend"] | None, optional
            Where the curve labels are shown, by default "ticks". If None, no labels are shown.
        subplot_groupby: str | None, optional
            Variable to group the subplot axes by, by default None (no grouping).
        subplot_label : Literal["legend"] | None, optional  # FIXME
            Where to label the subplot, by default None.
        subplot_label_format : str | None, optional
            Format of the subplot label, by default None.
        chi2_annotation : bool, optional
            If the χ²/point value is annotated, by default True.
        chi2_legend : LegendPos, optional
            Where on the figure to show a legend with the χ², by default "upper left". If None, no χ² legend is shown.
        info_legend: LegendPos, optional
            On which subplot to show a legend with info about the data set, by default "upper left". If None, no info legend is shown.
        curve_groupby : str | list[str] | Literal["fallback"] | None, optional
            Variable(s) to group the curves by, by default "fallback".
        apply_normalization : bool, optional
            If the normalization-corrected theory is plotted, by default True.
        shift_correlated : Literal["data", "theory"], optional
            Whether to shift data or theory when correlated errors are present, by default "theory". For details, see arXiv:hep-ph/0201195 appendix B.2.
        theory_min_width : float, optional
            Width of the theory curve (in units of axes fraction) if there is only one point, by default 0.06.
        plot_pdf_uncertainty : bool, optional
            If the PDF uncertainty is plotted around the theory curve, by default True.
        pdf_uncertainty_convention : Literal["sym", "asym"], optional
            If the PDF uncertainties should be symmetric ("sym") or asymmetric ("asym"), by default "asym".
        y_offset_add : float | None, optional
            Additive offset by which the curves are separated, by default 0 (no offset).
        y_offset_mul : float | None, optional
            Multiplicative offset by which the curves are separated, by default 0 (no offset). The factor between each curve is `10**y_offset_mul`.
        kwargs_subplots : dict[str, Any], optional
            Keyword arguments to pass to `plt.subplots` through `AxesGrid`.
        kwargs_data : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `plt.Axes.plot` for plotting the data.
        kwargs_theory : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.plot` for plotting the theory.
        kwargs_theory_unc : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.fill_between` for plotting the PDF uncertainty.
        kwargs_xlabel : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_xlabel`.
        kwargs_ylabel : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_ylabel`.
        kwargs_title : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_title`.
        kwargs_legend : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating data & theory.
        kwargs_legend_chi2 : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating the total χ².
        kwargs_legend_info : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating data set info.
        kwargs_legend_curves : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating the curve labels.
        kwargs_legend_info : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating data set info.
        kwargs_legend_subplots : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating subplot variables.
        kwargs_ticks_curves : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.set_yticks` for annotating the curve labels.
        kwargs_annotate_chi2 : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.annotate` for annotating the χ²/point values.
        kwargs_annotate_curves : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `plt.Axes.annotate` for annotating the curve labels.
        iterate : bool, optional
            If the plots should be returned as an iterator, by default False.
        ax: plt.Axes | Sequence[plt.Axes] | None, optional
            Axes to plot on, by default None. If None, an AxesGrid is created and returned. If not None, the subplots are filled row-wise w.r.t. the AxesGrid that would have been created, with each plot type being filled before moving on to the next subplot.

        Returns
        -------
        AxesGrid | list[AxesGrid]
            AxesGrid(s) with the data vs. theory plot(s).
        """

    @overload
    def plot_data_vs_theory(
        self,
        id_dataset: int | Sequence[int] | None = ...,
        type_experiment: str | None = ...,
        x_variable: str | list[str] | Literal["fallback"] | None = ...,
        xlabel: str | dict[str, str] | Literal["fallback"] | None = ...,
        ylabel: str | Literal["fallback"] | None = ...,
        xscale: str | None = ...,
        yscale: str | None = ...,
        title: str | None = ...,
        legend: LegendPos = ...,
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = ...,
        plot_types: DataVsTheoryType | Sequence[DataVsTheoryType] = ...,
        subplot_groupby: str | None = ...,
        subplot_label: Literal["legend"] | None = ...,
        subplot_label_format: str | None = ...,
        chi2_annotation: bool = ...,
        chi2_legend: LegendPos = ...,
        info_legend: LegendPos = ...,
        curve_groupby: str | list[str] | Literal["fallback"] | None = ...,
        apply_normalization: bool = ...,
        shift_correlated: Literal["data", "theory"] | None = ...,
        theory_min_width: float = ...,
        plot_pdf_uncertainty: bool = ...,
        pdf_uncertainty_convention: Literal["sym", "asym"] = ...,
        y_offset_add: float | None = ...,
        y_offset_mul: float | None = ...,
        kwargs_subplots: dict[str, Any] = ...,
        kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_theory_unc: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_xlabel: dict[str, Any] = ...,
        kwargs_ylabel: dict[str, Any] = ...,
        kwargs_title: dict[str, Any] = ...,
        kwargs_legend: dict[str, Any] = ...,
        kwargs_legend_chi2: dict[str, Any] = ...,
        kwargs_legend_info: dict[str, Any] = ...,
        kwargs_legend_curves: dict[str, Any] = ...,
        kwargs_legend_subplots: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_annotate_chi2: dict[str, Any] = ...,
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = ...,
        *,
        ax: None = ...,
        iterate: Literal[True],
        **kwargs: Any,
    ) -> Iterator[AxesGrid]:
        """Plot data vs. theory (one data set per figure).

        Parameters
        ----------
        id_dataset : int | Sequence[int] | None, optional
            Data sets for which to plot the data vs. theory, by default None. Must be of the same type_experiment. If None, `type_experiment` must be given.
        type_experiment : str | None, optional
            Process for which to plot the data vs. theory, by default None. If None, `id_dataset` must be given.
        x_variable : str | list[str] | None, optional
            The kinematic variable to put on the x axis, by default "fallback". Plots a binned distribution if a list is passed.
        xlabel : str | dict[str, str] | Literal["fallback"] | None, optional
            x label of the plot, by default None.
        ylabel : str | Literal["fallback"] | None, optional
            y label of the plot, by default None.
        xscale : str | None, optional
            x scale of the plot, by default no changing of x scale.
        yscale : str | None, optional
            y scale of the plot, by default no changing of y scale.
        title : str | None, optional
            Title of the plot, by default None.
        legend : bool, optional
            If a legend annotating data & theory is shown, by default True.
        curve_label : Literal["annotate above", "annotate right", "ticks", "legend"] | None, optional
            Where the curve labels are shown, by default "ticks". If None, no labels are shown.
        plot_types : DataVsTheoryType | Sequence[DataVsTheoryType], optional
            Types of the plots, by default "absolute". "absolute" plots the observable of the data set, "data over theory" plots the data over theory ratio, and vice versa for "theory over data". If a sequence is passed, each plot type becomes a subplot in the vertical direction.
        subplot_groupby: str | None, optional
            Variable to group the subplot axes by, by default None (no grouping).
        subplot_label : Literal["legend"] | None, optional  # FIXME
            Where to label the subplot, by default None.
        subplot_label_format : str | None, optional
            Format of the subplot label, by default None.
        chi2_annotation : bool, optional
            If the χ²/point value is annotated, by default True.
        chi2_legend : LegendPos, optional
            Where on the figure to show a legend with the χ², by default "upper left". If None, no χ² legend is shown.
        info_legend: LegendPos, optional
            On which subplot to show a legend with info about the data set, by default "upper left". If None, no info legend is shown.
        curve_groupby : str | list[str] | Literal["fallback"] | None, optional
            Variable(s) to group the curves by, by default "fallback".
        apply_normalization : bool, optional
            If the normalization-corrected theory is plotted, by default True.
        shift_correlated : Literal["data", "theory"], optional
            Whether to shift data or theory when correlated errors are present, by default "theory". For details, see arXiv:hep-ph/0201195 appendix B.2.
        theory_min_width : float, optional
            Width of the theory curve (in units of axes fraction) if there is only one point, by default 0.06.
        plot_pdf_uncertainty : bool, optional
            If the PDF uncertainty is plotted around the theory curve, by default True.
        pdf_uncertainty_convention : Literal["sym", "asym"], optional
            If the PDF uncertainties should be symmetric ("sym") or asymmetric ("asym"), by default "asym".
        y_offset_add : float | None, optional
            Additive offset by which the curves are separated, by default 0 (no offset).
        y_offset_mul : float | None, optional
            Multiplicative offset by which the curves are separated, by default 0 (no offset). The factor between each curve is `10**y_offset_mul`.
        kwargs_subplots : dict[str, Any], optional
            Keyword arguments to pass to `plt.subplots` through `AxesGrid`.
        kwargs_data : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `plt.Axes.plot` for plotting the data.
        kwargs_theory : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.plot` for plotting the theory.
        kwargs_theory_unc : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.fill_between` for plotting the PDF uncertainty.
        kwargs_xlabel : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_xlabel`.
        kwargs_ylabel : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_ylabel`.
        kwargs_title : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_title`.
        kwargs_legend : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating data & theory.
        kwargs_legend_chi2 : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating the total χ².
        kwargs_legend_info : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating data set info.
        kwargs_legend_subplots : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating subplot variables.
        kwargs_legend_curves : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating the curve labels.
        kwargs_ticks_curves : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.set_yticks` for annotating the curve labels.
        kwargs_annotate_chi2 : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.annotate` for annotating the χ²/point values.
        kwargs_annotate_curves : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `plt.Axes.annotate` for annotating the curve labels.
        iterate : bool, optional
            If the plots should be returned as an iterator, by default False.
        ax: plt.Axes | Sequence[plt.Axes] | None, optional
            Axes to plot on, by default None. If None, an AxesGrid is created and returned. If not None, the subplots are filled row-wise w.r.t. the AxesGrid that would have been created, with each plot type being filled before moving on to the next subplot.

        Returns
        -------
        Iterator[AxesGrid]
            AxesGrid(s) with the data vs. theory plot(s).
        """

    @overload
    def plot_data_vs_theory(
        self,
        id_dataset: int | Sequence[int] | None = ...,
        type_experiment: str | None = ...,
        x_variable: str | list[str] | Literal["fallback"] | None = ...,
        xlabel: str | dict[str, str] | Literal["fallback"] | None = ...,
        ylabel: str | Literal["fallback"] | None = ...,
        xscale: str | None = ...,
        yscale: str | None = ...,
        title: str | None = ...,
        legend: LegendPos = ...,
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = ...,
        plot_types: DataVsTheoryType | Sequence[DataVsTheoryType] = ...,
        subplot_groupby: str | None = ...,
        subplot_label: Literal["legend"] | None = ...,
        subplot_label_format: str | None = ...,
        chi2_annotation: bool = ...,
        chi2_legend: LegendPos = ...,
        info_legend: LegendPos = ...,
        curve_groupby: str | list[str] | Literal["fallback"] | None = ...,
        apply_normalization: bool = ...,
        shift_correlated: Literal["data", "theory"] | None = ...,
        theory_min_width: float = ...,
        plot_pdf_uncertainty: bool = ...,
        pdf_uncertainty_convention: Literal["sym", "asym"] = ...,
        y_offset_add: float | None = ...,
        y_offset_mul: float | None = ...,
        kwargs_subplots: dict[str, Any] = ...,
        kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_theory_unc: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_xlabel: dict[str, Any] = ...,
        kwargs_ylabel: dict[str, Any] = ...,
        kwargs_title: dict[str, Any] = ...,
        kwargs_legend: dict[str, Any] = ...,
        kwargs_legend_chi2: dict[str, Any] = ...,
        kwargs_legend_info: dict[str, Any] = ...,
        kwargs_legend_curves: dict[str, Any] = ...,
        kwargs_legend_subplots: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_annotate_chi2: dict[str, Any] = ...,
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = ...,
        *,
        ax: plt.Axes | Sequence[plt.Axes],
        iterate: bool = ...,
        **kwargs: Any,
    ) -> None:
        """Plot data vs. theory (one data set per figure).

        Parameters
        ----------
        id_dataset : int | Sequence[int] | None, optional
            Data sets for which to plot the data vs. theory, by default None. Must be of the same type_experiment. If None, `type_experiment` must be given.
        type_experiment : str | None, optional
            Process for which to plot the data vs. theory, by default None. If None, `id_dataset` must be given.
        x_variable : str | list[str] | None, optional
            The kinematic variable to put on the x axis, by default "fallback". Plots a binned distribution if a list is passed.
        xlabel : str | dict[str, str] | Literal["fallback"] | None, optional
            x label of the plot, by default None.
        ylabel : str | Literal["fallback"] | None, optional
            y label of the plot, by default None.
        xscale : str | None, optional
            x scale of the plot, by default no changing of x scale.
        yscale : str | None, optional
            y scale of the plot, by default no changing of y scale.
        title : str | None, optional
            Title of the plot, by default None.
        legend : bool, optional
            If a legend annotating data & theory is shown, by default True.
        curve_label : Literal["annotate above", "annotate right", "ticks", "legend"] | None, optional
            Where the curve labels are shown, by default "ticks". If None, no labels are shown.
        plot_types : DataVsTheoryType | Sequence[DataVsTheoryType], optional
            Types of the plots, by default "absolute". "absolute" plots the observable of the data set, "data over theory" plots the data over theory ratio, and vice versa for "theory over data". If a sequence is passed, each plot type becomes a subplot in the vertical direction.
        ax: plt.Axes | Sequence[plt.Axes] | None, optional
            Axes to plot on, by default None. If None, an AxesGrid is created and returned. If not None, the subplots are filled row-wise w.r.t. the AxesGrid that would have been created, with each plot type being filled before moving on to the next subplot.
        subplot_groupby: str | None, optional
            Variable to group the subplot axes by, by default None (no grouping).
        subplot_label : Literal["legend"] | None, optional  # FIXME
            Where to label the subplot, by default None.
        subplot_label_format : str | None, optional
            Format of the subplot label, by default None.
        chi2_annotation : bool, optional
            If the χ²/point value is annotated, by default True.
        chi2_legend : LegendPos, optional
            Where on the figure to show a legend with the χ², by default "upper left". If None, no χ² legend is shown.
        info_legend: LegendPos, optional
            On which subplot to show a legend with info about the data set, by default "upper left". If None, no info legend is shown.
        curve_groupby : str | list[str] | Literal["fallback"] | None, optional
            Variable(s) to group the curves by, by default "fallback".
        apply_normalization : bool, optional
            If the normalization-corrected theory is plotted, by default True.
        shift_correlated : Literal["data", "theory"], optional
            Whether to shift data or theory when correlated errors are present, by default "theory". For details, see arXiv:hep-ph/0201195 appendix B.2.
        theory_min_width : float, optional
            Width of the theory curve (in units of axes fraction) if there is only one point, by default 0.06.
        plot_pdf_uncertainty : bool, optional
            If the PDF uncertainty is plotted around the theory curve, by default True.
        pdf_uncertainty_convention : Literal["sym", "asym"], optional
            If the PDF uncertainties should be symmetric ("sym") or asymmetric ("asym"), by default "asym".
        y_offset_add : float | None, optional
            Additive offset by which the curves are separated, by default 0 (no offset).
        y_offset_mul : float | None, optional
            Multiplicative offset by which the curves are separated, by default 0 (no offset). The factor between each curve is `10**y_offset_mul`.
        kwargs_subplots : dict[str, Any], optional
            Keyword arguments to pass to `plt.subplots` through `AxesGrid`.
        kwargs_data : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `plt.Axes.plot` for plotting the data.
        kwargs_theory : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.plot` for plotting the theory.
        kwargs_theory_unc : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.fill_between` for plotting the PDF uncertainty.
        kwargs_xlabel : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_xlabel`.
        kwargs_ylabel : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_ylabel`.
        kwargs_title : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_title`.
        kwargs_legend : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating data & theory.
        kwargs_legend_chi2 : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating the total χ².
        kwargs_legend_info : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating data set info.
        kwargs_legend_subplots : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating subplot variables.
        kwargs_legend_curves : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating the curve labels.
        kwargs_ticks_curves : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.set_yticks` for annotating the curve labels.
        kwargs_annotate_chi2 : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.annotate` for annotating the χ²/point values.
        kwargs_annotate_curves : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `plt.Axes.annotate` for annotating the curve labels.
        iterate : bool, optional
            If the plots should be returned as an iterator, by default False.
        ax: plt.Axes | Sequence[plt.Axes] | None, optional
            Axes to plot on, by default None. If None, an AxesGrid is created and returned. If not None, the subplots are filled row-wise w.r.t. the AxesGrid that would have been created, with each plot type being filled before moving on to the next subplot.
        """

    def plot_data_vs_theory(
        self,
        id_dataset: int | Sequence[int] | None = None,
        type_experiment: str | None = None,
        x_variable: str | list[str] | Literal["fallback"] | None = "fallback",
        xlabel: str | dict[str, str] | Literal["fallback"] | None = "fallback",
        ylabel: str | Literal["fallback"] | None = "fallback",
        xscale: str | None = None,
        yscale: str | None = None,
        title: str | None = None,
        legend: LegendPos = "upper left",
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = "ticks",
        plot_types: DataVsTheoryType | Sequence[DataVsTheoryType] = ["absolute"],
        subplot_groupby: str | None = None,
        subplot_label: Literal["legend"] | None = None,
        subplot_label_format: str | None = None,
        chi2_annotation: bool = True,
        chi2_legend: LegendPos = "upper left",
        info_legend: LegendPos = "upper left",
        curve_groupby: str | list[str] | Literal["fallback"] | None = "fallback",
        apply_normalization: bool = True,
        shift_correlated: Literal["data", "theory"] | None = "theory",
        theory_min_width: float = 0.06,
        plot_pdf_uncertainty: bool = True,
        pdf_uncertainty_convention: Literal["sym", "asym"] = "asym",
        y_offset_add: float | None = None,
        y_offset_mul: float | None = None,
        kwargs_subplots: dict[str, Any] = {},
        kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_theory_unc: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_xlabel: dict[str, Any] = {},
        kwargs_ylabel: dict[str, Any] = {},
        kwargs_title: dict[str, Any] = {},
        kwargs_legend: dict[str, Any] = {},
        kwargs_legend_chi2: dict[str, Any] = {},
        kwargs_legend_info: dict[str, Any] = {},
        kwargs_legend_curves: dict[str, Any] = {},
        kwargs_legend_subplots: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_annotate_chi2: dict[str, Any] = {},
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        *,
        ax: plt.Axes | Sequence[plt.Axes] | None = None,
        iterate: bool = False,
        **kwargs: Any,
    ) -> AxesGrid | list[AxesGrid] | Iterator[AxesGrid] | None:
        res_iter = self._plot_data_vs_theory(
            id_dataset=id_dataset,
            type_experiment=type_experiment,
            x_variable=x_variable,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            title=title,
            legend=legend,
            curve_label=curve_label,
            plot_types=plot_types,
            ax=ax,
            subplot_groupby=subplot_groupby,
            subplot_label=subplot_label,
            subplot_label_format=subplot_label_format,
            chi2_annotation=chi2_annotation,
            chi2_legend=chi2_legend,
            info_legend=info_legend,
            curve_groupby=curve_groupby,
            apply_normalization=apply_normalization,
            shift_correlated=shift_correlated,
            theory_min_width=theory_min_width,
            plot_pdf_uncertainty=plot_pdf_uncertainty,
            pdf_uncertainty_convention=pdf_uncertainty_convention,
            y_offset_add=y_offset_add,
            y_offset_mul=y_offset_mul,
            kwargs_subplots=kwargs_subplots,
            kwargs_data=kwargs_data,
            kwargs_theory=kwargs_theory,
            kwargs_theory_unc=kwargs_theory_unc,
            kwargs_xlabel=kwargs_xlabel,
            kwargs_ylabel=kwargs_ylabel,
            kwargs_title=kwargs_title,
            kwargs_legend=kwargs_legend,
            kwargs_legend_chi2=kwargs_legend_chi2,
            kwargs_legend_info=kwargs_legend_info,
            kwargs_legend_curves=kwargs_legend_curves,
            kwargs_legend_subplots=kwargs_legend_subplots,
            kwargs_ticks_curves=kwargs_ticks_curves,
            kwargs_annotate_chi2=kwargs_annotate_chi2,
            kwargs_annotate_curves=kwargs_annotate_curves,
            **kwargs,
        )

        if ax is not None:
            # just consume iterator
            deque(res_iter, maxlen=0)
            return None
        elif iterate:
            return res_iter
        else:
            res = list(res_iter)

            return res[0] if len(res) == 1 else res

    # TODO: more sophisticated filtering
    # TODO: cuts
    # multiple figures with one dataset per figure
    def _plot_data_vs_theory(
        self,
        id_dataset: int | Sequence[int] | None = None,
        type_experiment: str | None = None,
        x_variable: str | list[str] | Literal["fallback"] | None = "fallback",
        xlabel: str | dict[str, str] | Literal["fallback"] | None = "fallback",
        ylabel: str | Literal["fallback"] | None = "fallback",
        xscale: str | None = None,
        yscale: str | None = None,
        title: str | None = None,
        legend: LegendPos = "upper left",
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = "ticks",
        plot_types: DataVsTheoryType | Sequence[DataVsTheoryType] = ["absolute"],
        ax: plt.Axes | Sequence[plt.Axes] | None = None,
        subplot_groupby: str | None = None,
        subplot_label: Literal["legend"] | None = None,
        subplot_label_format: str | None = None,
        chi2_annotation: bool = True,
        chi2_legend: LegendPos = "upper left",
        info_legend: LegendPos = "upper left",
        curve_groupby: str | list[str] | Literal["fallback"] | None = "fallback",
        apply_normalization: bool = True,
        shift_correlated: Literal["data", "theory"] | None = "theory",
        theory_min_width: float = 0.06,
        plot_pdf_uncertainty: bool = True,
        pdf_uncertainty_convention: Literal["sym", "asym"] = "asym",
        y_offset_add: float | None = None,
        y_offset_mul: float | None = None,
        kwargs_subplots: dict[str, Any] = {},
        kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_theory_unc: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_xlabel: dict[str, Any] = {},
        kwargs_ylabel: dict[str, Any] = {},
        kwargs_title: dict[str, Any] = {},
        kwargs_legend: dict[str, Any] = {},
        kwargs_legend_chi2: dict[str, Any] = {},
        kwargs_legend_info: dict[str, Any] = {},
        kwargs_legend_curves: dict[str, Any] = {},
        kwargs_legend_subplots: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_annotate_chi2: dict[str, Any] = {},
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        **kwargs: Any,
    ) -> Iterator[AxesGrid]:

        # bring id_dataset in Sequence[int] form (either directly from id_dataset or indirectly from collecting all IDs belonging to type_experiment)
        # the points variable holds all snapshot points belonging to the relevant dataset IDs
        # ValueErrors are thrown if id_dataset and type_experiment are not consistent
        if id_dataset is None:
            if type_experiment is None:
                raise ValueError(
                    "Please provide either `id_dataset` or `type_experiment`"
                )
            else:
                points = self.minimum_points.query(
                    "type_experiment == @type_experiment"
                )
                id_dataset = cast(
                    Sequence[int], points["id_dataset"].unique()
                )  # actually npt.NDArray[np.int_]
        else:
            if isinstance(id_dataset, int):
                id_dataset = [id_dataset]
            points = self.minimum_points.query("id_dataset in @id_dataset")
            types_experiment = points["type_experiment"].unique()
            if len(types_experiment) == 1:
                if type_experiment is None:
                    type_experiment = cast(str, types_experiment[0])
                elif not type_experiment == types_experiment[0]:
                    raise ValueError(
                        "Dataset(s) given by id_dataset are not of type_experiment"
                    )
            else:
                raise ValueError(
                    "Please provide datasets that are each of the same type_experiment"
                )

        if isinstance(plot_types, str):
            plot_types = [plot_types]

        if isinstance(ax, plt.Axes):
            ax = [ax]

        # group the relevant dataset IDs so we get one dataset per figure
        fig_gb = points.query("id_dataset in @id_dataset").groupby(
            "id_dataset", sort=True
        )

        for i, (id_dataset_i, data_i) in enumerate(fig_gb):

            subplot_gb = (
                data_i.groupby(subplot_groupby) if subplot_groupby is not None else None
            )

            kwargs_subplots_default = {
                "sharex": True,
                "sharey": "row",
                "unit_shape": (len(plot_types), 1),
                "unit_height_ratios": [
                    (2 if t == "absolute" else 1) for t in plot_types
                ],
                "layout": "constrained",
            }
            kwargs_subplots_updated = util.update_kwargs(
                kwargs_subplots_default, kwargs_subplots, i
            )

            n_real = (
                len(plot_types)
                if subplot_gb is None
                else len(subplot_gb) * len(plot_types)
            )

            if ax is None:
                ax_grid = AxesGrid(n_real=n_real, **kwargs_subplots_updated)
                ax_i = ax_grid.ax_unit_real
            else:
                if len(ax) != n_real:
                    raise ValueError(f"len(ax) must be {n_real}")

                ax_grid = None
                ax_i = ax[i * n_real : (i + 1) * n_real]

            if chi2_legend is not None:
                ax_legend_chi2 = (
                    ax_grid.locate_ax(chi2_legend) if ax_grid is not None else ax_i[0]
                )

                kwargs_legend_chi2_default = {
                    "order": 0,
                    "parent": ax_legend_chi2,
                    "handles": [Patch(), Patch(), Patch()],
                    "labels": [
                        f"$N_{{\\text{{data}}}} = {self.num_points[id_dataset_i]}$",
                        f"$\\chi^2_{{\\text{{total}}}} = {self.minimum_value_per_data[id_dataset_i]:.3f}$",
                        f"$\\chi^2_{{\\text{{total}}}}\\,/\\, N_{{\\text{{data}}}} = {self.minimum_value_per_data[id_dataset_i] / self.num_points[id_dataset_i]:.3f}$",
                    ],
                    "labelspacing": 0,
                    "handlelength": 0,
                    "handleheight": 0,
                    "handletextpad": 0,
                }
                kwargs_legend_chi2_updated = util.update_kwargs(
                    kwargs_legend_chi2_default, kwargs_legend_chi2
                )

                ax_legend_chi2.add_artist(
                    AdditionalLegend(**kwargs_legend_chi2_updated)
                )

            if info_legend is not None:
                ax_legend_info = (
                    ax_grid.locate_ax(info_legend) if ax_grid is not None else ax_i[0]
                )

                labels_info = self.datasets.index[
                    self.datasets.index["id_dataset"].isin(data_i["id_dataset"])
                ].iloc[0]
                labels_info_legend = [
                    f"{labels_info['experiment']} (ID {labels_info["id_dataset"]})",
                ]
                if not pd.isna(labels_info["reaction"]) and len(labels_info["reaction"]) > 0:
                    labels_info_legend.append(f"${labels.reaction_to_latex(labels_info["reaction"])}$")

                kwargs_legend_info_default = {
                    "order": -1,
                    "parent": ax_legend_info,
                    "handles": [Patch()] * len(labels_info_legend),
                    "labels": labels_info_legend,
                    "labelspacing": 0.25,
                    "handlelength": 0,
                    "handleheight": 0,
                    "handletextpad": 0,
                    "fontsize": "small",
                }
                kwargs_legend_info_updated = util.update_kwargs(
                    kwargs_legend_info_default, kwargs_legend_info
                )

                ax_legend_info.add_artist(
                    AdditionalLegend(**kwargs_legend_info_updated)
                )

            ax_iter = zip(
                (
                    (
                        np.atleast_2d(np.asarray(ax_ij)).T
                        for ax_ij in batched(ax_i, len(plot_types))
                    )
                    if ax_grid is None
                    else ax_grid.ax_unit_real
                ),
                (subplot_gb if subplot_gb is not None else [(np.nan, data_i)]),
            )
            for j, (ax_ij, (ax_gb_val_j, data_ij)) in enumerate(ax_iter):

                labels_x = (
                    self.datasets.labels_x.loc[data_ij["id_dataset"]].iloc[0].dropna()
                )
                labels_y = (
                    self.datasets.labels_y.loc[data_ij["id_dataset"]].iloc[0].dropna()
                )

                if xlabel == "fallback" and labels_x.size > 0:
                    xlabel_updated = {
                        str(k): str(s["LabelX"].iloc[0]).replace("\\frac", "\\dfrac")
                        for k, s in labels_x.groupby(level=1)
                    }
                    xunit = {
                        str(k): (
                            str(u).replace("\\frac", "\\dfrac")
                            if (u := s["UnitX"].iloc[0]) != 1.0
                            else ""
                        )
                        for k, s in labels_x.groupby(level=1)
                    }
                else:
                    xlabel_updated = xlabel
                    xunit = ""

                if ylabel == "fallback" and labels_y.size > 0:
                    ylabel_updated = str(labels_y["LabelY"]).replace(
                        "\\frac", "\\dfrac"
                    )
                    yunit = (
                        str(u).replace("\\frac", "\\dfrac")
                        if (u := labels_y["UnitY"]) != 1.0
                        else ""
                    )
                else:
                    ylabel_updated = ylabel
                    yunit = ""

                plot_legend = legend is not None and (
                    ax_grid is None
                    or ax_grid is not None
                    and ax_grid.locate_ax(legend) == ax_ij[0, 0]
                )

                data_vs_theory.plot(
                    type_experiment=type_experiment,
                    ax=ax_ij[:, 0],  # pyright: ignore[reportArgumentType]
                    points=data_ij,
                    x_variable=x_variable,
                    xlabel=xlabel_updated,
                    xunit=xunit,
                    ylabel=ylabel_updated,
                    yunit=yunit,
                    xscale=xscale,
                    yscale=yscale,
                    title=title,
                    legend=plot_legend,
                    curve_label=curve_label,
                    plot_types=plot_types,
                    # subplot_label=subplot_label,
                    subplot_label_format=subplot_label_format,
                    chi2_annotation=chi2_annotation,
                    chi2_legend=False,
                    curve_groupby=curve_groupby,
                    apply_normalization=apply_normalization,
                    shift_correlated=shift_correlated,
                    theory_min_width=theory_min_width,
                    plot_pdf_uncertainty=plot_pdf_uncertainty,
                    pdf_uncertainty_convention=pdf_uncertainty_convention,
                    y_offset_add=y_offset_add,
                    y_offset_mul=y_offset_mul,
                    kwargs_data=kwargs_data,
                    kwargs_theory=kwargs_theory,
                    kwargs_theory_unc=kwargs_theory_unc,
                    kwargs_xlabel=kwargs_xlabel,
                    kwargs_ylabel=kwargs_ylabel,
                    kwargs_title=kwargs_title,
                    kwargs_legend=kwargs_legend,
                    kwargs_legend_chi2=kwargs_legend_chi2,
                    kwargs_legend_curves=kwargs_legend_curves,
                    kwargs_ticks_curves=kwargs_ticks_curves,
                    kwargs_annotate_chi2=kwargs_annotate_chi2,
                    kwargs_annotate_curves=kwargs_annotate_curves,
                    # **kwargs_i,
                )

                if subplot_groupby is not None and subplot_label == "legend":
                    label = _format_curve_label(
                        subplot_groupby,
                        ax_gb_val_j,  # pyright: ignore[reportArgumentType]
                        variables_labels=xlabel_updated,
                        variables_units=xunit,
                    )

                    kwargs_legend_subplots_default = dict(
                        order=-2,
                        parent=ax_ij[0, 0],
                        handles=[Patch()],
                        labels=[label],
                        labelspacing=0,
                        handlelength=0,
                        handleheight=0,
                        handletextpad=0,
                        fontsize="small",
                    )
                    kwargs_legend_subplots_updated = util.update_kwargs(
                        kwargs_legend_subplots_default,
                        kwargs_legend_subplots,
                        i * len(fig_gb) + j,
                    )

                    subplot_legend = AdditionalLegend(**kwargs_legend_subplots_updated)

                    ax_ij[0, 0].add_artist(subplot_legend)

            if ax_grid is not None:
                ax_grid.prune_labels()
                yield ax_grid

    def plot_data_vs_theory_grouped(
        self,
        id_dataset: int | Sequence[int] | None = None,
        type_experiment: str | None = None,
        x_variable: str | list[str] | Literal["fallback"] | None = "fallback",
        xlabel: str | dict[str, str] | Literal["fallback"] | None = "fallback",
        ylabel: str | Literal["fallback"] | None = "fallback",
        xscale: str | None = None,
        yscale: str | None = None,
        title: str | None = None,
        legend: LegendPos = "upper left",
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = "ticks",
        plot_types: DataVsTheoryType | Sequence[DataVsTheoryType] = "absolute",
        ax: plt.Axes | Sequence[plt.Axes] | None = None,
        subplot_groupby: str | None = None,
        subplot_label: Literal["legend"] | None = "legend",
        subplot_label_format: str | None = None,
        chi2_annotation: bool = True,
        chi2_legend: LegendPos = "upper left",
        info_legend: LegendPos = "upper left",
        curve_groupby: str | list[str] | Literal["fallback"] | None = "fallback",
        apply_normalization: bool = True,
        theory_min_width: float = 0.06,
        plot_pdf_uncertainty: bool = True,
        pdf_uncertainty_convention: Literal["sym", "asym"] = "asym",
        y_offset_add: float | None = None,
        y_offset_mul: float | None = None,
        kwargs_subplots: dict[str, Any] = {},
        kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_theory_unc: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_xlabel: dict[str, Any] = {},
        kwargs_ylabel: dict[str, Any] = {},
        kwargs_title: dict[str, Any] = {},
        kwargs_legend: dict[str, Any] = {},
        kwargs_legend_chi2: dict[str, Any] = {},
        kwargs_legend_info: dict[str, Any] = {},
        kwargs_legend_curves: dict[str, Any] = {},
        kwargs_legend_subplots: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_annotate_chi2: dict[str, Any] = {},
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        **kwargs,
    ) -> AxesGrid | None:
        """Plot data vs. theory (one figure with multiple data sets).

        Parameters
        ----------
        id_dataset : int | Sequence[int] | None, optional
            Data sets for which to plot the data vs. theory, by default None. Must be of the same type_experiment. If None, `type_experiment` must be given.
        type_experiment : str | None, optional
            Process for which to plot the data vs. theory, by default None. If None, `id_dataset` must be given.
        x_variable : str | list[str] | None, optional
            The kinematic variable to put on the x axis, by default "fallback". Plots a binned distribution if a list is passed.
        xlabel : str | dict[str, str] | Literal["fallback"] | None, optional
            x label of the plot, by default None.
        ylabel : str | Literal["fallback"] | None, optional
            y label of the plot, by default None.
        xscale : str | None, optional
            x scale of the plot, by default no changing of x scale.
        yscale : str | None, optional
            y scale of the plot, by default no changing of y scale.
        title : str | None, optional
            Title of the plot, by default None.
        legend : bool, optional
            If a legend annotating data & theory is shown, by default True.
        curve_label : Literal["annotate above", "annotate right", "ticks", "legend"] | None, optional
            Where the curve labels are shown, by default "ticks". If None, no labels are shown.
        plot_types : DataVsTheoryType | Sequence[DataVsTheoryType], optional
            Types of the plots, by default "absolute". "absolute" plots the observable of the data set, "data over theory" plots the data over theory ratio, and vice versa for "theory over data". If a sequence is passed, each plot type becomes a subplot in the vertical direction.
        ax: plt.Axes | Sequence[plt.Axes] | None, optional
            Axes to plot on, by default None. If None, an AxesGrid is created and returned. If not None, the subplots are filled row-wise w.r.t. the AxesGrid that would have been created, with each plot type being filled before moving on to the next subplot.
        subplot_groupby: str | None, optional
            Variable to group the subplot axes by, by default None (no grouping).
        subplot_label : Literal["legend"] | None, optional
            Where to label the subplot, by default "legend".
        subplot_label_format : str | None, optional
            Format of the subplot label, by default None.
        chi2_annotation : bool, optional
            If the χ²/point value is annotated, by default True.
        chi2_legend : LegendPos, optional
            Where on the figure to show a legend with the χ², by default "upper left". If None, no χ² legend is shown.
        info_legend: LegendPos, optional
            On which subplot to show a legend with info about the data set, by default "upper left". If None, no info legend is shown.
        curve_groupby : str | list[str] | Literal["fallback"] | None, optional
            Variable(s) to group the curves by, by default "fallback".
        apply_normalization : bool, optional
            If the normalization-corrected theory is plotted, by default True.
        theory_min_width : float, optional
            Width of the theory curve (in units of axes fraction) if there is only one point, by default 0.06.
        plot_pdf_uncertainty : bool, optional
            If the PDF uncertainty is plotted around the theory curve, by default True.
        pdf_uncertainty_convention : Literal["sym", "asym"], optional
            If the PDF uncertainties should be symmetric ("sym") or asymmetric ("asym"), by default "asym".
        y_offset_add : float | None, optional
            Additive offset by which the curves are separated, by default 0 (no offset).
        y_offset_mul : float | None, optional
            Multiplicative offset by which the curves are separated, by default 0 (no offset). The factor between each curve is `10**y_offset_mul`.
        kwargs_subplots : dict[str, Any], optional
            Keyword arguments to pass to `plt.subplots` through `AxesGrid`.
        kwargs_data : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `plt.Axes.plot` for plotting the data.
        kwargs_theory : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.plot` for plotting the theory.
        kwargs_theory_unc : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.fill_between` for plotting the PDF uncertainty.
        kwargs_xlabel : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_xlabel`.
        kwargs_ylabel : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_ylabel`.
        kwargs_title : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.set_title`.
        kwargs_legend : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating data & theory.
        kwargs_legend_chi2 : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating the total χ².
        kwargs_legend_info : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating data set info.
        kwargs_legend_curves : dict[str, Any], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating the curve labels.
        kwargs_legend_subplots : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `AdditionalLegend` for annotating subplot variables.
        kwargs_ticks_curves : dict[str, Any] | list[dict[str, Any]  |  None], optional
            Keyword arguments to pass to `plt.Axes.set_yticks` for annotating the curve labels.
        kwargs_annotate_chi2 : dict[str, Any], optional
            Keyword arguments to pass to `plt.Axes.annotate` for annotating the χ²/point values.
        kwargs_annotate_curves : dict[str, Any] | list[dict[str, Any] | None], optional
            Keyword arguments to pass to `plt.Axes.annotate` for annotating the curve labels.
        iterate : bool, optional
            If the plots should be returned as an iterator, by default False.

        Returns
        -------
        Iterator[AxesGrid]
            AxesGrid(s) with the data vs. theory plot(s).
        """

        if id_dataset is None:
            if type_experiment is None:
                raise ValueError(
                    "Please provide either `id_dataset` or `type_experiment`"
                )
            else:
                points = self.minimum_points.query(
                    "type_experiment == @type_experiment"
                )
                id_dataset = cast(
                    Sequence[int], points["id_dataset"].unique()
                )  # actually npt.NDArray[np.int_]
        else:
            if isinstance(id_dataset, int):
                id_dataset = [id_dataset]

            points = self.minimum_points.query("id_dataset in @id_dataset")
            types_experiment = points["type_experiment"].unique()
            if len(types_experiment) == 1:
                if type_experiment is None:
                    type_experiment = cast(str, types_experiment[0])
                elif not type_experiment == types_experiment[0]:
                    raise ValueError(
                        "Dataset(s) given by id_dataset are not of type_experiment"
                    )
            else:
                raise ValueError(
                    "Please provide datasets that are each of the same type_experiment"
                )

        if isinstance(plot_types, str):
            plot_types = [plot_types]

        if isinstance(ax, plt.Axes):
            ax = [ax]

        subplot_gb = (
            points.groupby(subplot_groupby) if subplot_groupby is not None else None
        )

        kwargs_subplots_default = {
            "sharex": True,
            "sharey": "row",
            "unit_shape": (len(plot_types), 1),
            "unit_height_ratios": [(2 if t == "absolute" else 1) for t in plot_types],
            "layout": "constrained",
        }
        kwargs_subplots_updated = util.update_kwargs(
            kwargs_subplots_default, kwargs_subplots
        )

        n_real = (
            len(plot_types) if subplot_gb is None else len(subplot_gb) * len(plot_types)
        )

        if ax is None:
            ax_grid = AxesGrid(n_real=n_real, **kwargs_subplots_updated)
            ax_i = ax_grid.ax_unit_real
        else:
            if len(ax) != n_real:
                raise ValueError(f"len(ax) must be {n_real}")

            ax_grid = None
            ax_i = ax

        if chi2_legend is not None:
            ax_legend_chi2 = (
                ax_grid.locate_ax(chi2_legend)
                if ax_grid is not None
                else ax[0]  # pyright: ignore[reportOptionalSubscript]
            )
            kwargs_legend_chi2_default = {
                "order": 0,
                "parent": ax_legend_chi2,
                "handles": [Patch(), Patch(), Patch()],
                "labels": [
                    f"$N_{{\\text{{data}}}} = {self.num_points[id_dataset].sum()}$",  # pyright: ignore[reportCallIssue,reportArgumentType]
                    f"$\\chi^2_{{\\text{{total}}}} = {self.minimum_value_per_data[id_dataset].sum():.3f}$",  # pyright: ignore[reportCallIssue,reportArgumentType]
                    f"$\\chi^2_{{\\text{{total}}}}\\,/\\, N_{{\\text{{data}}}} = {self.minimum_value_per_data[id_dataset].sum() / self.num_points[id_dataset].sum():.3f}$",  # pyright: ignore[reportCallIssue,reportArgumentType]
                ],
                "labelspacing": 0,
                "handlelength": 0,
                "handleheight": 0,
                "handletextpad": 0,
            }
            kwargs_legend_chi2_updated = util.update_kwargs(
                kwargs_legend_chi2_default, kwargs_legend_chi2
            )

            ax_legend_chi2.add_artist(AdditionalLegend(**kwargs_legend_chi2_updated))

        if info_legend is not None:
            ax_legend_info = (
                ax_grid.locate_ax(info_legend) if ax_grid is not None else ax_i[0]
            )

            labels_info = self.datasets.index[
                self.datasets.index["id_dataset"].isin(points["id_dataset"])
            ]
            # labels_info_legend = [
            #     f"{labels_info['experiment']} (ID {labels_info["id_dataset"]})",
            #     f"${labels.reaction_to_latex(labels_info["reaction"])}$",
            # ]
            labels_info_legend = format_dataset_info(labels_info)

            kwargs_legend_info_default = {
                "order": -1,
                "parent": ax_legend_info,
                "handles": [Patch()] * len(labels_info_legend),
                "labels": labels_info_legend,
                "labelspacing": 0.25,
                "handlelength": 0,
                "handleheight": 0,
                "handletextpad": 0,
                "fontsize": "small",
            }
            kwargs_legend_info_updated = util.update_kwargs(
                kwargs_legend_info_default, kwargs_legend_info
            )

            ax_legend_info.add_artist(AdditionalLegend(**kwargs_legend_info_updated))

        ax_iter = zip(
            (
                (
                    np.atleast_2d(np.asarray(ax_ij)).T
                    for ax_ij in batched(ax_i, len(plot_types))
                )
                if ax_grid is None
                else ax_grid.ax_unit_real
            ),
            (subplot_gb if subplot_gb is not None else [(np.nan, points)]),
        )

        for i, (ax_ij, (_, points_i)) in enumerate(ax_iter):

            labels_x = (
                self.datasets.labels_x.loc[points_i["id_dataset"]].iloc[0].dropna()
            )
            labels_y = (
                self.datasets.labels_y.loc[points_i["id_dataset"]].iloc[0].dropna()
            )

            if xlabel == "fallback" and labels_x.size > 0:
                xlabel = {
                    str(k): str(s["LabelX"].iloc[0]).replace("\\frac", "\\dfrac")
                    for k, s in labels_x.groupby(level=1)
                }
                xunit = {
                    str(k): (
                        str(u).replace("\\frac", "\\dfrac")
                        if (u := s["UnitX"].iloc[0]) != 1.0
                        else ""
                    )
                    for k, s in labels_x.groupby(level=1)
                }
            else:
                xunit = ""

            if ylabel == "fallback" and labels_y.size > 0:
                ylabel = str(labels_y["LabelY"]).replace("\\frac", "\\dfrac")
                yunit = (
                    str(u).replace("\\frac", "\\dfrac")
                    if (u := labels_y["UnitY"]) != 1.0
                    else ""
                )
            else:
                yunit = ""

            if subplot_groupby is not None and subplot_label_format is None:
                subplot_label_format_i = (
                    "$"
                    + ", ".join(
                        f"{labels.kinvars_py_to_tex.get(s_gb, s_gb)} = {{{s_gb}}}"
                        for s_gb in subplot_groupby
                    )
                    + "$"
                )
            else:
                subplot_label_format_i = subplot_label_format

            plot_legend = legend is not None and (
                ax_grid is None
                or ax_grid is not None
                and ax_grid.locate_ax(legend) == ax_ij[0, 0]
            )

            data_vs_theory.plot(
                type_experiment=type_experiment,
                points=points_i,
                ax=ax_ij[:, 0],  # pyright: ignore[reportArgumentType]
                x_variable=x_variable,
                xlabel=xlabel,
                xunit=xunit,
                ylabel=ylabel,
                yunit=yunit,
                xscale=xscale,
                yscale=yscale,
                title=title,
                legend=plot_legend,
                curve_label=curve_label,
                plot_types=plot_types,
                subplot_label=subplot_label if subplot_groupby is not None else None,
                subplot_label_format=subplot_label_format_i,
                chi2_annotation=chi2_annotation,
                chi2_legend=False,
                curve_groupby=curve_groupby,
                apply_normalization=apply_normalization,
                theory_min_width=theory_min_width,
                plot_pdf_uncertainty=plot_pdf_uncertainty,
                pdf_uncertainty_convention=pdf_uncertainty_convention,
                y_offset_add=y_offset_add,
                y_offset_mul=y_offset_mul,
                kwargs_data=kwargs_data,
                kwargs_theory=kwargs_theory,
                kwargs_theory_unc=kwargs_theory_unc,
                kwargs_xlabel=kwargs_xlabel,
                kwargs_ylabel=kwargs_ylabel,
                kwargs_title=kwargs_title,
                kwargs_legend=kwargs_legend,
                kwargs_legend_chi2=kwargs_legend_chi2,
                kwargs_legend_curves=kwargs_legend_curves,
                kwargs_legend_subplots=util.get_kwargs(kwargs_legend_subplots, i),
                kwargs_ticks_curves=kwargs_ticks_curves,
                kwargs_annotate_chi2=kwargs_annotate_chi2,
                kwargs_annotate_curves=kwargs_annotate_curves,
                **kwargs,
            )

        if ax_grid is not None:
            ax_grid.prune_labels()

        return ax_grid


def format_dataset_info(info: pd.DataFrame) -> list[str]:
    experiments = []

    for exp_i, info_i in info.sort_values("id_dataset").groupby(
        "experiment", sort=False
    ):
        experiments.append(
            f"{exp_i} ({"IDs" if info_i.shape[0] > 1 else "ID"} {util.format_indices(info_i["id_dataset"])})"  # pyright: ignore[reportArgumentType]
        )

    return [
        ", ".join(experiments),
        *(
            f"${labels.reaction_to_latex(r)}$"
            for r in info.sort_values("id_dataset")["reaction"].unique()
        ),
    ]
