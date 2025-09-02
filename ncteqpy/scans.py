import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence, cast, override, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ncteqpy.jaml as jaml
from ncteqpy.data import Datasets
from ncteqpy.data_groupby import DatasetsGroupBy
from ncteqpy.plot.scan import plot_scan_1d, plot_scan_2d


def _all_equal(s: Sequence[Any]) -> bool:
    """Returns True if all elements of `s` are equal"""
    return s.count(s[0]) == len(s)


class ParameterScan(ABC, jaml.YAMLWrapper):
    """Abstract base class for parameter scans. Subclasses must override `_load_ranges` and `_load_profile`, and should implement the properties `parameters_scanned` and `parameters_scanned_indices`"""

    _parameters_all: list[str] | None = None
    _parameters_all_indices: dict[str, int] | None = None

    _target_delta_chi2: float | None = None
    _margin: float | None = None
    _num_steps: int | None = None

    _parameters_range: pd.DataFrame | None = None

    _profile_params: pd.DataFrame | None = None
    _profile_chi2: pd.DataFrame | None = None
    _profile_chi2_per_data: pd.DataFrame | None = None

    _minimum_params: pd.DataFrame | None = None
    _minimum_chi2: float | None = None

    def __init__(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        cache_path: str | os.PathLike[str] = "./.jaml_cache",
        retain_yaml: bool = False,
    ) -> None:
        super().__init__(paths, cache_path, retain_yaml)

    def _load_params(self) -> None:
        """Initialize `_parameters_names` and `_parameters_indices` by reading `Chi2Fcn: IndexOfInputParams` and `Chi2Fcn: ParamIndices`"""

        pattern = jaml.Pattern(
            {"Chi2Fcn": {"IndexOfInputParams": None, "ParamIndices": None}}
        )
        yaml = cast(
            dict[str, jaml.YAMLType] | list[tuple[Path, dict[str, jaml.YAMLType]]],
            self._load_yaml(pattern),
        )

        if not isinstance(yaml, list):
            yaml = [(Path(), yaml)]

        # read parameter names and indices
        index_of_input_params: list[list[int]] = []
        param_indices: list[dict[str, int]] = []
        for y in map(lambda x: x[1], yaml):
            index_of_input_params.append(
                cast(list[int], jaml.nested_get(y, ["Chi2Fcn", "IndexOfInputParams"]))
            )
            param_indices.append(
                cast(dict[str, int], jaml.nested_get(y, ["Chi2Fcn", "ParamIndices"]))
            )

        # check that all files have the same parameter names and indices
        if not _all_equal(index_of_input_params) or not _all_equal(param_indices):
            raise ValueError(
                "Incompatible scans: All files must have the same parameter names and indices"
            )

        # order the parameter name to parameter index mapping by the index (dicts preserve ordering in Python 3.7+)
        self._parameters_all_indices = {
            p: i_p
            for i in index_of_input_params[0]
            for p, i_p in param_indices[0].items()
            if i_p == i
        }
        self._parameters_all = list(self._parameters_all_indices.keys())

    def _load_scan_info(self) -> None:
        """Initialize `_target_delta_chi2`, `_margin`, `_num_steps`, and `_minimum_chi2`"""

        pattern = jaml.Pattern(
            {
                "Scans": {
                    "targetDeltaChi2": None,
                    "rangeMargin": None,
                    "numberOfSteps": None,
                    "Chi2AtMinimum": None,
                }
            }
        )

        yaml = cast(
            dict[str, jaml.YAMLType] | list[tuple[Path, dict[str, jaml.YAMLType]]],
            self._load_yaml(pattern),
        )

        if not isinstance(yaml, list):
            yaml = [(Path(), yaml)]

        target_delta_chi2: list[float] = []
        margin: list[float] = []
        num_steps: list[int] = []
        min_chi2: list[float] = []
        for y in map(lambda x: x[1], yaml):
            target_delta_chi2.append(
                cast(float, jaml.nested_get(y, ["Scans", "targetDeltaChi2"]))
            )
            margin.append(cast(float, jaml.nested_get(y, ["Scans", "rangeMargin"])))
            num_steps.append(cast(int, jaml.nested_get(y, ["Scans", "numberOfSteps"])))
            min_chi2.append(cast(float, jaml.nested_get(y, ["Scans", "Chi2AtMinimum"])))

        # check that all files have the same values
        # TODO: support multiple values per parameter?
        if (
            not _all_equal(target_delta_chi2)
            or not _all_equal(margin)
            or not _all_equal(num_steps)
            or not _all_equal(min_chi2)
        ):
            raise ValueError(
                "Incompatible scans: All files must have the same targetDeltaChi2, rangeMargin, numberOfSteps and chi2AtMinimum"
            )

        self._target_delta_chi2 = target_delta_chi2[0]
        self._margin = margin[0]
        self._num_steps = num_steps[0]
        self._minimum_chi2 = min_chi2[0]

    @abstractmethod
    def _load_ranges(self) -> None:
        pass

    @abstractmethod
    def _load_profile(self) -> None:
        pass

    @property
    def parameters_all(self) -> list[str]:
        if self._parameters_all is None or self._yaml_changed():
            self._load_params()

        assert self._parameters_all is not None

        return self._parameters_all

    @property
    def parameters_all_indices(self) -> dict[str, int]:
        if self._parameters_all_indices is None or self._yaml_changed():
            self._load_params()

        assert self._parameters_all_indices is not None

        return self._parameters_all_indices

    @property
    def target_delta_chi2(self) -> float:
        if self._target_delta_chi2 is None or self._yaml_changed():
            self._load_scan_info()

        assert self._target_delta_chi2 is not None

        return self._target_delta_chi2

    @property
    def margin(self) -> float:
        if self._margin is None or self._yaml_changed():
            self._load_scan_info()

        assert self._margin is not None

        return self._margin

    @property
    def num_steps(self) -> int:
        if self._num_steps is None or self._yaml_changed():
            self._load_scan_info()

        assert self._num_steps is not None

        return self._num_steps

    @property
    def minimum_chi2(self) -> float:
        if self._minimum_chi2 is None or self._yaml_changed():
            self._load_scan_info()

        assert self._minimum_chi2 is not None

        return self._minimum_chi2

    @property
    def minimum_params(self) -> pd.DataFrame:
        if self._minimum_params is None or self._yaml_changed():
            self._load_ranges()

        assert self._minimum_params is not None

        return self._minimum_params

    @property
    def parameters_range(self) -> pd.DataFrame:
        if self._parameters_range is None or self._yaml_changed():
            self._load_ranges()

        assert self._parameters_range is not None

        return self._parameters_range

    @property
    def profile_params(self) -> pd.DataFrame:
        if self._profile_params is None or self._yaml_changed():
            self._load_profile()

        assert self._profile_params is not None

        return self._profile_params

    @property
    def profile_chi2(self) -> pd.DataFrame:
        if self._profile_chi2 is None or self._yaml_changed():
            self._load_profile()

        assert self._profile_chi2 is not None

        return self._profile_chi2

    @property
    def profile_chi2_per_data(self) -> pd.DataFrame:
        if self._profile_chi2_per_data is None or self._yaml_changed():
            self._load_profile()

        assert self._profile_chi2_per_data is not None

        return self._profile_chi2_per_data


class ParameterScan1D(ParameterScan):

    _parameters_scanned: list[str] | None = None
    _parameters_scanned_indices: dict[str, int] | None = None

    _datasets: set[int] | None = None

    def __init__(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        cache_path: str | os.PathLike[str] = "./.jaml_cache",
        retain_yaml: bool = False,
    ) -> None:
        super().__init__(paths, cache_path, retain_yaml)

    @override
    def _load_ranges(self) -> None:
        """Initialize `_parameters_scanned`, `_parameters_scanned_indices` and `_parameters_range`"""

        pattern = jaml.Pattern(
            {
                "Scans": {
                    "ParameterScans": [
                        {
                            "parameterIndex": None,
                            "parameterName": None,
                            "upperRange": None,
                            "lowerRange": None,
                            "paramValueInMinimum": None,
                        }
                    ]
                }
            }
        )
        yaml = cast(
            dict[str, jaml.YAMLType] | list[tuple[Path, dict[str, jaml.YAMLType]]],
            self._load_yaml(pattern),
        )

        if not isinstance(yaml, list):
            yaml = [(Path(), yaml)]

        param_indices: list[int] = []
        param_names: list[str] = []
        ranges: list[list[float]] = []
        param_min: list[float] = []
        for _, y in yaml:

            for scan in cast(
                list[dict[str, Any]],
                jaml.nested_get(y, ["Scans", "ParameterScans"]),
            ):
                param_indices_i = cast(int, scan["parameterIndex"])
                param_names_i = cast(str, scan["parameterName"])
                ranges_upper_i = cast(float, scan["upperRange"])
                ranges_lower_i = cast(float, scan["lowerRange"])
                param_min_i = cast(float, scan["paramValueInMinimum"])

                param_indices.append(param_indices_i)
                param_names.append(param_names_i)
                ranges.append([ranges_lower_i, ranges_upper_i])
                param_min.append(param_min_i)

        assert _all_equal(
            [
                len(param_indices),
                len(param_names),
                len(ranges),
                len(param_min),
            ]
        )

        for i, p in zip(param_indices, param_names):
            # check that the parameter indices match the ones in Chi2Fcn
            if self.parameters_all_indices[p] != i:
                raise ValueError(
                    f"Parameter index mismatch: {p} is at index {self.parameters_all_indices[p]} in `Scans` but at index {i} in `Chi2Fcn`"
                )

            # check for duplicates
            if param_indices.count(i) > 1:
                raise ValueError(f"Duplicate parameter {p} (index {i})")

        # sort the columns by the parameter indices
        param_indices, param_names, ranges, param_min = zip(
            *sorted(
                zip(param_indices, param_names, ranges, param_min),
                key=lambda x: x[0],
            )
        )

        self._parameters_scanned = list(param_names)
        self._parameters_scanned_indices = {p: i for i, p in enumerate(param_names)}
        self._parameters_range = pd.DataFrame(np.array(ranges).T, columns=param_names)
        self._minimum_params = pd.DataFrame([param_min], columns=param_names)

    @property
    def parameters_scanned(self) -> list[str]:
        if self._parameters_scanned is None or self._yaml_changed():
            self._load_ranges()

        assert self._parameters_scanned is not None

        return self._parameters_scanned

    @override
    def _load_profile(self) -> None:
        """Initialize `_profile_params`, `_profile_chi2` and `_profile_chi2_per_data`"""

        pattern = jaml.Pattern(
            {
                "Scans": {
                    "ParameterScans": [
                        {
                            "profile": None,
                            "parameterName": None,
                            "snapshots": [
                                {
                                    "par": None,
                                    "outputPerPointBreakdown": None,
                                    "outputPerDataBreakdown": None,
                                    "chi2Value": None,
                                    "perDataBreakdown": None,
                                }
                            ],
                        }
                    ],
                }
            }
        )

        yaml = cast(
            dict[str, jaml.YAMLType] | list[tuple[Path, dict[str, jaml.YAMLType]]],
            self._load_yaml(pattern),
        )

        if not isinstance(yaml, list):
            yaml = [(Path(), yaml)]

        params: list[str] = []
        datasets: list[list[int]] = []
        profile_params: list[list[float]] = []
        profile_chi2: list[list[float]] = []
        profile_chi2_per_data: list[list[list[float]]] = []
        for y in map(lambda x: x[1], yaml):

            for scan in cast(
                list[dict[str, Any]],
                jaml.nested_get(y, ["Scans", "ParameterScans"]),
            ):
                params.append(scan["parameterName"])
                profile_params.append(
                    [x[0] for x in cast(list[list[float]], scan["profile"])]
                )
                profile_chi2.append(
                    [x[1] for x in cast(list[list[float]], scan["profile"])]
                )

                snapshots_per_data: list[list[float]] = []
                for snapshot in cast(
                    list[dict[str, Any]],
                    scan["snapshots"],
                ):
                    if cast(bool, snapshot["outputPerDataBreakdown"]):
                        datasets.append(
                            list(
                                cast(
                                    dict[int, float], snapshot["perDataBreakdown"]
                                ).keys()
                            )
                        )
                        snapshots_per_data.append(
                            list(
                                cast(
                                    dict[int, float], snapshot["perDataBreakdown"]
                                ).values()
                            )
                        )

                # transpose snapshots_per_data before appending to profile_chi2_per_data
                profile_chi2_per_data.append(list(map(list, zip(*snapshots_per_data))))

        self._datasets = set(y for x in datasets for y in x)

        # sort the parameter columns by the parameter indices
        params, profile_params, profile_chi2, profile_chi2_per_data = zip(
            *sorted(
                zip(params, profile_params, profile_chi2, profile_chi2_per_data),
                key=lambda x: self.parameters_all_indices[x[0]],
            )
        )

        self._profile_params = pd.DataFrame(
            np.array(profile_params).T + self.minimum_params.values[0],
            columns=pd.Index(params, name="parameter"),
        )
        self._profile_chi2 = pd.DataFrame(
            np.array(profile_chi2).T, columns=pd.Index(params, name="parameter")
        )
        self._profile_chi2_per_data = pd.DataFrame(
            np.array([y for x in profile_chi2_per_data for y in x]).T,
            columns=pd.MultiIndex.from_tuples(
                [(p, d) for p, ds in zip(params, datasets) for d in ds],
                names=("parameter", "id_dataset"),
            ),
        )

    @property
    def datasets(self) -> set[int]:
        if self._datasets is None or self._yaml_changed():
            self._load_profile()

        assert (
            self._datasets is not None
        )  # TODO: may be None if no per-data breakdown was loaded

        return self._datasets

    @property
    def profile_chi2_per_data(self) -> pd.DataFrame:
        if self._profile_chi2_per_data is None or self._yaml_changed():
            self._load_profile()

        assert (
            self._profile_chi2_per_data is not None
        )  # TODO: may be None if no per-data breakdown was loaded

        return self._profile_chi2_per_data

    def plot(
        self,
        ax: plt.Axes | Sequence[plt.Axes],
        parameter: str | Sequence[str] | None = None,
        datasets: Datasets | DatasetsGroupBy | None = None,
        data_groupby: str | None = None,
        groups_labels: dict[str, str] | None = None,
        highlight_groups: str | list[str] | None = None,
        highlight_important_groups: int | None = None,
        legend: bool | Literal["last","first"] = "last",
        kwargs_chi2_total: dict[str, Any] | None = None,
        kwargs_chi2_minimum: dict[str, Any] | None = None,
        kwargs_chi2_groups: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot the parameter scan(s)

        Parameters
        ----------
        ax : plt.Axes or sequence of plt.Axes
            Axes to plot on. If a sequence is given, its length must match the number of eigenvectors to plot.
        parameter :
            Parameters to plot. If `None`, all scanned parameters are plotted.
        datasets : Datasets, optional
            Datasets object containing the dataset metadata. Required if `data_groupby` is not `None`.
        data_groupby : str, optional
            Column name in `datasets.index` to group datasets by. If `None`, no grouping is applied.
        groups_labels : dict, optional
            Mapping of group names (as in `data_groupby`) to labels to use in the legend. If `None`, the group names are used as labels.
        highlight_groups : str or list of str, optional
            Group name or list of group names (as in `data_groupby`) to highlight in the plot. Highlighted groups are drawn with thicker lines and higher opacity. 
            If `None`, no groups are highlighted.
        highlight_important_groups : int, optional
            Number of most important groups (by maximum chi2 contribution) to highlight in the plot. If `None`, no groups are highlighted.
        plot_minimum : bool, default True
            Whether to mark the minimum chi2 point on the plot.
        legend : bool or 'last' or 'first', default 'last'
            Whether to show a legend. If 'last', the legend is shown only on the last subplot. If 'first', the legend is shown only on the first subplot.
        kwargs_chi2_total : dict, optional
            Additional keyword arguments passed to `ax.plot` when plotting the total chi2 curve.
        kwargs_chi2_minimum : dict, optional
            Additional keyword arguments passed to when plotting the minimum chi2
        kwargs_chi2_groups : dict or list of dict, optional
            Additional keyword arguments passed to `ax.plot` when plotting the grouped chi2 curves. If a list is given, its length must match the number of groups. If a dict is given, the same arguments are used for all groups.
        """
        if data_groupby is not None:
            if self.profile_chi2_per_data is None:
                raise ValueError(
                    "Grouping data not available since no per-data breakdown was loaded"
                )

            if datasets is None:
                raise ValueError("Grouping data requires passing `datasets`")

            # construct dataset grouper as pd.Series with index "id_dataset" and values `data_groupby`
            # TODO: Better treatment of datasets with the same ID but different values for `data_groupby`
            if isinstance(datasets, Datasets):
                grouper = (
                    datasets.index[["id_dataset", data_groupby]]
                    .drop_duplicates()
                    .set_index("id_dataset")[data_groupby]
                )
            if isinstance(datasets, DatasetsGroupBy):
                grouper = (
                    datasets.datasets_index[["id_dataset", data_groupby]]
                    .drop_duplicates()
                    .set_index("id_dataset")[data_groupby]
                )
            # group profile_chi2_per_data by the grouper and sum the chi2 values of the groups
            profile_chi2_groups = (
                self.profile_chi2_per_data.T.groupby(lambda x: (x[0], grouper[x[1]]))
                .sum()
                .T
            )
            profile_chi2_groups.columns = pd.MultiIndex.from_tuples(
                profile_chi2_groups.columns, names=["parameters", data_groupby]
            )
            if isinstance(datasets, Datasets):
                data_groups_labels = dict(
                    zip(datasets.index[data_groupby], datasets.index[data_groupby].map(str))
                )
            if isinstance(datasets, DatasetsGroupBy):
                data_groups_labels = dict(
                    zip(datasets.datasets_index[data_groupby], datasets.datasets_index[data_groupby].map(str))
                )
        else:
            profile_chi2_groups = None
            data_groups_labels = None

        minimum = self.minimum_params.copy()
        minimum["chi2"] = self.minimum_chi2

        if data_groups_labels is not None and groups_labels is not None:
            data_groups_labels = data_groups_labels | groups_labels

        plot_scan_1d(
            ax=ax,
            profile_params=self.profile_params,
            profile_chi2=self.profile_chi2,
            modus="Parameter",
            parameter=parameter,
            minimum=minimum,
            profile_chi2_groups=profile_chi2_groups,
            groups_labels=data_groups_labels,
            legend=legend,
            highlight_groups=highlight_groups,
            highlight_important_groups=highlight_important_groups,
            kwargs_chi2_total=kwargs_chi2_total,
            kwargs_chi2_minimum=kwargs_chi2_minimum,
            kwargs_chi2_groups=kwargs_chi2_groups,
            **kwargs,
        )


class ParameterScan2D(ParameterScan):

    _parameters_scanned: list[tuple[str, str]] | None = None
    _parameters_scanned_indices: dict[tuple[str, str], int] | None = None

    _pattern_all = jaml.Pattern(
        {
            "Chi2Fcn": {"IndexOfInputParams": None, "ParamIndices": None},
            "Scans": {
                "targetDeltaChi2": None,
                "rangeMargin": None,
                "numberOfSteps": None,
                "Chi2AtMinimum": None,
                "ParameterScans2D": {
                    None: {
                        "upperRange1": None,
                        "lowerRange1": None,
                        "upperRange2": None,
                        "lowerRange2": None,
                        "parameterIndex1": None,
                        "parameterIndex2": None,
                        "parameterName1": None,
                        "parameterName2": None,
                        "paramValueInMinimum1": None,
                        "paramValueInMinimum2": None,
                        "ParamValues": None,
                        "Chi2Values": None,
                    }
                },
            },
        }
    )

    def __init__(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        cache_path: str | os.PathLike[str] = "./.jaml_cache",
        retain_yaml: bool = False,
    ) -> None:
        super().__init__(paths, cache_path, retain_yaml)

    @override
    def _load_ranges(self) -> None:
        """Initialize `_parameters_scanned`, `_parameters_scanned_indices`, `_range_upper` and `_range_lower`"""

        pattern = jaml.Pattern(
            {
                "Scans": {
                    "ParameterScans2D": {
                        None: {
                            "upperRange1": None,
                            "lowerRange1": None,
                            "upperRange2": None,
                            "lowerRange2": None,
                            "parameterIndex1": None,
                            "parameterIndex2": None,
                            "parameterName1": None,
                            "parameterName2": None,
                            "paramValueInMinimum1": None,
                            "paramValueInMinimum2": None,
                            "ParamValues":None,
                        }
                    }
                }
            }
        )

        yaml = cast(
            dict[str, jaml.YAMLType] | list[tuple[Path, dict[str, jaml.YAMLType]]],
            self._load_yaml(pattern),
        )

        if not isinstance(yaml, list):
            yaml = [(Path(), yaml)]

        param_indices: list[tuple[int, int]] = []
        param_names: list[tuple[str, str]] = []
        param_min: dict[str, float] = {}
        ranges: list[list[tuple[float, float]]] = []
        for y in map(lambda x: x[1], yaml):

            for scan in cast(
                dict[str, dict[str, Any]],
                jaml.nested_get(y, ["Scans", "ParameterScans2D"]),
            ).values():
                param_indices_i = cast(
                    tuple[int, int], (scan["parameterIndex1"], scan["parameterIndex2"])
                )
                param_names_i = cast(
                    tuple[str, str], (scan["parameterName1"], scan["parameterName2"])
                )

                param_min_i = cast(
                    tuple[float, float],
                    (scan["ParamValues"][int((len(scan["ParamValues"])-1)/2)][int((len(scan["ParamValues"])-1)/2)][0],
                    scan["ParamValues"][int((len(scan["ParamValues"])-1)/2)][int((len(scan["ParamValues"])-1)/2)][1]),
                )
                ranges_upper_i = cast(
                    tuple[float, float], (scan["upperRange1"], scan["upperRange2"])
                )
                ranges_lower_i = cast(
                    tuple[float, float], (scan["lowerRange1"], scan["lowerRange2"])
                )

                for i in range(2):
                    if not param_names_i[i] in param_min:
                        param_min[param_names_i[i]] = param_min_i[i]
                    elif param_min[param_names_i[i]] != param_min_i[i]:
                        raise ValueError(
                            f"Multiple minimum values for parameter {param_names_i[i]}"
                        )

                # sort the parameter column levels by the parameter indices
                param_indices_i, param_names_i, ranges_upper_i, ranges_lower_i = zip(
                    *sorted(
                        zip(
                            param_indices_i,
                            param_names_i,
                            ranges_upper_i,
                            ranges_lower_i,
                        ),
                        key=lambda x: x[0],
                    )
                )

                param_indices.append(param_indices_i)
                param_names.append(param_names_i)
                ranges.append(list(zip(ranges_lower_i, ranges_upper_i)))

        assert _all_equal(
            [
                len(param_indices),
                len(param_names),
                len(ranges),
            ]
        )

        for (i1, i2), (p1, p2) in zip(param_indices, param_names):
            # check that the parameter indices match the ones in Chi2Fcn
            for i, p in (i1, p1), (i2, p2):
                if self.parameters_all_indices[p] != i:
                    raise ValueError(
                        f"Parameter index mismatch: {p} is at index {self.parameters_all_indices[p]} in `Scans` but at index {i} in `Chi2Fcn`"
                    )

            # check for duplicates
            if param_indices.count((i1, i2)) > 1:
                raise ValueError(f"Duplicate parameters {(p1, p2)} (index {(i1, i2)})")

        # sort the columns by the parameter indices
        param_indices, param_names, ranges = zip(
            *sorted(
                zip(param_indices, param_names, ranges),
                key=lambda x: x[0],
            )
        )

        self._parameters_scanned = list(param_names)
        self._parameters_scanned_indices = {
            p: i for i, p in enumerate(self._parameters_scanned)
        }
        param_names_flat = sorted(
            set(y for x in param_names for y in x),
            key=lambda x: self.parameters_all_indices[x],
        )
        self._minimum_params = pd.DataFrame([param_min], columns=param_names_flat)
        self._parameters_range = pd.DataFrame(
            np.array([y for x in ranges for y in x]).T,
            columns=pd.MultiIndex.from_tuples(
                [(p1, p2, i) for p1, p2 in param_names for i in range(2)]
            ),
        )

    @override
    def _load_profile(self) -> None:
        """Initialize `_profile_params` and `_profile_chi2`"""

        pattern = jaml.Pattern(
            {
                "Scans": {
                    "ParameterScans2D": {
                        None: {
                            "parameterName1": None,
                            "parameterName2": None,
                            "ParamValues": None,
                            "Chi2Values": None,
                        }
                    }
                }
            }
        )

        yaml = cast(
            dict[str, jaml.YAMLType] | list[tuple[Path, dict[str, jaml.YAMLType]]],
            self._load_yaml(pattern),
        )

        if not isinstance(yaml, list):
            yaml = [(Path(), yaml)]

        params: list[tuple[str, str]] = []
        profile_params: list[list[float]] = []
        profile_chi2: list[list[float]] = []
        for y in map(lambda x: x[1], yaml):

            for scan in cast(
                dict[str, dict[str, Any]],
                jaml.nested_get(y, ["Scans", "ParameterScans2D"]),
            ).values():

                params_i = cast(
                    tuple[str, str], (scan["parameterName1"], scan["parameterName2"])
                )
                profile_params_i = list(
                    map(
                        list,
                        zip(
                            *(
                                y
                                for x in cast(
                                    list[list[list[int]]], scan["ParamValues"]
                                )
                                for y in x
                            )
                        ),
                    )
                )
                profile_chi2_i = [
                    y for x in cast(list[list[float]], scan["Chi2Values"]) for y in x
                ]

                # sort the parameter column levels by the parameter indices
                params_i, profile_params_i = zip(
                    *sorted(
                        zip(params_i, profile_params_i),
                        key=lambda x: self.parameters_all_indices[x[0]],
                    )
                )

                params.append(params_i)
                profile_params.append(profile_params_i)
                profile_chi2.append(profile_chi2_i)

        assert _all_equal([len(params), len(profile_params), len(profile_chi2)])

        # sort the columns by the parameter indices
        params, profile_params, profile_chi2 = zip(
            *sorted(
                zip(params, profile_params, profile_chi2),
                key=lambda x: self.parameters_scanned_indices[x[0]],
            )
        )

        self._profile_params = pd.DataFrame(
            np.array([y for x in profile_params for y in x]).T,
            columns=pd.MultiIndex.from_tuples(
                [(p1, p2, i) for p1, p2 in params for i in range(2)],
                names=("parameter1", "parameter2", "parameter_index"),
            ),
        )
        self._profile_chi2 = pd.DataFrame(
            np.array(profile_chi2).T,
            columns=pd.MultiIndex.from_tuples(
                params, names=("parameter1", "parameter2")
            ),
        )

    @property
    def parameters_scanned(self) -> list[tuple[str, str]]:
        if self._parameters_scanned is None or self._yaml_changed():
            self._load_ranges()

        assert self._parameters_scanned is not None

        return self._parameters_scanned

    @property
    def parameters_scanned_indices(self) -> dict[tuple[str, str], int]:
        if self._parameters_scanned_indices is None or self._yaml_changed():
            self._load_ranges()

        assert self._parameters_scanned_indices is not None

        return self._parameters_scanned_indices

    def plot(
        self,
        ax: plt.Axes | Sequence[plt.Axes],
        parameters: tuple[str, str] | list[tuple[str, str]] | None = None,
        draw_contour: bool = True,
        plot_minimum: bool = True,
        levels: list | None = None,
        cbar_scale: Literal["linear", "log"] = "linear",
        vmax: float = 100,
        colormap: str = "Spectral_r",
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot the 2D eigenvector scan.

        Parameters
        ----------
        ax : plt.Axes or Sequence[plt.Axes]
            Axes to plot on.
        parameters : tuple[str, str] or list[tuple[str, str]], optional
            Parameters to plot, by default None (all).
        draw_contour : bool, optional
            Whether to draw contour lines, by default True.
            If True and levels!=None: contours at levels. 
            If True and levels==None: Contours at tolerance/2, tolerance and 2*tolerance. 
            If tolerance==None contour at vmax/2
        plot_minimum : bool, optional
            Whether to plot the minimum point, by default True.
        levels : list, optional
            Contour levels, by default None (uses tolerance or vmax).
        cbar_scale : Literal["linear", "log"], optional
            Colorbar scale, by default "linear".
        vmax : float, optional
            Maximum value for colorbar, by default 1000.
        colormap : str, optional
            Colormap to use, by default "Spectral_r".
        tolerance : float, optional
            Tolerance for chi2 contour, if no levels are chosen, by default None (uses target_delta_chi2).
            if cbar_scale=="linear": tolerance is the center of the color spectrum, e.g. the brightest color.
            if cbar_scale=="log", choice of tolerance has no influence on colors
        """
        minimum = self.minimum_params.copy()
        minimum["chi2"] = self.minimum_chi2

        plot_scan_2d(
            ax=ax,
            profile_params=self.profile_params,
            profile_chi2=self.profile_chi2,
            parameters=parameters,
            modus="Parameter",
            minimum=minimum,
            tolerance=tolerance,
            draw_contour=draw_contour,
            levels=levels,
            plot_minimum=plot_minimum,
            cbar_scale=cbar_scale,
            vmax=vmax,
            colormap=colormap,
            **kwargs,
        )


class EVScan(ABC, jaml.YAMLWrapper):
    """Abstract base class for eigenvector scans. Subclasses must override `_load_ranges` and `_load_profile`, and should implement the properties `evs_scanned`"""

    _ev_all: list[str] | None = None
    _ev_all_indices: dict[str, int] | None = None

    _target_delta_chi2: float | None = None
    _margin: float | None = None
    _num_steps: int | None = None

    _ev_range: pd.DataFrame | None = None

    _profile_evs: pd.DataFrame | None = None
    _profile_chi2: pd.DataFrame | None = None
    _profile_chi2_per_data: pd.DataFrame | None = None

    _minimum_params: pd.DataFrame | None = None
    _minimum_chi2: float | None = None

    def __init__(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        cache_path: str | os.PathLike[str] = "./.jaml_cache",
        retain_yaml: bool = False,
    ) -> None:
        super().__init__(paths, cache_path, retain_yaml)

    def _load_evs(self) -> None:
        """Initialize `_parameters_names` and `_parameters_indices` by reading `Chi2Fcn: IndexOfInputParams` and `Chi2Fcn: ParamIndices`"""

        pattern = jaml.Pattern(
            {"Chi2Fcn": {"IndexOfInputParams": None, "ParamIndices": None}}
        )
        yaml = cast(
            dict[str, jaml.YAMLType] | list[tuple[Path, dict[str, jaml.YAMLType]]],
            self._load_yaml(pattern),
        )

        if not isinstance(yaml, list):
            yaml = [(Path(), yaml)]

        # read parameter names and indices
        index_of_input_params: list[list[int]] = []
        param_indices: list[dict[str, int]] = []
        for y in map(lambda x: x[1], yaml):
            index_of_input_params.append(
                cast(list[int], jaml.nested_get(y, ["Chi2Fcn", "IndexOfInputParams"]))
            )
            param_indices.append(
                cast(dict[str, int], jaml.nested_get(y, ["Chi2Fcn", "ParamIndices"]))
            )

        # check that all files have the same parameter names and indices
        if not _all_equal(index_of_input_params) or not _all_equal(param_indices):
            raise ValueError(
                "Incompatible scans: All files must have the same parameter names and indices"
            )

        # order the parameter name to parameter index mapping by the index (dicts preserve ordering in Python 3.7+)
        self._ev_all_indices = {
            i_p: i_p
            for i in index_of_input_params[0]
            for p, i_p in param_indices[0].items()
            if i_p == i
        }
        self._evs_all = list(self._ev_all_indices.keys())

    def _load_scan_info(self) -> None:
        """Initialize `_target_delta_chi2`, `_margin`, `_num_steps`, and `_minimum_chi2`"""

        pattern = jaml.Pattern(
            {
                "Scans": {
                    "targetDeltaChi2": None,
                    "rangeMargin": None,
                    "numberOfSteps": None,
                    "Chi2AtMinimum": None,
                }
            }
        )

        yaml = cast(
            dict[str, jaml.YAMLType] | list[tuple[Path, dict[str, jaml.YAMLType]]],
            self._load_yaml(pattern),
        )

        if not isinstance(yaml, list):
            yaml = [(Path(), yaml)]

        target_delta_chi2: list[float] = []
        margin: list[float] = []
        num_steps: list[int] = []
        min_chi2: list[float] = []
        for y in map(lambda x: x[1], yaml):
            target_delta_chi2.append(
                cast(float, jaml.nested_get(y, ["Scans", "targetDeltaChi2"]))
            )
            margin.append(cast(float, jaml.nested_get(y, ["Scans", "rangeMargin"])))
            num_steps.append(cast(int, jaml.nested_get(y, ["Scans", "numberOfSteps"])))
            min_chi2.append(cast(float, jaml.nested_get(y, ["Scans", "Chi2AtMinimum"])))

        # check that all files have the same values
        # TODO: support multiple values per parameter?
        if (
            not _all_equal(target_delta_chi2)
            or not _all_equal(margin)
            or not _all_equal(num_steps)
            or not _all_equal(min_chi2)
        ):
            raise ValueError(
                "Incompatible scans: All files must have the same targetDeltaChi2, rangeMargin, numberOfSteps and chi2AtMinimum"
            )

        self._target_delta_chi2 = target_delta_chi2[0]
        self._margin = margin[0]
        self._num_steps = num_steps[0]
        self._minimum_chi2 = min_chi2[0]

#    @abstractmethod
#    def _load_ranges(self) -> None:
#        pass

    @abstractmethod
    def _load_profile(self) -> None:
        pass

    @property
    def parameters_all(self) -> list[str]:
        if self._parameters_all is None or self._yaml_changed():
            self._load_params()

        assert self._parameters_all is not None

        return self._parameters_all

    @property
    def ev_all_indices(self) -> dict[str, int]:
        if self._ev_all_indices is None or self._yaml_changed():
            self._load_evs()

        assert self._ev_all_indices is not None

        return self._ev_all_indices

    @property
    def target_delta_chi2(self) -> float:
        if self._target_delta_chi2 is None or self._yaml_changed():
            self._load_scan_info()

        assert self._target_delta_chi2 is not None

        return self._target_delta_chi2

    @property
    def margin(self) -> float:
        if self._margin is None or self._yaml_changed():
            self._load_scan_info()

        assert self._margin is not None

        return self._margin

    @property
    def num_steps(self) -> int:
        if self._num_steps is None or self._yaml_changed():
            self._load_scan_info()

        assert self._num_steps is not None

        return self._num_steps

    @property
    def minimum_chi2(self) -> float:
        if self._minimum_chi2 is None or self._yaml_changed():
            self._load_scan_info()

        assert self._minimum_chi2 is not None

        return self._minimum_chi2

    @property
    def parameters_range(self) -> pd.DataFrame:
        if self._parameters_range is None or self._yaml_changed():
            self._load_ranges()

        assert self._parameters_range is not None

        return self._parameters_range

    @property
    def profile_evs(self) -> pd.DataFrame:
        if self._profile_evs is None or self._yaml_changed():
            self._load_profile()

        assert self._profile_evs is not None

        return self._profile_evs

    @property
    def profile_chi2(self) -> pd.DataFrame:
        if self._profile_chi2 is None or self._yaml_changed():
            self._load_profile()

        assert self._profile_chi2 is not None

        return self._profile_chi2

    @property
    def profile_chi2_per_data(self) -> pd.DataFrame:
        if self._profile_chi2_per_data is None or self._yaml_changed():
            self._load_profile()

        assert self._profile_chi2_per_data is not None

        return self._profile_chi2_per_data


class EVScan1D(EVScan):

    _evs_scanned: list[int] | None = None

    _datasets: set[int] | None = None

    def __init__(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        cache_path: str | os.PathLike[str] = "./.jaml_cache",
        retain_yaml: bool = False,
    ) -> None:
        super().__init__(paths, cache_path, retain_yaml)

    @property
    def evs_scanned(self) -> list[str]:
        if self._evs_scanned is None or self._yaml_changed():
            self._load_profile()

        assert self._evs_scanned is not None

        return self._evs_scanned

    @override
    def _load_profile(self) -> None:
        """Initialize `_profile_evs`, `_profile_chi2` and `_profile_chi2_per_data`"""

        pattern = jaml.Pattern(
            {
                "Scans": {
                    "EVScans": [
                        {
                            "profile": None,
                            "evIndex": None,
                            "snapshots": [
                                {
                                    "par": None,
                                    "outputPerPointBreakdown": None,
                                    "outputPerDataBreakdown": None,
                                    "chi2Value": None,
                                    "perDataBreakdown": None,
                                }
                            ],
                        }
                    ],
                }
            }
        )

        yaml = cast(
            dict[str, jaml.YAMLType] | list[tuple[Path, dict[str, jaml.YAMLType]]],
            self._load_yaml(pattern),
        )

        if not isinstance(yaml, list):
            yaml = [(Path(), yaml)]

        evs: list[str] = []
        datasets: list[list[int]] = []
        profile_evs: list[list[float]] = []
        profile_chi2: list[list[float]] = []
        profile_chi2_per_data: list[list[list[float]]] = []
        for y in map(lambda x: x[1], yaml):

            for scan in cast(
                list[dict[str, Any]],
                jaml.nested_get(y, ["Scans", "EVScans"]),
            ):

                profile_evs.append(
                    [x[0] for x in cast(list[list[float]], scan["profile"])]
                )
                profile_chi2.append(
                    [x[1] for x in cast(list[list[float]], scan["profile"])]
                )
                evs.append(scan["evIndex"])
                snapshots_per_data: list[list[float]] = []
                for snapshot in cast(
                    list[dict[str, Any]],
                    scan["snapshots"],
                ):
                    if cast(bool, snapshot["outputPerDataBreakdown"]):
                        datasets.append(
                            list(
                                cast(
                                    dict[int, float], snapshot["perDataBreakdown"]
                                ).keys()
                            )
                        )
                        snapshots_per_data.append(
                            list(
                                cast(
                                    dict[int, float], snapshot["perDataBreakdown"]
                                ).values()
                            )
                        )

                # transpose snapshots_per_data before appending to profile_chi2_per_data
                profile_chi2_per_data.append(list(map(list, zip(*snapshots_per_data))))

        self._datasets = set(y for x in datasets for y in x)
        evs, profile_evs, profile_chi2, profile_chi2_per_data = zip(
            *sorted(
                zip(evs, profile_evs, profile_chi2, profile_chi2_per_data),
                key=lambda x: self.ev_all_indices[x[0]],
            )
        )
        self._evs_scanned = evs
        self._profile_evs = pd.DataFrame(
            np.array(profile_evs).T,
            columns=pd.Index(evs, name="eigenvector"),
        )
        self._profile_chi2 = pd.DataFrame(
            np.array(profile_chi2).T, columns=pd.Index(evs, name="eigenvector")
        )
        self._profile_chi2_per_data = pd.DataFrame(
            np.array([y for x in profile_chi2_per_data for y in x]).T,
            columns=pd.MultiIndex.from_tuples(
                [(p, d) for p, ds in zip(evs, datasets) for d in ds],
                names=("eigenvector", "id_dataset"),
            ),
        )

    @property
    def datasets(self) -> set[int]:
        if self._datasets is None or self._yaml_changed():
            self._load_profile()

        assert (
            self._datasets is not None
        )  # TODO: may be None if no per-data breakdown was loaded

        return self._datasets

    @property
    def profile_chi2_per_data(self) -> pd.DataFrame:
        if self._profile_chi2_per_data is None or self._yaml_changed():
            self._load_profile()

        assert (
            self._profile_chi2_per_data is not None
        )  # TODO: may be None if no per-data breakdown was loaded

        return self._profile_chi2_per_data

    def plot(
        self,
        ax: plt.Axes | Sequence[plt.Axes],
        eigenvector: int | Sequence[int] | None = None,
        datasets: Datasets | DatasetsGroupBy | None = None,
        data_groupby: str | None = None,
        groups_labels: dict[str, str] | None = None,
        highlight_groups: str | list[str] | None = None,
        highlight_important_groups: int | None = None,
        plot_minimum: bool = True, 
        legend: bool | Literal["last","first"] = "last",
        kwargs_chi2_total: dict[str, Any] | None = None,
        kwargs_chi2_minimum: dict[str, Any] | None = None,
        kwargs_chi2_groups: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot the eigenvector scan(s)

        Parameters
        ----------
        ax : plt.Axes or sequence of plt.Axes
            Axes to plot on. If a sequence is given, its length must match the number of eigenvectors to plot.
        eigenvector :
            Eigenvector index or list of indices to plot. If `None`, all scanned eigenvectors are plotted.
        datasets : Datasets, optional
            Datasets object containing the dataset metadata. Required if `data_groupby` is not `None`.
        data_groupby : str, optional
            Column name in `datasets.index` to group datasets by. If `None`, no grouping is applied.
        groups_labels : dict, optional
            Mapping of group names (as in `data_groupby`) to labels to use in the legend. If `None`, the group names are used as labels.
        highlight_groups : str or list of str, optional
            Group name or list of group names (as in `data_groupby`) to highlight in the plot. Highlighted groups are drawn with thicker lines and higher opacity. 
            If `None`, no groups are highlighted.
        highlight_important_groups : int, optional
            Number of most important groups (by maximum chi2 contribution) to highlight in the plot. If `None`, no groups are highlighted.
        plot_minimum : bool, default True
            Whether to mark the minimum chi2 point on the plot.
        legend : bool or 'last' or 'first', default 'last'
            Whether to show a legend. If 'last', the legend is shown only on the last subplot. If 'first', the legend is shown only on the first subplot.
        kwargs_chi2_total : dict, optional
            Additional keyword arguments passed to `ax.plot` when plotting the total chi2 curve.
        kwargs_chi2_minimum : dict, optional
            Additional keyword arguments passed to when plotting the minimum chi2
        kwargs_chi2_groups : dict or list of dict, optional
            Additional keyword arguments passed to `ax.plot` when plotting the grouped chi2 curves. If a list is given, its length must match the number of groups. If a dict is given, the same arguments are used for all groups.
        """

        if data_groupby is not None:
            if self.profile_chi2_per_data is None:
                raise ValueError(
                    "Grouping data not available since no per-data breakdown was loaded"
                )

            if datasets is None:
                raise ValueError("Grouping data requires passing `datasets`")

            # construct dataset grouper as pd.Series with index "id_dataset" and values `data_groupby`
            # TODO: Better treatment of datasets with the same ID but different values for `data_groupby`
            if isinstance(datasets, Datasets):
                grouper = (
                    datasets.index[["id_dataset", data_groupby]]
                    .drop_duplicates()
                    .set_index("id_dataset")[data_groupby]
                )
            if isinstance(datasets, DatasetsGroupBy):
                grouper = (
                    datasets.datasets_index[["id_dataset", data_groupby]]
                    .drop_duplicates()
                    .set_index("id_dataset")[data_groupby]
                )

            # group profile_chi2_per_data by the grouper and sum the chi2 values of the groups
            profile_chi2_groups = (
                self.profile_chi2_per_data.T.groupby(lambda x: (x[0], grouper[x[1]]))
                .sum()
                .T
            )
            profile_chi2_groups.columns = pd.MultiIndex.from_tuples(
                profile_chi2_groups.columns, names=["eigenvector", data_groupby]
            )

            if isinstance(datasets, Datasets):
                data_groups_labels = dict(
                    zip(datasets.index[data_groupby], datasets.index[data_groupby].map(str))
                )
            if isinstance(datasets, DatasetsGroupBy):
                data_groups_labels = dict(
                    zip(datasets.datasets_index[data_groupby], datasets.datasets_index[data_groupby].map(str))
                )

        else:
            profile_chi2_groups = None
            data_groups_labels = None

        minimum_chi2 = self.minimum_chi2

        if data_groups_labels is not None and groups_labels is not None:
            data_groups_labels = data_groups_labels | groups_labels

        plot_scan_1d(
            ax=ax,
            profile_evs=self.profile_evs,
            profile_chi2=self.profile_chi2,
            eigenvector=eigenvector,
            modus="EV",
            minimum=minimum_chi2,
            plot_minimum= plot_minimum, 
            profile_chi2_groups=profile_chi2_groups,
            groups_labels=data_groups_labels,
            legend=legend,
            highlight_groups=highlight_groups,
            highlight_important_groups=highlight_important_groups,
            kwargs_chi2_total=kwargs_chi2_total,
            kwargs_chi2_minimum=kwargs_chi2_minimum,
            kwargs_chi2_groups=kwargs_chi2_groups,
            **kwargs,
        )


#
#
class EVScan2D(EVScan):

    _evs_scanned: list[tuple[str, str]] | None = None

    _pattern_all = jaml.Pattern(
        {
            "Chi2Fcn": {"IndexOfInputParams": None, "ParamIndices": None},
            "Scans": {
                "targetDeltaChi2": None,
                "rangeMargin": None,
                "numberOfSteps": None,
                "Chi2AtMinimum": None,
                "EVScans2D": [
                    {
                        "negDirBound1": None,
                        "negDirBound2": None,
                        "posDirBound1": None,
                        "posDirBound2": None,
                        "negDirStep1": None,
                        "negDirStep2": None,
                        "posDirStep1": None,
                        "posDirStep2": None,
                        "zmax": None,
                        "evIndex1": None,
                        "evIndex2": None,
                        "paramValues": None,
                        "Chi2Values": None,
                    },
                ],
            },
        }
    )

    def __init__(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        cache_path: str | os.PathLike[str] = "./.jaml_cache",
        retain_yaml: bool = False,
    ) -> None:
        super().__init__(paths, cache_path, retain_yaml)

    @property
    def evs_scanned(self) -> list[str]:
        if self._evs_scanned is None or self._yaml_changed():
            self._load_profile()

        assert self._evs_scanned is not None

        return self._evs_scanned

    @override
    def _load_profile(self) -> None:
        """Initialize `_profile_evs` and `_profile_chi2`"""

        pattern = jaml.Pattern(
            {
                "Scans": {
                    "EVScans2D": [
                        {
                            "profile": None,
                            "negDirBound1": None,
                            "negDirBound2": None,
                            "posDirBound1": None,
                            "posDirBound2": None,
                            "negDirStep1": None,
                            "negDirStep2": None,
                            "posDirStep1": None,
                            "posDirStep2": None,
                            "zmax": None,
                            "evIndex1": None,
                            "evIndex2": None,
                            "paramValues": None,
                            "Chi2Values": None,
                        }
                    ],
                }
            }
        )

        yaml = cast(
            dict[str, jaml.YAMLType] | list[tuple[Path, dict[str, jaml.YAMLType]]],
            self._load_yaml(pattern),
        )

        if not isinstance(yaml, list):
            yaml = [(Path(), yaml)]

        ev_ids: list[tuple[int, int]] = []
        # ranges: list[list[float]] = []
        # params: list[tuple[str, str]] = []
        # profile_params: list[list[float]] = []
        profile_evs: list[list[tuple[tuple[float, float], float]]] = []
        profile_chi2: list[list[float]] = []
        for y in map(lambda x: x[1], yaml):

            for scan in cast(
                dict[str, dict[str, Any]],
                jaml.nested_get(y, ["Scans", "EVScans2D"]),
            ):

                ev_i = cast(tuple[int, int], (scan["evIndex1"], scan["evIndex2"]))
                profile_evs_i = list(
                    map(
                        list,
                        zip(
                            *(
                                y
                                for x in cast(list[list[list[int]]], scan["profile"])
                                for y in x
                            )
                        ),
                    )
                )
                profile_chi2_i = [
                    y for x in cast(list[list[float]], scan["Chi2Values"]) for y in x
                ]

                ev_ids.append(ev_i)

                profile_evs_i_sorted = [
                    sorted(profile_evs_i[0], key=lambda x: (x[0], x[1]))
                ]
                profile_evs.append(profile_evs_i_sorted)

                profile_chi2_i_sorted = [
                    chi2
                    for _, chi2 in sorted(
                        zip(profile_evs_i[0], profile_chi2_i),
                        key=lambda pair: (pair[0][0], pair[0][1]),
                    )
                ]

                # profile_chi2_i_sorted=[x for _, x in sorted(zip([sub[1] for sub in profile_evs_i[0]], profile_chi2_i))]
                profile_chi2.append(profile_chi2_i_sorted)
        assert _all_equal([len(ev_ids), len(profile_chi2)])
        self._evs_scanned = ev_ids
        # sort the columns by the parameter indices
        # ev_ids,  profile_chi2 = zip(
        #     *sorted(
        #         zip(ev_ids, profile_chi2),
        #         key=lambda x: self._evs_scanned[x],
        #     )
        # )

        self._profile_evs = pd.DataFrame(
            [
                list(np.array([y[i] for x in profile_evs for y in x]).flatten())
                for i in range(len(profile_evs[0][0]))
            ],
            columns=pd.MultiIndex.from_tuples(
                [(p1, p2, i) for p1, p2 in ev_ids for i in range(1, 3)],
                names=("EV1", "EV2", "parameter_index"),
            ),
        )

        self._profile_chi2 = pd.DataFrame(
            np.array(profile_chi2).T,
            columns=pd.MultiIndex.from_tuples(ev_ids),
        )

    def plot(
        self,
        ax: plt.Axes | Sequence[plt.Axes],
        eigenvectors: tuple[int, int] | list[tuple[int, int]] | None = None,
        draw_contour: bool = True,
        plot_minimum: bool = True,
        levels: list | None = None,
        cbar_scale: Literal["linear", "log"] = "linear",
        vmax: float = 1000,
        colormap: str = "Spectral_r",
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot the 2D eigenvector scan.

        Parameters
        ----------
        ax : plt.Axes or Sequence[plt.Axes]
            Axes to plot on.
        eigenvectors : tuple[int, int] or list[tuple[int, int]], optional
            Eigenvectors to plot, by default None (all).
        draw_contour : bool, optional
            Whether to draw contour lines, by default True.
            If True and levels!=None: contours at levels. 
            If True and levels==None: Contours at tolerance/2, tolerance and 2*tolerance. 
            If tolerance==None contour at vmax/2
        plot_minimum : bool, optional
            Whether to plot the minimum point, by default True.
        levels : list, optional
            Contour levels, by default None (uses tolerance or vmax).
        cbar_scale : Literal["linear", "log"], optional
            Colorbar scale, by default "linear".
        vmax : float, optional
            Maximum value for colorbar, by default 1000.
        colormap : str, optional
            Colormap to use, by default "Spectral_r".
        tolerance : float, optional
            Tolerance for chi2 contour, if no levels are chosen, by default None (uses target_delta_chi2).
            if cbar_scale=="linear": tolerance is the center of the color spectrum, e.g. the brightest color.
            if cbar_scale=="log", choice of tolerance has no influence on colors
        """
        self._load_profile()
        minimum_chi2 = self.minimum_chi2

        plot_scan_2d(
            ax=ax,
            profile_evs=self.profile_evs,
            profile_chi2=self.profile_chi2,
            eigenvectors=eigenvectors,
            modus="EV",
            minimum=minimum_chi2,
            tolerance=tolerance,
            draw_contour=draw_contour,
            levels=levels,
            plot_minimum=plot_minimum,
            cbar_scale=cbar_scale,
            vmax=vmax,
            colormap=colormap,
            **kwargs,
        )
