from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from typing_extensions import Any, Iterator, Literal, Sequence, cast, overload

import ncteqpy.data as data
import ncteqpy.jaml as jaml
import ncteqpy.labels as labels
import ncteqpy.util as util
from ncteqpy.data_groupby import DatasetsGroupBy
from ncteqpy.plot import data_vs_theory
from ncteqpy.plot.chi2_histograms import (
    plot_chi2_data_breakdown,
    plot_chi2_histogram,
    plot_S_E_histogram,
)
from ncteqpy.plot.grid import AxesGrid
from ncteqpy.plot.util import AdditionalLegend


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
    _normalizations: pd.DataFrame | None = None
    _snapshots_parameters: pd.DataFrame | None = None
    _snapshots_values: npt.NDArray[np.float64] | None = None
    _snapshots_breakdown_points: pd.DataFrame | None = None
    _snapshots_breakdown_datasets: pd.DataFrame | None = None
    _num_points: pd.Series[int] | None = None

    _S_E: pd.Series[float] | None = None
    _points: pd.DataFrame | None = None

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
            pd.DataFrame, self._unpickle("chi2_snapshots_parameters")
        )
        self._snapshots_values = cast(
            npt.NDArray[np.float64], self._unpickle("chi2_snapshots_values")
        )
        self._snapshots_breakdown_datasets = cast(
            pd.DataFrame, self._unpickle("chi2_snapshots_breakdown_datasets")
        )

        if (
            self._snapshots_parameters is None
            or self._snapshots_values is None
            or self._snapshots_breakdown_datasets is None
        ):
            pattern = jaml.Pattern(
                {
                    "Chi2Fcn": {
                        "Snapshots": [
                            {"par": None, "chi2Value": None, "perDataBreakdown": None}
                        ]
                    }
                }
            )
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            snapshots = cast(
                list[dict[str, object]], jaml.nested_get(yaml, ["Chi2Fcn", "Snapshots"])
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

            self._pickle(self._snapshots_parameters, "chi2_snapshots_parameters")
            self._pickle(self._snapshots_values, "chi2_snapshots_values")
            self._pickle(
                self._snapshots_breakdown_datasets,
                "chi2_snapshots_breakdown_datasets",
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
                "id_bin",
            ]  # no A and Z here since they are sometimes non-integer
            self._snapshots_breakdown_points[int_cols] = (
                self._snapshots_breakdown_points[int_cols].astype("Int64", copy=False)
            )

            self._snapshots_breakdown_points.set_index(
                ["id_snapshot", "id_point"], inplace=True
            )

            # multiply the theory for which normalization is fitted with its corresponding factor
            mask = self._snapshots_breakdown_points["id_dataset"].isin(
                self.normalizations.index
            )
            # `theory_with_normalization_only` is NaN for data sets that are not normalization-corrected
            self._snapshots_breakdown_points.loc[
                mask, "theory_with_normalization_only"
            ] = (
                self._snapshots_breakdown_points.loc[mask, "theory"]
                * self.normalizations.loc[
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

    @property
    def normalizations(self) -> pd.DataFrame:
        if self._normalizations is None or self._yaml_changed():
            pattern = jaml.Pattern({"Chi2Fcn": {"NormInfo": None}})
            yaml = cast(
                dict[str, dict[str, list[dict[str, object]]]], self._load_yaml(pattern)
            )

            norm_records = []
            for norm_info in yaml["Chi2Fcn"]["NormInfo"]:
                for id_dataset in cast(list[int], norm_info["IDs"]):
                    norm_records.append(
                        {
                            "id_dataset": id_dataset,
                            "factor": norm_info["Value"],
                            "penalty": norm_info["Penalty"],
                            "scheme": norm_info["Scheme"],
                        }
                    )

            self._normalizations = pd.DataFrame.from_records(
                norm_records, index="id_dataset"
            )

        return self._normalizations

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
    def points(self) -> pd.DataFrame:
        if self._points is None or self._yaml_changed():
            points_snapshots = cast(
                pd.DataFrame, self.snapshots_breakdown_points.loc[1].copy()
            ).sort_values(
                "id_dataset"
            )  # FIXME figure out which snapshots to read

            datasets_index = self.datasets.index.set_index("id_dataset")
            datasets_points = self.datasets.points.set_index("id_dataset")

            points_list = []

            for id_dataset, points1_i in points_snapshots.groupby(
                "id_dataset", sort=True
            ):

                kinematic_variables = cast(
                    list[str], datasets_index.loc[id_dataset]["kinematic_variables"]
                )

                # drop nan columns when matching since we cannot match on them anyway
                points1_i_notna = points1_i.dropna(axis=1)

                points2_i = datasets_points.loc[[id_dataset]].reset_index("id_dataset")

                assert isinstance(kinematic_variables, list)

                match_cols = (
                    ["id_dataset"]
                    + points1_i_notna.columns[
                        points1_i_notna.columns.isin(kinematic_variables)
                    ].to_list()
                    + ["data"]
                )

                cols = (
                    match_cols
                    + list(labels.uncertainties_yaml_to_py.values())
                    + ["unc_tot"]
                )

                points_list.append(
                    pd.merge(points1_i, points2_i[cols], how="left", on=match_cols)
                )
            self._points = pd.concat(points_list)
            self._points.index = self.snapshots_breakdown_points.loc[1].index.copy()

            # assert self._points["unc_tot"].notna().sum() == datasets_points["unc_tot"]

            # compute PDF uncertainties for each point
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
                    self._points[f"{col_in}_{col_out}"] = (
                        self.snapshots_breakdown_points.loc[1:]
                        .groupby("id_point")[col_in]
                        .apply(func)
                    )

        return self._points

    @property
    def S_E(self) -> pd.Series[float]:
        if self._S_E is None or self._yaml_changed():
            self._S_E = cast(
                pd.Series,
                np.sqrt(2 * self.last_value_per_data)
                - np.sqrt(2 * self.num_points - 1),
            )

        return self._S_E

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
        kwargs_bar: dict[str, Any] = {},
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

        chi2 = (
            self.last_value_per_data.loc[id_dataset]
            if id_dataset is not None
            else self.last_value_per_data
        )

        plot_chi2_data_breakdown(
            ax=ax,
            chi2=chi2,
            per_point=per_point,
            num_points=self.num_points,
            chi2_line_1=chi2_line_1,
            chi2_drop_0=chi2_drop_0,
            bar_orientation=bar_orientation,
            bar_groupby=bar_groupby,
            bar_props_groupby=bar_props_groupby,
            bar_order_groupby=bar_order_groupby,
            kwargs_bar=kwargs_bar,
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
            chi2=self.points,
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
            S_E=self.S_E,
            bin_width=bin_width,
            subplot_groupby=subplot_groupby,
            kwargs_subplots=kwargs_subplots,
            kwargs_histogram=kwargs_histogram,
            kwargs_gaussian=kwargs_gaussian,
            kwargs_fit=kwargs_fit,
            kwargs_gaussian_fit=kwargs_gaussian_fit,
        )

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
        legend: bool = ...,
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = ...,
        subplot_groupby: str | None = ...,
        subplot_label: Literal["legend"] | None = ...,
        subplot_label_format: str | None = ...,
        chi2_annotation: bool = ...,
        chi2_legend: bool = ...,
        curve_groupby: str | list[str] | Literal["fallback"] | None = ...,
        apply_normalization: bool = ...,
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
        kwargs_legend_curves: dict[str, Any] = ...,
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_annotate_chi2: dict[str, Any] = ...,
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = ...,
        *,
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
        chi2_legend : bool, optional
            If a legend with the total χ² is shown, by default True.
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
        legend: bool = ...,
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = ...,
        subplot_groupby: str | None = ...,
        subplot_label: Literal["legend"] | None = ...,
        subplot_label_format: str | None = ...,
        chi2_annotation: bool = ...,
        chi2_legend: bool = ...,
        curve_groupby: str | list[str] | Literal["fallback"] | None = ...,
        apply_normalization: bool = ...,
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
        kwargs_legend_curves: dict[str, Any] = ...,
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = ...,
        kwargs_annotate_chi2: dict[str, Any] = ...,
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = ...,
        *,
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
        subplot_groupby: str | None, optional
            Variable to group the subplot axes by, by default None (no grouping).
        subplot_label : Literal["legend"] | None, optional  # FIXME
            Where to label the subplot, by default None.
        subplot_label_format : str | None, optional
            Format of the subplot label, by default None.
        chi2_annotation : bool, optional
            If the χ²/point value is annotated, by default True.
        chi2_legend : bool, optional
            If a legend with the total χ² is shown, by default True.
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

        Returns
        -------
        Iterator[AxesGrid]
            AxesGrid(s) with the data vs. theory plot(s).
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
        legend: bool = True,
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = "ticks",
        subplot_groupby: str | None = None,
        subplot_label: Literal["legend"] | None = None,
        subplot_label_format: str | None = None,
        chi2_annotation: bool = True,
        chi2_legend: bool = True,
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
        kwargs_legend_curves: dict[str, Any] = {},
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_annotate_chi2: dict[str, Any] = {},
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        *,
        iterate: bool = False,
        **kwargs: Any,
    ) -> AxesGrid | list[AxesGrid] | Iterator[AxesGrid]:
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
            subplot_groupby=subplot_groupby,
            subplot_label=subplot_label,
            subplot_label_format=subplot_label_format,
            chi2_annotation=chi2_annotation,
            chi2_legend=chi2_legend,
            curve_groupby=curve_groupby,
            apply_normalization=apply_normalization,
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
            kwargs_legend_curves=kwargs_legend_curves,
            kwargs_ticks_curves=kwargs_ticks_curves,
            kwargs_annotate_chi2=kwargs_annotate_chi2,
            kwargs_annotate_curves=kwargs_annotate_curves,
            **kwargs,
        )

        if iterate:
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
        legend: bool = True,
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = "ticks",
        subplot_groupby: str | None = None,
        subplot_label: Literal["legend"] | None = None,
        subplot_label_format: str | None = None,
        chi2_annotation: bool = True,
        chi2_legend: bool = True,
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
        kwargs_legend_curves: dict[str, Any] = {},
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
                points = self.points.query("type_experiment == @type_experiment")
                id_dataset = cast(
                    Sequence[int], points["id_dataset"].unique()
                )  # actually npt.NDArray[np.int_]
        else:
            if isinstance(id_dataset, int):
                id_dataset = [id_dataset]
            points = self.points.query("id_dataset in @id_dataset")
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

        # group the relevant dataset IDs so we get one dataset per figure
        fig_gb = points.query("id_dataset in @id_dataset").groupby(
            "id_dataset", sort=True
        )

        found_dataset = False
        for i, (id_dataset_i, data_i) in enumerate(fig_gb):
            try:
                dataset = self.datasets.by_id(
                    id_dataset_i  # pyright: ignore[reportArgumentType]
                )
                found_dataset = True
            except:
                dataset = None

            subplot_gb = (
                data_i.groupby(subplot_groupby) if subplot_groupby is not None else None
            )

            kwargs_subplots_default = {
                "sharex": True,
                "sharey": True,
                "layout": "constrained",
            }
            kwargs_subplots_updated = util.update_kwargs(
                kwargs_subplots_default, kwargs_subplots, i
            )

            n_real = 1 if subplot_gb is None else len(subplot_gb)

            ax_grid = AxesGrid(n_real=n_real, **kwargs_subplots_updated)

            ax_iter = zip(
                ax_grid.ax_real.flatten(),
                (subplot_gb if subplot_gb is not None else [(np.nan, data_i)]),
            )
            for ax_i, (ax_gb_val_i, data_ij) in ax_iter:

                if dataset is not None:

                    kwargs_i: dict[str, Any] = {}
                    if (
                        dataset.plotting_labels_kinematic_variables is not None
                        and dataset.plotting_units_kinematic_variables is not None
                    ):
                        kwargs_i["xlabel"] = {
                            kin_var: util.format_unit(
                                f"${label}$",
                                dataset.plotting_units_kinematic_variables[kin_var],
                            )
                            for kin_var, label in dataset.plotting_labels_kinematic_variables.items()
                        }
                    if (
                        dataset.plotting_label_theory is not None
                        and dataset.plotting_unit_theory is not None
                    ):
                        kwargs_i["ylabel"] = util.format_unit(
                            dataset.plotting_label_theory, dataset.plotting_unit_theory
                        )

                    kwargs_i = kwargs_i | kwargs
                else:
                    kwargs_i = {}

                data_vs_theory.plot(
                    type_experiment=type_experiment,
                    ax=ax_i,
                    points=data_ij,
                    x_variable=x_variable,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    xscale=xscale,
                    yscale=yscale,
                    title=title,
                    legend=legend,
                    curve_label=curve_label,
                    subplot_label=subplot_label,
                    subplot_label_format=subplot_label_format,
                    chi2_annotation=chi2_annotation,
                    chi2_legend=chi2_legend,
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
                    kwargs_ticks_curves=kwargs_ticks_curves,
                    kwargs_annotate_chi2=kwargs_annotate_chi2,
                    kwargs_annotate_curves=kwargs_annotate_curves,
                    **kwargs_i,
                )

                if subplot_groupby is not None and subplot_label == "legend":
                    label = "$"
                    if (
                        dataset is not None
                        and dataset.plotting_labels_kinematic_variables is not None
                    ):
                        label += dataset.plotting_labels_kinematic_variables[
                            subplot_groupby
                        ]
                    else:
                        label += labels.kinvars_py_to_tex[subplot_groupby]

                    label += f" = {ax_gb_val_i}"

                    if (
                        dataset is not None
                        and dataset.plotting_units_kinematic_variables is not None
                    ):
                        label += dataset.plotting_units_kinematic_variables[
                            subplot_groupby
                        ]

                    label += "$"

                    subplot_legend = AdditionalLegend(
                        -1,
                        ax_i,
                        handles=[Patch()],
                        labels=[label],
                        labelspacing=0,
                        handlelength=0,
                        handleheight=0,
                        handletextpad=0,
                        fontsize="small",
                    )
                    ax_i.add_artist(subplot_legend)

            ax_grid.prune_labels()

            if dataset is not None:
                ax_grid.fig.suptitle(
                    f"{(dataset.plotting_short_info + ',  ') if dataset.plotting_short_info is not None else ''}{(dataset.plotting_process + ',  ') if dataset.plotting_process is not None else ''}Dataset ID: {dataset.id}",
                    fontsize="medium",
                )

            yield ax_grid

        if not found_dataset:
            raise ValueError(
                "No datasets found for the given values of id_dataset and type_experiment"
            )

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
        legend: bool = True,
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = "ticks",
        subplot_groupby: str | None = None,
        subplot_label: Literal["legend"] | None = "legend",
        subplot_label_format: str | None = None,
        chi2_annotation: bool = True,
        chi2_legend: bool = True,
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
        kwargs_legend_curves: dict[str, Any] = {},
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_annotate_chi2: dict[str, Any] = {},
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        **kwargs,
    ) -> AxesGrid:
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
        subplot_groupby: str | None, optional
            Variable to group the subplot axes by, by default None (no grouping).
        subplot_label : Literal["legend"] | None, optional
            Where to label the subplot, by default "legend".
        subplot_label_format : str | None, optional
            Format of the subplot label, by default None.
        chi2_annotation : bool, optional
            If the χ²/point value is annotated, by default True.
        chi2_legend : bool, optional
            If a legend with the total χ² is shown, by default True.
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
                points = self.points.query("type_experiment == @type_experiment")
                id_dataset = cast(
                    Sequence[int], points["id_dataset"].unique()
                )  # actually npt.NDArray[np.int_]
        else:
            if isinstance(id_dataset, int):
                id_dataset = [id_dataset]
            points = self.points.query("id_dataset in @id_dataset")
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

        subplot_gb = (
            points.groupby(subplot_groupby) if subplot_groupby is not None else None
        )

        kwargs_suplots_default = {
            "sharex": True,
            "sharey": True,
        }
        kwargs_subplots_updated = util.update_kwargs(
            kwargs_suplots_default, kwargs_subplots
        )

        ax_grid = AxesGrid(
            len(subplot_gb) if subplot_gb is not None else 1, **kwargs_subplots_updated
        )

        ax_iter = zip(
            ax_grid.ax_real.flatten(),
            (subplot_gb if subplot_gb is not None else [(np.nan, points)]),
        )

        for ax_i, (_, points_i) in ax_iter:

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

            data_vs_theory.plot(
                type_experiment=type_experiment,
                points=points_i,
                ax=ax_i,
                x_variable=x_variable,
                xlabel=xlabel,
                ylabel=ylabel,
                xscale=xscale,
                yscale=yscale,
                title=title,
                legend=legend,
                curve_label=curve_label,
                subplot_label=subplot_label if subplot_groupby is not None else None,
                subplot_label_format=subplot_label_format_i,
                chi2_annotation=chi2_annotation,
                chi2_legend=chi2_legend,
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
                kwargs_ticks_curves=kwargs_ticks_curves,
                kwargs_annotate_chi2=kwargs_annotate_chi2,
                kwargs_annotate_curves=kwargs_annotate_curves,
                **kwargs,
            )

        ax_grid.prune_labels()

        return ax_grid
