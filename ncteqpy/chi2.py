from __future__ import annotations

import os
from typing import Any, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import ncteqpy.data as data
import ncteqpy.jaml as jaml
import ncteqpy.labels as labels
import ncteqpy.plot as plot
import ncteqpy.util as util


# TODO: implement pickling for the other members
# TODO: implement some functionality to record the parsing time for each variable so they can be grouped together systematically
class Chi2(jaml.YAMLWrapper):

    _parameters_names: list[str] | None = None
    _parameters_indices: dict[str, int] | None = None
    _parameters_last_values: npt.NDArray[np.float64] | None = None
    _parameters_input_values: npt.NDArray[np.float64] | None = None
    _parameters_values_at_min: npt.NDArray[np.float64] | None = None
    _last_value: float | None = None
    _last_value_with_penalty: float | None = None
    _last_value_per_data: dict[int, float] | None = None
    _snapshots_parameters: pd.DataFrame | None = None
    _snapshots_values: npt.NDArray[np.float64] | None = None
    _snapshots_breakdown_points: pd.DataFrame | None = None
    _snapshots_breakdown_datasets: pd.DataFrame | None = None

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
            self._snapshots_breakdown_points.set_index(
                ["id_snapshot", "id_point"], inplace=True
            )
            self._pickle(self._snapshots_breakdown_points, pickle_name)

    @property
    def parameters_names(self) -> list[str]:
        if self._parameters_names is None or self._yaml_changed():
            pattern = jaml.Pattern(
                {"Chi2Fcn": {"IndexOfInputParams": None, "ParamIndices": None}}
            )
            yaml = self._load_yaml(pattern)

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

            self._parameters_last_values = np.array(
                cast(list[float], jaml.nested_get(yaml, ["Chi2Fcn", "LastParams"]))
            )

        return self._parameters_last_values

    @property
    def parameters_input_values(self) -> npt.NDArray[np.float64]:
        if self._parameters_input_values is None or self._yaml_changed():
            pattern = jaml.Pattern({"Chi2Fcn": {"InputParametrizationParams": None}})
            yaml = self._load_yaml(pattern)

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

            self._last_value = cast(
                float, jaml.nested_get(yaml, ["Chi2Fcn", "LastValue"])
            )

        return self._last_value

    @property
    def last_value_with_penalty(self) -> float:
        if self._last_value_with_penalty is None or self._yaml_changed():
            pattern = jaml.Pattern({"Chi2Fcn": {"LastValueWithPenalty": None}})
            yaml = self._load_yaml(pattern)

            self._last_value_with_penalty = cast(
                float, jaml.nested_get(yaml, ["Chi2Fcn", "LastValueWithPenalty"])
            )

        return self._last_value_with_penalty

    @property
    def last_value_per_data(self) -> dict[int, float]:
        if self._last_value_per_data is None or self._yaml_changed():
            pattern = jaml.Pattern({"Chi2Fcn": {"LastValuePerData": None}})
            yaml = self._load_yaml(pattern)

            self._last_value_per_data = cast(
                dict[int, float], jaml.nested_get(yaml, ["Chi2Fcn", "LastValuePerData"])
            )

        return self._last_value_per_data

    @property
    def snapshots_parameters(self) -> pd.DataFrame:
        if self._snapshots_parameters is None or self._yaml_changed():
            self._load_snapshots_without_breakdown_points()

        return self._snapshots_parameters  # type: ignore[return-value] # value cannot be None since it is set in the if clause

    @property
    def snapshots_values(self) -> pd.DataFrame:
        if self._snapshots_values is None or self._yaml_changed():
            self._load_snapshots_without_breakdown_points()

        return self._snapshots_values  # type: ignore[return-value] # value cannot be None since it is set in the if clause

    @property
    def snapshots_breakdown_datasets(self) -> pd.DataFrame:
        if self._snapshots_breakdown_datasets is None or self._yaml_changed():
            self._load_snapshots_without_breakdown_points()

        return self._snapshots_breakdown_datasets  # type: ignore[return-value] # value cannot be None since it is set in the if clause

    @property
    def snapshots_breakdown_points(self) -> pd.DataFrame:
        if self._snapshots_breakdown_points is None or self._yaml_changed():
            self._load_snapshots_breakdown_points()

        return self._snapshots_breakdown_points  # type: ignore[return-value] # value cannot be None since it is set in the if clause

    # TODO: more sophisticated filtering
    # TODO: cuts
    # TODO: PDF uncertainties
    def plot_data_vs_theory(
        self,
        data: data.Datasets,
        id_dataset: int | Sequence[int] | None = None,
        type_experiment: str | None = None,
        id_snapshot: int = 0,
        ax: plt.Axes | Sequence[plt.Axes] | None = None,
        subplot_groupby: str | None = None,
        curve_groupby: str | None = None,
        pdf_path: str | os.PathLike | None = None,
        kwargs_subplots: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes] | list[tuple[plt.Figure, plt.Axes]]:
        if id_dataset is None:
            if type_experiment is None:
                raise ValueError(
                    "Please provide either `id_dataset` or `type_experiment`"
                )
            else:
                points = self.snapshots_breakdown_points.loc[id_snapshot].query(
                    "type_experiment == @type_experiment"
                )
                id_dataset = cast(
                    Sequence[int], points["id_dataset"].unique()
                )  # actually npt.NDArray[np.int_]
        else:
            if isinstance(id_dataset, int):
                id_dataset = [id_dataset]
            points = self.snapshots_breakdown_points.loc[id_snapshot].query(
                "id_dataset in @id_dataset"
            )
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

        # if ax is None and pdf_path is None:
        #     ax = [plt.gca()]
        # elif ax is None and pdf_path is not None:
        #     pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

        # if save_path is not None:
        #     save_path = pathlib.Path(save_path)
        #     pdf = mpl.backends.backend_pdf.PdfPages(save_path)

        # gb = self.snapshots_breakdown_points.loc[id_snapshot].sort_values(["type_experiment", "id_dataset"]).groupby(["type_experiment", "id_dataset"])

        if ax is None:
            kwargs_subplots = {"layout": "tight"} | kwargs_subplots

            if len(id_dataset) == 1:
                fig_ax = [
                    cast(tuple[plt.Figure, plt.Axes], plt.subplots(**kwargs_subplots))
                ]
            else:
                if subplot_groupby is None:
                    fig_ax = [
                        cast(
                            tuple[plt.Figure, plt.Axes], plt.subplots(**kwargs_subplots)
                        )
                        for _ in id_dataset
                    ]
                else:
                    raise NotImplementedError("subplot_groupby not yet implemented")
        else:
            if isinstance(ax, Sequence):
                fig_ax = [(ax_i.figure, ax_i) for ax_i in ax]
            else:
                fig_ax = [(ax.figure, ax)]

        if subplot_groupby is not None:
            raise NotImplementedError("subplot_groupby not yet implemented")
        else:
            gb = points.groupby("id_dataset")

        found_dataset = False
        for (fig_i, ax_i), (id_dataset_i, data_i) in zip(
            fig_ax, points.groupby("id_dataset", sort=True)
        ):
            found_dataset = True

            try:
                dataset = data.by_id(id_dataset_i)
            except:
                dataset = None

            # plot only the same bins: match on the kinematic variables
            if dataset is not None:
                match_indices = []
                for row_data in dataset.points.itertuples():
                    for row_theory in data_i.itertuples():
                        row_match = True
                        for kv in dataset.kinematic_variables:
                            if (
                                abs(getattr(row_data, kv) - getattr(row_theory, kv))
                                > 1e-6
                            ):
                                row_match = False
                                break
                        if row_match:
                            match_indices.append(getattr(row_data, "Index"))
                data_matched = dataset.points.iloc[match_indices]

                kwargs_i: dict[str, Any] = {}
                if (
                    dataset.plotting_labels_kinematic_variables is not None
                    and dataset.plotting_units_kinematic_variables is not None
                ):
                    kwargs_i["xlabel"] = {
                        kin_var: util.format_unit(
                            label, dataset.plotting_units_kinematic_variables[kin_var]
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
                kwargs_i["title"] = (
                    f"{(dataset.plotting_short_info + ',  ') if dataset.plotting_short_info is not None else ''}{(dataset.plotting_process + ',  ') if dataset.plotting_process is not None else ''}Dataset ID: {dataset.id}"
                )

                kwargs_i = kwargs_i | kwargs
            else:
                data_matched = data_i

            plot.plot(
                type_experiment,
                data=data_matched,
                theory=data_i,
                ax=ax_i,
                **kwargs_i,
            )

        if not found_dataset:
            raise ValueError(
                "No datasets found for the given values of id_dataset and type_experiment"
            )

        return fig_ax

    def plot_data_vs_theory_grouped(self) -> None:
        pass
