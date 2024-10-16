from __future__ import annotations

import os
from typing import Any, Literal, Sequence, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

import ncteqpy.data as data
import ncteqpy.jaml as jaml
import ncteqpy.labels as labels
import ncteqpy.util as util
from ncteqpy.plot import data_vs_theory
from ncteqpy.plot import util as p_util
from ncteqpy.plot.grid import AxesGrid


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
        ax: Axes | Sequence[Axes] | None = None,
        subplot_groupby: str | None = None,
        subplot_label: Literal["legend"] | None = "legend",
        curve_groupby: str | None = None,
        pdf_path: str | os.PathLike | None = None,
        kwargs_subplots: dict[str, Any] = {},
        **kwargs: Any,
    ) -> tuple[Figure, Axes] | list[tuple[Figure, Axes | npt.NDArray[Axes | None]]]:
        if pdf_path is not None:
            raise NotImplementedError("Saving as pdf not implemented yet")

        if ax is not None:
            raise NotImplementedError("Passing your own Axes not implemented yet")

        # bring id_dataset in Sequence[int] form (either directly from id_dataset or indirectly from collecting all IDs belonging to type_experiment)
        # the points variable holds all snapshot points belonging to the relevant dataset IDs
        # ValueErrors are thrown if id_dataset and type_experiment are not consistent
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

        # group the relevant dataset IDs so we get one dataset per figure
        fig_gb = points.query("id_dataset in @id_dataset").groupby(
            "id_dataset", sort=True
        )

        if ax is None:
            kwargs_subplots = {"layout": "tight"} | kwargs_subplots

            if subplot_groupby is None:
                ax_grid = [AxesGrid(1, **kwargs_subplots) for _ in id_dataset]
            else:
                ax_grid = []
                for _, data_i in fig_gb:
                    # group for different axes in one figure
                    data_i_gb = data_i.groupby(subplot_groupby)
                    ax_grid.append(AxesGrid(n_real=len(data_i_gb), **kwargs_subplots))

        else:
            if isinstance(ax, Sequence):
                fig_ax = [(ax_i.figure, ax_i) for ax_i in ax]
            else:
                fig_ax = [(ax.figure, ax)]

        found_dataset = False
        for grid_i, (id_dataset_i, data_i) in zip(ax_grid, fig_gb):
            try:
                dataset = data.by_id(id_dataset_i)
                found_dataset = True
            except:
                dataset = None

            ax_iter = zip(
                grid_i.ax_real.flatten(),
                (
                    data_i.groupby(subplot_groupby)
                    if subplot_groupby is not None
                    else [(np.nan, data_i)]
                ),
            )
            for ax_i, (ax_gb_val_i, data_ij) in ax_iter:

                # plot only the same bins: match on the kinematic variables
                if dataset is not None:

                    match_indices = []
                    for row_data in dataset.points.itertuples():
                        for row_theory in data_ij.itertuples():
                            row_match = True
                            for kv in dataset.kinematic_variables:
                                if (
                                    abs(getattr(row_data, kv) - getattr(row_theory, kv))
                                    > 1e-10
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
                    data_matched = data_ij
                    kwargs_i = {}

                assert len(data_ij) == len(data_matched)

                data_vs_theory.plot(
                    type_experiment,
                    data=data_matched,
                    theory=data_ij,
                    ax=ax_i,
                    curve_groupby=curve_groupby,
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

                    subplot_legend = p_util.AdditionalLegend(
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

            grid_i.prune_labels()

            if dataset is not None:
                grid_i.fig.suptitle(
                    f"{(dataset.plotting_short_info + ',  ') if dataset.plotting_short_info is not None else ''}{(dataset.plotting_process + ',  ') if dataset.plotting_process is not None else ''}Dataset ID: {dataset.id}",
                    fontsize="medium",
                )

            grid_i.tight_layout()

        if not found_dataset:
            raise ValueError(
                "No datasets found for the given values of id_dataset and type_experiment"
            )

        return [(g.fig, g.ax) for g in ax_grid]

    def plot_data_vs_theory_grouped(self) -> None:
        pass
