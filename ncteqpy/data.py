from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import yaml.parser

import ncteqpy.jaml as jaml
import ncteqpy.labels as labels


# TODO: make possible to load only subdirs or list of paths
class Datasets(jaml.YAMLWrapper):

    _index: pd.DataFrame | None = None

    _dataIDs: dict[str, dict[int, list[str]]] | None = (
        None  # TODO: get rid of the dataIDs.yaml file
    )

    @property
    def index(self) -> pd.DataFrame:
        if self._index is None or self._yaml_changed():
            self._load_dataset_index()

        return self._index  # type: ignore[return-value] # self._index is set in self.index_datasets())

    @property
    def dataIDs(self) -> dict[str, dict[int, list[str]]]:
        if self._dataIDs is None:
            with open(Path(__file__).parent / "dataIDs.yaml") as f:
                self._dataIDs = yaml.safe_load(f)

        return self._dataIDs

    def filtered_index(
        self,
        id_dataset: int | Sequence[int] | None = None,
        type_experiment: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:  # TODO: more options for filtering
        res = self.index

        if id_dataset is not None:
            if isinstance(id_dataset, int):
                id_dataset = [id_dataset]
            res = res.query("id_dataset == @id_dataset")

        if type_experiment is not None:
            if isinstance(type_experiment, str):
                type_experiment = [type_experiment]
            res = res.query("type_experiment in @type_experiment")

        return res

    def by_id(self, id_dataset: int) -> Dataset:
        datasets = self.index.query("id_dataset == @id_dataset")
        match len(datasets):
            case 0:
                raise ValueError(f"Dataset with ID {id} not found")
            case 1:
                path = datasets.iloc[0]["path"]
            case n if 2 <= n:
                if (
                    id_dataset in self.dataIDs["dataIDs"]
                ):  # TODO: get rid of dataIDs.yaml
                    path = self.path / Path(self.dataIDs["dataIDs"][id_dataset][0])
                else:
                    dataset_variants = "\n".join(map(str, datasets["path"]))
                    raise ValueError(
                        f"Multiple dataset files with ID {id_dataset} found. Please load one of them by its path:\n{dataset_variants}"
                    )

        return Dataset(path)

    # def by_path(self, path: str | os.PathLike) -> Dataset:
    #     pass

    # def by_name(self, name: str) -> Dataset:
    #     pass

    def _load_dataset_index(
        self, verbose: bool | int = False, settings: None = None
    ) -> None:
        pickle_name = "dataset_index"

        self._index = cast(pd.DataFrame, self._unpickle(pickle_name))

        if self._index is None:
            index_list = []

            index_pattern = jaml.Pattern(
                {
                    "Description": {
                        "TypeExp": None,
                        "FinalState": None,
                        "IDDataSet": None,
                        "AZValues1": None,
                        "AZValues2": None,
                    },
                    "GridSpec": {
                        "NumberOfCorrSysErr": None,
                        "TypeTheory": None,
                        "TypeUncertainties": None,
                        "TypeKinVar": None,
                        # "Grid": None,
                    },
                }
            )
            index_yaml = self._load_yaml(index_pattern)

            assert isinstance(index_yaml, list)

            for p, data in cast(list[tuple[Path, jaml.YAMLType]], index_yaml):
                if not jaml.nested_in(data, ["Description", "IDDataSet"]):
                    continue

                # we have to deal with A and Z first because we can't index None
                az1 = (
                    az
                    if isinstance(
                        az := jaml.nested_get(data, ["Description", "AZValues1"]), list
                    )
                    else (None, None)
                )
                az2 = (
                    az
                    if isinstance(
                        az := jaml.nested_get(data, ["Description", "AZValues2"]), list
                    )
                    else (None, None)
                )
                # if a field is not in the data file we set it to None in the dataframe
                index_list.append(
                    {
                        "id_dataset": jaml.nested_get(
                            data, ["Description", "IDDataSet"]
                        ),
                        "path": p,
                        "type_experiment": jaml.nested_get(
                            data, ["Description", "TypeExp"]
                        ),
                        "A1": az1[0],
                        "Z1": az1[1],
                        "A2": az2[0],
                        "Z2": az2[1],
                        "final_state": jaml.nested_get(
                            data, ["Description", "FinalState"]
                        ),
                        "correlated_systematic_uncertainties": jaml.nested_get(
                            data, ["GridSpec", "NumberOfCorrSysErr"]
                        ),
                        "kinematic_variables": jaml.nested_get(
                            data, ["GridSpec", "TypeKinVar"]
                        ),
                        "type_theory": jaml.nested_get(
                            data, ["GridSpec", "TypeTheory"]
                        ),
                        "types_uncertainties": jaml.nested_get(
                            data, ["GridSpec", "TypeUncertainties"]
                        ),
                        # "grid": np.array(jaml.nested_get(data, ["GridSpec", "Grid"])),
                    }
                )
            self._index = pd.DataFrame.from_records(data=index_list)
            int_cols = [
                "id_dataset",
                "correlated_systematic_uncertainties",
            ]  # no A and Z here since they are sometimes non-integer
            self._index[int_cols] = self._index[int_cols].astype("Int64", copy=False)
            self._index.sort_values(by="path", inplace=True)
            self._pickle(self._index, pickle_name)


class Dataset:

    _points: pd.DataFrame
    _yaml: jaml.YAMLType

    _id: int | None = None
    _type_experiment: str | None = None
    _kinematic_variables: list[str] | None = None
    # TODO add rest of fields

    _plotting_short_info: str | None = None
    _plotting_reference_id: str | None = None
    _plotting_process: str | None = None
    _plotting_labels_kinematic_variables: dict[str, str] | None = None
    _plotting_units_kinematic_variables: dict[str, str] | None = None
    _plotting_label_theory: str | None = None
    _plotting_unit_theory: str | None = None

    def __init__(self, path: str | os.PathLike) -> None:
        path = Path(path)

        with open(path, "r") as file:
            self._yaml = yaml.safe_load(file)

        cols = (
            labels.kinvars_yaml_to_py
            | labels.theory_yaml_to_py
            | labels.uncertainties_yaml_to_py
        )

        self._points = pd.DataFrame(
            data=cast(
                list[list[float]], jaml.nested_get(self.yaml, ["GridSpec", "Grid"])
            ),
            columns=list(
                map(
                    lambda k: cols.get(k, k),
                    cast(
                        list[str],
                        jaml.nested_get(self.yaml, ["GridSpec", "TypeColumns"]),
                    ),
                )
            ),
        )

        if "pT_min" in self.points.columns and "pT_max" in self.points.columns:
            i = self.points.columns.get_loc("pT_max")
            if not isinstance(i, int):
                raise ValueError("multiple columns called 'pT_max'")
            self._points.insert(
                i + 1, "pT_mid", (self.points["pT_min"] + self.points["pT_max"]) / 2
            )
            self._points.insert(
                i + 2, "pT_err", (self.points["pT_max"] - self.points["pT_min"]) / 2
            )

        if "y_min" in self.points.columns and "y_max" in self.points.columns:
            i = self.points.columns.get_loc("y_max")
            if not isinstance(i, int):
                raise ValueError("multiple columns called 'y_max'")
            self._points.insert(
                i + 1, "y_mid", (self.points["y_min"] + self.points["y_max"]) / 2
            )
            self._points.insert(
                i + 2, "y_err", (self.points["y_max"] - self.points["y_min"]) / 2
            )

        if not "unc_tot" in self._points.columns:
            errs = list(set(self.points.columns) & set(["unc_stat", "unc_sys"]))
            self._points["unc_tot"] = np.sqrt(np.sum(self.points[errs] ** 2, axis=1))

    @property
    def points(self) -> pd.DataFrame:
        return self._points

    @property
    def yaml(self) -> jaml.YAMLType:
        return self._yaml

    @property
    def id(self) -> int:
        if self._id is None:
            self._id = cast(
                int, jaml.nested_get(self.yaml, ["Description", "IDDataSet"])
            )

        return self._id

    @property
    def type_experiment(self) -> str:
        if self._type_experiment is None:
            self._type_experiment = cast(
                str, jaml.nested_get(self.yaml, ["Description", "TypeExp"])
            )

        return self._type_experiment

    @property
    def kinematic_variables(self) -> list[str]:
        if self._kinematic_variables is None:
            self._kinematic_variables = list(
                map(
                    lambda l: labels.kinvars_yaml_to_py.get(l, l),
                    cast(
                        list[str],
                        jaml.nested_get(self.yaml, ["GridSpec", "TypeKinVar"]),
                    ),
                )
            )

        return self._kinematic_variables

    @property
    def plotting_short_info(
        self,
    ) -> (
        str | None
    ):  # TODO: plotting info not always included in data file: better handling of None
        if self._plotting_short_info is None and jaml.nested_in(
            self.yaml, ["Plotting", "ShortInfo"]
        ):
            self._plotting_short_info = cast(
                str, jaml.nested_get(self.yaml, ["Plotting", "ShortInfo"])
            )

        return self._plotting_short_info

    @property
    def plotting_reference_id(self) -> str | None:
        if self._plotting_reference_id is None and jaml.nested_in(
            self.yaml, ["Plotting", "ReferenceID"]
        ):
            self._plotting_reference_id = cast(
                str, jaml.nested_get(self.yaml, ["Plotting", "ReferenceID"])
            )

        return self._plotting_reference_id

    @property
    def plotting_process(self) -> str | None:
        if self._plotting_process is None and jaml.nested_in(
            self.yaml, ["Plotting", "Process"]
        ):
            self._plotting_process = cast(
                str, jaml.nested_get(self.yaml, ["Plotting", "Process"])
            )

        return self._plotting_process

    @property
    def plotting_labels_kinematic_variables(self) -> dict[str, str] | None:
        if self._plotting_labels_kinematic_variables is None and jaml.nested_in(
            self.yaml, ["Plotting", "LabelsKinVar"]
        ):
            self._plotting_labels_kinematic_variables = dict(
                zip(
                    self.kinematic_variables,
                    cast(
                        list[str],
                        jaml.nested_get(self.yaml, ["Plotting", "LabelsKinVar"]),
                    ),
                )
            )

        return self._plotting_labels_kinematic_variables

    @property
    def plotting_units_kinematic_variables(self) -> dict[str, str] | None:
        if self._plotting_units_kinematic_variables is None and jaml.nested_in(
            self.yaml, ["Plotting", "UnitsKinVar"]
        ):
            self._plotting_units_kinematic_variables = dict(
                zip(
                    self.kinematic_variables,
                    cast(
                        list[str],
                        jaml.nested_get(self.yaml, ["Plotting", "UnitsKinVar"]),
                    ),
                )
            )

        return self._plotting_units_kinematic_variables

    @property
    def plotting_label_theory(self) -> str | None:
        if self._plotting_label_theory is None and jaml.nested_in(
            self.yaml, ["Plotting", "LabelTheory"]
        ):
            self._plotting_label_theory = cast(
                str, jaml.nested_get(self.yaml, ["Plotting", "LabelTheory"])
            )

        return self._plotting_label_theory

    @property
    def plotting_unit_theory(self) -> str | None:
        if self._plotting_unit_theory is None and jaml.nested_in(
            self.yaml, ["Plotting", "LabelTheory"]
        ):
            self._plotting_unit_theory = cast(
                str, jaml.nested_get(self.yaml, ["Plotting", "UnitTheory"])
            )

        return self._plotting_unit_theory

    def plot(
        self,
        ax: plt.Axes | None = None,
        kinematic_variable: str | list[str] | None = None,  # TODO: implement
        bin_label: Literal["legend", "annotate"] | None = None,
        ratio_to: Dataset | None = None,  # TODO: add option for data/theory?
        stat_uncertainty: bool = False,
        xlabel: bool = False,
        ylabel: bool = False,
        legend: bool = False,
        **kwargs: object,
    ) -> None:
        if ax is None:
            ax = plt.gca()

        match self.type_experiment:
            case "OPENHEAVY" | "QUARKONIUM":
                for (y_min, y_max), data in self.points.groupby(["y_min", "y_max"]):
                    kwargs = {**kwargs}

                    if ratio_to is not None:
                        data_denom = ratio_to.points.query(
                            "abs(y_min - @y_min) < 1e-6 and abs(y_max - @y_max) < 1e-6"
                        )
                        if not np.all(
                            np.abs(
                                data_denom["pT_min"].to_numpy()
                                - data["pT_min"].to_numpy()
                            )
                            < 1e-6
                        ) or not np.all(
                            np.abs(
                                data_denom["pT_max"].to_numpy()
                                - data["pT_max"].to_numpy()
                            )
                            < 1e-6
                        ):
                            raise ValueError(
                                f"Ratio denominator has incompatible pT bins for y ∈ [{y_min}, {y_max}]"
                            )

                    if not "label" in kwargs:
                        kwargs["label"] = (
                            f"${y_min:.1f} < y < {y_max:.1f}$"
                            if bin_label == "legend"
                            else None
                        )

                    ax.plot(
                        np.append(data["pT_min"].iloc[0], data["pT_max"]),
                        np.append(data["sigma"], data["sigma"].iloc[-1])
                        / (
                            np.append(data_denom["sigma"], data_denom["sigma"].iloc[-1])
                            if ratio_to is not None
                            else 1
                        ),
                        drawstyle="steps-post",
                        **kwargs,
                    )
                    if stat_uncertainty:
                        ax.errorbar(
                            data["pT_mid"],
                            data["sigma"]
                            / (data_denom["sigma"] if ratio_to is not None else 1),
                            yerr=data["sigma_err"]
                            / (data_denom["sigma"] if ratio_to is not None else 1),
                            ls="",
                            capsize=1.5,
                            color=ax.get_lines()[-1].get_color(),
                        )

                    if bin_label == "annotate":
                        raise NotImplementedError(
                            "bin_label='annotate' not implemented yet"
                        )

                if xlabel:
                    ax.set_xlabel(r"$p_{\mathrm{T}}$")
                if ylabel:
                    if ratio_to is not None:
                        ax.set_ylabel(r"$\dfrac{\text{GM-VFNS}}{\text{Data}}$")
                    else:
                        ax.set_ylabel(
                            r"$\dfrac{\mathrm{d}^2\sigma}{\mathrm{d}p_{\mathrm{T}}\,\mathrm{d}y}$ [nb/GeV]"
                        )

                if legend and bin_label == "legend":
                    ax.legend()

            case _:
                raise NotImplementedError(
                    f"Plotting not implemented for experiment type {self.type_experiment}"
                )