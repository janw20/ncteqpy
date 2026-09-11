from __future__ import annotations

import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import sympy as sp
import yaml
from typing_extensions import Any, Hashable, Literal, Sequence, cast

import ncteqpy.jaml as jaml
import ncteqpy.labels as labels
from ncteqpy._typing import SequenceNotStr
from ncteqpy.cuts import Cuts, cut_accepts
from ncteqpy.data_groupby import DatasetsGroupBy
from ncteqpy.kinematic_variables import (
    Q2_disdimu,
    Q2_hq_pT_bin,
    Q2_sih,
    W2_dis,
    W2_disdimu,
    x_hq_bin,
    x_sih,
    x_wzprod_bin,
)
from ncteqpy.plot.kinematic_coverage import plot_kinematic_coverage
from ncteqpy.settings import Settings
from ncteqpy.plot.data_vs_theory import _plot_data, _add_curve_labels, _set_labels, _get_kinematic_variable, _check_ax_offsets, _calculate_ratio
from ncteqpy.util import update_kwargs
import ncteqpy.labels as nc_labels
from ncteqpy.plot.util import AdditionalLegend




# TODO: make possible to load only subdirs or list of paths
class Datasets(jaml.YAMLWrapper):

    _cuts: Cuts | None = None
    _index: pd.DataFrame | None = None
    _labels_x: dict | None = None
    _labels_y: str | None = None
    _points: pd.DataFrame | None = None
    _points_after_cuts: pd.DataFrame | None = None
    _num_points: pd.Series[int] | None = None
    _num_points_after_cuts: pd.Series[int] | None = None
    _cached_datasets: dict[Path, Dataset] = {}
    _cached_datasets_with_cuts: dict[Path, Dataset] = {}

    duplicate_fallback: list[Path]

    def __init__(
        self,
        path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        settings: Settings | None = None,
        duplicate_fallback: (
            str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
        ) = "NON-ISO",
        cache_path: str | os.PathLike = "./.jaml_cache",
        retain_yaml: bool = False,
    ) -> None:
        if settings is not None:
            if isinstance(path, Sequence) and not isinstance(path, str):
                raise ValueError(
                    "If you pass `settings`, please pass only one path to the directory where the data is saved"
                )
            else:
                path = Path(path)

                if not path.is_dir():
                    raise ValueError(
                        "If you pass `settings`, please pass the path to the directory where the data is saved"
                    )

                path = [path / p for p in settings.datasets]

            self._cuts = settings.cuts

        super().__init__(path, cache_path, retain_yaml)

        # bring duplicate_fallback into list[Path] form. we need the second check because str is a Sequence
        if not isinstance(duplicate_fallback, Sequence) or isinstance(
            duplicate_fallback, str
        ):
            duplicate_fallback = [duplicate_fallback]
        self.duplicate_fallback = [Path(p) for p in duplicate_fallback]

    @property
    def cuts(self) -> Cuts | None:
        return self._cuts

    def apply(self, cuts: Cuts) -> None:
        self._cuts = cuts
        if self._points is not None:
            self._points_after_cuts = self._points.drop(
                self._points.loc[~cuts.accept(self._points)].index
            )

    @property
    def index(self) -> pd.DataFrame:
        if self._index is None or self._yaml_changed():
            self._load_dataset_index()

        assert self._index is not None

        return self._index

    @property
    def labels_x(self) -> pd.DataFrame:
        if self._labels_x is None or self._yaml_changed():
            self._load_labels()

        assert self._labels_x is not None

        return self._labels_x

    @property
    def labels_y(self) -> pd.DataFrame:
        if self._labels_y is None or self._yaml_changed():
            self._load_labels()

        assert self._labels_y is not None

        return self._labels_y

    @property
    def points(self) -> pd.DataFrame:
        if self._points is None or self._yaml_changed():
            self._load_points()

        assert self._points is not None

        return self._points

    @property
    def points_after_cuts(self) -> pd.DataFrame:
        if self._points_after_cuts is None or self._yaml_changed():
            if self.cuts is None:
                raise ValueError(
                    "No cuts provided. Please pass a Settings object to the Datasets constructor or set the cuts attribute manually."
                )

            self._load_points()

        assert self._points_after_cuts is not None

        return self._points_after_cuts

    @property
    def num_points(self) -> pd.Series[int]:
        if self._num_points is None or self._yaml_changed():
            self._num_points = self.points.groupby("id_dataset")["id_dataset"].count()
            self._num_points.name = "num_points"

        assert self._num_points is not None

        return self._num_points

    @property
    def num_points_after_cuts(self) -> pd.Series[int]:
        if self._num_points_after_cuts is None or self._yaml_changed():
            self._num_points_after_cuts = self.points_after_cuts.groupby("id_dataset")[
                "id_dataset"
            ].count()
            self._num_points_after_cuts.name = "num_points_after_cuts"

            # fill missing IDs with 0
            self._num_points_after_cuts, _ = self.num_points_after_cuts.align(
                self.num_points, fill_value=0
            )

        assert self._num_points_after_cuts is not None

        return self._num_points_after_cuts

    @property
    def cached_datasets(self) -> dict[Path, Dataset]:
        return self._cached_datasets

    @property
    def cached_datasets_with_cuts(self) -> dict[Path, Dataset]:
        return self._cached_datasets_with_cuts

    def filter(
        self, include: Sequence[int] | None = None, exclude: Sequence[int] | None = None
    ) -> Datasets:
        """Return new `Datasets` with filtered IDs.

        Parameters
        ----------
        include : Sequence[int] | None, optional
            Data set IDs to include, by default None
        exclude : Sequence[int] | None, optional
            Data set IDs to exclude, by default None

        Returns
        -------
        Datasets
            Data sets with filtered IDs
        """
        new_datasets = copy.deepcopy(self)

        def filter_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
            if df is None:
                return None

            mask = df["id_dataset"].isin(include) if include is not None else True
            if exclude is not None:
                mask &= ~df["id_dataset"].isin(exclude)

            return df[slice(None) if mask is True else mask]

        # we need to load the index, otherwise we cannot resolve the data set IDs
        new_datasets._index = filter_df(new_datasets.index)

        new_datasets._points = filter_df(new_datasets._points)
        new_datasets._points_after_cuts = filter_df(new_datasets._points_after_cuts)

        new_datasets.paths = new_datasets.index["path"].to_list()

        if new_datasets._cuts is not None:
            new_datasets._cuts = new_datasets._cuts.filter(include, exclude)

        new_datasets._cached_datasets = {}
        new_datasets._cached_datasets_with_cuts = {}

        return new_datasets

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

    def by_id(
        self,
        id_dataset: int,
        duplicate_fallback: (
            str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None
        ) = None,
        apply_cuts: bool = False,
    ) -> Dataset:
        datasets = self.index.query("id_dataset == @id_dataset")
        match len(datasets):
            case 0:
                raise ValueError(f"Dataset with ID {id} not found")
            case 1:
                path = datasets.iloc[0]["path"]
            case n if n >= 2:
                # use self.duplicate_fallback as default, or bring duplicate_fallback in list[Path] form
                if duplicate_fallback is None:
                    duplicate_fallback = self.duplicate_fallback
                elif isinstance(duplicate_fallback, Sequence) and not isinstance(
                    duplicate_fallback, str
                ):  # we need the second check since str is a Sequence
                    duplicate_fallback = [Path(p) for p in duplicate_fallback]
                else:
                    duplicate_fallback = [Path(duplicate_fallback)]

                # to check if a duplicate_fallback is a subdirectory we need the absolute paths
                duplicate_fallback_abs = []
                for f in duplicate_fallback:
                    print(f)
                    duplicate_fallback_abs.extend((p / f for p in self.paths) if not f.is_absolute() else [f])  # type: ignore[union-attr] # p is of type Path

                print(duplicate_fallback_abs)

                # keep only the datasets that are in the paths in duplicate_fallback
                datasets = datasets[
                    datasets["path"].apply(
                        lambda p: any(q in p.parents for q in duplicate_fallback_abs)
                    )
                ]

                if len(datasets) == 1:
                    path = datasets.iloc[0]["path"]
                else:
                    dataset_variants = "\n".join(map(str, datasets["path"]))
                    raise ValueError(
                        f"Multiple dataset files with ID {id_dataset} found. Please provide the duplicate_fallback argument or load one of the files by its path:\n{dataset_variants}"
                    )
            case _:
                raise ValueError(
                    f"Unexpected number of datasets found: {len(datasets)}"
                )  # we need this because otherwise pyright complains that the path variable might not be initialized

        if apply_cuts:
            if self.cuts is None:
                raise ValueError(
                    "No cuts provided. Please pass a Settings object to the Datasets constructor or set the cuts attribute manually."
                )

            cut = None
            if id_dataset in self.cuts.by_dataset_id:
                cut = self.cuts.by_dataset_id[id_dataset]

            type_experiment = datasets[datasets["path"] == path][
                "type_experiment"
            ].iloc[0]
            if type_experiment in self.cuts.by_type_experiment:
                cut = (
                    self.cuts.by_type_experiment[type_experiment]
                    if cut is None
                    else cut & self.cuts.by_type_experiment[type_experiment]
                )

            dataset = Dataset(path, cut)
            self._cached_datasets_with_cuts[path] = dataset

            return dataset

        dataset = Dataset(path)
        self._cached_datasets[path] = dataset

        return dataset

    def groupby(
        self,
        by: str | list[str],
        grouper: pd.Series[Any] | None = None,
        sort: Literal["ascending", "descending"] | None = None,
        order: SequenceNotStr[Hashable] | None = None,
        labels: dict[Hashable, str] | None = None,
        label_format: str | None = None,
        props: dict[Hashable, dict[str, Any]] | None = None,
    ) -> DatasetsGroupBy:
        """Groups the data sets by one or more variables.

        Parameters
        ----------
        by : str | list[str]
            Key(s) to group the data sets by, must be column labels of `Datasets.index`, e.g., `"type_experiment"` or `["A_heavier", "Z_heavier"]`.
        grouper : pd.Series | None, optional
            `Series` mapping some or all dataset IDs to group values. This takes precedence over the values in `Datasets.index`.
        sort : Literal["ascending", "descending"] | None, optional
            If the group values should be sorted in ascending or descending order, by default no sorting.
        order : SequenceNotStr[Hashable] | None, optional
            Custom ordering of the group values, by default None. If `by` is a list, this must be a list of tuples.
        labels : dict[Hashable, str] | None, optional
            Manual relabeling of some or all group values, by default None. Must be given as a map from the group values to the new labels and takes precedence over `label_format`.
        label_format : str | None, optional
            Format string applied to all group values, by default None. Fields in the format string must be labeled by column names in `Datasets.index`, or by `A1_sym`, `A2_sym`, `A_lighter_sym`, `A_heavier_sym`, which formats the element symbol of the respective nucleus.
        props : dict[Hashable, dict[str, Any]] | None, optional
            Matplotlib properties for some or all group values, by default None. Must be given as a map from the group values to a dictionary with matplotlib properties, e.g., the latter could be `{"color": "red"}`. If `props` is None, the property cycle in `matplotlib.rcParams["axes.prop_cycle"]` is used.

        Returns
        -------
        DatasetsGroupBy
            The grouped data sets.
        """
        return DatasetsGroupBy(
            datasets_index=self.index,
            by=by,
            grouper=grouper,
            sort=sort,
            order=order,
            labels=labels,
            label_format=label_format,
            props=props,
        )

    def _load_points(self, verbose: bool | int = False, settings: None = None) -> None:
        points_list = []

        points_pattern = jaml.Pattern(
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
                    "TypeColumns": None,
                    "Grid": None,
                },
            }
        )
        points_yaml = self._load_yaml(points_pattern)

        if not isinstance(points_yaml, list):
            points_yaml = [(self.paths[0], points_yaml)]

        info: dict[str, Path | jaml.YAMLType] = {
            "id_dataset": None,
            "path": None,
            "type_experiment": None,
            "A1": None,
            "Z1": None,
            "A2": None,
            "Z2": None,
            "final_state": None,
            "correlated_systematic_uncertainties": None,
        }

        for p, data in cast(list[tuple[Path, jaml.YAMLType]], points_yaml):
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
            info["id_dataset"] = jaml.nested_get(data, ["Description", "IDDataSet"])
            info["path"] = p
            info["type_experiment"] = jaml.nested_get(data, ["Description", "TypeExp"])
            info["A1"] = az1[0]
            info["Z1"] = az1[1]
            info["A2"] = az2[0]
            info["Z2"] = az2[1]
            info["final_state"] = jaml.nested_get(data, ["Description", "FinalState"])
            info["correlated_systematic_uncertainties"] = jaml.nested_get(
                data, ["GridSpec", "NumberOfCorrSysErr"]
            )
            try:
                grid_row_labels = [
                    labels.data_yaml_to_py[c]
                    for c in cast(
                        list[str], jaml.nested_get(data, ["GridSpec", "TypeColumns"])
                    )
                ]
            except KeyError as e:
                e.add_note(f"Unknown field {e.args[0]} in the data file {p}")
                raise e

            observable = cast(str, jaml.nested_get(data, ["GridSpec", "TypeTheory"]))

            points_list.extend(
                {
                    **info,
                    **dict(zip(grid_row_labels, grid_row)),
                    # add data column that holds the actual observable
                    "data": float(
                        grid_row[
                            grid_row_labels.index(labels.data_yaml_to_py[observable])
                        ]
                    ),
                }
                for grid_row in cast(
                    list[list[float]], jaml.nested_get(data, ["GridSpec", "Grid"])
                )
            )

        self._points = pd.DataFrame.from_records(
            data=points_list,
            columns=[
                *info.keys(),
                *labels.kinvars_yaml_to_py.values(),
                *labels.theory_yaml_to_py.values(),
                "data",
                *labels.uncertainties_yaml_to_py.values(),
            ],
        )
        for col in labels.data_yaml_to_py.values():
            self._points[col] = self._points[col].astype("Float64", copy=False)

        int_cols = [
            "id_dataset",
            "correlated_systematic_uncertainties",
            "id_bin",
        ]  # no A and Z here since they are sometimes non-integer
        self._points[int_cols] = self._points[int_cols].astype("Int64", copy=False)

        # add mid and err columns for binned variables, i.e. pT_mid and pT_err
        for var in ("pT", "y"):
            i = self._points.columns.get_loc(f"{var}_max")
            assert isinstance(i, int)
            self._points.insert(
                i + 1,
                f"{var}_mid",
                (self._points[f"{var}_min"] + self._points[f"{var}_max"]) / 2,
            )
            self._points.insert(
                i + 2,
                f"{var}_err",
                (self._points[f"{var}_max"] - self._points[f"{var}_min"]) / 2,
            )

        m_proton = 0.938

        # calculate W2 if it is NaN
        mask_dis = (
            self._points["type_experiment"].isin(["DIS", "DISNEU"])
            & self._points["W2"].isna()
        )
        self._points.loc[mask_dis, "W2"] = sp.lambdify(
            tuple(W2_dis.free_symbols), W2_dis
        )(**self._points.loc[mask_dis, ["Q2", "x"]])

        # calculate Q2 and W2 for DISDIMU
        mask_disdimu = self._points["type_experiment"] == "DISDIMU"
        mask_disdimu_w2 = mask_disdimu & self._points["W2"].isna()
        mask_disdimu_q2 = mask_disdimu & self._points["Q2"].isna()

        self._points.loc[mask_disdimu_w2, "W2"] = sp.lambdify(
            tuple(W2_disdimu.free_symbols), W2_disdimu
        )(
            **self._points.loc[mask_disdimu_w2, ["x", "y", "E_had"]],
        )
        self._points.loc[mask_disdimu_q2, "Q2"] = sp.lambdify(
            tuple(Q2_disdimu.free_symbols), Q2_disdimu
        )(
            **self._points.loc[mask_disdimu_q2, ["x", "y", "E_had"]],
        )

        # calculate x and Q2 for WZPROD
        mask_wzprod = self._points["type_experiment"] == "WZPROD"
        mask_wzprod_Q2 = mask_wzprod & self._points["Q2"].isna()
        mask_wzprod_x = mask_wzprod & self._points["x"].isna()

        m2_W = 80.4**2
        m2_Z = 91.2**2
        self._points.loc[mask_wzprod_Q2, "Q2"] = self._points.loc[
            mask_wzprod_Q2, "final_state"
        ].map(
            {
                "WPLUS": m2_W,
                "WMINUS": m2_W,
                "Z": m2_Z,
            }
        )
        self._points.loc[mask_wzprod_x, "x"] = sp.lambdify(
            tuple(x_wzprod_bin.free_symbols), x_wzprod_bin
        )(**self._points.loc[mask_wzprod_x, ["Q2", "sqrt_s", "eta_min", "eta_max"]])

        # calculate x and Q2 for SIH
        mask_sih = self._points["type_experiment"] == "SIH"
        mask_sih_Q2 = mask_sih & self._points["Q2"].isna()
        mask_sih_x = mask_sih & self._points["x"].isna()

        self._points.loc[mask_sih_Q2, "Q2"] = sp.lambdify(
            tuple(Q2_sih.free_symbols), Q2_sih
        )(pT=self._points.loc[mask_sih_Q2, "pT"])
        self._points.loc[mask_sih_x, "x"] = sp.lambdify(
            tuple(x_sih.free_symbols), x_sih
        )(**self._points.loc[mask_sih_x, ["Q2", "sqrt_s", "y"]])

        # calculate x and Q2 for HQ, OPENHEAVY and QUARKONIUM
        for type_exp in "HQ", "OPENHEAVY", "QUARKONIUM":
            mask_hq = self._points["type_experiment"] == type_exp
            mask_hq_Q2 = mask_hq & self._points["Q2"].isna()
            mask_hq_x = mask_hq & self._points["x"].isna()

            mask_hq_x_diff = mask_hq_x & self._points["sigma"].notna()
            # TODO: figure out how to treat sigma_pT_integrated
            mask_hq_x_int = mask_hq_x & self._points["sigma_pT_integrated"].notna()

            self._points.loc[mask_hq_Q2, "Q2"] = sp.lambdify(
                tuple(Q2_hq_pT_bin.free_symbols), Q2_hq_pT_bin
            )(**self._points.loc[mask_hq_Q2, ["pT_min", "pT_max"]])
            self._points.loc[mask_hq_x_diff, "x"] = sp.lambdify(
                tuple(x_hq_bin.free_symbols), x_hq_bin
            )(**self._points.loc[mask_hq_x_diff, ["Q2", "sqrt_s", "y_min", "y_max"]])

        # assert self._points[["x", "Q2"]].notna().all().all()
        self._points["unc_tot"] = (
            self._points[["unc_stat", "unc_sys_uncorr"]] ** 2
        ).sum(
            axis=1, skipna=True
        ) ** 0.5  # FIXME: include unc_sys_corr

        # add A_lighter, Z_lighter, A_heavier, Z_heavier columns (lighter and heavier nucleus out of the 2, ignoring nan)
        i = cast(int, self._points.columns.get_loc("Z2"))

        idx_lighter, _ = pd.factorize(
            self._points[["A1", "A2"]].idxmin(axis=1), sort=True
        )
        self._points.insert(
            i + 1,
            "A_lighter",
            self._points[["A1", "A2"]].to_numpy()[
                np.arange(len(self._points)), idx_lighter
            ],
        )
        self._points.insert(
            i + 2,
            "Z_lighter",
            self._points[["Z1", "Z2"]].to_numpy()[
                np.arange(len(self._points)), idx_lighter
            ],
        )

        idx_heavier, _ = pd.factorize(
            self._points[["A1", "A2"]].idxmax(axis=1), sort=True
        )
        self._points.insert(
            i + 3,
            "A_heavier",
            self._points[["A1", "A2"]].to_numpy()[
                np.arange(len(self._points)), idx_heavier
            ],
        )
        self._points.insert(
            i + 4,
            "Z_heavier",
            self._points[["Z1", "Z2"]].to_numpy()[
                np.arange(len(self._points)), idx_heavier
            ],
        )

        # to group by A, these cannot be nan, because GroupBy.get_group does not find keys that include nan. so we fill with np.inf
        self._points.loc[:, ["A1", "Z1", "A2", "Z2"]] = self._points.loc[
            :, ["A1", "Z1", "A2", "Z2"]
        ].fillna(value=np.inf)

        # apply cuts
        if self.cuts is not None:
            self.apply(self.cuts)

    def _load_dataset_index(
        self, verbose: bool | int = False, settings: None = None
    ) -> None:
        index_list = []

        index_pattern = jaml.Pattern(
            {
                "Description": {
                    "IDDataSet": None,
                    "TypeExp": None,
                    "Reaction": None,
                    "Experiment": None,
                    "Collaboration": None,
                    "CiteKey": None,
                    "FinalState": None,
                    "Observable": None,
                    "SqrtS": None,
                    "AZValues1": None,
                    "AZValues2": None,
                    "Reference": None,
                },
                "GridSpec": {
                    "NumberOfCorrSysErr": None,
                    "TypeTheory": None,
                    "TypeUncertainties": None,
                    "TypeKinVar": None,
                    "Dim": None,
                    # "Grid": None,
                },
            }
        )
        index_yaml = self._load_yaml(index_pattern)

        if isinstance(index_yaml, dict):
            index_yaml = [(self.paths[0], index_yaml)]

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
                    "id_dataset": jaml.nested_get(data, ["Description", "IDDataSet"]),
                    "path": p,
                    "type_experiment": jaml.nested_get(
                        data, ["Description", "TypeExp"]
                    ),
                    "reaction": jaml.nested_get(data, ["Description", "Reaction"]),
                    "experiment": jaml.nested_get(data, ["Description", "Experiment"]),
                    "collaboration": jaml.nested_get(
                        data, ["Description", "Collaboration"]
                    ),
                    "reference": jaml.nested_get(data, ["Description", "Reference"]),
                    "cite_key": jaml.nested_get(data, ["Description", "CiteKey"]),
                    "A1": az1[0],
                    "Z1": az1[1],
                    "A2": az2[0],
                    "Z2": az2[1],
                    "final_state": jaml.nested_get(data, ["Description", "FinalState"]),
                    "observable": jaml.nested_get(data, ["Description", "Observable"]),
                    "sqrt_s": jaml.nested_get(data, ["Description", "SqrtS"]),
                    "correlated_systematic_uncertainties": jaml.nested_get(
                        data, ["GridSpec", "NumberOfCorrSysErr"]
                    ),
                    "kinematic_variables": [
                        labels.kinvars_yaml_to_py[k]
                        for k in cast(
                            list[str], jaml.nested_get(data, ["GridSpec", "TypeKinVar"])
                        )
                    ],
                    "type_theory": labels.theory_yaml_to_py[
                        cast(str, jaml.nested_get(data, ["GridSpec", "TypeTheory"]))
                    ],
                    "types_uncertainties": [
                        labels.uncertainties_yaml_to_py[k]
                        for k in cast(
                            list[str],
                            jaml.nested_get(data, ["GridSpec", "TypeUncertainties"]),
                        )
                    ],
                    "num_points": jaml.nested_get(data, ["GridSpec", "Dim", 0]),
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

        # add A_lighter, Z_lighter, A_heavier, Z_heavier columns (lighter and heavier nucleus out of the 2, ignoring nan)
        i = cast(int, self._index.columns.get_loc("Z2"))

        idx_lighter, _ = pd.factorize(
            self._index[["A1", "A2"]].idxmin(axis=1), sort=True
        )
        self._index.insert(
            i + 1,
            "A_lighter",
            self._index[["A1", "A2"]].to_numpy()[
                np.arange(len(self._index)), idx_lighter
            ],
        )
        self._index.insert(
            i + 2,
            "Z_lighter",
            self._index[["Z1", "Z2"]].to_numpy()[
                np.arange(len(self._index)), idx_lighter
            ],
        )

        idx_heavier, _ = pd.factorize(
            self._index[["A1", "A2"]].idxmax(axis=1), sort=True
        )
        self._index.insert(
            i + 3,
            "A_heavier",
            self._index[["A1", "A2"]].to_numpy()[
                np.arange(len(self._index)), idx_heavier
            ],
        )
        self._index.insert(
            i + 4,
            "Z_heavier",
            self._index[["Z1", "Z2"]].to_numpy()[
                np.arange(len(self._index)), idx_heavier
            ],
        )

        # to group by A, these cannot be nan, because GroupBy.get_group does not find keys that include nan. so we fill with np.inf
        self._index.loc[:, ["A1", "Z1", "A2", "Z2"]] = self._index.loc[
            :, ["A1", "Z1", "A2", "Z2"]
        ].fillna(value=np.inf)

    def _load_labels(self) -> None:
        labels_pattern = jaml.Pattern(
            {
                "Description": {
                    "IDDataSet": None
                },
                "Labels": {
                    "LabelX": None,
                    "LabelY": None,
                    "UnitX": None,
                    "UnitY": None,
                },
            }
        )
        labels_yaml = cast(
            list[tuple[Path, dict[str, dict[str, Any]]]],
            self._load_yaml(labels_pattern),
        )

        self._labels_x = pd.DataFrame.from_records(
            [
                (
                    l[1]["Description"]["IDDataSet"],
                    l[0],
                    *(
                        l[1].get("Labels", {}).get("LabelX", {}).get(v, np.nan)
                        for v in labels.kinvars_yaml_to_py.keys()
                    ),
                    *(
                        l[1].get("Labels", {}).get("UnitX", {}).get(v, np.nan)
                        for v in labels.kinvars_yaml_to_py.keys()
                    ),
                )
                for l in labels_yaml if "Description" in l[1] and "IDDataSet" in l[1]["Description"]
            ],
            index=[0, 1],
        )
        self._labels_x.columns = pd.MultiIndex.from_product(
            [["LabelX", "UnitX"], list(labels.kinvars_yaml_to_py.values())]
        )
        self._labels_x.index.names = ["id_dataset", "path"]
        self._labels_x.sort_index()

        self._labels_y = pd.DataFrame.from_records(
            [
                (
                    l[1]["Description"]["IDDataSet"],
                    l[0],
                    l[1].get("Labels", {}).get("LabelY", np.nan),
                    l[1].get("Labels", {}).get("UnitY", np.nan),
                )
                for l in labels_yaml if "Description" in l[1] and "IDDataSet" in l[1]["Description"]
            ],
            index=[0, 1],
        )
        self._labels_y.columns = ["LabelY", "UnitY"]
        self._labels_y.index.names = ["id_dataset", "path"]
        self._labels_y.sort_index()

    def plot_kinematic_coverage(
        self,
        ax: plt.Axes,
        kinematic_variables: tuple[str, str] = ("x", "Q2"),
        filter_query: str | None = None,
        groupby: DatasetsGroupBy | None = None,
        show_cut_points: Literal["before", "after", "both"] = "both",
        cuts: (
            list[tuple[float | sp.Rel | sp.Expr, npt.NDArray[np.floating]]] | None
        ) = None,
        cuts_labels: list[tuple[float, str] | None] | None = None,
        cuts_labels_offset: float | list[float | None] | None = None,
        kwargs_points: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        kwargs_points_before_cuts: (
            dict[str, Any] | list[dict[str, Any] | None] | None
        ) = None,
        kwargs_cuts: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        kwargs_cuts_labels: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ) -> None:
        """Plots the data points in the plane of 2 kinematic variables (by default x and Q²).

        Parameters
        ----------
        ax : plt.Axes
            The axes to plot on.
        kinematic_variables : tuple[str, str], optional
            Kinematic variables to display on the x and y axis, by default ("x", "Q2"). Must be in the columns of `points` and `points_before_cuts`.
        groupby : DatasetsGroupby | None, optional
            How to group the points, by default no grouping.
        show_cut_points : Literal["before", "after", "both"], optional.
            If the points before or after cuts should be shown, by default both.
        cuts : list[tuple[float | sp.Rel | sp.Expr, npt.NDArray[np.floating]]] | None, optional
            Cuts to display as curves, by default None. A cut is given as a tuple with first element a float (for a constant cut on the y axis) or a sympy expression that has at most one free variable which represents the x axis values, and second element a numpy array that gives the x axis values of the curve. For example, to display a W² cut on the (x, Q²) plane, pass
            ```
                cuts=[(nc.Q2_dis.subs({nc.W2: 1.7**2}), np.logspace(-0.7, -1e-3, 200))]
            ```
            (where `ncteqpy` is imported as `nc` and `numpy` is imported as `np`).
        cuts_labels : list[tuple[float, str]  |  None] | None, optional
            Labels to annotate the cuts by, by default None. These must be in the same ordering as `cuts`.
        cuts_labels_offset : float | list[float  |  None] | None, optional
            Offset in units of font size to shift the label orthogonally away from the curve representing a cut, by default None. Must be in the same order as `cuts` and `cuts_labels`.
        kwargs_points : dict[str, Any] | list[dict[str, Any] | None] | None, optional
            Keyword arguments to adjust plotting the points, passed to `ax.plot`, by default None.
        kwargs_points_before_cuts : dict[str, Any] | list[dict[str, Any] | None] | None, optional
            Keyword arguments to adjust plotting the points before cuts, passed to `ax.plot`, by default None.
        kwargs_cuts : dict[str, Any] | list[dict[str, Any] | None] | None, optional
            Keyword arguments to adjust plotting the cuts, passed to `ax.plot`, by default None. If a `list` is passed, it must be in the same order as `cuts`.
        kwargs_cuts_labels : dict[str, Any] | list[dict[str, Any] | None] | None, optional
            Keyword arguments to adjust plotting the labels of the cuts, passed to `ax.annotate`, by default None. If a `list` is passed, it must be in the same order as `cuts_labels`.
        """

        filtered_points = (
            self.points.query(filter_query) if filter_query is not None else self.points
        )
        filtered_points_after_cuts = (
            self.points_after_cuts.query(filter_query)
            if filter_query is not None
            else self.points_after_cuts
        )

        if show_cut_points == "before":
            points = filtered_points
            points_before_cuts = None
        elif show_cut_points == "after":
            points = filtered_points_after_cuts
            points_before_cuts = None
        else:
            points = filtered_points_after_cuts
            points_before_cuts = filtered_points

        plot_kinematic_coverage(
            ax=ax,
            points=points,
            points_before_cuts=points_before_cuts,
            kinematic_variables=kinematic_variables,
            groupby=groupby,
            cuts=cuts,
            cuts_labels=cuts_labels,
            cuts_labels_offset=cuts_labels_offset,
            kwargs_points=kwargs_points,
            kwargs_points_before_cuts=kwargs_points_before_cuts,
            kwargs_cuts=kwargs_cuts,
            kwargs_cuts_labels=kwargs_cuts_labels,
        )


class Dataset:

    _points: pd.DataFrame
    _yaml: jaml.YAMLType

    _path: Path
    _id: int | None = None
    _type_experiment: str | None = None
    _type_theory: str | None = None
    _A1: int | None = None
    _Z1: int | None = None
    _A2: int | None = None
    _Z2: int | None = None
    _kinematic_variables: list[str] | None = None
    _labelsx: dict[str, str] | None = None
    _labelsy: str | None = None
    _unitsx: dict[str, str] | None = None
    _unitsy: str | None = None
    # TODO add rest of fields

    _cut: sp.Rel | None = None

    _plotting_short_info: str | None = None
    _plotting_reference_id: str | None = None
    _plotting_process: str | None = None
    _plotting_labels_kinematic_variables: dict[str, str] | None = None
    _plotting_units_kinematic_variables: dict[str, str] | None = None
    _plotting_label_theory: str | None = None
    _plotting_unit_theory: str | None = None

    def __init__(self, path: str | os.PathLike, cut: sp.Rel | None = None) -> None:
        path = Path(path)
        self._path = path

        with open(path, "r") as file:
            self._yaml = yaml.safe_load(file)

        self.cut = cut

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

        if not "W2" in self.points:
            m_proton = 0.938

            if (
                self.type_experiment in ("DIS", "DISNEU")
                and "x" in self.points
                and "Q2" in self.points
            ):

                i = self.points.columns.get_loc("Q2")
                if not isinstance(i, int):
                    raise ValueError("multiple columns called 'Q2'")

                self._points.insert(
                    i + 1,
                    "W2",
                    self.points["Q2"] * (1.0 / self.points["x"] - 1.0) + m_proton**2,
                )

            elif (
                self.type_experiment == "DISDIMU"
                and "x" in self.points
                and "y" in self.points
                and "E_had" in self.points
            ):

                i = self.points.columns.get_loc("E_had")
                if not isinstance(i, int):
                    raise ValueError("multiple columns called 'E_had'")

                self._points.insert(
                    i + 1,
                    "W2",
                    m_proton**2
                    + 2.0
                    * m_proton
                    * (1.0 - self.points["x"])
                    * self.points["y"]
                    * self.points["E_had"],
                )

        if self.cut is not None:
            self._points = self._points[cut_accepts(self.cut, self._points)]

        if not "unc_tot" in self._points.columns:
            errs = list(set(self.points.columns) & set(["unc_stat", "unc_sys_uncorr", "unc_sys_corr"]))
            self._points["unc_tot"] = np.sqrt(np.sum(self.points[errs] ** 2, axis=1))

    def __repr__(self):
        return (
            f"Dataset(path='{self.path}')"
            if self.cut is None
            else f"Dataset(path='{self.path}', cut={self.cut})"
        )

    @property
    def points(self) -> pd.DataFrame:
        return self._points

    @property
    def yaml(self) -> jaml.YAMLType:
        return self._yaml

    @property
    def path(self) -> Path:
        return self._path

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
    def type_theory(self) -> str:
        if self._type_theory is None:
            self._type_theory = cast(
                str, jaml.nested_get(self.yaml, ["GridSpec", "TypeTheory"])
            )

        return self._type_theory

    @property
    def labelsx(self) -> dict[str, str]:
        if self._labelsx is None:
            self._labelsx = cast(
                dict[str, str], jaml.nested_get(self.yaml, ["Labels", "LabelX"])
            )
        print(self._labelsx)

        return self._labelsx

    @property
    def labelsy(self) -> str:
        if self._labelsy is None:
            self._labelsy = cast(
                str, jaml.nested_get(self.yaml, ["Labels", "LabelY"])
            )

        return self._labelsy

    @property
    def unitsx(self) -> dict[str, str]:
        if self._unitsx is None:
            self._unitsx = cast(
                dict[str, str], jaml.nested_get(self.yaml, ["Labels", "UnitX"])
            )

        return self._unitsx

    @property
    def unitsy(self) -> str:
        if self._unitsy is None:
            self._unitsy = cast(
                str, jaml.nested_get(self.yaml, ["Labels", "UnitY"])
            )

        return self._unitsy
    
   
    @property
    def A1(self) -> int | None:
        if self._A1 is None:
            self._A1 = cast(
                int, jaml.nested_get(self.yaml, ["Description", "AZValues1", 0])
            )

        return self._A1

    @property
    def A2(self) -> int | None:
        if self._A2 is None:
            self._A2 = cast(
                int, jaml.nested_get(self.yaml, ["Description", "AZValues2", 0])
            )

        return self._A2

    @property
    def Z1(self) -> int | None:
        if self._Z1 is None:
            self._Z1 = cast(
                int, jaml.nested_get(self.yaml, ["Description", "AZValues1", 1])
            )

        return self._Z1

    @property
    def Z2(self) -> int | None:
        if self._Z2 is None:
            self._Z2 = cast(
                int, jaml.nested_get(self.yaml, ["Description", "AZValues2", 1])
            )

        return self._Z2

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

    def apply(self, cut: sp.Rel) -> None:
        self._points = self._points[cut_accepts(cut, self._points)]

    def plot(
        self,
        ax: plt.Axes | None = None,
        x_variable: str | list[str] = "fallback",
        kinematic_variable: str | list[str] | None = None,  # TODO: implement
        bin_label: Literal["legend", "annotate"] | None = None,
        ratio_to: Dataset | None = None,  # TODO: add option for data/theory?
        stat_uncertainty: bool = False,
        xlabel: str | dict[str, str] | None = "fallback",
        xunit: str | dict[str, str] = "fallback",
        ylabel: (
            str
            | Literal["fallback"]
            | None
            | SequenceNotStr[str | Literal["fallback"] | None]
        ) = "fallback",        
        yunit: str = "fallback",
        title: str | None = None,
        curve_label: (
            Literal[
                "annotate above",
                "annotate right",
                "ticks",
                "legend",
            ]
            | None
        ) = "ticks",
        plot_types: DataVsTheoryType | Sequence[DataVsTheoryType] = [
            "absolute",
            "theory over data",
        ],
        xscale: str | None = None,
        yscale: str | None = None,
        y_offset_add: float | Sequence[float] = 0,
        y_offset_mul: float | Sequence[float] = 0,
        legend: bool = False,
        subplot_label: Literal["legend"] | None = None,
        subplot_label_format: str | None = None,
        curve_groupby: str | list[str] | None = None,
        kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_xlabel: dict[str, Any] = {},
        kwargs_ylabel: dict[str, Any] = {},
        kwargs_title: dict[str, Any] = {},
        kwargs_legend: dict[str, Any] = {},
        kwargs_legend_curves: dict[str, Any] = {},
        kwargs_legend_subplots: dict[str, Any] = {},
        kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
        **kwargs: object,
    ) -> None:
        if ax is None:
            ax = plt.gca()

        match self.type_experiment:
            case "DIS" | "DY":
                if x_variable == "fallback":
                    x_variable = "x"

                if y_offset_add is None:
                    y_offset_add = 0.5

                if y_offset_mul is None:
                    y_offset_mul = 0

            case "DISNEU":
                if x_variable == "fallback":
                    x_variable = "Q2"
                    if xscale is None:
                        xscale = "log"

                if curve_groupby == "fallback":
                    curve_groupby = "x"

                if y_offset_add is None:
                    y_offset_add = 0.5

                if y_offset_mul is None:
                    y_offset_mul = 0

            case "DISDIMU":
                if x_variable == "fallback":
                    x_variable = "x"

                if curve_groupby == "fallback":
                    curve_groupby = "y"

                if y_offset_add is None:
                    y_offset_add = 0.5

                if y_offset_mul is None:
                    y_offset_mul = 0

            case "HQ" | "OPENHEAVY" | "QUARKONIA":
                if x_variable == "fallback":
                    x_variable = ["pT_min", "pT_max"]

                if curve_groupby == "fallback":
                    curve_groupby = ["y_min", "y_max"]

                if yscale is None:
                    yscale = "log"

                if y_offset_add is None:
                    y_offset_add = [0, 1]

                if y_offset_mul is None:
                    y_offset_mul = [1, 0]

            case "SIH":
                if x_variable == "fallback":
                    x_variable = "pT"

                if yscale is None:
                    yscale = "log"

                if y_offset_add is None:
                    y_offset_add = 0

                if y_offset_mul is None:
                    y_offset_mul = 1

            case "WZPROD"|"WCPROD":
                if x_variable == "fallback":
                    x_variable = "eta"

                if y_offset_add is None:
                    y_offset_add = 0

                if y_offset_mul is None:
                    y_offset_mul = 1

        assert x_variable is not None
    
        if curve_groupby == "fallback":
            curve_groupby = None
    
        if y_offset_add is None:
            y_offset_add = 0
    
        if y_offset_mul is None:
            y_offset_mul = 0
    
        if isinstance(xlabel, dict) and "fallback" in xlabel:
            xlabel[_get_kinematic_variable(x_variable)] = xlabel["fallback"]
    
        # if isinstance(xunit, dict) and "fallback" in xunit:
        #     xunit[_get_kinematic_variable(x_variable)] = xunit["fallback"]

        points = self.points

        labels_x = self.labelsx
        labels_y = self.labelsy
        units_x = self.unitsx
        units_y = self.unitsy


        if xlabel == "fallback":
            xlabel_updated = {}
        elif isinstance(xlabel, str):
            if x_variable is None or x_variable == "fallback":
                xlabel_updated = {"fallback": xlabel}
            else:
                xlabel_updated = {
                    _get_kinematic_variable(x_variable): xlabel
                }
        elif xlabel is None:
            if x_variable is None or x_variable == "fallback":
                xlabel_updated = {"fallback": ""}
            else:
                xlabel_updated = {
                    _get_kinematic_variable(x_variable): ""
                }
        else:
            xlabel_updated = xlabel

        if xunit == "fallback":
            xunit_updated = {}
        elif isinstance(xunit, str):
            if x_variable is None or x_variable == "fallback":
                xunit_updated = {"fallback": xunit}
            else:
                xunit_updated = {
                    _get_kinematic_variable(x_variable): xunit
                }
        elif xunit is None:
            if x_variable is None or x_variable == "fallback":
                xunit_updated = {"fallback": ""}
            else:
                xunit_updated = {
                    _get_kinematic_variable(x_variable): ""
                }
        else:
            xunit_updated = xunit

        if len(labels_x) > 0:
            xlabel_updated = {
                nc_labels.kinvars_yaml_to_py[k]: str(s).replace("\\frac", "\\dfrac")
                for k, s in labels_x.items()
            } | xlabel_updated

            xunit_updated = {
                nc_labels.kinvars_yaml_to_py[k]: (
                    str(u).replace("\\frac", "\\dfrac")
                    if (u := s) not in [1.0, "1", "1.0"]
                    else ""
                )
                for k, s in units_x.items()
            } | xunit_updated
            print(xunit_updated)


        if ylabel == "fallback":
            ylabel_updated = str(labels_y).replace(
                "\\frac", "\\dfrac"
            )
            yunit = (
                str(u).replace("\\frac", "\\dfrac")
                if (u := units_y) not in [1.0, "1", "1.0"]
                else ""
            )
        else:
            ylabel_updated = ylabel
            yunit = ""



        plot_common_data(
            ax=ax,
            points=points,
            x_variable=x_variable,
            xlabel=xlabel_updated,
            ylabel=ylabel_updated,
            xunit= xunit_updated,
            yunit=yunit,
            xscale=xscale,
            yscale=yscale,
            title=title,
            plot_types=plot_types,
            theory_col=self.type_theory,
            legend=legend,
            curve_label=curve_label,
            subplot_label=subplot_label,
            subplot_label_format=subplot_label_format,
            curve_groupby=curve_groupby,
            y_offset_add=y_offset_add,
            y_offset_mul=y_offset_mul,
            kwargs_data=kwargs_data,
            kwargs_xlabel=kwargs_xlabel,
            kwargs_ylabel=kwargs_ylabel,
            kwargs_title=kwargs_title,
            kwargs_legend=kwargs_legend,
            kwargs_legend_curves=kwargs_legend_curves,
            kwargs_legend_subplots=kwargs_legend_subplots,
            kwargs_ticks_curves=kwargs_ticks_curves,
            kwargs_annotate_curves=kwargs_annotate_curves,
            **kwargs,
        )


def plot_common_data(
    ax: plt.Axes | Sequence[plt.Axes],
    points: pd.DataFrame,
    x_variable: str | list[str],
    plot_types: DataVsTheoryType | Sequence[DataVsTheoryType] = [
        "absolute",
    ],
    xlabel: str | dict[str, str] | None = None,
    xunit: str | dict[str, str] = "",
    theory_col : str = "",
    ylabel: (
        str
        | Literal["fallback"]
        | None
        | SequenceNotStr[str | Literal["fallback"] | None]
    ) = None,
    yunit: str = "",
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
    subplot_label: Literal["legend"] | None = None,
    subplot_label_format: str | None = None,
    curve_groupby: str | list[str] | None = None,
    apply_normalization: bool = True,
    shift_correlated: Literal["data", "theory"] | None = "theory",
    y_offset_add: float | Sequence[float] = 0,
    y_offset_mul: float | Sequence[float] = 0,
    kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_xlabel: dict[str, Any] = {},
    kwargs_ylabel: dict[str, Any] = {},
    kwargs_title: dict[str, Any] = {},
    kwargs_legend: dict[str, Any] = {},
    kwargs_legend_curves: dict[str, Any] = {},
    kwargs_legend_subplots: dict[str, Any] = {},
    kwargs_ticks_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
    kwargs_annotate_curves: dict[str, Any] | list[dict[str, Any] | None] = {},
) -> None:
    """Plot data vs. theory in a way that is not experiment-specific.

    Parameters
    ----------
    ax : plt.Axes | Sequence[plt.Axes]
        Axes to plot on. If more than one axes are given, a ratio plot is drawn on the second one.
    points : pd.DataFrame
        The points to be plotted. Needs to have the columns `x_variable`, `"data"` (to plot the data points) and `"theory"` or `"theory_with_normalization"` to plot the theory curves (normalization-uncorrected or corrected, respectively).
    x_variable : str | list[str]
        The kinematic variable to put on the x axis. Plots a binned distribution if a list is passed.
    xlabel : str | None, optional
        x label of the plot, by default None.
    ylabel : str | None, optional
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
    subplot_label : Literal["legend"] | None, optional  # FIXME
        Where to label the subplot, by default None.
    subplot_label_format : str | None, optional
        Format of the subplot label, by default None.
    chi2_annotation : bool, optional
        If the χ²/point value is annotated, by default True.
    chi2_legend : bool, optional
        If a legend with the total χ² is shown, by default True.
    curve_groupby : str | list[str] | None, optional
        Variable(s) to group the curves by, by default no grouping.
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
    y_offset_add : float, optional
        Additive offset by which the curves are separated, by default 0 (no offset).
    y_offset_mul : float, optional
        Multiplicative offset by which the curves are separated, by default 0 (no offset). The factor between each curve is `10**y_offset_mul`.
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
    """

    if isinstance(ax, plt.Axes):
        ax = [ax]

    if not isinstance(y_offset_add, Sequence):
        y_offset_add = len(ax) * [y_offset_add]

    if not isinstance(y_offset_mul, Sequence):
        y_offset_mul = len(ax) * [y_offset_mul]

    if len(y_offset_add) < len(ax):
        raise ValueError("len(y_offset_add) must be equal to len(ax)")

    if len(y_offset_mul) < len(ax):
        raise ValueError("len(y_offset_mul) must be equal to len(ax)")

    # Data & Theory legend
    legend_handles = []
    legend_labels = []

    # bin labels for legend or ticks
    legend_curve_handles = []

    if curve_groupby is not None:
        ascending = curve_groupby != "x"  # TODO: replace with PointsGroupBy
        points = points.sort_values(curve_groupby, ascending=ascending)

    curve_gb = (
        points.groupby(curve_groupby, sort=False) if curve_groupby is not None else None
    )

    # enumerate groupby if curve_groupby is given, else just enumerate one tuple (i.e. no iteration)
    iter_curves_data = (
        enumerate(curve_gb) if curve_gb is not None else enumerate([(0.0, points)])
    )
    for i, (label_i, points_i) in iter_curves_data:
        points_i = points_i.sort_values(x_variable)
        


        l = None

        kwargs_data_default = (
            {"markerfacecolor": l.get_color()} if l is not None else {}
        )
        kwargs_data_updated = update_kwargs(kwargs_data_default, kwargs_data, i)
        e = _plot_data_data(
            ax=ax,
            points=points_i,
            x_col=x_variable,
            theory_col=nc_labels.theory_yaml_to_py[theory_col],
            index_curve=i,
            y_offset_add=y_offset_add,
            y_offset_mul=y_offset_mul,
            shift_correlated=(shift_correlated == "data"),
            plot_types=plot_types,
            kwargs_data=kwargs_data_updated,
        )

        if e is not None and i == 0:
            legend_handles.insert(0, e)
            legend_labels.insert(0, "Data")


    if yscale is not None:
        ax[0].set_yscale(yscale)  # pyright: ignore[reportArgumentType]

    if curve_gb is not None:
        #print(xlabel)
        _add_curve_labels(
            ax,
            types=plot_types,
            curve_label_position=curve_label,
            curve_groupby=curve_gb,
            x_col=x_variable,
            y_col=nc_labels.theory_yaml_to_py[theory_col],
            label=curve_groupby,
            unit=xunit,
            y_offset_add=y_offset_add,
            y_offset_mul=y_offset_mul,
            legend_curve_handles=legend_curve_handles,
            kwargs_annotate_curves=kwargs_annotate_curves,
            kwargs_legend_curves=kwargs_legend_curves,
            kwargs_tick_curves=kwargs_ticks_curves,
        )

    x_fallback = x_variable if isinstance(x_variable, str) else x_variable[0]
    if x_fallback in nc_labels.kinvars_py_to_tex:
        x_fallback = f"${nc_labels.kinvars_py_to_tex[x_fallback]}$"
    print(xlabel)
    print(ylabel)
    print(xunit)
    print(yunit)
    _set_labels(
        ax=ax,
        x_variable=x_variable,
        x_fallback=x_fallback,
        y_fallback={
            "absolute": "",
            "data over theory": r"\dfrac{\text{Data}}{\text{Theory}}",
            "theory over data": r"\dfrac{\text{Theory}}{\text{Data}}",
        },
        x_label=xlabel,
        x_unit=xunit,
        y_label=ylabel,
        y_unit=yunit,
        title=title,
        plot_types=plot_types,
        kwargs_xlabel=kwargs_xlabel,
        kwargs_ylabel=kwargs_ylabel,
        kwargs_title=kwargs_title,
    )

    if legend:
        kwargs_legend_default = {
            "order": 2,
            "parent": ax[0],
            "handles": legend_handles,
            "labels": legend_labels,
        }
        kwargs_legend_updated = update_kwargs(kwargs_legend_default, kwargs_legend)

        legend_data_theory = AdditionalLegend(**kwargs_legend_updated)
        ax[0].add_artist(legend_data_theory)


    if subplot_label == "legend":
        if subplot_label_format is None:
            raise ValueError(
                "subplot_label_format must be given if subplot_label is 'legend'"
            )

        A_symbols = {}

        if "data" in points:
            if points.iloc[0][["Z1", "A1"]].notna().all():
                A_symbols["A1_sym"] = nc_labels.nucleus_to_latex(
                    Z=points.iloc[0]["Z1"], A=points.iloc[0]["A1"], show_A=True
                )
            if points.iloc[0][["Z2", "A2"]].notna().all():
                A_symbols["A2_sym"] = nc_labels.nucleus_to_latex(
                    Z=points.iloc[0]["Z2"], A=points.iloc[0]["A2"], show_A=True
                )

            subplot_label_str = subplot_label_format.format(
                **points.iloc[0].to_dict(), **A_symbols
            )
        else:
            raise ValueError(
                "Either data or theory must be given if subplot_label is 'legend'"
            )

        kwargs_legend_subplots_default = dict(
            order=1,
            parent=ax[0],
            handles=[Patch()],
            labels=[subplot_label_str],
            labelspacing=0,
            handlelength=0,
            handleheight=0,
            handletextpad=0,
        )
        kwargs_legend_subplots_updated = update_kwargs(
            kwargs_legend_subplots_default, kwargs_legend_subplots
        )

        leg3 = AdditionalLegend(**kwargs_legend_subplots_updated)
        ax[0].add_artist(leg3)

    if xscale is not None:
        ax[0].set_xscale(xscale)  # pyright: ignore[reportArgumentType]
        ax[0].relim()  # otherwise axes autoscaling does not work for some reason
        if len(ax) > 1:
            ax[1].set_xscale(xscale)  # pyright: ignore[reportArgumentType]
            ax[1].relim()



def _plot_data_data(
    ax: plt.Axes | Sequence[plt.Axes],
    points: pd.DataFrame,
    x_col: str | list[str],
    theory_col: str = "theory",
    index_curve: int = 0,
    num_curves: int = 1,
    y_offset_add: float | Sequence[float] = 0,
    y_offset_mul: float | Sequence[float] = 0,
    shift_correlated: bool = False,
    ratio_offset: bool = True,
    plot_types: DataVsTheoryType | Sequence[DataVsTheoryType] = [
        "absolute",
        "theory over data",
    ],
    kwargs_data: dict[str, Any] | list[dict[str, Any] | None] = {},
) -> ErrorbarContainer | None:

    ax, plot_types, y_offset_add, y_offset_mul = _check_ax_offsets(
        ax=ax, types=plot_types, y_offset_add=y_offset_add, y_offset_mul=y_offset_mul
    )

    # print(ax, plot_types, y_offset_add, y_offset_mul)

    res = None

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if theory_col in points:
        for i, (ax_i, plot_type_i, y_offset_add_i, y_offset_mul_i) in enumerate(
            zip(ax, plot_types, y_offset_add, y_offset_mul)
        ):
           # print(i)
            offset_factor = 10 ** (np.array(y_offset_mul_i) * index_curve)
            offset_summand = np.array(y_offset_add_i) * index_curve

            kwargs_data_default: dict[str, Any] = {
                "capsize": 1.7,
                "marker": ".",
                "markersize": 7,
                "ls": "",
                "color": "black",
                "markerfacecolor": colors[index_curve % len(colors)],
            }

            if isinstance(x_col, list):
                xerr = (points[x_col[1]] - points[x_col[0]]) / 2
                kwargs_data_default["xerr"] = (xerr,)
            else:
                xerr = None

            points_data = (
                points[theory_col] - points["shift_correlated"]
                if shift_correlated and "shift_correlated" in points
                else points[theory_col]
            )

            if "unc_tot" in points:
                kwargs_data_default["yerr"] = (
                    _calculate_ratio(
                        plot_type_i, points["unc_tot"], points_data, points[theory_col]
                    )
                    * offset_factor
                )

            x = points[x_col].mean(axis=1) if isinstance(x_col, list) else points[x_col]

            y = _calculate_ratio(
                plot_type_i, points_data, points_data, points[theory_col]
            )

            kwargs_data_updated = update_kwargs(
                kwargs_data_default, kwargs_data, index_curve
            )

            e = ax_i.errorbar(
                x,
                y * offset_factor + offset_summand,
                **kwargs_data_updated,
            )

            e.lines[0].set_markeredgewidth(0.4)

            if i == 0:
                res = e

            # if len(ax) >= 2:
            #     if "unc_tot" in points:
            #         kwargs_data_updated["yerr"] = points["unc_tot"] / y

            #     # TODO: make ratio_offset work if x_col is not a list, i.e. when not plotting a binned distribution
            #     # if xerr is not None:
            #     #     x_offsets = (
            #     #         1.9 * xerr * (1 / (num_curves + 1) * (index_curve + 0.5) - 0.5)
            #     #     )
            #     #     x += x_offsets

            #     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            #     kwargs_data_updated["markerfacecolor"] = colors[
            #         index_curve % len(colors)
            #     ]
            #     kwargs_data_updated["markeredgecolor"] = kwargs_data_updated[
            #         "markerfacecolor"
            #     ]
            #     # kwargs_data_updated.pop("xerr")
            #     ax[1].errorbar(
            #         x,
            #         y / y * offset_factors[1] + offset_summands[1],
            #         **kwargs_data_updated,
            #     )

    return res