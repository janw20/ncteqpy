from __future__ import annotations

import os
from pathlib import Path
from typing_extensions import Any, Hashable, Literal, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import sympy as sp
import yaml

import ncteqpy.jaml as jaml
import ncteqpy.labels as labels
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


# TODO: make possible to load only subdirs or list of paths
class Datasets(jaml.YAMLWrapper):

    _cuts: Cuts | None = None
    _index: pd.DataFrame | None = None
    _points: pd.DataFrame | None = None
    _points_after_cuts: pd.DataFrame | None = None
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
    def cached_datasets(self) -> dict[Path, Dataset]:
        return self._cached_datasets

    @property
    def cached_datasets_with_cuts(self) -> dict[Path, Dataset]:
        return self._cached_datasets_with_cuts

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
        order: list[Hashable] | None = None,
        labels: dict[Hashable, str] | None = None,
        label_format: str | None = None,
        props: dict[Hashable, dict[str, Any]] | None = None,
    ) -> DatasetsGroupBy:
        return DatasetsGroupBy(
            datasets_index=self.index,
            by=by,
            grouper=grouper,
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

        assert isinstance(points_yaml, list)

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
                    "data": grid_row[
                        grid_row_labels.index(labels.data_yaml_to_py[observable])
                    ],
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
                    "Dim": None,
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
                    "id_dataset": jaml.nested_get(data, ["Description", "IDDataSet"]),
                    "path": p,
                    "type_experiment": jaml.nested_get(
                        data, ["Description", "TypeExp"]
                    ),
                    "A1": az1[0],
                    "Z1": az1[1],
                    "A2": az2[0],
                    "Z2": az2[1],
                    "final_state": jaml.nested_get(data, ["Description", "FinalState"]),
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
    _A1: int | None = None
    _Z1: int | None = None
    _A2: int | None = None
    _Z2: int | None = None
    _kinematic_variables: list[str] | None = None
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
            errs = list(set(self.points.columns) & set(["unc_stat", "unc_sys"]))
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
