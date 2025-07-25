from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TypeAlias, cast, override, reveal_type

import matplotlib.transforms as mtransforms
import numpy as np
import numpy.typing as npt
import pandas as pd
import sympy as sp
from matplotlib import pyplot as plt

from ncteqpy.chi2 import Chi2
from ncteqpy.data import Datasets
from ncteqpy.labels import nucleus_to_latex
from ncteqpy.util import update_kwargs

pdfplotter_imported = False
try:
    import pdfplotter as pp

    pdfplotter_imported = True
except ImportError:
    pass


idx = pd.IndexSlice

Scalar = str | float
GroupByLabels = Scalar | tuple[Scalar, ...] | list[Scalar] | list[tuple[Scalar, ...]]


class Chi2PDFCorrelation(ABC):
    """Abstract base class for x- and Q²-dependent correlations between the χ² function and the PDFs."""

    _datasets: Datasets
    _chi2: Chi2
    _pdf_set: pp.PDFSet
    _groupby: str | list[str]
    _grouper: pd.Series
    _data: pd.DataFrame
    _data_total: pd.DataFrame

    def __init__(
        self,
        datasets: Datasets,
        chi2: Chi2,
        pdfs: pp.PDFSet,
        groupby: str | list[str] = "id_dataset",
    ) -> None:
        """Creates an instance to calculate and plot the correlation.

        Parameters
        ----------
        datasets : Datasets
            Datasets to supply the information needed for grouping.
        chi2 : Chi2
            χ² function, needs to have the dataset breakdown to each snapshot.
        pdfs : pp.PDFSet
            The PDF set.
        groupby : str | list[str], optional
            Column of `datasets.index` to group the datasets by, by default "id_dataset" (no grouping).
        """

        if not pdfplotter_imported:
            raise ValueError("Please install pdfplotter to use L2Sensitivity")

        self._datasets = datasets
        self._chi2 = chi2
        self._pdf_set = pdfs
        self._groupby = groupby

        if groupby == "id_dataset":
            self._grouper = pd.Series(
                datasets.index["id_dataset"].to_numpy(),
                index=datasets.index["id_dataset"],
            )
        elif not isinstance(groupby, list):
            self._grouper = datasets.index.set_index("id_dataset")[groupby]
        else:
            self._grouper = datasets.index.set_index("id_dataset")[groupby].apply(
                tuple, axis=1
            )

        self._data = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [np.atleast_1d(pdfs.Q2), np.atleast_1d(pdfs.x)],
                names=["Q2", "x"],
            ),
            columns=pd.MultiIndex.from_product(
                (len(self._grouper.index.names) + 1) * [[]],
                names=["observable", *self._grouper.index.names],
            ),
        )

        self._data_total = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [np.atleast_1d(pdfs.Q2), np.atleast_1d(pdfs.x)],
                names=["Q2", "x"],
            ),
            columns=pd.MultiIndex.from_product(
                [[], []],
                names=["observable", "total"],
            ),
        )

    @property
    def chi2(self) -> Chi2:
        """The χ² function to correlate"""
        return self._chi2

    @property
    def pdf_set(self) -> pp.PDFSet:
        """The PDF set to correlate"""
        return self._pdf_set

    @property
    def data(self) -> pd.DataFrame:
        """Values that have been calculated so far by calling `get`"""
        return self._data

    @property
    def data_total(self) -> pd.DataFrame:
        """Values that have been calculated so far by calling `get_total`"""
        return self._data_total

    @property
    def groupby(self) -> str | list[str]:
        """The column(s) the correlation curves are grouped by"""
        return self._groupby

    @property
    def grouper(self) -> pd.Series:
        """pandas.Series that maps each `id_dataset` (Series index) to the values of the groupby column (Series values). If multiple columns are chosen for grouping, their values are stored as tuples."""
        return self._grouper

    @staticmethod
    @abstractmethod
    def calculate(
        pdf_data: pd.DataFrame,
        chi2_data: pd.DataFrame,
        chi2_min: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Member functions that calculates the correlation and needs to be overridden in each subclass. Both `pdf_data` and `chi2_data` must have an index level `id_eigendirection` containing the eigenvectors with positive and negative directions alternating and in the same order in each index. The other index levels and the columns can be arbitrary and will be preserved in the output DataFrame.

        Parameters
        ----------
        pdf_data : pd.DataFrame
            Values of the PDFs. Index: `(..., id_eigendirection, ...)`, where the `id_eigendirection` level contains the eigenvectors with positive and negative directions alternating and in the same order as in the `id_eigendirection` index level of `chi_data`

        chi2_data : pd.DataFrame
            Values of the chi-squared function. Index: `(..., id_eigendirection, ...)`, where the `id_eigendirection` level contains the eigenvectors with positive and negative directions alternating and in the same order as in the `id_eigendirection` index level of `pdf_data`

        Returns
        -------
        pd.DataFrame
            Values of the correlation, with index and columns levels being the cartesian product of the ones in `pdf_data` and `chi2_data`.
        """
        ...

    def get(
        self,
        observable: sp.Basic,
        group_labels: GroupByLabels | None = None,
        x: float | npt.NDArray[np.float64] | None = None,  # FIXME
        Q: float | npt.NDArray[np.float64] | None = None,  # FIXME
        Q2: float | npt.NDArray[np.float64] | None = None,  # FIXME
    ) -> pd.DataFrame:
        """Get the actual values of the correlation between the χ² and the PDFs. Calculated values are cached in `data`.

        Parameters
        ----------
        observable : sp.Basic
            The PDF observable to correlate to the χ² function.
        group_labels : GroupByType | None, optional
            Filter for which `groupby` labels to get the correlation, by default all. E.g., if the `groupby` is `id_dataset`, `group_labels` can be a data set ID or a list of multiple data set IDs. If multiple columns are chosen for grouping, i.e., `groupby` is a list, `group_labels` must be given as a tuple or a list of tuples.
        x : float | npt.NDArray[np.float64] | None, optional
            x values to get the correlation for, by default all. Not implemented yet
        Q : float | npt.NDArray[np.float64] | None, optional
            Q values to get the correlation for, by default all. Not implemented yet
        Q2 : float | npt.NDArray[np.float64] | None, optional
            Q² values to get the correlation for, by default all. Not implemented yet

        Returns
        -------
        pd.DataFrame
            Calculated values of the correlation for PDF observable `observable` and groups `group_labels`.
        """
        if group_labels is None:
            group_labels = self.grouper.to_list()
        elif not isinstance(group_labels, list):
            group_labels = cast(list[Scalar] | list[tuple[Scalar, ...]], [group_labels])

        if not str(observable) in self.data.columns.get_level_values("observable"):
            pdf_data = pd.DataFrame(
                np.array(
                    [
                        self.pdf_set.get_member(observable, i + 1)
                        for i in range(self.pdf_set.num_errors)
                    ]
                ).flatten(),
                index=pd.MultiIndex.from_product(
                    [
                        range(self.pdf_set.num_errors),
                        np.atleast_1d(self.pdf_set.Q2),
                        np.atleast_1d(self.pdf_set.x),
                    ],
                    names=["id_eigendirection", "Q2", "x"],
                ),
                columns=pd.Index([str(observable)], name="observable"),
            )

            chi2_data = (
                self.chi2.snapshots_breakdown_datasets.iloc[
                    -2 * len(self.chi2.parameters_names) - 1 :
                ]
                .T.groupby(
                    self.grouper,
                    dropna=False,
                    sort=True,
                )
                .sum()
                .T.reset_index(drop=True)
            )
            chi2_data.columns.set_names(self._data.columns.names[1:], inplace=True)

            chi2_min = chi2_data.loc[0]
            chi2_data.index = pd.Index(chi2_data.index - 1, name="id_eigendirection")

            res = self.calculate(
                pdf_data=pdf_data,
                chi2_data=chi2_data.loc[
                    0:
                ],  # [0:] is needed because of above index shift
                chi2_min=chi2_min,
            )

            self._data = pd.concat([self._data, res], axis=1)

        return self._data.loc[:, idx[str(observable), group_labels]]

    def get_total(
        self,
        observable: sp.Basic,
        x: float | npt.NDArray[np.float64] | None = None,  # FIXME
        Q: float | npt.NDArray[np.float64] | None = None,  # FIXME
        Q2: float | npt.NDArray[np.float64] | None = None,  # FIXME
    ) -> pd.DataFrame:
        """Get the actual values of the correlation between the total χ² and the PDFs. Calculated values are cached in `data_total`.

        Parameters
        ----------
        observable : sp.Basic
            The PDF observable to correlate to the χ² function.
        x : float | npt.NDArray[np.float64] | None, optional
            x values to get the correlation for, by default all. Not implemented yet
        Q : float | npt.NDArray[np.float64] | None, optional
            Q values to get the correlation for, by default all. Not implemented yet
        Q2 : float | npt.NDArray[np.float64] | None, optional
            Q² values to get the correlation for, by default all. Not implemented yet

        Returns
        -------
        pd.DataFrame
            Calculated values of the correlation for PDF observable `observable` and groups `group_labels`.
        """

        if not str(observable) in self.data_total.columns.get_level_values(
            "observable"
        ):
            pdf_data = pd.DataFrame(
                np.array(
                    [
                        self.pdf_set.get_member(observable, i + 1)
                        for i in range(self.pdf_set.num_errors)
                    ]
                ).flatten(),
                index=pd.MultiIndex.from_product(
                    [
                        range(self.pdf_set.num_errors),
                        np.atleast_1d(self.pdf_set.Q2),
                        np.atleast_1d(self.pdf_set.x),
                    ],
                    names=["id_eigendirection", "Q2", "x"],
                ),
                columns=pd.Index([str(observable)], name="observable"),
            )

            chi2_data = (
                self.chi2.snapshots_breakdown_datasets.iloc[
                    -2 * len(self.chi2.parameters_names) - 1 :
                ]
                .sum(axis=1)
                .reset_index(drop=True)
            ).to_frame(name="total")
            chi2_data.columns.name = "total"

            chi2_min = chi2_data.loc[0]
            chi2_data.index = pd.Index(chi2_data.index - 1, name="id_eigendirection")

            res = self.calculate(
                pdf_data=pdf_data,
                chi2_data=chi2_data.loc[0:],
                chi2_min=chi2_min,
            )

            self._data_total = pd.concat([self._data_total, res], axis=1)

        return self._data_total.loc[:, idx[str(observable)]]

    def plot(
        self,
        ax: plt.Axes,
        observable: sp.Basic,
        group_labels: GroupByLabels | None = None,
        group_highlights: GroupByLabels | None = None,
        group_label_style: Literal["legend", "curve"] = "curve",
        group_label_format: str | None = None,
        x: float | npt.NDArray[np.float64] | None = None,  # FIXME
        Q: float | npt.NDArray[np.float64] | None = None,  # FIXME
        Q2: float | npt.NDArray[np.float64] | None = None,  # FIXME
        kwargs_curves: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        kwargs_curves_background: (
            dict[str, Any] | list[dict[str, Any] | None] | None
        ) = None,
        kwargs_curve_total: dict[str, Any] | None = None,
        kwargs_labels: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ) -> None:
        """Plot the correlation curves for observable `observable` and groupby labels `group_labels`

        Parameters
        ----------
        ax : plt.Axes
            The axes to plot on.
        observable : sp.Basic
            The PDF observable to correlate to the χ² function.
        group_labels : str | float | list[str  |  float] | None, optional
            Filter for which `groupby` labels to plot the correlation curves, by default all. E.g., if the `groupby` is `id_dataset`, `group_labels` can be a data set ID or a list of multiple data set IDs. If multiple columns are chosen for grouping, i.e., `groupby` is a list, `group_labels` must be given as a tuple or a list of tuples.
        group_highlights : str | float | list[str  |  float] | None, optional
            Which curves to highlight, by default all. E.g., if the `groupby` is `id_dataset`, `group_labels` can be a data set ID or a list of multiple data set IDs. If multiple columns are chosen for grouping, i.e., `groupby` is a list, `group_labels` must be given as a tuple or a list of tuples.
        group_label_style : Literal["legend", "curve"], optional
            If the labels of the group are shown in the legend or annotated on each curve, by default "curve".
        group_label_format : str | None, optional
            Format string to control formatting of each label. By default no formatting is applied, i.e., only the label is shown. Fields must be named, where possible names are the columns of the `index` of the `Datasets` that were passed to the constructor, as well as `A1_sym`, `A2_sym`, `A_heavier_sym`, and `A_lighter_sym`, giving the symbol of the respective nucleus.
        x : float | npt.NDArray[np.float64] | None, optional
            x values to get the correlation for, by default all. Not implemented yet
        Q : float | npt.NDArray[np.float64] | None, optional
            Q values to get the correlation for, by default all. Not implemented yet
        Q2 : float | npt.NDArray[np.float64] | None, optional
            Q² values to get the correlation for, by default all. Not implemented yet
        kwargs_curves : dict[str, Any]  |  list[dict[str, Any]  |  None]  |  None, optional
            Keyword arguments to adjust plotting the correlation curves, passed to `ax.plot`, by default None. If a `list` is passed, it must be in the same order as the index of `grouper`.
        kwargs_curves_background : dict[str, Any]  |  list[dict[str, Any]  |  None]  |  None, optional
            Keyword arguments to adjust plotting the correlation curves that are in the background if `group_highlights` is given, passed to `ax.plot`, by default None. If a `list` is passed, it must be in the same order as the index of `grouper`.
        kwargs_curve_total : dict[str, Any] | None, optional
            Keyword arguments to adjust plotting the correlation curve for the total χ², passed to `ax.plot`, by default None.
        kwargs_labels : dict[str, Any] | list[dict[str, Any] | None] | None, optional
            Keyword arguments to adjust the label of each correlation, passed to `ax.annotate`, by default None.
        """
        values = self.get(observable=observable, group_labels=group_labels).T
        values_total = self.get_total(observable=observable)

        if group_highlights is not None and not isinstance(group_highlights, list):
            group_highlights = [group_highlights]

        num_curves = values.shape[0]
        num_labels = (
            len(group_highlights) if group_highlights is not None else num_curves
        )

        # to place the labels evenly on the curves, we transform values in [0, 1] onto the x values of the curves
        label_transform: mtransforms.Transform = (
            ax.transScale  # pyright: ignore[reportAttributeAccessIssue]
            + mtransforms.BboxTransformFrom(
                mtransforms.TransformedBbox(
                    mtransforms.Bbox(
                        np.array([[self.pdf_set.x.min(), 1], [self.pdf_set.x.max(), 2]])
                    ),
                    ax.transScale,  # pyright: ignore[reportAttributeAccessIssue]
                )
            )
        ).inverted()

        # offset of the labels from the edges (in [0, 1] space)
        labels_offset = 1 / (2 * num_labels)
        # transform evenly spaced values, need to do it this way since x_label_transform is a 2D transform
        labels_x = iter(
            label_transform.transform(
                np.stack(
                    [
                        np.linspace(
                            labels_offset, 1 - labels_offset, num_labels, endpoint=True
                        ),
                        np.ones(num_labels),
                    ]
                ).T.reshape(-1, 2)
            )[:, 0]
        )

        kwargs_default = {
            "color": "black",
            "zorder": 1.3,
            "label": "total",
            "lw": 1.2,
        }
        kwargs = update_kwargs(kwargs_default, kwargs_curve_total)

        ax.plot(self.pdf_set.x, values_total, **kwargs)

        gb_data = self._datasets.index.set_index("id_dataset").groupby(
            self.grouper, dropna=False
        )

        i_curves = 0
        i_curves_background = 0

        for label, l2 in values.iterrows():

            group_label: str = label[1]

            data = gb_data.get_group(group_label)

            A_symbols = {
                "A1_sym": nucleus_to_latex(
                    A=data.iloc[0]["A1"], Z=data.iloc[0]["Z1"], superscript=True
                ),
                "A2_sym": nucleus_to_latex(
                    A=data.iloc[0]["A2"], Z=data.iloc[0]["Z2"], superscript=True
                ),
                "A_lighter_sym": nucleus_to_latex(
                    A=data.iloc[0]["A_lighter"],
                    Z=data.iloc[0]["Z_lighter"],
                    superscript=True,
                ),
                "A_heavier_sym": nucleus_to_latex(
                    A=data.iloc[0]["A_heavier"],
                    Z=data.iloc[0]["Z_heavier"],
                    superscript=True,
                ),
            }

            group_label_annotation: str = (
                group_label
                if group_label_format is None
                else group_label_format.format(
                    **data.iloc[0].to_dict(),
                    **A_symbols,
                )
            )

            if (
                group_highlights is None
                or group_highlights is not None
                and group_label in group_highlights
            ):
                kwargs_default = {
                    "zorder": 1.2,
                    "lw": 1.2,
                    "label": group_label_annotation,
                }
                kwargs = update_kwargs(kwargs_default, kwargs_curves, i_curves)
                i_curves += 1
            else:
                kwargs_default = {
                    "color": "silver",
                    "zorder": 1.1,
                    "lw": 0.5,
                }
                kwargs = update_kwargs(
                    kwargs_default, kwargs_curves_background, i_curves_background
                )
                i_curves_background += 1

            l = ax.plot(self.pdf_set.x, l2, **kwargs)[0]

            if group_label_style == "curve" and (
                group_highlights is None
                or group_highlights is not None
                and group_label in group_highlights
            ):
                label_x = next(labels_x)
                label_y = np.interp(label_x, self.pdf_set.x, l2)

                kwargs_default = {
                    "text": group_label_annotation,
                    "xy": (label_x, label_y),
                    "zorder": 3,
                    "ha": "center",
                    "va": "center",
                    "color": l.get_color(),
                    "bbox": dict(
                        facecolor=(1, 1, 1),
                        alpha=0.8,
                        lw=0,
                        boxstyle="round,pad=0.1,rounding_size=0.2",
                    ),
                    "fontsize": "x-small",
                }
                kwargs = update_kwargs(kwargs_default, kwargs_labels)

                ax.annotate(**kwargs)

        if group_label_style == "legend":
            ax.legend()


class L2Sensitivity(Chi2PDFCorrelation):

    @override
    @staticmethod
    def calculate(
        pdf_data: pd.DataFrame,
        chi2_data: pd.DataFrame,
        chi2_min: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Calculates the L2 sensitivities of the PDFs and the χ² as shown in arXiv:2306.03918 section 2.A. Both `pdf_data` and `chi2_data` must have an index level `id_eigendirection` containing the eigenvectors with positive and negative directions alternating and in the same order in each index. The other index levels and the columns can be arbitrary and will be preserved in the output DataFrame.

        Parameters
        ----------
        pdf_data : pd.DataFrame
            Values of the PDFs. Index: `(..., id_eigendirection, ...)`, where the `id_eigendirection` level contains the eigenvectors with positive and negative directions alternating and in the same order as in the `id_eigendirection` index level of `chi_data`

        chi2_data : pd.DataFrame
            Values of the chi-squared function. Index: `(..., id_eigendirection, ...)`, where the `id_eigendirection` level contains the eigenvectors with positive and negative directions alternating and in the same order as in the `id_eigendirection` index level of `pdf_data`

        Returns
        -------
        pd.DataFrame
            Values of the L2 sensitivities, with index and columns levels being the cartesian product of the ones in `pdf_data` and `chi2_data`
        """

        # indexer that selects two consecutive rows
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)

        # DataFrame containing the difference of the positive and negative eigendirection PDF as columns
        pdf_diff = cast(
            pd.DataFrame,
            pdf_data.stack(
                pdf_data.columns.names,  # pyright: ignore[reportArgumentType]
                future_stack=True,
            )
            .unstack("id_eigendirection")
            .T.rolling(  # group by windows of two consecutive rows
                indexer,
                step=2,
                min_periods=1,
            )
            .agg(lambda x: x.iloc[0] - x.iloc[1])  # and subtract each two rows
            .T,
        )

        # Series containing the PDF uncertainties
        delta_pdf = 0.5 * np.sqrt((pdf_diff**2).sum(axis=1))

        # same for chi2_data
        chi2_diff = (
            chi2_data.stack(
                chi2_data.columns.names,  # pyright: ignore[reportArgumentType]
                future_stack=True,
            )
            .unstack("id_eigendirection")
            .T.rolling(
                indexer,
                step=2,
                min_periods=1,
            )
            .agg(lambda x: x.iloc[0] - x.iloc[1])
            .T
        )

        # merge pdf_diff/delta_pdf and chi2_diff
        diffs = pd.merge(pdf_diff.div(4 * delta_pdf, axis=0), chi2_diff, how="cross").T

        # multiply PDF and chi2 eigendirections and sum over eigendirections
        result = (
            diffs.filter(regex="[0-9]+_x", axis=0).rename(lambda x: x.replace("_x", ""))
            * diffs.filter(regex="[0-9]+_y", axis=0).rename(
                lambda x: x.replace("_y", "")
            )
        ).sum(axis=0)

        # set the indices of pdf_data and chi2_data
        pdf_index = (
            pdf_diff.index.levels
            if isinstance(pdf_diff.index, pd.MultiIndex)
            else [pdf_diff.index]
        )
        chi2_index = (
            chi2_diff.index.levels
            if isinstance(chi2_diff.index, pd.MultiIndex)
            else [chi2_diff.index]
        )
        result.index = pd.MultiIndex.from_product(
            [*pdf_index, *chi2_index],  # pyright: ignore[reportArgumentType]
            names=pdf_diff.index.names + chi2_diff.index.names,
        )

        result = cast(
            pd.DataFrame,
            result.unstack(
                pdf_data.columns.names  # pyright: ignore[reportArgumentType]
            ).unstack(
                chi2_data.columns.names  # pyright: ignore[reportArgumentType]
            ),
        )

        return result


class CosPhi(Chi2PDFCorrelation):

    @override
    @staticmethod
    def calculate(
        pdf_data: pd.DataFrame,
        chi2_data: pd.DataFrame,
        chi2_min: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Calculates cos(ϕ) of the PDFs and the χ² as shown in the nCTEQ15 paper (arXiv:1509.00792) eq. 4.2. Both `pdf_data` and `chi2_data` must have an index level `id_eigendirection` containing the eigenvectors with positive and negative directions alternating and in the same order in each index. The other index levels and the columns can be arbitrary and will be preserved in the output DataFrame.

        Parameters
        ----------
        pdf_data : pd.DataFrame
            Values of the PDFs. Index: `(..., id_eigendirection, ...)`, where the `id_eigendirection` level contains the eigenvectors with positive and negative directions alternating and in the same order as in the `id_eigendirection` index level of `chi_data`

        chi2_data : pd.DataFrame
            Values of the chi-squared function. Index: `(..., id_eigendirection, ...)`, where the `id_eigendirection` level contains the eigenvectors with positive and negative directions alternating and in the same order as in the `id_eigendirection` index level of `pdf_data`

        Returns
        -------
        pd.DataFrame
            Values of the L2 sensitivities, with index and columns levels being the cartesian product of the ones in `pdf_data` and `chi2_data`
        """

        # indexer that selects two consecutive rows
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)

        # DataFrame containing the difference of the positive and negative eigendirection PDF as columns
        pdf_diff = cast(
            pd.DataFrame,
            pdf_data.stack(
                pdf_data.columns.names,  # pyright: ignore[reportArgumentType]
                future_stack=True,
            )
            .unstack("id_eigendirection")
            .T.rolling(  # group by windows of two consecutive rows
                indexer,
                step=2,
                min_periods=1,
            )
            .agg(lambda x: x.iloc[0] - x.iloc[1])  # and subtract each two rows
            .T,
        )

        # Series containing the PDF uncertainties
        delta_pdf = 0.5 * np.sqrt((pdf_diff**2).sum(axis=1))

        # same for chi2_data
        chi2_diff = cast(
            pd.DataFrame,
            chi2_data.stack(
                chi2_data.columns.names,  # pyright: ignore[reportArgumentType]
                future_stack=True,
            )
            .unstack("id_eigendirection")
            .T.rolling(
                indexer,
                step=2,
                min_periods=1,
            )
            .agg(lambda x: x.iloc[0] - x.iloc[1])
            .T,
        )

        # Series containing the chi2 uncertainties
        delta_chi2 = 0.5 * np.sqrt((chi2_diff**2).sum(axis=1))

        # merge pdf_diff/delta_pdf and chi2_diff
        diffs = pd.merge(
            pdf_diff.div(4 * delta_pdf, axis=0),
            chi2_diff.div(delta_chi2, axis=0),
            how="cross",
        ).T

        # multiply PDF and chi2 eigendirections and sum over eigendirections
        result = (
            diffs.filter(regex="[0-9]+_x", axis=0).rename(lambda x: x.replace("_x", ""))
            * diffs.filter(regex="[0-9]+_y", axis=0).rename(
                lambda x: x.replace("_y", "")
            )
        ).sum(axis=0)

        # set the indices of pdf_data and chi2_data
        pdf_index = (
            pdf_diff.index.levels
            if isinstance(pdf_diff.index, pd.MultiIndex)
            else [pdf_diff.index]
        )
        chi2_index = (
            chi2_diff.index.levels
            if isinstance(chi2_diff.index, pd.MultiIndex)
            else [chi2_diff.index]
        )
        result.index = pd.MultiIndex.from_product(
            [*pdf_index, *chi2_index],
            names=pdf_diff.index.names + chi2_diff.index.names,
        )

        result = cast(
            pd.DataFrame,
            result.unstack(
                pdf_data.columns.names  # pyright: ignore[reportArgumentType]
            ).unstack(
                chi2_data.columns.names  # pyright: ignore[reportArgumentType]
            ),
        )

        return result


class DeltaChi2Eff(Chi2PDFCorrelation):

    @override
    @staticmethod
    def calculate(
        pdf_data: pd.DataFrame,
        chi2_data: pd.DataFrame,
        chi2_min: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Calculates Δχ²_eff of the PDFs and the χ² as shown in the nCTEQ15 paper (arXiv:1509.00792) eq. 4.3. Both `pdf_data` and `chi2_data` must have an index level `id_eigendirection` containing the eigenvectors with positive and negative directions alternating and in the same order in each index. The other index levels and the columns can be arbitrary and will be preserved in the output DataFrame.

        Parameters
        ----------
        pdf_data : pd.DataFrame
            Values of the PDFs. Index: `(..., id_eigendirection, ...)`, where the `id_eigendirection` level contains the eigenvectors with positive and negative directions alternating and in the same order as in the `id_eigendirection` index level of `chi_data`

        chi2_data : pd.DataFrame
            Values of the chi-squared function. Index: `(..., id_eigendirection, ...)`, where the `id_eigendirection` level contains the eigenvectors with positive and negative directions alternating and in the same order as in the `id_eigendirection` index level of `pdf_data`

        Returns
        -------
        pd.DataFrame
            Values of the L2 sensitivities, with index and columns levels being the cartesian product of the ones in `pdf_data` and `chi2_data`
        """

        if chi2_min is None:
            raise ValueError("chi2_min is required to calculate delta chi2 eff")

        # indexer that selects two consecutive rows
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)

        # DataFrame containing the difference of the positive and negative eigendirection PDF as columns
        pdf_diff = cast(
            pd.DataFrame,
            pdf_data.stack(
                pdf_data.columns.names,  # pyright: ignore[reportArgumentType]
                future_stack=True,
            )
            .unstack("id_eigendirection")
            .T.rolling(  # group by windows of two consecutive rows
                indexer,
                step=2,
                min_periods=1,
            )
            .agg(lambda x: x.iloc[0] - x.iloc[1])  # and subtract each two rows
            .T
            ** 2,
        )

        # Series containing the PDF uncertainties
        delta_pdf = pdf_diff.sum(axis=1)

        # same for chi2_data
        chi2_diff = cast(
            pd.DataFrame,
            chi2_data.sub(chi2_min)
            .abs()
            .stack(
                chi2_data.columns.names,  # pyright: ignore[reportArgumentType]
                future_stack=True,
            )
            .unstack("id_eigendirection")
            .T.rolling(
                indexer,
                step=2,
                min_periods=1,
            )
            # .agg(lambda x: x.iloc[0] + x.iloc[1])
            .sum().T,
        )

        # merge pdf_diff/delta_pdf and chi2_diff
        diffs = pd.merge(pdf_diff.div(2 * delta_pdf, axis=0), chi2_diff, how="cross").T

        # multiply PDF and chi2 eigendirections and sum over eigendirections
        result = (
            diffs.filter(regex="[0-9]+_x", axis=0).rename(lambda x: x.replace("_x", ""))
            * diffs.filter(regex="[0-9]+_y", axis=0).rename(
                lambda x: x.replace("_y", "")
            )
        ).sum(axis=0)

        # set the indices of pdf_data and chi2_data
        pdf_index = (
            pdf_diff.index.levels
            if isinstance(pdf_diff.index, pd.MultiIndex)
            else [pdf_diff.index]
        )
        chi2_index = (
            chi2_diff.index.levels
            if isinstance(chi2_diff.index, pd.MultiIndex)
            else [chi2_diff.index]
        )
        result.index = pd.MultiIndex.from_product(
            [*pdf_index, *chi2_index],
            names=pdf_diff.index.names + chi2_diff.index.names,
        )

        result = cast(
            pd.DataFrame,
            result.unstack(
                pdf_data.columns.names  # pyright: ignore[reportArgumentType]
            ).unstack(
                chi2_data.columns.names  # pyright: ignore[reportArgumentType]
            ),
        )

        return result
