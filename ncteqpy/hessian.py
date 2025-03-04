from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import ncteqpy.jaml as jaml
from ncteqpy.plot.hessian import plot_eigenvectors


class Hessian(jaml.YAMLWrapper):

    _minimum: npt.NDArray[np.float64] | None = None

    _parameters_names: list[str] | None = None
    _parameters_indices: pd.Series | None = None
    _parameters_minimum: pd.Series | None = None

    _tolerance: float | None = None
    _hessian: npt.NDArray[np.float64] | None = None
    _eigenvalues: pd.Series | None = None
    _eigenvectors: pd.DataFrame | None = None
    _eigenvectors_rescaled: pd.DataFrame | None = None

    def __init__(
        self,
        path: str | PathLike[str],
        cache_path: str | PathLike[str] = Path("./.jaml_cache/"),
        retain_yaml: bool = False,
    ) -> None:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"{path} is not a file")

        super().__init__(path, cache_path, retain_yaml)

    def _load_parameters(self) -> None:
        pattern = jaml.Pattern({"HessianErrorAnalysis": {"ParamsAtMin": None}})
        yaml = self._load_yaml(pattern)

        assert isinstance(yaml, dict)

        params = cast(
            list[tuple[str, float]],
            jaml.nested_get(yaml, ["HessianErrorAnalysis", "ParamsAtMin"]),
        )

        self._parameters_minimum = pd.Series(dict(params))
        self._parameters_minimum.index.name = "parameter"
        self._parameters_names = self._parameters_minimum.index.tolist()
        self._parameters_indices = pd.Series(
            range(len(self._parameters_names)),
            index=self._parameters_minimum.index,
            name="id_parameter",
        )

    @property
    def parameters_names(self) -> list[str]:
        if self._parameters_names is None or self._yaml_changed():
            self._load_parameters()

        assert self._parameters_names is not None

        return self._parameters_names

    @property
    def parameters_indices(self) -> pd.Series:
        if self._parameters_indices is None or self._yaml_changed():
            self._load_parameters()

        assert self._parameters_indices is not None

        return self._parameters_indices

    @property
    def parameters_minimum(self) -> pd.Series:
        if self._parameters_minimum is None or self._yaml_changed():
            self._load_parameters()

        assert self._parameters_minimum is not None

        return self._parameters_minimum

    @property
    def eigenvalues(self) -> pd.Series:
        if self._eigenvalues is None or self._yaml_changed():
            pattern = jaml.Pattern({"HessianErrorAnalysis": {"Eigenvalues": None}})
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            ev = cast(
                list[float],
                jaml.nested_get(yaml, ["HessianErrorAnalysis", "Eigenvalues"]),
            )

            self._eigenvalues = pd.Series(
                ev, index=pd.RangeIndex(1, len(ev) + 1, name="id_eigenvector")
            )

        return self._eigenvalues

    @property
    def eigenvectors(self) -> pd.DataFrame:
        if self._eigenvectors is None or self._yaml_changed():
            pattern = jaml.Pattern({"HessianErrorAnalysis": {"Eigenvectors": None}})
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            ev = np.array(
                cast(
                    list[list[float]],
                    jaml.nested_get(yaml, ["HessianErrorAnalysis", "Eigenvectors"]),
                )
            ).T

            self._eigenvectors = pd.DataFrame(
                ev, columns=pd.RangeIndex(1, ev.shape[1] + 1, name="id_eigenvector")
            )

        return self._eigenvectors

    @property
    def eigenvectors_rescaled(self) -> pd.DataFrame:
        if self._eigenvectors_rescaled is None or self._yaml_changed():
            pattern = jaml.Pattern(
                {"HessianErrorAnalysis": {"EigenvectorParams": None}}
            )
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            # 0th component is the minimum, and we have to subtract the minimum from each EigenvectorParams endpoint to get the actual rescaled eigenvector endpoint
            ev_params = np.array(
                cast(
                    list[list[float]],
                    jaml.nested_get(
                        yaml, ["HessianErrorAnalysis", "EigenvectorParams"]
                    ),
                )
            )

            self._eigenvectors_rescaled = pd.DataFrame(
                (ev_params[1:] - ev_params[0]).T,
                columns=pd.MultiIndex.from_product(
                    [range(1, (ev_params.shape[0] - 1) // 2 + 1), ["-", "+"]],
                    names=["id_eigenvector", "direction"],
                ),
                index=pd.RangeIndex(0, ev_params.shape[1], name="id_parameter"),
            )
            # print(self._eigenvectors_rescaled)

        assert self._eigenvectors_rescaled is not None

        return self._eigenvectors_rescaled

    def plot_eigenvectors(
        self,
        ax: plt.Axes,
        parameters: tuple[str, str],
        id_eigenvectors: int | Sequence[int] | None = None,
    ) -> None:
        plot_eigenvectors(
            ax=ax,
            eigenvectors=self.eigenvectors_rescaled,
            parameter_indices=tuple(self.parameters_indices.loc[list(parameters)]),
            eigenvector_indices=id_eigenvectors,
            minimum=self.parameters_minimum,
        )
