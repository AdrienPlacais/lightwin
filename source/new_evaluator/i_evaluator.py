"""Define the base object for every evaluator."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.list_of_elements.list_of_elements import ListOfElements
from plotter.pd_plotter import PandasPlotter
from util.dicts_output import markdown


class IEvaluator(ABC):
    """Base class for all evaluators."""

    _x_quantity: str
    _y_quantity: str
    _fignum: int
    _plot_kwargs: dict[str, str | bool | float]
    _axes_index: int = 0

    def __init__(self, plotter: PandasPlotter | None = None) -> None:
        """Instantiate the ``plotter`` object."""
        self._plotter = plotter
        if not hasattr(self, "_plot_kwargs"):
            self._plot_kwargs = {}
        self._ref_xdata: Iterable[float]

    def __str__(self) -> str:
        """Give a detailed description of what this class does."""
        return self.__repr__()

    @abstractmethod
    def __repr__(self) -> str:
        """Give a short description of what this class does."""

    @property
    def _markdown(self) -> str:
        """Give a markdown representation of object, with units."""
        return markdown[self._y_quantity]

    @abstractmethod
    def get(self, *args: Any, **kwargs: Any) -> Iterable[float]:
        """Get the base data."""
        pass

    def post_treat(self, ydata: Iterable[float]) -> Iterable[float]:
        """Perform operations on data. By default, return data as is."""
        return ydata

    def to_pandas(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Give the post-treated data as a pandas dataframe."""
        data = self.get(*args, **kwargs)
        post_treated = self.post_treat(data)
        assert isinstance(post_treated, np.ndarray)
        assert hasattr(self, "_ref_xdata")
        as_df = pd.DataFrame(data=post_treated, index=self._ref_xdata)
        return as_df

    def plot(
        self,
        *args: Any,
        elts: ListOfElements | None = None,
        save_path: Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot the post treated data using ``plotter``."""
        assert isinstance(
            self._plotter, PandasPlotter
        ), "Please provide a plotter object."
        self._plotter.plot(
            self.to_pandas(*args, **kwargs),
            ylabel=self._markdown,
            fignum=self._fignum,
            axes_index=self._axes_index,
            elts=elts,
            save_path=save_path,
            title=str(self),
            **self._plot_kwargs,
        )

    def run(self, *args: Any, **kwargs: Any) -> Iterable[bool | np.bool_]:
        """Test if the object(s) under evaluation pass(es) the test."""
        return (True,)
