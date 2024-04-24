"""Define the base object for every evaluator."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from util.dicts_output import markdown


class IEvaluator(ABC):
    """Base class for all evaluators."""

    _x_quantity: str
    _y_quantity: str
    _fignum: int
    _plot_kwargs: dict[str, str | bool | float]
    _axes_num: int = 0

    def __init__(self, plotter: object | None = None) -> None:
        """Instantiate the ``plotter`` object."""
        self._plotter = plotter

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
        as_df = pd.DataFrame(data=post_treated)
        return as_df

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Plot the post treated data using ``plotter``."""
        data = self.get(*args, **kwargs)
        post_treated_data = self.post_treat(data, *args, **kwargs)

        assert self._plotter is not None, "Please provide a plotter object."
        plot_2d = getattr(self._plotter, "plot_2d", None)
        assert isinstance(
            plot_2d, Callable
        ), "plotter object must have a Callable plot_2d method"

        plot_2d(
            post_treated_data,
            y_label=self._markdown,
            fignum=self._fignum,
            axes_num=self._axes_num,
            **self.plot_kwargs,
        )

    def run(self, *args: Any, **kwargs: Any) -> Iterable[bool | np.bool_]:
        """Test if the object(s) under evaluation pass(es) the test."""
        return (True,)
