"""Define the base object for :class:`.SimulationOutput` evaluators."""

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, final

import numpy as np
import numpy.typing as npt
import pandas as pd

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.elements.element import Element
from core.list_of_elements.list_of_elements import ListOfElements
from new_evaluator.i_evaluator import IEvaluator
from plotter.pd_plotter import PandasPlotter


class ISimulationOutputEvaluator(IEvaluator):
    """Base class for :class:`.SimulationOutput` evaluations."""

    _x_quantity = "z_abs"
    _to_deg: bool = True
    _elt: str | Element | None = None
    _pos: str | None = None
    _get_kwargs: dict[str, bool | str | None]
    _constant_limits: bool
    _dump_no_numerical_data_to_plot: bool = False

    def __init__(
        self, reference: SimulationOutput, plotter: PandasPlotter
    ) -> None:
        """Instantiate with a reference simulation output."""
        super().__init__(plotter)
        if not hasattr(self, "_get_kwargs"):
            self._get_kwargs = {}
        self._ref_xdata = self._getter(reference, self._x_quantity)
        self._n_points = len(self._ref_xdata)
        self._ref_ydata = self._getter(reference, self._y_quantity)

    @final
    def _default_dummy(self, quantity: str) -> npt.NDArray[np.float64]:
        """Give dummy ydata, with expected shape if possible.

        Also, set ``_dump_no_numerical_data_to_plot`` to avoid future pandas
        plotter errors.

        """
        self._dump_no_numerical_data_to_plot = True
        if hasattr(self, "_ref_ydata"):
            logging.error(
                f"{quantity = } was not found in the simulation output. "
                "Maybe the simulation was interrupted? Returning dummy data."
            )
            return np.full_like(self._ref_ydata, np.NaN)
        if hasattr(self, "_ref_xdata"):
            logging.error(
                f"Reference {quantity = } was not found in the simulation"
                " output. Maybe the simulation parameters are invalid, or "
                "the BeamCalculator does not produce this data? Returning "
                "dummy data."
            )
            return np.full_like(self._ref_xdata, np.NaN)
        logging.critical(
            f"Reference {quantity = } data was not found and I could not "
            "find fallback array ({self._x_quantity}). Returning a very dummy "
            "array."
        )
        return np.full((10,), np.NaN)

    def _getter(
        self, simulation_output: SimulationOutput, quantity: str
    ) -> npt.NDArray[np.float64]:
        """Call the ``get`` method with proper kwarguments."""
        data = simulation_output.get(
            quantity,
            to_deg=self._to_deg,
            elt=self._elt,
            pos=self._pos,
            **self._get_kwargs,
        )
        if data.ndim == 0 or data is None:
            return self._default_dummy(quantity)
        return data

    def _get_n_interpolate(
        self,
        simulation_output: SimulationOutput,
        interp: bool = True,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """Give ydata from one simulation, with proper number of points."""
        new_ydata = self._getter(simulation_output, self._y_quantity)
        if not interp or len(new_ydata) == self._n_points:
            return new_ydata

        new_xdata = self._getter(simulation_output, self._x_quantity)
        new_ydata = np.interp(self._ref_xdata, new_xdata, new_ydata)
        return new_ydata

    def get(
        self, *simulation_outputs: SimulationOutput, **kwargs
    ) -> npt.NDArray[np.float64]:
        """Get the data from the simulation outputs."""
        y_data = [
            self._get_n_interpolate(x, **kwargs) for x in simulation_outputs
        ]
        return np.column_stack(y_data)

    def plot(
        self,
        post_treated: npt.NDArray[np.float64],
        elts: Sequence[ListOfElements] | None = None,
        png_folders: Sequence[Path] | None = None,
        lower_limits: (
            Sequence[Iterable[float]] | Sequence[float] | None
        ) = None,
        upper_limits: (
            Sequence[Iterable[float]] | Sequence[float] | None
        ) = None,
        keep_nan: bool = False,
        style: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot all the post treated data using ``plotter``."""
        if style is None:
            style = ["-", "r--", "r:"]
        for i, data in enumerate(post_treated.T):
            elements = elts[i] if elts is not None else None

            data_as_dict = {
                "Data": data,
                "Lower limit": (
                    lower_limits[i] if lower_limits is not None else np.NaN
                ),
                "Upper limit": (
                    upper_limits[i] if upper_limits is not None else np.NaN
                ),
            }
            data_as_pd = pd.DataFrame(data_as_dict, index=self._ref_xdata)
            if not keep_nan:
                data_as_pd = data_as_pd.dropna(axis=1)
            axes = self._plot_single(
                data_as_pd,
                elements,
                style=style,
                dump_no_numerical_data_to_plot=self._dump_no_numerical_data_to_plot,
                **kwargs,
            )
            if png_folders is not None:
                self._plotter.save_figure(
                    axes, png_folders[i] / self.png_filename
                )

    @final
    def _evaluate_single(
        self,
        post_treated: npt.NDArray[np.float64],
        lower_limit: npt.NDArray[np.float64] | float = np.NaN,
        upper_limit: npt.NDArray[np.float64] | float = np.NaN,
        nan_in_data_is_allowed: bool = False,
        **kwargs,
    ) -> bool:
        """Check that ``post_treated`` is within limits.

        Parameters
        ----------
        post_treated: npt.NDArray[np.float64]
            Data, already post-treated. If there is ``np.NaN`` in this array,
            we consider that the test if failed.
        lower_limit, upper_limit : npt.NDArray[np.float64] | float, optional
            Min/max value for data. Where it is ``np.NaN``, the test is passed.
        nan_in_data_is_allowed : boolean, optional
            If the test is valid where ``post_treated`` is NaN. Use for example
            with synchronous phases, which is Nan when not in a cavity.

        Returns
        -------
        test : bool
            If the data is always within the given limits.

        """
        is_under_upper = np.full_like(post_treated, True, dtype=bool)
        where = ~np.isnan(upper_limit)
        if nan_in_data_is_allowed:
            where = where & ~np.isnan(post_treated)
        np.less_equal(
            post_treated,
            upper_limit,
            where=where,
            out=is_under_upper,
        )

        is_above_lower = np.full_like(post_treated, True, dtype=bool)
        where = ~np.isnan(lower_limit)
        if nan_in_data_is_allowed:
            where = where & ~np.isnan(post_treated)
        np.greater_equal(
            post_treated,
            lower_limit,
            where=where,
            out=is_above_lower,
        )
        test = np.all(is_above_lower & is_under_upper, axis=0)
        return bool(test)

    @property
    def png_filename(self) -> str:
        """Give a filename for consistent saving of figures."""
        return f"{self._y_quantity}.png"
