"""Define the base object for :class:`.SimulationOutput` evaluators."""

import numpy as np
import numpy.typing as npt

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.elements.element import Element
from new_evaluator.i_evaluator import IEvaluator


class ISimulationOutputEvaluator(IEvaluator):
    """Base class for :class:`.SimulationOutput` evaluations."""

    _x_quantity = "z_abs"
    _to_deg: bool = True
    _elt: str | Element | None = None
    _pos: str | None = None
    _get_kwargs: dict[str, bool | str | None]

    def __init__(
        self, reference: SimulationOutput, plotter: object | None = None
    ) -> None:
        """Instantiate with a reference simulation output."""
        super().__init__(plotter)
        self._ref_xdata = reference.get(self._x_quantity)
        self._n_points = len(self._ref_xdata)
        self._ref_ydata = reference.get(self._y_quantity)

        if not hasattr(self, "_get_kwargs"):
            self._get_kwargs = {}

    def _getter(
        self, simulation_output, quantity: str
    ) -> npt.NDArray[np.float64]:
        """Call the ``get`` method with proper kwarguments."""
        return simulation_output.get(
            quantity,
            to_deg=self._to_deg,
            elt=self._elt,
            pos=self._pos,
            **self._get_kwargs,
        )

    def _get_n_interpolate(
        self,
        simulation_output: SimulationOutput,
        interp: bool = True,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """Give ydata from one simulation, with proper number of points."""
        new_ydata = self._getter(simulation_output, self._y_quantity)
        if not interp or len(new_ydata == self._n_points):
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
