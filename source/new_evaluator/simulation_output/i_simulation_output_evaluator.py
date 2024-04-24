"""Define the base object for :class:`.SimulationOutput` evaluators."""

import numpy as np

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.elements.element import Element
from new_evaluator.i_evaluator import IEvaluator


class ISimulationOutputEvaluator(IEvaluator):
    """Base class for :class:`.SimulationOutput` evaluations."""

    _to_deg: bool = True
    _elt: str | Element | None = None
    _pos: str | None = None
    _get_kwargs: dict[str, bool | str | None]

    def __init__(
        self, reference: SimulationOutput, plotter: object | None = None
    ) -> None:
        """Instantiate with a reference simulation output."""
        super().__init__(plotter)
        self._reference_data = reference.get(self._quantity)

        if not hasattr(self, "_get_kwargs"):
            self._get_kwargs = {}

    def get(
        self, *simulation_outputs: SimulationOutput, **kwargs
    ) -> np.ndarray:
        """Get the data from the simulation output."""
        data = [
            x.get(
                self._quantity,
                to_deg=self._to_deg,
                elt=self._elt,
                pos=self._pos,
                **self._get_kwargs,
            )
            for x in simulation_outputs
        ]
        return np.column_stack(data)
