"""Create some generic evaluators for :class:`.SimulationOutput.`"""

from collections.abc import Iterable
from typing import override

import numpy as np
import numpy.typing as npt

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from new_evaluator.simulation_output.i_simulation_output_evaluator import (
    ISimulationOutputEvaluator,
)


class PowerLoss(ISimulationOutputEvaluator):
    """Check that the power loss is acceptable."""

    _quantity = "pow_lost"
    _fignum = 101

    def __init__(
        self,
        max_percentage_increase: float,
        reference: SimulationOutput,
        plotter: object | None = None,
    ) -> None:
        """Instantiate with a reference simulation output."""
        super().__init__(reference, plotter)

        # First point is sometimes very high
        self._reference_data = self.post_treat(self._reference_data)

        self._max_percentage_increase = max_percentage_increase
        self._max_loss = (
            1e-2 * max_percentage_increase * np.sum(self._reference_data)
        )

    def __repr__(self) -> str:
        """Give a short description of what this class does."""
        return (
            self._markdown
            + f"< {self._max_loss:.2f}W "
            + f"(+{self._max_percentage_increase:.2f}%)"
        )

    @override
    def post_treat(self, data: Iterable[float]) -> npt.NDArray[np.float32]:
        """Set the first point to 0 (sometimes it is inf in TW)."""
        assert isinstance(data, np.ndarray)
        if data.ndim == 1:
            data[0] = 0.0
            return data
        if data.ndim == 2:
            data[:, 0] = 0.0
            return data
        raise ValueError

    def run(self, *simulation_outputs, **kwargs) -> list[bool]:
        """Assert that lost power is lower than maximum."""
        data = self.get(*simulation_outputs, **kwargs)
        post_treated = self.post_treat(data)
        sums = np.sum(post_treated, axis=1)
        tests = [x <= self._max_loss for x in sums]
        return tests


class LongitudinalEmittance(ISimulationOutputEvaluator):
    """Check that the longitudinal emittance is acceptable."""

    _quantity = "eps_phiw"
    _to_deg = False
    _fignum = 110

    def __init__(
        self,
        max_percentage_rel_increase: float,
        reference: SimulationOutput,
        plotter: object | None = None,
    ) -> None:
        """Instantiate with a reference simulation output."""
        super().__init__(reference, plotter)

        self._reference_data = self._reference_data[0]
        self._max_percentage_rel_increase = max_percentage_rel_increase

    @property
    @override
    def _markdown(self) -> str:
        """Give the proper markdown."""
        return r"$\Delta\epsilon_{\phi W} / \epsilon_{\phi W}$ (ref $z=0$) [%]"

    def __repr__(self) -> str:
        """Give a short description of what this class does."""
        return (
            "Relative increase of eps_phiw < "
            f"{self._max_percentage_rel_increase:0.2f}%"
        )

    @override
    def post_treat(self, data: Iterable[float]) -> npt.NDArray[np.float32]:
        """Compute relative diff w.r.t. reference value @ z = 0."""
        assert isinstance(data, np.ndarray)
        if data.ndim in (1, 2):
            post_treated = (data - self._reference_data) / self._reference_data
            assert isinstance(data, np.ndarray)
            return post_treated
        raise ValueError

    def run(self, *simulation_outputs, **kwargs) -> Iterable[np.bool_]:
        """Assert that longitudinal emittance does not grow too much."""
        data = self.get(*simulation_outputs, **kwargs)
        post_treated = self.post_treat(data)

        tests = np.all(
            post_treated < 1e2 * self._max_percentage_rel_increase, axis=0
        )
        if isinstance(tests, bool):
            tests = np.array([tests])
        return tests
