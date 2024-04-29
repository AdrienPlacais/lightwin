"""Create some generic evaluators for :class:`.SimulationOutput.`"""

from collections.abc import Iterable
from typing import Any, override

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

    _y_quantity = "pow_lost"
    _fignum = 101
    _constant_limits = True

    def __init__(
        self,
        max_percentage_increase: float,
        reference: SimulationOutput,
        plotter: object | None = None,
    ) -> None:
        """Instantiate with a reference simulation output."""
        super().__init__(reference, plotter)

        # First point is sometimes very high
        self._ref_ydata = self.post_treat(self._ref_ydata)

        self._max_percentage_increase = max_percentage_increase
        self._max_loss = (
            1e-2 * max_percentage_increase * np.sum(self._ref_ydata)
        )

    def __repr__(self) -> str:
        """Give a short description of what this class does."""
        return (
            self._markdown
            + f"< {self._max_loss:.2f}W "
            + f"(+{self._max_percentage_increase:.2f}%)"
        )

    @override
    def post_treat(self, ydata: Iterable[float]) -> npt.NDArray[np.float64]:
        """Set the first point to 0 (sometimes it is inf in TW)."""
        assert isinstance(ydata, np.ndarray)
        if ydata.ndim == 1:
            ydata[0] = 0.0
            return ydata
        if ydata.ndim == 2:
            ydata[:, 0] = 0.0
            return ydata
        raise ValueError

    def run(self, *simulation_outputs, **kwargs) -> list[bool]:
        """Assert that lost power is lower than maximum."""
        ydata = self.get(*simulation_outputs, **kwargs)
        post_treated = self.post_treat(ydata)
        sums = np.sum(post_treated, axis=1)
        tests = [x <= self._max_loss for x in sums]
        return tests


class LongitudinalEmittance(ISimulationOutputEvaluator):
    """Check that the longitudinal emittance is acceptable."""

    _y_quantity = "eps_phiw"
    _to_deg = False
    _fignum = 110
    _constant_limits = True

    def __init__(
        self,
        max_percentage_rel_increase: float,
        reference: SimulationOutput,
        plotter: object | None = None,
    ) -> None:
        """Instantiate with a reference simulation output."""
        super().__init__(reference, plotter)

        self._ref_ydata = self._ref_ydata[0]
        self._max_percentage_rel_increase = max_percentage_rel_increase

    @property
    @override
    def _markdown(self) -> str:
        """Give the proper markdown."""
        return r"$\Delta\epsilon_{\phi W} / \epsilon_{\phi W}$ (ref $z=0$) [%]"

    def __repr__(self) -> str:
        """Give a short description of what this class does."""
        return (
            r"Relative increase of $\epsilon_{\phi W} < "
            f"{self._max_percentage_rel_increase:0.2f}$%"
        )

    @override
    def post_treat(self, ydata: Iterable[float]) -> npt.NDArray[np.float64]:
        """Compute relative diff w.r.t. reference value @ z = 0."""
        assert isinstance(ydata, np.ndarray)
        if ydata.ndim in (1, 2):
            post_treated = (ydata - self._ref_ydata) / self._ref_ydata
            assert isinstance(ydata, np.ndarray)
            return post_treated
        raise ValueError

    def run(
        self,
        *simulation_outputs,
        plot_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> Iterable[np.bool_]:
        """Assert that longitudinal emittance does not grow too much."""
        ydata = self.get(*simulation_outputs, **kwargs)
        post_treated = self.post_treat(ydata)
        upper_limit = 1e2 * self._max_percentage_rel_increase
        tests = np.all(post_treated < upper_limit, axis=0)

        if isinstance(tests, bool):
            tests = np.array([tests])

        if plot_kwargs is not None:
            upper_limits = [(upper_limit,) for _ in simulation_outputs]
            limits_kw = {"label": ("Upper limit",)}
            self.plot(
                *simulation_outputs,
                limits=upper_limits,
                limits_kw=limits_kw,
                **plot_kwargs,
            )
        return tests
