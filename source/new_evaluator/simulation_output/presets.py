"""Create some generic evaluators for :class:`.SimulationOutput.`"""

from collections.abc import Iterable, Sequence
from typing import Any, override

import numpy as np
import numpy.typing as npt

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.list_of_elements.list_of_elements import ListOfElements
from new_evaluator.simulation_output.i_simulation_output_evaluator import (
    ISimulationOutputEvaluator,
)
from plotter.pd_plotter import PandasPlotter


class PowerLoss(ISimulationOutputEvaluator):
    """Check that the power loss is acceptable."""

    _y_quantity = "pow_lost"
    _fignum = 101
    _constant_limits = True

    def __init__(
        self,
        max_percentage_increase: float,
        reference: SimulationOutput,
        plotter: PandasPlotter = PandasPlotter(),
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
        raise ValueError(f"{ydata = } not understood.")

    def evaluate(
        self,
        *simulation_outputs,
        elts: Sequence[ListOfElements] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[list[bool], npt.NDArray[np.float64]]:
        """Assert that lost power is lower than maximum."""
        all_post_treated = self.post_treat(
            self.get(*simulation_outputs, **kwargs)
        )
        tests: list[bool] = []

        if plot_kwargs is None:
            plot_kwargs = {}

        used_for_eval = np.sum(all_post_treated, axis=0)
        for data in used_for_eval:
            test = self._evaluate_single(
                data,
                lower_limit=np.NaN,
                upper_limit=self._max_loss,
                **kwargs,
            )
            tests.append(test)

        self.plot(
            all_post_treated,
            elts,
            lower_limits=None,
            upper_limits=[self._max_loss for _ in simulation_outputs],
            **plot_kwargs,
            **kwargs,
        )
        return tests, used_for_eval


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
        plotter: PandasPlotter = PandasPlotter(),
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
            f"{self._max_percentage_rel_increase:0.4f}$%"
        )

    @override
    def post_treat(self, ydata: Iterable[float]) -> npt.NDArray[np.float64]:
        """Compute relative diff w.r.t. reference value @ z = 0."""
        assert isinstance(ydata, np.ndarray)
        if ydata.ndim in (1, 2):
            post_treated = 1e2 * (ydata - self._ref_ydata) / self._ref_ydata
            assert isinstance(ydata, np.ndarray)
            return post_treated
        raise ValueError

    def evaluate(
        self,
        *simulation_outputs,
        elts: Sequence[ListOfElements] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[list[bool], npt.NDArray[np.float64]]:
        """Assert that longitudinal emittance does not grow too much."""
        all_post_treated = self.post_treat(
            self.get(*simulation_outputs, **kwargs)
        )
        tests: list[bool] = []

        if plot_kwargs is None:
            plot_kwargs = {}

        for post_treated in all_post_treated.T:
            test = self._evaluate_single(
                post_treated,
                lower_limit=np.NaN,
                upper_limit=self._max_percentage_rel_increase,
                **kwargs,
            )
            tests.append(test)

        self.plot(
            all_post_treated,
            elts,
            lower_limits=None,
            upper_limits=[
                self._max_percentage_rel_increase for _ in simulation_outputs
            ],
            **plot_kwargs,
            **kwargs,
        )
        return tests, all_post_treated[-1, :]


class TransverseMismatchFactor(ISimulationOutputEvaluator):
    """Check that mismatch factor at end is not too high."""

    _y_quantity = "mismatch_factor_t"
    _to_deg = False
    _fignum = 111
    _constant_limits = True

    def __init__(
        self,
        max_mismatch: float,
        reference: SimulationOutput,
        plotter: PandasPlotter = PandasPlotter(),
    ) -> None:
        """Instantiate with a reference simulation output."""
        super().__init__(reference, plotter)

        self._ref_ydata = [0.0, 0.0]
        self._max_mismatch = max_mismatch

    def __repr__(self) -> str:
        """Give a short description of what this class does."""
        return (
            f"At end of linac, {self._markdown} $< "
            f"{self._max_mismatch:0.2f}$"
        )

    @override
    def _getter(
        self, simulation_output: SimulationOutput, quantity: str
    ) -> npt.NDArray[np.float64]:
        """Call the ``get`` method with proper kwarguments.

        Also skip calculation with reference accelerator, as mismatch will not
        be defined.

        """
        data = simulation_output.get(
            quantity,
            to_deg=self._to_deg,
            elt=self._elt,
            pos=self._pos,
            **self._get_kwargs,
        )
        if data.ndim == 0 or data is None:
            if simulation_output.out_path.parent.stem == "000000_ref":
                self._dump_no_numerical_data_to_plot = True
                return np.full_like(self._ref_xdata, np.NaN)
            return self._default_dummy(quantity)
        return data

    @override
    def post_treat(self, ydata: Iterable[float]) -> npt.NDArray[np.float64]:
        """Return the unaltered ``ydata``."""
        assert isinstance(ydata, np.ndarray)
        return ydata

    def evaluate(
        self,
        *simulation_outputs,
        elts: Sequence[ListOfElements] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[list[bool], npt.NDArray[np.float64]]:
        """Assert that longitudinal emittance does not grow too much."""
        all_post_treated = self.post_treat(
            self.get(*simulation_outputs, **kwargs)
        )
        tests: list[bool] = []

        if plot_kwargs is None:
            plot_kwargs = {}

        used_for_eval = all_post_treated[-1, :]
        for data in used_for_eval:
            test = self._evaluate_single(
                data,
                lower_limit=np.NaN,
                upper_limit=self._max_mismatch,
                **kwargs,
            )
            tests.append(test)

        self.plot(
            all_post_treated,
            elts,
            lower_limits=None,
            upper_limits=[self._max_mismatch for _ in simulation_outputs],
            **plot_kwargs,
            **kwargs,
        )
        return tests, used_for_eval


class LongitudinalMismatchFactor(ISimulationOutputEvaluator):
    """Check that mismatch factor at end is not too high."""

    _y_quantity = "mismatch_factor_zdelta"
    _to_deg = False
    _fignum = 112
    _constant_limits = True

    def __init__(
        self,
        max_mismatch: float,
        reference: SimulationOutput,
        plotter: PandasPlotter = PandasPlotter(),
    ) -> None:
        """Instantiate with a reference simulation output."""
        super().__init__(reference, plotter)

        self._ref_ydata = [0.0, 0.0]
        self._max_mismatch = max_mismatch

    def __repr__(self) -> str:
        """Give a short description of what this class does."""
        return (
            f"At end of linac, {self._markdown} $< "
            f"{self._max_mismatch:0.2f}$"
        )

    @override
    def post_treat(self, ydata: Iterable[float]) -> npt.NDArray[np.float64]:
        """Return the unaltered ``ydata``."""
        assert isinstance(ydata, np.ndarray)
        return ydata

    @override
    def _getter(
        self, simulation_output: SimulationOutput, quantity: str
    ) -> npt.NDArray[np.float64]:
        """Call the ``get`` method with proper kwarguments.

        Also skip calculation with reference accelerator, as mismatch will not
        be defined.

        """
        data = simulation_output.get(
            quantity,
            to_deg=self._to_deg,
            elt=self._elt,
            pos=self._pos,
            **self._get_kwargs,
        )
        if data.ndim == 0 or data is None:
            if simulation_output.out_path.parent.stem == "000000_ref":
                self._dump_no_numerical_data_to_plot = True
                return np.full_like(self._ref_xdata, np.NaN)
            return self._default_dummy(quantity)
        return data

    def evaluate(
        self,
        *simulation_outputs,
        elts: Sequence[ListOfElements] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[list[bool], npt.NDArray[np.float64]]:
        """Assert that longitudinal emittance does not grow too much."""
        all_post_treated = self.post_treat(
            self.get(*simulation_outputs, **kwargs)
        )
        tests: list[bool] = []

        if plot_kwargs is None:
            plot_kwargs = {}

        used_for_eval = all_post_treated[-1, :]
        for data in used_for_eval:
            test = self._evaluate_single(
                data,
                lower_limit=np.NaN,
                upper_limit=self._max_mismatch,
                **kwargs,
            )
            tests.append(test)

        self.plot(
            all_post_treated,
            elts,
            lower_limits=None,
            upper_limits=[self._max_mismatch for _ in simulation_outputs],
            **plot_kwargs,
            **kwargs,
        )
        return tests, used_for_eval


class SynchronousPhases(ISimulationOutputEvaluator):
    """Check that synchronous phases are within [-90deg, 0deg]."""

    _x_quantity = "elt_idx"
    _y_quantity = "phi_s"
    _to_deg = True
    _fignum = 120
    _constant_limits = True

    def __init__(
        self,
        min_phi_s_deg: float,
        max_phi_s_deg: float,
        reference: SimulationOutput,
        plotter: PandasPlotter = PandasPlotter(),
    ) -> None:
        """Instantiate with a reference simulation output."""
        super().__init__(reference, plotter)

        self._min_phi_s = min_phi_s_deg
        self._max_phi_s = max_phi_s_deg

    def __repr__(self) -> str:
        """Give a short description of what this class does."""
        return (
            f"All {self._markdown} are within [{self._min_phi_s:0.2f}, "
            f"{self._max_phi_s:-.2f}] (deg)"
        )

    def _getter(
        self, simulation_output: SimulationOutput, quantity: str
    ) -> npt.NDArray[np.float64]:
        """Call the ``get`` method with proper kwarguments."""
        data = super()._getter(simulation_output, quantity)
        if quantity != "phi_s":
            return data

        data = [phi_s if phi_s is not None else np.NaN for phi_s in data]
        return np.array(data)

    @override
    def post_treat(self, ydata: Iterable[float]) -> npt.NDArray[np.float64]:
        """Remove the None."""
        assert isinstance(ydata, np.ndarray)
        return ydata

    def evaluate(
        self,
        *simulation_outputs,
        elts: Sequence[ListOfElements] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[list[bool], npt.NDArray[np.float64]]:
        """Assert that longitudinal emittance does not grow too much."""
        all_post_treated = self.post_treat(
            self.get(*simulation_outputs, **kwargs)
        )
        tests: list[bool] = []

        if plot_kwargs is None:
            plot_kwargs = {}

        for data in all_post_treated.T:
            test = self._evaluate_single(
                data,
                lower_limit=self._min_phi_s,
                upper_limit=self._max_phi_s,
                nan_in_data_is_allowed=True,
                **kwargs,
            )
            tests.append(test)

        self.plot(
            all_post_treated,
            elts,
            lower_limits=[self._min_phi_s for _ in simulation_outputs],
            upper_limits=[self._max_phi_s for _ in simulation_outputs],
            keep_nan=True,
            style=["o", "r--", "r:"],
            x_axis=self._x_quantity,
            **plot_kwargs,
            **kwargs,
        )
        return tests, np.array([np.NaN for _ in simulation_outputs])


SIMULATION_OUTPUT_EVALUATORS = {
    "PowerLoss": PowerLoss,
    "LongitudinalEmittance": LongitudinalEmittance,
    "TransverseMismatchFactor": TransverseMismatchFactor,
    "LongitudinalMismatchFactor": LongitudinalMismatchFactor,
    "SynchronousPhases": SynchronousPhases,
}
