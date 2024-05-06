"""Wrap-up creation and execution of :class:`.ISimulationOutputEvaluator`.

.. todo::
    Maybe should inherit from a more generic factory.

"""

from collections.abc import Collection, Sequence
from typing import Any

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.accelerator.accelerator import Accelerator
from new_evaluator.simulation_output.i_simulation_output_evaluator import (
    ISimulationOutputEvaluator,
)
from new_evaluator.simulation_output.presets import (
    SIMULATION_OUTPUT_EVALUATORS,
)
from plotter.i_plotter import IPlotter
from plotter.pd_plotter import PandasPlotter
from util.helper import get_constructors


class SimulationOutputEvaluatorsFactory:
    """Define a class to create and execute multiple evaluators."""

    def __init__(
        self,
        evaluator_kwargs: Collection[dict[str, str | float | bool]],
        user_evaluators: dict[str, type] | None = None,
        plotter: IPlotter = PandasPlotter(),
    ) -> None:
        """Instantiate object with basic attributes.

        Parameters
        ----------
        evaluator_kwargs : Collection[EvaluatorKwargs]
            Dictionaries holding necessary information to instantiate the
            evaluators. The only mandatory key-value pair is "name" of type
            str.
        user_evaluators : dict[str, type] | None, optional
            Additional user-defined evaluators; keys should be in PascalCase,
            values :class:`.ISimulationOutputEvaluator` constructors.
        plotter : IPlotter, optional
            An object used to produce plots. The default is
            :class:`.PandasPlotter`.

        """
        self._plotter = plotter
        self._constructors_n_kwargs = _constructors_n_kwargs(
            evaluator_kwargs, user_evaluators
        )

    def run(
        self,
        accelerators: Sequence[Accelerator],
        beam_solver_id: str,
    ) -> list[ISimulationOutputEvaluator]:
        """Instantiate all the evaluators."""
        reference = accelerators[0].simulation_outputs[beam_solver_id]
        evaluators = self._instantiate_evaluators(reference)
        return evaluators

    def _instantiate_evaluators(
        self, reference: SimulationOutput
    ) -> list[ISimulationOutputEvaluator]:
        """Create all the evaluators.

        Parameters
        ----------
        reference : SimulationOutput
            The reference simulation output.

        Returns
        -------
        list[ISimulationOutputEvaluator]
            All the created evaluators.

        """
        evaluators = [
            constructor(reference=reference, plotter=self._plotter, **kwargs)
            for constructor, kwargs in self._constructors_n_kwargs.items()
        ]
        return evaluators

    def batch_evaluate(
        self,
        evaluators: Collection[ISimulationOutputEvaluator],
        accelerators: Collection[Accelerator],
        beam_solver_id: str,
        plot_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[list[bool]]:
        """Evaluate several evaluators."""
        simulation_outputs = [
            x.simulation_outputs[beam_solver_id] for x in accelerators
        ]
        elts = [x.elts for x in accelerators]
        tests = [
            evaluator.evaluate(
                *simulation_outputs,
                elts=elts,
                plot_kwargs=plot_kwargs,
                **kwargs,
            )
            for evaluator in evaluators
        ]
        return tests


def _constructors_n_kwargs(
    evaluator_kwargs: Collection[dict[str, str | float | bool]],
    user_evaluators: dict[str, type] | None = None,
) -> dict[type, dict[str, bool | float | str]]:
    """Take and associate every evaluator class with its kwargs.

    We also remove the "name" key from the kwargs.

    Parameters
    ----------
    evaluator_kwargs : Collection[dict[str, str | float | bool]]
        Dictionaries holding necessary information to instantiate the
        evaluators. The only mandatory key-value pair is "name" of type str.
    user_evaluators : dict[str, type] | None, optional
        Additional user-defined evaluators; keys should be in PascalCase,
        values :class:`.ISimulationOutputEvaluator` constructors.

    Returns
    -------
    dict[type, dict[str | float | bool]]
        Keys are class constructor, values associated keyword arguments.

    """
    evaluator_ids = []
    for kwargs in evaluator_kwargs:
        assert "name" in kwargs
        name = kwargs.pop("name")
        assert isinstance(name, str)
        evaluator_ids.append(name)

    if user_evaluators is None:
        user_evaluators = {}
    evaluator_constructors = user_evaluators | SIMULATION_OUTPUT_EVALUATORS

    constructors = get_constructors(evaluator_ids, evaluator_constructors)

    constructors_n_kwargs = {
        constructor: kwargs
        for constructor, kwargs in zip(
            constructors, evaluator_kwargs, strict=True
        )
    }
    return constructors_n_kwargs
