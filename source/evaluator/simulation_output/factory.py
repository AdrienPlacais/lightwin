#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Set an object that can create :class:`.SimulationOutputEvaluator`.

.. todo::
    maybe create a mother class more generic, also for FaultScenarioEvaluator?

"""
from typing import Any, Sequence

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from evaluator.list_of_simulation_output_evaluators import (
    ListOfSimulationOutputEvaluators,
)
from evaluator.simulation_output.presets import (
    SIMULATION_OUTPUT_EVALUATOR_PRESETS,
)
from evaluator.simulation_output.simulation_output_evaluator import (
    SimulationOutputEvaluator,
)


class SimulationOutputEvaluatorFactory:
    """Hold methods to create and run :class:`.SimulationOutputEvaluator`."""

    def __init__(self, ref_simulation_output: SimulationOutput) -> None:
        """Instantiate the factory."""
        self.ref_simulation_output = ref_simulation_output
        return

    def run_from_kw(self, **evaluator_kw: Any) -> SimulationOutputEvaluator:
        """Create an evaluator."""
        evaluator = SimulationOutputEvaluator(
            ref_simulation_output=self.ref_simulation_output,
            **evaluator_kw,
        )
        return evaluator

    def run_from_preset(self, preset_name: str) -> SimulationOutputEvaluator:
        """Create an evaluator from a preset."""
        assert preset_name in SIMULATION_OUTPUT_EVALUATOR_PRESETS, \
            f"{preset_name = } was not found in evaluator.simulation_output."\
            "SIMULATION_OUTPUT_EVALUATOR_PRESETS."
        evaluator_kw = SIMULATION_OUTPUT_EVALUATOR_PRESETS[preset_name]
        return self.run_from_kw(**evaluator_kw)

    def run_all(self,
                presets_names: Sequence[str],
                evaluators_kw: Sequence[dict[str, Any]]
                ) -> ListOfSimulationOutputEvaluators:
        """Create all the simulation output evaluators."""
        evaluators_from_presets = [self.run_from_preset(preset_name)
                                   for preset_name in presets_names]
        evaluators_from_kw = [self.run_from_kw(**evaluator_kw)
                              for evaluator_kw in evaluators_kw]
        all_evaluators = evaluators_from_presets + evaluators_from_kw
        return ListOfSimulationOutputEvaluators(all_evaluators)
