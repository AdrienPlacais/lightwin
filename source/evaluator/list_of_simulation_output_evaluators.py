#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:07:54 2023.

@author: placais

In this module we define `ListOfSimulationOutputEvaluators`, to regroup several
`SimulationOutputEvaluator`s.

We also define some factory functions to facilitate their creation.

"""
import logging

from core.elements import _Element
from beam_calculation.output import SimulationOutput
from evaluator.simulation_output_evaluator import SimulationOutputEvaluator
from evaluator.simulation_output_evaluator_presets import (
    PRESETS,
    preset_to_evaluate_fit_once_it_is_over,
)


class ListOfSimulationOutputEvaluators(list):
    """
    A list of `SimulationOutputEvaluator`s.

    """

    def __init__(self, evaluators: list[SimulationOutputEvaluator]) -> None:
        """Create the objects (factory)."""
        super().__init__(evaluators)

    def run(self, *simulation_outputs: SimulationOutput
            ) -> list[bool | float | None]:
        """Call `run` method of every `SimulationOutputEvaluator`."""
        for evaluator in self:
            logging.info(f"{evaluator}")
            for simulation_output in simulation_outputs:
                logging.info(evaluator.run(simulation_output))


def factory_simulation_output_evaluators_from_presets(
    *evaluator_names: str,
    ref_simulation_output: SimulationOutput | None = None
) -> ListOfSimulationOutputEvaluators:
    """Create the `ListOfSimulationOutputEvaluators` using PRESETS."""
    all_kwargs = [PRESETS[name] for name in evaluator_names]

    evaluators = [SimulationOutputEvaluator(
        ref_simulation_output=ref_simulation_output,
        **kwargs)
                  for kwargs in all_kwargs]
    list_of_evaluators = ListOfSimulationOutputEvaluators(evaluators)
    return list_of_evaluators


def factory_simulation_output_evaluators_after_a_fit(
        ref_simulation_output: SimulationOutput,
        elements: tuple[str | _Element]) -> ListOfSimulationOutputEvaluators:
    """Create a `ListOfSimulationOutputEvaluators` @ different elts exits."""
    all_kwargs = [
        preset_to_evaluate_fit_once_it_is_over(quantity, elt, error,
                                               ref_simulation_output)
        for elt, error in zip([], [])
        for quantity in []
    ]
    evaluators = [SimulationOutputEvaluator(**kwargs) for kwargs in all_kwargs]
    list_of_evaluators = ListOfSimulationOutputEvaluators(evaluators)
    return list_of_evaluators
