#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:07:54 2023.

@author: placais

In this module we define `ListOfSimulationOutputEvaluators`, to regroup several
`SimulationOutputEvaluator`s.

"""
import logging

from beam_calculation.output import SimulationOutput
from evaluator.simulation_output_evaluator import (SimulationOutputEvaluator,
                                                   PRESETS)


class ListOfSimulationOutputEvaluators(list):
    """
    A list of `SimulationOutputEvaluator`s.

    """

    def __init__(self, *evaluator_names: str,
                 ref_simulation_output: SimulationOutput | None = None
                 ) -> None:
        """Create the objects (factory)."""
        kwarguments = [PRESETS[name] for name in evaluator_names]

        evaluators = [SimulationOutputEvaluator(
            ref_simulation_output=ref_simulation_output,
            fignum=i + 100,
            **kwarg)
                      for i, kwarg in enumerate(kwarguments)]
        super().__init__(evaluators)

    def run(self, *simulation_outputs: SimulationOutput
            ) -> list[bool | float | None]:
        """Call `run` method of every `SimulationOutputEvaluator`."""
        for evaluator in self:
            logging.info(f"{evaluator}")
            for simulation_output in simulation_outputs:
                logging.info(evaluator.run(simulation_output))
