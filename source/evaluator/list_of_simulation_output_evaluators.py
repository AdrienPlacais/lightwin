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

import pandas as pd

from core.elements import _Element
from failures.fault import Fault
from beam_calculation.output import SimulationOutput
from evaluator.simulation_output_evaluator import SimulationOutputEvaluator
from evaluator.simulation_output_evaluator_presets import (
    PRESETS,
    presets_for_fault_scenario_rel_diff_at_some_element,
    presets_for_fault_scenario_rms_over_full_linac
)
from util.dicts_output import markdown
from util.helper import pd_output


class ListOfSimulationOutputEvaluators(list):
    """
    A simple list of `SimulationOutputEvaluator`s.

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


class FaultScenarioSimulationOutputEvaluators:
    """
    A more specific class to evaluate settings found for a `FaultScenario`.

    It allows to have a compact pd.DataFrame where several performance
    indicators at different positions are stored.

    """

    def __init__(self, quantities: tuple[str], faults: list[Fault],
                 simulation_outputs: tuple[SimulationOutputEvaluator],
                 additional_elts: tuple[_Element | str] | None = None
                 ) -> None:
        self.quantities = quantities

        self.elts, self.columns = self._set_evaluation_elements(
            faults, additional_elts)

        ref_simulation_output = simulation_outputs[0]
        self.simulation_output = simulation_outputs[1]

        self.evaluators = \
            self._create_simulation_output_evaluators(ref_simulation_output)

    def _set_evaluation_elements(
        self, faults: list[Fault],
        additional_elts: tuple[_Element | str] | None = None
    ) -> tuple[list[_Element | str], list[str]]:
        """
        Set where the relative difference of `quantities` will be evaluated.

        It is at the end of each compensation zone, plus at the exit of
        additional elements if given.
        Also set `columns` to  ease pandas DataFrame creation.

        """
        elts = [fault.elts[-1] for fault in faults]
        columns = [f"end comp zone ({elt})" for elt in elts]
        if additional_elts is not None:
            elts += list(additional_elts)
            columns += [f"user-defined ({elt})"
                        for elt in list(additional_elts)]
        elts.append('last')
        columns.append("end linac")
        columns.append("RMS [usual units]")
        return elts, columns

    def _create_simulation_output_evaluators(
            self, ref_simulation_output: SimulationOutput
    ) -> list[SimulationOutputEvaluator]:
        """Create the proper `SimulationOutputEvaluator`s."""
        evaluators = []
        for qty in self.quantities:
            for elt in self.elts:
                kwargs = presets_for_fault_scenario_rel_diff_at_some_element(
                    qty, elt, ref_simulation_output)
                evaluators.append(SimulationOutputEvaluator(**kwargs))

            kwargs = presets_for_fault_scenario_rms_over_full_linac(
                qty, ref_simulation_output)
            evaluators.append(SimulationOutputEvaluator(**kwargs))
        return evaluators

    def run(self, output: bool = True) -> pd.DataFrame:
        """Perform all the simulation output evaluations."""
        evaluations = [evaluator.run(self.simulation_output)
                       for evaluator in self.evaluators]
        evaluations = self._to_pandas_dataframe(evaluations)
        if output:
            self._output(evaluations)
        return evaluations

    def _to_pandas_dataframe(self, evaluations: list[float | bool | None],
                             precision: int = 3) -> pd.DataFrame:
        """Convert all the evaluations to a compact pd.DataFrame."""
        lines_labels = [markdown[qty].replace('deg', 'rad')
                        for qty in self.quantities]

        evaluations_nice_output = pd.DataFrame(columns=self.columns,
                                               index=lines_labels)

        formatted_evaluations = self._format_evaluations(evaluations,
                                                         precision)
        n_columns = len(self.columns)
        evaluations_sorted_by_qty = chunks(formatted_evaluations, n_columns)
        logging.warning("still need to handle units, %")

        for line_label, evaluation in zip(lines_labels,
                                          evaluations_sorted_by_qty):
            evaluations_nice_output.loc[line_label] = evaluation

        return evaluations_nice_output

    def _format_evaluations(self, evaluations: list[float | bool | None],
                            precision: int = 3) -> list[str]:
        """Prepare the `evaluations` array for a nice output."""
        units = []
        for qty in self.quantities:
            for elt in self.elts:
                if 'mismatch' in qty:
                    units.append('')
                    continue
                units.append('%')
            units.append('')

        fmt = f".{precision}f"
        formatted_evaluations = [f"{evaluation:{fmt}}"
                                 for evaluation in evaluations]
        formatted_evaluations = [evaluation + unit
                                 for evaluation, unit
                                 in zip(formatted_evaluations, units)]
        return formatted_evaluations

    def _output(self, evaluations: pd.DataFrame) -> None:
        """Print out the given pd.DataFrame."""
        title = "Fit quality (settings in ??????)"
        logging.info(pd_output(evaluations, header=title))


def chunks(lst: list, n_size: int) -> list[list]:
    """
    Yield successive n-sized chunks from lst.

    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n_size):
        yield lst[i:i + n_size]
