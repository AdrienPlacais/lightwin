"""Define an object to regroup several :class:`.SimulationOutputEvaluator`.

We also define some factory functions to facilitate their creation.

"""

import datetime
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.elements.element import Element
from evaluator.simulation_output.presets import (
    presets_for_fault_scenario_rel_diff_at_some_element,
    presets_for_fault_scenario_rms_over_full_linac,
)
from evaluator.simulation_output.simulation_output_evaluator import (
    SimulationOutputEvaluator,
)
from failures.fault import Fault
from util.dicts_output import markdown
from util.helper import chunks, pd_output


class ListOfSimulationOutputEvaluators(list):
    """A simple list of :class:`.SimulationOutputEvaluator`."""

    def __init__(self, evaluators: list[SimulationOutputEvaluator]) -> None:
        """Create the objects (factory)."""
        super().__init__(evaluators)

    def run(
        self,
        *simulation_outputs: SimulationOutput,
        other_evals: dict[str, list[Any]] | None = None,
        project_folder: Path | None = None,
        **files_kw,
    ) -> pd.DataFrame:
        """Run all the evaluations."""
        index = self._set_indexes(*simulation_outputs)
        other_columns, other_data = self._unpack_other_evals(other_evals)
        columns = self._set_columns(other_columns)
        data = self._get_evaluations(other_data, *simulation_outputs)

        evaluations = pd.DataFrame(data=data, columns=columns, index=index)

        if project_folder is not None:
            csv_path = Path(project_folder, "evaluations.csv")
            evaluations.to_csv(csv_path)
            logging.info(f"Saved all evaluations in {str(csv_path)}.")
        return evaluations

    def _unpack_other_evals(
        self,
        other_evals: dict[str, list[Any]] | None,
    ) -> tuple[list[str], list[list[Any]]]:
        """Extract column names and data."""
        if other_evals is None:
            return [], []
        other_columns = list(other_evals.keys())

        for other_column in other_columns:
            if other_column in markdown:
                other_column = markdown[other_column]

        other_data = [
            [dat for dat in other_dat] for other_dat in other_evals.values()
        ]

        # Transpose array
        other_data = list(zip(*other_data))
        other_data = [list(data) for data in other_data]
        return other_columns, other_data

    def _set_indexes(
        self,
        *simulation_outputs: SimulationOutput,
    ) -> list[str]:
        """Set the indexes of the pandas dataframe."""
        index = [
            str(simulation_output.beam_calculator_information)
            for simulation_output in simulation_outputs
        ]
        return index

    def _set_columns(
        self,
        other_columns: list[str],
    ) -> list[str]:
        """Set the columns of the pandas dataframe."""
        columns = [evaluator.descriptor for evaluator in self]
        if other_columns is None:
            return columns
        return columns + other_columns

    def _get_evaluations(
        self,
        other_data: list[list[Any]],
        *simulation_outputs: SimulationOutput,
    ) -> list[list[float | bool | datetime.timedelta]]:
        # data = [
        #     [evaluator.run(simulation_output) for evaluator in self]
        #     for simulation_output in simulation_outputs
        # ]
        # if len(other_data) > 0:
        #     data = [line + other_dat
        #             for line, other_dat in zip(data, other_data)]

        data = [
            [evaluator.run(simulation_output) for evaluator in self]
            + other_dat
            for simulation_output, other_dat in zip(
                simulation_outputs, other_data
            )
        ]
        return data


class FaultScenarioSimulationOutputEvaluators:
    """
    A more specific class to evaluate settings found for a `FaultScenario`.

    This class was designed to be used when all the faults of a `FaultScenario`
    are fixed, to output several performance indicators in a compact way. No
    plot is produced.

    """

    def __init__(
        self,
        quantities: tuple[str],
        faults: list[Fault],
        simulation_outputs: tuple[SimulationOutputEvaluator],
        additional_elts: tuple[Element | str] | None = None,
    ) -> None:
        self.quantities = quantities

        self.elts, self.columns = self._set_evaluation_elements(
            faults, additional_elts
        )

        ref_simulation_output = simulation_outputs[0]
        self.simulation_output = simulation_outputs[1]

        self.evaluators = self._create_simulation_output_evaluators(
            ref_simulation_output
        )

    def _set_evaluation_elements(
        self,
        faults: list[Fault],
        additional_elts: tuple[Element | str] | None = None,
    ) -> tuple[list[Element | str], list[str]]:
        """
        Set where the relative difference of `quantities` will be evaluated.

        It is at the end of each compensation zone, plus at the exit of
        additional elements if given.
        Also set `columns` to  ease `pandas` `DataFrame` creation.

        """
        elts = [fault.elts[-1] for fault in faults]
        columns = [f"end comp zone ({elt})" for elt in elts]
        if additional_elts is not None:
            elts += list(additional_elts)
            columns += [
                f"user-defined ({elt})" for elt in list(additional_elts)
            ]
        elts.append("last")
        columns.append("end linac")
        columns.append("RMS [usual units]")
        return elts, columns

    def _create_simulation_output_evaluators(
        self, ref_simulation_output: SimulationOutput
    ) -> list[SimulationOutputEvaluator]:
        """Create the proper `SimulationOutputEvaluator` s."""
        evaluators = []
        for qty in self.quantities:
            for elt in self.elts:
                kwargs = presets_for_fault_scenario_rel_diff_at_some_element(
                    qty, elt, ref_simulation_output
                )
                evaluators.append(SimulationOutputEvaluator(**kwargs))

            kwargs = presets_for_fault_scenario_rms_over_full_linac(
                qty, ref_simulation_output
            )
            evaluators.append(SimulationOutputEvaluator(**kwargs))
        return evaluators

    def run(self, output: bool = True) -> pd.DataFrame:
        """Perform all the simulation output evaluations."""
        evaluations = [
            evaluator.run(self.simulation_output)
            for evaluator in self.evaluators
        ]
        evaluations = self._to_pandas_dataframe(evaluations)
        if output:
            self._output(evaluations)
        return evaluations

    def _to_pandas_dataframe(
        self, evaluations: list[float | bool | None], precision: int = 3
    ) -> pd.DataFrame:
        """Convert all the evaluations to a compact `pd.DataFrame`."""
        lines_labels = [
            markdown[qty].replace("deg", "rad") for qty in self.quantities
        ]

        evaluations_nice_output = pd.DataFrame(
            columns=self.columns, index=lines_labels
        )

        formatted_evaluations = self._format_evaluations(
            evaluations, precision
        )
        n_columns = len(self.columns)
        evaluations_sorted_by_qty = chunks(formatted_evaluations, n_columns)

        for line_label, evaluation in zip(
            lines_labels, evaluations_sorted_by_qty
        ):
            evaluations_nice_output.loc[line_label] = evaluation

        return evaluations_nice_output

    def _format_evaluations(
        self, evaluations: list[float | bool | None], precision: int = 3
    ) -> list[str]:
        """Prepare the `evaluations` array for a nice output."""
        units = []
        for qty in self.quantities:
            for elt in self.elts:
                if "mismatch" in qty:
                    units.append("")
                    continue
                units.append("%")
            units.append("")

        fmt = f".{precision}f"
        formatted_evaluations = [
            f"{evaluation:{fmt}}" if evaluation is not None else "skipped"
            for evaluation in evaluations
        ]
        formatted_evaluations = [
            evaluation + unit
            for evaluation, unit in zip(formatted_evaluations, units)
        ]
        return formatted_evaluations

    def _output(self, evaluations: pd.DataFrame) -> None:
        """Print out the given `pd.DataFrame`."""
        title = "Fit quality:"
        # FIXME
        title += "(FIXME: settings in FaultScenario, not config_manager)"
        logging.info(pd_output(evaluations, header=title))
