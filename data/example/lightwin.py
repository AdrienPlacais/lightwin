#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a generic compensation workflow."""
import tomllib
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

import config_manager
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import BeamCalculatorsFactory
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import WithFaults
from experimental.new_evaluator.simulation_output.factory import (
    SimulationOutputEvaluatorsFactory,
)
from experimental.plotter.pd_plotter import PandasPlotter
from failures.fault_scenario import FaultScenario, fault_scenario_factory
from visualization import plot


def _set_up_solvers(
    config: dict[str, Any]
) -> tuple[tuple[BeamCalculator, ...], list[str]]:
    """Create the beam calculators."""
    factory = BeamCalculatorsFactory(**config)
    beam_calculators = factory.run_all()
    beam_calculators_id = factory.beam_calculators_id
    return beam_calculators, beam_calculators_id


def _set_up_accelerators(
    config: dict[str, Any], beam_calculators: tuple[BeamCalculator, ...]
) -> list[Accelerator]:
    """Create the accelerators."""
    factory = WithFaults(
        beam_calculators=beam_calculators, **config["files"], **config["wtf"]
    )
    accelerators = factory.run_all()
    return accelerators


def _set_up_faults(
    config: dict[str, Any],
    beam_calculator: BeamCalculator,
    accelerators: list[Accelerator],
) -> list[FaultScenario]:
    """Create the failures."""
    beam_calculator.compute(accelerators[0])
    fault_scenarios = fault_scenario_factory(
        accelerators,
        beam_calculator,
        config["wtf"],
        config["design_space"],
    )
    return fault_scenarios


def set_up(
    config: dict[str, Any],
) -> tuple[
    tuple[BeamCalculator, ...],
    list[Accelerator],
    list[FaultScenario],
    list[str],
]:
    """Set up everything."""
    beam_calculators, beam_calculator_ids = _set_up_solvers(config)
    accelerators = _set_up_accelerators(config, beam_calculators)

    fault_scenarios = _set_up_faults(config, beam_calculators[0], accelerators)

    for beam_calculator in beam_calculators:
        beam_calculator.compute(accelerators[0])

    return beam_calculators, accelerators, fault_scenarios, beam_calculator_ids


def fix(fault_scenarios: Collection[FaultScenario]) -> None:
    """Fix the faults."""
    for fault_scenario in fault_scenarios:
        fault_scenario.fix_all()


def recompute(
    beam_calculators: Collection[BeamCalculator],
    references: Collection[SimulationOutput],
    *accelerators: Accelerator,
) -> list[list[SimulationOutput]]:
    """Recompute accelerator after a fix with more precision."""
    simulation_outputs = [
        [
            beam_calculator.compute(
                accelerator, ref_simulation_output=reference
            )
            for accelerator in accelerators
        ]
        for beam_calculator, reference in zip(
            beam_calculators, references, strict=True
        )
    ]
    return simulation_outputs


def _perform_evaluations_new_implementation(
    accelerators: Sequence[Accelerator],
    beam_calculators_ids: Sequence[str],
    evaluator_kw: Collection[dict[str, str | float | bool]] | None = None,
) -> pd.DataFrame:
    """Post-treat, with new implementation. Still not fully implemented."""
    if evaluator_kw is None:
        with open("lightwin.toml", "rb") as f:
            config = tomllib.load(f)
        evaluator_kw = config["evaluators"]["simulation_output"]
    assert evaluator_kw is not None
    factory = SimulationOutputEvaluatorsFactory(
        evaluator_kw, plotter=PandasPlotter()
    )
    evaluators = factory.run(accelerators, beam_calculators_ids[0])
    tests = factory.batch_evaluate(
        evaluators, accelerators, beam_calculators_ids[0]
    )
    return tests


def main(config: dict[str, dict[str, Any]]) -> None:
    """Set up the various faults and fix it."""
    beam_calculators, accelerators, fault_scenarios, beam_calculator_ids = (
        set_up(config)
    )
    fix(fault_scenarios)

    references = [
        accelerators[0].simulation_outputs[solver_id]
        for solver_id in beam_calculator_ids
    ]
    simulation_outputs = recompute(
        beam_calculators[1:], references[1:], *accelerators[1:]
    )
    del simulation_outputs

    kwargs = {"save_fig": True, "clean_fig": True}
    _ = plot.factory(accelerators, config["plots"], **kwargs)

    tests = _perform_evaluations_new_implementation(
        accelerators,
        beam_calculator_ids,
        evaluator_kw=None,
    )
    del tests


if __name__ == "__main__":
    toml_filepath = Path("lightwin.toml")
    toml_keys = {
        "files": "files",
        "plots": "plots_minimal",
        "beam_calculator": "generic_envelope1d",
        "beam": "beam",
        "wtf": "generic_wtf",
        "design_space": "tiny_design_space",
    }
    config = config_manager.process_config(toml_filepath, toml_keys)
    main(config)
