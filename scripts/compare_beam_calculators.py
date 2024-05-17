#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an utility function to compare two :class:`.BeamCalculator`.

.. todo::
    Allow for undetermined number of BeamCalculator in the config, and update
    here.

"""
import logging
import sys
from collections.abc import Collection
from pathlib import Path

from scripts_shorthands import compute_beams

import config_manager
from beam_calculation.factory import BeamCalculatorsFactory
from beam_calculation.simulation_output.simulation_output import SimulationOutput
from core.elements.element import Element
from visualization import plot


def output_comparison(
    sim_1: SimulationOutput,
    sim_2: SimulationOutput,
    element: Element | str,
    qty: str,
    single_value: bool,
    **kwargs,
) -> str:
    """Compare two simulation outputs.

    Parameters
    ----------
    sim1, sim2 : SimulationOutput
        Objects to compate.
    element : Element | str
        Element at which look for ``qty``.
    qty : str
        Quantity that will be compared.
    single_value : bool
        True if a single value is expected, False if it is an array.

    Returns
    -------
    msg : str
        Holds requested information.

    """
    kwargs = {"to_deg": True, "elt": element}

    if single_value:
        msg = f"""
        Comparing {qty} in {element}.
        1: {sim_1.get(qty, **kwargs)}
        2: {sim_2.get(qty, **kwargs)}
        """
        return msg

    msg = f"""
    Comparing {qty} in {element}.
    0: {sim_1.get(qty, **kwargs)[0]} to {sim_1.get(qty, **kwargs)[-1]}
    1: {sim_2.get(qty, **kwargs)[0]} to {sim_2.get(qty, **kwargs)[-1]}
    """
    return msg


def compare_beam_calculators(
    toml_filepath: Path,
    toml_keys: dict[str, str],
    tests: Collection[dict[str, str | bool | Element]],
) -> None:
    """Compute beam with two beam calculators and compare them.

    Parameters
    ----------
    toml_filepath : Path
        Path to the configuration file.
    toml_keys : dict[str, str]
        Keys in the configuration file.
    tests : Collection[dict[str, str | bool]]
        kwargs for the :func:`output_comparison`.

    """
    configuration = config_manager.process_config(toml_filepath, toml_keys)

    beam_calculator_factory = BeamCalculatorsFactory(**configuration)
    beam_calculators = beam_calculator_factory.run_all()

    accelerators, simulation_outputs = compute_beams(
        beam_calculators, configuration["files"]
    )

    kwargs = {"save_fig": True, "clean_fig": True}
    _ = plot.factory(accelerators, configuration["plots"], **kwargs)

    for test in tests:
        msg = output_comparison(simulation_outputs[0],simulation_outputs[1], **test)
        logging.info(msg)


if __name__ == "__main__":
    this_file_path = Path(__file__)
    lw_dir = this_file_path.parents[1]
    sys.path.append(str(lw_dir / "source"))
    toml_filepath = lw_dir / "data/example/lightwin.toml"
    toml_keys = {
        "files": "files",
        "plots": "plots_minimal",
        "beam_calculator": "generic_envelope1d",
        "beam_calculator_post": "generic_tracewin",
        "beam": "beam",
    }
    tests = (
        {"element": "FM4", "qty": "phi_abs", "single_value": False},
        {"element": "FM4", "qty": "w_kin", "single_value": False},
        {"element": "FM4", "qty": "phi_s", "single_value": True},
    )
    compare_beam_calculators(toml_filepath, toml_keys, tests)
