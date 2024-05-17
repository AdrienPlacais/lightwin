#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate ``variables.csv`` and ``constraints.csv``.

These files hold the initial value and bounds for every cavity :math:`k_e`,
:math:`\phi_{0,\mathrm{\,abs}}`, :math:`\phi_{0,\mathrm{\,rel}}` and
:math:`\phi_s`. You can then alter every cavity settings to your needs.

"""
import logging
from pathlib import Path

from scripts.scripts_shorthands import compute_beam

import config_manager
from beam_calculation.factory import BeamCalculatorsFactory
from optimisation.design_space.factory import get_design_space_factory


def generate_design_space_files(toml_filepath: Path, toml_keys: dict[str, str]) -> None:
    """Generate the ``variables.csv`` and ``constraints.csv`` files.

    Parameters
    ----------
    toml_filepath : Path
        Path to the configuration file.
    toml_keys : dict[str, str]
        Keys of the configuration file.

    """
    configuration = config_manager.process_config(toml_filepath, toml_keys)
    if configuration["design_space"]["design_space_preset"] != "everything":
        logging.warning(
            "Modifying the design_space_preset entry to have all the possible "
            "variables and constraints in the output file."
        )
        configuration["design_space"]["design_space_preset"] = "everything"

    beam_calculator_factory = BeamCalculatorsFactory(**configuration)
    beam_calculator = beam_calculator_factory.run_all()[0]
    accelerator, _ = compute_beam(beam_calculator, configuration["files"])

    # Take only the FieldMap objects
    cavities = accelerator.elts.l_cav

    design_space_factory = get_design_space_factory(
        **configuration["design_space"]
    )
    design_space = design_space_factory.run(
            compensating_elements=cavities,
            reference_elements=cavities,
    )

    project_folder = toml_filepath.parent
    design_space.to_files(
        basepath=project_folder,
        variables_filename=Path("variables"),
        constraints_filename=Path("constraints"),
        overwrite=True,
    )

if __name__ == "__main__":
    this_file_path = Path(__file__)
    toml_filepath = this_file_path.parents[1] / "data/example/lightwin.toml"
    toml_keys = {
        "files": "files",
        "plots": "plots_minimal",
        "beam_calculator": "generic_envelope1d",
        "beam_calculator_post": "generic_tracewin",
        "beam": "beam",
        "design_space": "generic_design_space",
    }
    generate_design_space_files(toml_filepath, toml_keys)
