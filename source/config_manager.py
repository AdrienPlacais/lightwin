#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handle simulation parameters.

In particular:
    - what are the initial properties of the beam?
    - which cavities are broken?
    - how should they be fixed?
    - simulation parameters to give to TW for a 'post' simulation?

.. todo::
    Maybe make test and config to dict more compact?

.. todo::
    position:
        - element name
        - element number
        - end_section
    variable:
        - maybe add this? Unnecessary at this point

.. todo::
    Remove global variables.

.. todo::
    Find a way to override the entries in the ``.toml`` before testing.

"""
import configparser
import logging
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

import config.ini.beam
import config.ini.beam_calculator
import config.ini.evaluators
import config.ini.files
import config.ini.optimisation.design_space
import config.ini.plots
import config.ini.wtf
import config.toml.beam
import config.toml.beam_calculator
import config.toml.design_space
import config.toml.evaluators
import config.toml.files
import config.toml.plots
import config.toml.wtf
from config.toml.helper import dict_for_pretty_output

# Values that will be available everywhere
FLAG_CYTHON = bool
METHOD = str
N_STEPS_PER_CELL = int()

E_MEV, E_REST_MEV, INV_E_REST_MEV = float(), float(), float()
GAMMA_INIT = float()
F_BUNCH_MHZ, OMEGA_0_BUNCH, LAMBDA_BUNCH = float(), float(), float()
Q_ADIM, Q_OVER_M, M_OVER_Q = float(), float(), float()
SIGMA = np.full((6, 6), np.NaN)


MANDATORY_CONFIG_ENTRIES = ("files", "beam_calculator", "beam")  #:
OPTIONAL_CONFIG_ENTRIES = (
    "beam_calculator_post",
    "evaluators",
    "plots",
    "wtf",
    "design_space",
)  #:


def process_config(
    config_path: Path,
    config_keys: dict[str, str],
    warn_mismatch: bool = False,
    override: (
        dict[str, dict[str, dict[str, str | int | float | bool | list]]] | None
    ) = None,
) -> dict[str, dict[str, Any]]:
    """Load and test the configuration file.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file. It must be a ``.ini`` or a ``.toml``
        file. ``.toml`` is prefered, as ``.ini`` will soon be deprecated.
    config_keys : dict[str, str]
        Associate the name of LightWin's group of parameters to the entry in
        the configuration file.
    warn_mismatch : bool, optional
        Raise a warning if a key in a ``override`` sub-dict is not found.
    override : dict[str, dict[str, dict[str, str | int | float | bool | list]]]
        To override entries in the ``.toml``. If not provided, we keep
        defaults.

    Returns
    -------
    configuration : dict[str, dict[str, Any]]
        A dictonary holding all the keyword arguments that will be passed to
        LightWin objects, eg ``beam_calculator`` will be passed to
        :class:`.BeamCalculator`.

    """
    if config_path.suffix == ".ini":
        logging.warning(
            "ini configuration format will soon be deprecated. "
            "Please switch to .toml, it will be easier for "
            "everyone."
        )
        return _process_config_ini(config_path, config_keys)

    if config_path.suffix == ".toml":
        configuration = _load_correct_toml_entries(config_path, config_keys)
        if override is None:
            override = {}
        _override_some_toml_entries(configuration, warn_mismatch, **override)
        config_folder = config_path.parent
        _process_config_toml(configuration, config_folder)
        return configuration

    raise IOError(f"{config_path.suffix = } while it should be .ini or .toml")


def _load_correct_toml_entries(
    config_path: Path,
    config_keys: dict[str, str],
) -> dict[str, dict[str, str | int | float | bool | list]]:
    """Load the ``.toml`` and extract the dicts asked by user."""
    all_toml: dict[str, dict[str, str | int | float | bool | list]]
    with open(config_path, "rb") as f:
        all_toml = tomllib.load(f)

    desired_toml = {key: all_toml[value] for key, value in config_keys.items()}
    for key in MANDATORY_CONFIG_ENTRIES:
        assert key in desired_toml, f"{key = } is mandatory and missing."

    return desired_toml


def _override_some_toml_entries(
    configuration: dict[str, dict[str, str | int | float | bool | list]],
    warn_mismatch: bool = False,
    **override: dict[str, str | int | float | bool | list],
) -> None:
    """Override some entries before testing."""
    for over_key, over_subdict in override.items():
        assert over_key in configuration, (
            f"You want to override entries in {over_key = }, which was not "
            f"found in {configuration.keys() = }"
        )
        conf_subdict = configuration[over_key]

        for key, val in over_subdict.items():
            if warn_mismatch and key not in conf_subdict:
                logging.warning(
                    f"You want to override {key = }, which was "
                    f"not found in {conf_subdict.keys() = }"
                )
            conf_subdict[key] = val


def _process_config_toml(
    toml_entries: dict[str, dict[str, Any]], config_folder: Path
) -> None:
    """Test all the given configuration keys."""
    associated_modules = {
        "files": config.toml.files,
        "plots": config.toml.plots,
        "beam_calculator": config.toml.beam_calculator,
        "beam": config.toml.beam,
        "wtf": config.toml.wtf,
        "design_space": config.toml.design_space,
        "beam_calculator_post": config.toml.beam_calculator,
        "evaluators": config.toml.evaluators,
    }

    for key, config_dict in toml_entries.items():
        if config_dict is None or config_dict == {}:
            continue

        associated_module = associated_modules[key]
        associated_module.test(config_folder=config_folder, **config_dict)

        if hasattr(associated_module, "edit_configuration_dict_in_place"):
            associated_module.edit_configuration_dict_in_place(
                config_dict, config_folder=config_folder
            )

        logging.info(
            f"Config dict {key} successfully tested. After potential "
            " modifications, it looks like:\n"
            f"{dict_for_pretty_output(config_dict)}"
        )
    _make_global(**toml_entries)


def _process_config_ini(
    config_path: Path,
    config_keys: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """
    Frontend for config: load ``.ini``, test it, return its content as dicts.

    Parameters
    ----------
    config_path : Path
        Path to the .ini file.
    config_keys : dict[str, str]
        Associate the name of the Sections in the config_file to the proper
        configurations.
        Mandatory keys are:
            - ``files``: related to input/output files.
            - ``plots``: what should be plotted.
            - ``beam_calculator``: everything related to the tool that will\
                compute the propagation of the beam.
            - ``beam``: the initial beam properties.
            - ``wtf``: for 'what to fit'. Everything related to the fault\
                compensation methodology.
            - ``evaluators_post``: to set the tests that are run on the newly\
                found settings. Can be empty.
        Optional keys are:
            - ``beam_calculator_post``: for an additional simulation once the\
                fault are compensated. Usually, this simulation should be more\
                precise but take more time.

    Returns
    -------
    output_dict : dict
        A dict of dicts. The 'sub' dicts are:
            - ``files`` : dict
                Information on the files, project folders.
            - ``plots`` : dict
                The quantities to plot.
            - ``beam_calculator`` : dict
                Holds the beam_calculator used for simulation.
            - ``beam`` : dict
                Dictionary holding all beam parameters.
            - ``wtf`` : dict
                Dictionary holding all wtf parameters.
            - ``beam_calculator_post`` : dict
                Holds beam_calculator parameters for the post treatment
                simulation.
            - ``evaluators`` : dict
                Holds the name of the tests/evaluations presets that will be
                run during and after the simulation.

    """
    # Load config
    # the converters key allows to have methods to directly convert the strings
    # in the .ini to the proper type
    config = configparser.ConfigParser(
        converters={
            "liststr": lambda x: [i.strip() for i in x.split(",")],
            "tuplestr": lambda x: tuple([i.strip() for i in x.split(",")]),
            "listint": lambda x: [int(i.strip()) for i in x.split(",")],
            "listfloat": lambda x: [float(i.strip()) for i in x.split(",")],
            "faults": lambda x: [
                [int(i.strip()) for i in y.split(",")] for y in x.split(",\n")
            ],
            "groupedfaults": lambda x: [
                [[int(i.strip()) for i in z.split(",")] for z in y.split("|")]
                for y in x.split(",\n")
            ],
            "matrixfloat": lambda x: np.array(
                [
                    [float(i.strip()) for i in y.split(",")]
                    for y in x.split(",\n")
                ]
            ),
            "path": lambda x: Path(x),
        },
        allow_no_value=True,
    )
    # FIXME listlistint and matrixfloat: same kind of input, but very different
    # outputs!!
    config.read(config_path)

    _test_config(config, config_keys)
    output_dict = _config_to_dict(config, config_keys)

    # Remove unused Sections, save resulting file
    _ = [
        config.remove_section(key)
        for key in list(config.keys())
        if key not in config_keys.keys()
    ]

    with open(
        Path(output_dict["files"]["project_folder"], "lighwin.ini"),
        "w",
        encoding="utf-8",
    ) as file:
        config.write(file)

    _make_global(**output_dict)
    return output_dict


def _test_config(
    config: configparser.ConfigParser, config_keys: dict[str, str]
) -> None:
    """Run all the configuration tests, and save the config if ok."""
    for key, val in config_keys.items():
        if val is None:
            continue
        tester = TESTERS[key]
        tester(config[val])


def _config_to_dict(
    config: configparser.ConfigParser, config_keys: dict[str, str]
) -> dict:
    """To convert the configparser into the formats required by LightWin."""
    output_dict = {}
    for key, val in config_keys.items():
        to_dict = DICTIONARIZERS[key]
        if val is None:
            continue
        output_dict[key] = to_dict(config[val])
    return output_dict


def _make_global(
    beam: dict, beam_calculator: dict | None = None, **kwargs
) -> None:
    """Update the values of some variables so they can be used everywhere."""
    global Q_ADIM, E_REST_MEV, INV_E_REST_MEV, OMEGA_0_BUNCH, GAMMA_INIT, LAMBDA_BUNCH, Q_OVER_M, M_OVER_Q, F_BUNCH_MHZ, E_MEV, SIGMA, LINAC
    Q_ADIM = beam["q_adim"]
    E_REST_MEV = beam["e_rest_mev"]
    INV_E_REST_MEV = beam["inv_e_rest_mev"]
    OMEGA_0_BUNCH = beam["omega_0_bunch"]
    GAMMA_INIT = beam["gamma_init"]
    LAMBDA_BUNCH = beam["lambda_bunch"]
    Q_OVER_M = beam["q_over_m"]
    M_OVER_Q = beam["m_over_q"]
    F_BUNCH_MHZ = beam["f_bunch_mhz"]
    E_MEV = beam["e_mev"]
    SIGMA = beam["sigma"]

    if beam_calculator is None:
        return

    global FLAG_CYTHON, N_STEPS_PER_CELL, METHOD
    FLAG_CYTHON = beam_calculator.get("flag_cython", None)
    N_STEPS_PER_CELL = beam_calculator.get("n_steps_per_cell", None)
    METHOD = beam_calculator.get("method", None)
    logging.warning("default flags for tracewin")


# =============================================================================
# Dictionaries
# =============================================================================
TESTERS = {
    "files": config.ini.files.test,
    "plots": config.ini.plots.test,
    "beam_calculator": config.ini.beam_calculator.test,
    "beam": config.ini.beam.test,
    "wtf": config.ini.wtf.test,
    "design_space": config.ini.optimisation.design_space.test,
    "beam_calculator_post": config.ini.beam_calculator.test,
    "evaluators": config.ini.evaluators.test,
}

DICTIONARIZERS = {
    "files": config.ini.files.config_to_dict,
    "plots": config.ini.plots.config_to_dict,
    "beam_calculator": config.ini.beam_calculator.config_to_dict,
    "beam": config.ini.beam.config_to_dict,
    "wtf": config.ini.wtf.config_to_dict,
    "design_space": config.ini.optimisation.design_space.config_to_dict,
    "beam_calculator_post": config.ini.beam_calculator.config_to_dict,
    "evaluators": config.ini.evaluators.config_to_dict,
}
