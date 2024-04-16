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

import config.beam
import config.beam_calculator
import config.design_space
import config.evaluators
import config.files
import config.plots
import config.wtf
from config.helper import dict_for_pretty_output

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
        dict[str, dict[str, str | int | float | bool | list]] | None
    ) = None,
) -> dict[str, dict[str, Any]]:
    """Load and test the configuration file.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file. It must be a ```.toml`` file.
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
    configuration = _load_correct_toml_entries(config_path, config_keys)
    if override is None:
        override = {}
    _override_some_toml_entries(configuration, warn_mismatch, **override)
    config_folder = config_path.parent
    _process_config(configuration, config_folder)
    return configuration


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


def _process_config(
    toml_entries: dict[str, dict[str, Any]], config_folder: Path
) -> None:
    """Test all the given configuration keys."""
    associated_modules = {
        "files": config.files,
        "plots": config.plots,
        "beam_calculator": config.beam_calculator,
        "beam": config.beam,
        "wtf": config.wtf,
        "design_space": config.design_space,
        "beam_calculator_post": config.beam_calculator,
        "evaluators": config.evaluators,
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
