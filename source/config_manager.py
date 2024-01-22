#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handle simulation parameters.

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

"""
from typing import Any
from pathlib import Path
import logging
import configparser

import numpy as np

from config.ini import files, plots, beam_calculator, beam, wtf, evaluators
from config.ini.optimisation import design_space


# Values that will be available everywhere
FLAG_CYTHON, FLAG_PHI_ABS = bool, bool
METHOD = str
N_STEPS_PER_CELL = int()

E_MEV, E_REST_MEV, INV_E_REST_MEV = float(), float(), float()
GAMMA_INIT = float()
F_BUNCH_MHZ, OMEGA_0_BUNCH, LAMBDA_BUNCH = float(), float(), float()
Q_ADIM, Q_OVER_M, M_OVER_Q = float(), float(), float()
SIGMA = np.full((6, 6), np.NaN)


def process_config(config_path: Path,
                   config_keys: dict[str, str],
                   ) -> dict[str, dict[str, Any]]:
    """Load and test the configuration file."""
    if config_path.suffix == '.ini':
        logging.warning("ini configuration format will soon be deprecated. "
                        "Please switch to .toml, it will be easier for "
                        "everyone.")
        return _process_config_ini(config_path, config_keys)

    if config_path.suffix == '.toml':
        logging.warning(".toml not implemented yet. Now who's the fool?")
        return _process_config_ini(config_path, config_keys)

    raise IOError(f"{config_path.suffix = } while it should be .ini or .toml")


def _process_config_ini(config_path: Path,
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
            'liststr': lambda x: [i.strip() for i in x.split(',')],
            'tuplestr': lambda x: tuple([i.strip() for i in x.split(',')]),
            'listint': lambda x: [int(i.strip()) for i in x.split(',')],
            'listfloat': lambda x: [float(i.strip()) for i in x.split(',')],
            'faults': lambda x: [[int(i.strip()) for i in y.split(',')]
                                 for y in x.split(',\n')],
            'groupedfaults': lambda x: [[[int(i.strip()) for i in z.split(',')]
                                         for z in y.split('|')]
                                        for y in x.split(',\n')],
            'matrixfloat': lambda x: np.array(
                [[float(i.strip()) for i in y.split(',')]
                 for y in x.split(',\n')]),
            'path': lambda x: Path(x),
        },
        allow_no_value=True,
    )
    # FIXME listlistint and matrixfloat: same kind of input, but very different
    # outputs!!
    config.read(config_path)

    _test_config(config, config_keys)
    output_dict = _config_to_dict(config, config_keys)

    # Remove unused Sections, save resulting file
    _ = [config.remove_section(key) for key in list(config.keys())
         if key not in config_keys.keys()]

    with open(Path(output_dict['files']['project_folder'], 'lighwin.ini'),
              'w', encoding='utf-8') as file:
        config.write(file)

    _make_global(**output_dict)
    return output_dict


def _test_config(config: configparser.ConfigParser, config_keys: dict[str, str]
                 ) -> None:
    """Run all the configuration tests, and save the config if ok."""
    for key, val in config_keys.items():
        if val is None:
            continue
        tester = TESTERS[key]
        tester(config[val])


def _config_to_dict(config: configparser.ConfigParser,
                    config_keys: dict[str, str]) -> dict:
    """To convert the configparser into the formats required by LightWin."""
    output_dict = {}
    for key, val in config_keys.items():
        to_dict = DICTIONARIZERS[key]
        if val is None:
            continue
        output_dict[key] = to_dict(config[val])
    return output_dict


def _make_global(beam: dict,
                 beam_calculator: dict | None = None,
                 **kwargs) -> None:
    """Update the values of some variables so they can be used everywhere."""
    global Q_ADIM, E_REST_MEV, INV_E_REST_MEV, OMEGA_0_BUNCH, GAMMA_INIT, \
        LAMBDA_BUNCH, Q_OVER_M, M_OVER_Q, F_BUNCH_MHZ, E_MEV, \
        SIGMA, LINAC
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

    global FLAG_CYTHON, FLAG_PHI_ABS, N_STEPS_PER_CELL, METHOD
    FLAG_CYTHON = beam_calculator.get("flag_cython", None)
    FLAG_PHI_ABS = beam_calculator.get("flag_phi_abs", True)
    N_STEPS_PER_CELL = beam_calculator.get("n_steps_per_cell", None)
    METHOD = beam_calculator.get("method", None)
    logging.warning('default flags for tracewin')


# =============================================================================
# Dictionaries
# =============================================================================
TESTERS = {
    'files': files.test,
    'plots': plots.test,
    'beam_calculator': beam_calculator.test,
    'beam': beam.test,
    'wtf': wtf.test,
    'design_space': design_space.test,
    'beam_calculator_post': beam_calculator.test,
    'evaluators': evaluators.test,
}

DICTIONARIZERS = {
    'files': files.config_to_dict,
    'plots': plots.config_to_dict,
    'beam_calculator': beam_calculator.config_to_dict,
    'beam': beam.config_to_dict,
    'wtf': wtf.config_to_dict,
    'design_space': design_space.config_to_dict,
    'beam_calculator_post': beam_calculator.config_to_dict,
    'evaluators': evaluators.config_to_dict,
}
