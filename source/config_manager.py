#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:36:08 2023.

@author: placais

Handle simulation parameters. In particular:
    - what are the initial properties of the beam?
    - which cavities are broken?
    - how should they be fixed?
    - simulation parameters to give to TW for a 'post' simulation?

TODO: maybe make test and config to dict more compact?

TODO strategy:
    global_section
    global_section_downstream
    flag to select priority in k out of n when k odd
TODO position:
    element name
    element number
    end_section
TODO allow for different objectives at different positions.
    quickfix for now: simply set some scales to 0.

TODO variable: maybe add this? Unnecessary at this point
"""
import os
import configparser
import numpy as np

from config import files, beam_calculator, beam, wtf


# Values that will be available everywhere
FLAG_CYTHON, FLAG_PHI_ABS = bool, bool
METHOD = str
N_STEPS_PER_CELL = int()

LINAC = str
E_MEV, E_REST_MEV, INV_E_REST_MEV = float(), float(), float()
GAMMA_INIT = float()
F_BUNCH_MHZ, OMEGA_0_BUNCH, LAMBDA_BUNCH = float(), float(), float()
Q_ADIM, Q_OVER_M, M_OVER_Q = float(), float(), float()
SIGMA_ZDELTA = np.ndarray(shape=(2, 2))


def process_config(config_path: str, config_keys: dict[str, str],
                   ) -> dict[str, dict[str, str | None]]:
    """
    Frontend for config: load .ini, test it, return its content as dicts.

    Parameters
    ----------
    config_path : str
        Path to the .ini file.
    config_keys : dict[str, str]
        Associate the name of the Sections in the config_file to the proper
        configurations.
        Mandatory keys are:
            - files: related to input/output files.
            - plots: what should be plotted.
            - beam_calculator: everything related to the tool that will compute
                               the propagation of the beam.
            - beam: the initial beam properties.
            - wtf: for 'what to fit'. Everything related to the fault
                   compensation methodology.
        Optional keys are:
            - beam_calculator_post: for an additional simulation once the fault
                                    are compensated. Usually, this simulation
                                    should be more precise but take more time.

    Returns
    -------
    output_dict : dict
        A dict of dicts. The 'sub' dicts are:
        files : dict
            Information on the files, project folders.
        plot : dict
            The quantities to plot.
        beam_calculator : dict
            Holds the beam_calculator used for simulation.
        beam : dict
            Dictionary holding all beam parameters.
        wtf : dict
            Dictionary holding all wtf parameters.
        beam_calculator_post : dict
            Holds beam_calculator parameters for the post treatment simulation.

    """
    # Load config
    # the converters key allows to have methods to directly convert the strings
    # in the .ini to the proper type
    config = configparser.ConfigParser(
        converters={
            'liststr': lambda x: [i.strip() for i in x.split(',')],
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

    with open(os.path.join(output_dict['files']['project_folder'],
                           'lighwin.ini'),
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


def _make_global(beam_calculator: dict, beam: dict, **kwargs) -> None:
    """Update the values of some variables so they can be used everywhere."""
    global FLAG_CYTHON, FLAG_PHI_ABS, N_STEPS_PER_CELL, METHOD
    FLAG_CYTHON = beam_calculator["FLAG_CYTHON"]
    FLAG_PHI_ABS = beam_calculator["FLAG_PHI_ABS"]
    N_STEPS_PER_CELL = beam_calculator["N_STEPS_PER_CELL"]
    METHOD = beam_calculator["METHOD"]

    global Q_ADIM, E_REST_MEV, INV_E_REST_MEV, OMEGA_0_BUNCH, GAMMA_INIT, \
        LAMBDA_BUNCH, Q_OVER_M, M_OVER_Q, F_BUNCH_MHZ, E_MEV, SIGMA_ZDELTA, \
        LINAC
    Q_ADIM = beam["Q_ADIM"]
    E_REST_MEV = beam["E_REST_MEV"]
    INV_E_REST_MEV = beam["INV_E_REST_MEV"]
    OMEGA_0_BUNCH = beam["OMEGA_0_BUNCH"]
    GAMMA_INIT = beam["GAMMA_INIT"]
    LAMBDA_BUNCH = beam["LAMBDA_BUNCH"]
    Q_OVER_M = beam["Q_OVER_M"]
    M_OVER_Q = beam["M_OVER_Q"]
    F_BUNCH_MHZ = beam["F_BUNCH_MHZ"]
    E_MEV = beam["E_MEV"]
    SIGMA_ZDELTA = beam["SIGMA_ZDELTA"]
    LINAC = beam["LINAC"]


# =============================================================================
# Dictionaries
# =============================================================================
TESTERS = {
    'files': files.test,
    'beam_calculator': beam_calculator.test,
    'beam': beam.test,
    'wtf': wtf.test,
    'beam_calculator_post': beam_calculator.test
}

DICTIONARIZERS = {
    'files': files.config_to_dict,
    'beam_calculator': beam_calculator.config_to_dict,
    'beam': beam.config_to_dict,
    'wtf': wtf.config_to_dict,
    'beam_calculator_post': beam_calculator.config_to_dict
}


# =============================================================================
# Main func
# =============================================================================
if __name__ == '__main__':
    # Init paths
    CONFIG_PATH = 'jaea.ini'
    PROJECT_PATH = 'bla/'

    # Load config
    wtfs = process_config(
        CONFIG_PATH, PROJECT_PATH,
        key_beam_calculator='beam_calculator.lightwin.envelope_longitudinal',
        key_beam='beam.jaea',
        key_wtf='wtf.k_out_of_n',
        key_beam_calculator_post='post_tracewin.quick_debug')
    print(f"{wtfs}")
