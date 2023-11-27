#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we set the function to test validity of the ``design_space`` entries.

The documentation relative to every preset is in
:mod:`optimisation.design_space.factory`. This is also where you should add you
own presets.

"""
import logging
from pathlib import Path
import configparser


MANDATORY_DESIGN_SPACE_KEYS = (
    'from_file',
    'design_space_preset',
)  # :

MANDATORY_DESIGN_SPACE_KEYS_FROM_FILE = (
    'variables_filepath',
)
FACULTATIVE_DESIGN_SPACE_KEYS_FROM_FILE = (
    'constraints_filepath',
)

MANDATORY_DESIGN_SPACE_KEYS_NOT_FROM_FILE = (
    'max_increase_sync_phase_in_percent',
    'max_decrease_k_e_in_percent',
    'max_increase_k_e_in_percent',
)  #:
FACULTATIVE_DESIGN_SPACE_KEYS_NOT_FROM_FILE = (
    'max_absolute_sync_phase_in_deg',  # default 0 deg
    'min_absolute_sync_phase_in_deg',  # default -90 deg
    'maximum_k_e_is_calculated_wrt_maximum_k_e_of_section',  # default False
)  #:

implemented_design_space_presets = ('unconstrained',
                                    'constrained_sync_phase',
                                    'sync_phase_as_variable',
                                    'FM4_MYRRHA',
                                    'experimental',
                                    'everything',
                                    )


def test(c_design_space: configparser.SectionProxy) -> bool:
    """Specific test for the keys of 'design_space'."""
    passed = True

    key = 'from_file'
    if key not in c_design_space.keys():
        logging.error(f"{key} is mandatory and missing.")
        passed = False

    from_file = c_design_space.getboolean('from_file')
    if from_file:
        passed_bis = _test_from_file(c_design_space)
    else:
        passed_bis = _test_not_from_file(c_design_space)

    design_space_preset = c_design_space.get('design_space_preset')
    if design_space_preset not in implemented_design_space_presets:
        logging.error(f"{design_space_preset = } not recognized. Possible "
                      f"values are {implemented_design_space_presets = }.")
        passed = False

    if not (passed and passed_bis):
        raise IOError("Wrong value in c_design_space.")

    logging.info(f"design space {c_design_space.name} tested with success.")
    return True


def _test_not_from_file(c_design_space: configparser.SectionProxy) -> bool:
    """Ensure that phase space file will not raise errors."""
    passed_bis = True
    for key in MANDATORY_DESIGN_SPACE_KEYS_NOT_FROM_FILE:
        if key not in c_design_space.keys():
            logging.error(f"{key} is mandatory and missing.")
            passed_bis = False
    return passed_bis


def _test_from_file(c_design_space: configparser.SectionProxy) -> bool:
    """Ensure that phase space file will not raise errors."""
    passed_bis = True

    for key in MANDATORY_DESIGN_SPACE_KEYS_FROM_FILE:
        if key not in c_design_space.keys():
            logging.error(f"{key} is mandatory and missing.")
            passed_bis = False

    for path in ('variables_filepath', 'constraints_filepath'):
        filepath: Path = c_design_space.getpath(path, fallback=None)
        if filepath is None:
            continue
        filepath = filepath.absolute()
        if not filepath.is_file():
            logging.error(f"The design space path {filepath} does not exists.")
            passed_bis = False
        c_design_space[path] = filepath

    return passed_bis


def config_to_dict(c_design_space: configparser.SectionProxy) -> dict:
    """Convert design_space configparser into a dict."""
    design_space = {}
    # Special getters
    getter = {
        'from_file': c_design_space.getboolean,
        'variables_filepath': c_design_space.getpath,
        'constraints_filepath': c_design_space.getpath,
        'max_increase_sync_phase_in_percent': c_design_space.getfloat,
        'max_decrease_k_e_in_percent': c_design_space.getfloat,
        'max_increase_k_e_in_percent': c_design_space.getfloat,
        'max_absolute_sync_phase_in_deg': c_design_space.getfloat,
        'min_absolute_sync_phase_in_deg': c_design_space.getfloat,
        'maximum_k_e_is_calculated_wrt_maximum_k_e_of_section':
        c_design_space.getboolean,
    }

    for key in c_design_space.keys():
        if key in getter:
            design_space[key] = getter[key](key)
            continue

        design_space[key] = c_design_space.get(key)
    return design_space
