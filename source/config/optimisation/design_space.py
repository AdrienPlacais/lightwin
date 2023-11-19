#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we set the function to test validity of the ``design_space`` entries.

The documentation relative to every preset is in
:mod:`optimisation.design_space.factory`. This is also where you should add you
own presets.

"""
import logging
import configparser


MANDATORY_DESIGN_SPACE_KEYS = (
    'design_space_preset',
    'max_increase_sync_phase_in_percent',
    'max_decrease_k_e_in_percent',
    'max_increase_k_e_in_percent',
)  # :
FACULTATIVE_DESIGN_SPACE_KEYS = (
    'max_absolute_sync_phase_in_deg',  # default 0 deg
    'min_absolute_sync_phase_in_deg',  # default -90 deg
    'maximum_k_e_is_calculated_wrt_maximum_k_e_of_section',  # default False
)  #:

implemented_design_space_presets = ('unconstrained',
                                    'constrained_sync_phase',
                                    'sync_phase_as_variable',
                                    'FM4_MYRRHA',
                                    'experimental'
                                    )


def test(c_design_space: configparser.SectionProxy) -> bool:
    """Specific test for the keys of 'design_space'."""
    passed = True

    for key in MANDATORY_DESIGN_SPACE_KEYS:
        if key not in c_design_space.keys():
            logging.error(f"{key} is mandatory and missing.")
            passed = False

    design_space_preset = c_design_space.get('design_space_preset')
    if design_space_preset not in implemented_design_space_presets:
        logging.error(f"{design_space_preset = } not recognized. Possible "
                      f"values are {implemented_design_space_presets = }.")
        passed = False

    if not passed:
        raise IOError("Wrong value in c_design_space.")

    logging.info(f"design space {c_design_space.name} tested with success.")
    return True


def config_to_dict(c_design_space: configparser.SectionProxy) -> dict:
    """Convert design_space configparser into a dict."""
    design_space = {}
    # Special getters
    getter = {
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
