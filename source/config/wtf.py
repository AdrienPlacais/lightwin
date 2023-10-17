#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:54:54 2023.

@author: placais

All the functions to test the ``wtf`` (what to fit) key of the config file.

"""
import logging
import configparser

from config.failures.failed_cavities import test_failed_cavities
from config.failures.strategy import test_strategy
from config.failures.position import test_position

from config.optimisation.objective import test_objective_preset
from config.optimisation.design_space import test_design_space_preset
from config.optimisation.algorithm import test_optimisation_algorithm


# =============================================================================
# Front end
# =============================================================================
def test(c_wtf: configparser.SectionProxy) -> None:
    """Test the 'what_to_fit' dictionaries."""
    tests = {'failed_cavities': test_failed_cavities,
             'strategy': test_strategy,
             'objective_preset': test_objective_preset,
             'design_space_preset': test_design_space_preset,
             'optimisation_algorithm': test_optimisation_algorithm,
             'misc': _test_misc,
             # 'position': test_position,
             }
    for key, test in tests.items():
        if not test(c_wtf):
            raise IOError(f"What to fit {c_wtf.name}: error in entry {key}.")
    logging.info(f"what to fit {c_wtf.name} tested with success.")


def config_to_dict(c_wtf: configparser.SectionProxy) -> dict:
    """Convert wtf configparser into a dict."""
    wtf = {}
    # Special getters
    getter = {
        'objective_preset': c_wtf.get,
        'design_space_preset': c_wtf.get,
        'position': c_wtf.getliststr,
        'failed': c_wtf.getfaults,
        'manual list': c_wtf.getgroupedfaults,
        'k': c_wtf.getint,
        'l': c_wtf.getint,
        'phi_s fit': c_wtf.getboolean,
    }
    if c_wtf.get('strategy') == 'manual':
        getter['failed'] = c_wtf.getgroupedfaults

    for key in c_wtf.keys():
        if key in getter:
            wtf[key] = getter[key](key)
            continue

        wtf[key] = c_wtf.get(key)

    return wtf


def _test_misc(c_wtf: configparser.SectionProxy) -> bool:
    """Perform some other tests."""
    if 'phi_s fit' not in c_wtf.keys():
        logging.error("Please explicitely precise if you want to fit synch "
                      + "phases (recommended for least squares, which do not "
                      + "handle constraints) or not (for algorithms that can "
                      + "handle it).")
        return False

    try:
        c_wtf.getboolean("phi_s fit")
    except ValueError:
        logging.error("Not a boolean.")
        return False
    return True
