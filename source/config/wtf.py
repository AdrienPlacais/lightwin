#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:54:54 2023.

@author: placais

All the functions to test the ``wtf`` (what to fit) key of the config file.

"""
import logging
import configparser

from config.failures.strategy import test_strategy
from config.failures.position import test_position
from config.optimisation.objective import test_objective_preset


# =============================================================================
# Front end
# =============================================================================
def test(c_wtf: configparser.SectionProxy) -> None:
    """Test the 'what_to_fit' dictionaries."""
    tests = {'failed and idx': _test_failed_and_idx,
             'strategy': test_strategy,
             'objective_preset': test_objective_preset,
             'design_space_preset': _test_design_space_preset,
             'opti method': _test_opti_method,
             'misc': _test_misc,
             'position': test_position,
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


# =============================================================================
# Test
# =============================================================================
def _test_failed_and_idx(c_wtf: configparser.SectionProxy) -> bool:
    """
    Check that failed cavities are given.

    Required keys are:
        - idx:
            If 'cavity', 'failed' and 'manual list' are cavity numbers.
            If 'element', 'failed' and 'manual list' are element numbers. If
            these indexes correspond to element that are not a cavity, an error
            will be raised at the initialisaton of the Fault object.
        - failed:
            The indexes of the cavities that fail.

    Example
    -------
    (we consider that idx is 'cavity')
        1, 2,
        8,
        1, 2, 8
    In this case, LW will first fix the linac with the 1st and 2nd cavity
    failed. In a second time, LW will fix an error with the 8th failed cavity.
    In the last simulation, it will fix together the 1st, 2nd and 8th cavity.
    From LightWin's point of view: one line = one FaultScenario object.
    Each FaultScenario has a list of Fault objects. This is handled by
    the Faults are sorted by the FaultScenario._sort_faults method.

    Example for manual
    ------------------
    If strategy is manual, you must specify which cavities are fixed
    together by adding pipes.
        1, 2, 3 | 98, 99
    In this example, 1 2 and 3 are fixed together. The beam is then propagated
    up to 98 and 99 (even if it is not perfectly matched), and 98 and 99 are
    then fixed together.

    """
    for key in ['failed', 'idx']:
        if key not in c_wtf.keys():
            logging.error(f"You must provide {key}.")
            return False

    val = c_wtf.get('idx')
    if val not in ['cavity', 'element']:
        logging.error(f"'idx' key is {val}, while it must be 'cavity' or "
                      + "element.")
        return False

    return True


def _test_design_space_preset(c_wtf: configparser.SectionProxy) -> bool:
    """Specific test for the key 'design_space_preset' of what_to_fit."""
    if 'design_space_preset' not in c_wtf.keys():
        logging.error("You must provide 'design_space_preset' to tell LightWin"
                      " what it should fit.")
        return False

    design_space_preset = c_wtf.get('design_space_preset')
    implemented = ('unconstrained',
                   'constrained_sync_phase',
                   'sync_phase_as_variable',
                   'FM4_MYRRHA',
                   'one_cavity_mega_power',
                   'experimental'
                   )
    if design_space_preset not in implemented:
        logging.error(f"Objective preset {design_space_preset} was not "
                      "recognized. Check that is is implemented in "
                      "optimisation.design_space.factory and that you added it"
                      " to the list of implemented in config.wtf.")
        return False
    return True


def _test_opti_method(c_wtf: configparser.SectionProxy) -> bool:
    """Test the optimisation method."""
    if 'opti method' not in c_wtf.keys():
        logging.error("You must provide 'opti method' to tell LightWin what "
                      + "optimisation algorithm it should use.")
        return False

    implemented = ('least_squares', 'least_squares_penalty', 'nsga',
                   'downhill_simplex', 'nelder_mead', 'differential_evolution',
                   'explorator', 'experimental')
    # TODO: specific testing for each method (look at the kwargs)
    if c_wtf['opti method'] not in implemented:
        logging.error("Algorithm not implemented.")
        return False
    return True


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
