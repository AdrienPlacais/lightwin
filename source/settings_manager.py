#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:36:08 2023.

@author: placais

Handle simulation parameters. In particular:
    - which cavities are broken?
    - how should they be fixed?
"""

import json

from util.helper import printc

class SettingsManager():
    def __init__(self, settings_path: str) -> None:
        """Reads a settings file."""
        # settings = json.load(open(settings_path))

    def generate_wtf(self):
        """Generate a proper wtf (list of) dict."""
        wtf = None
        return wtf


    def test_wtf(self, l_wtf):
        """Test the 'what_to_fit' dictionaries."""
        if isinstance(l_wtf, dict):
            printc("settings_manager.test_wtf info:", "a single wtf dict was",
                   "provided and will be used for all the faults.")
            l_wtf = [l_wtf]

        for wtf in l_wtf:
            if not _test_strategy(wtf):
                raise IOError("Wrong argument in wtf['strategy'].")

            if not _test_objective(wtf):
                raise IOError("Wrong argument in wtf['objective'].")

            if not _test_opti_method(wtf):
                raise IOError("Wrong argument in wtf['opti method'].")

            if not _test_position(wtf):
                raise IOError("Wrong argument in wtf['position'].")

            assert 'phi_s fit' in wtf.keys() and isinstance(wtf['phi_s fit'], bool)

            if 'scale_objective' in wtf.keys():
                assert(len(wtf['scale_objective']) == len(wtf['objective']))

    def generate_list_of_faults(self):
        """Generate a proper (list of) faults."""
        failed = None
        return failed


# =============================================================================
# Testing of wtf
# =============================================================================
def _test_strategy(wtf):
    """Specific test for the key 'strategy' of what_to_fit."""
    if 'strategy' not in wtf.keys():
        printc("settings_manager._test_strategy error:", "you must",
               "provide 'strategy' to tell LightWin how compensating",
               "cavities are chosen.", color='red')
        return False

    # You must provide the number of compensating cavities per faulty
    # cavity. Close broken cavities are automatically gathered and fixed
    #  together.
    if wtf['strategy'] == 'k out of n':
        if not ('k' in wtf.keys() and isinstance(wtf['k'], int)):
            printc("settings_manager._test_strategy error:", "you must",
                   "provide k, the number of compensating cavities per",
                   "failed cavity.", color='red')
            return False
        return True

    # You must provide a list of lists of broken cavities, and the
    # corresponding list of lists of compensating cavities. Broken cavities
    # in a sublist are fixed together with the provided sublist of
    # compensating cavities.
    # example:
    # failed = [
    #   [12, 14], -> two cavities that will be fixed together
    #   [155]     -> a single error, fixed after [12, 14] is dealt with
    # ]
    # manual_list = [
    #   [8, 10, 16, 18],    -> compensate errors at idx 12 & 14
    #   [153, 157]          -> compensate error at 155
    # ]
    if wtf['strategy'] == 'manual':
        test = 'manual list' in wtf.keys() \
            and isinstance(wtf['manual list'], list)
        if not test:
            printc("settings_manager._test_strategy error:", "you must",
                   "provide a list of lists of compensating cavities",
                   "corresponding to each list of failed cavities.",
                   color='red')
            return False
        print('also check failed cavities')
        return True

     # You must provide the number of compensating lattices per faulty
     #  cavity. Close broken cavities are gathered and fixed together.
    if wtf['strategy'] == 'l neighboring lattices':
        if not ('l' in wtf.keys() and isinstance(wtf['l'], int)):
            printc("settings_manager._test_strategy error:", "you must",
                   "provide l, the number of compensating lattices.",
                   color='red')
            return False
        return True

    printc("settings_manager._test_strategy error:", "the 'strategy' key",
           f"did not match any authorized value ({wtf['strategy']}).",
           color='red')
    return False

def _test_objective(wtf):
    """Specific test for the key 'objective' of what_to_fit."""
    if 'objective' not in wtf.keys():
        printc("settings_manager._test_objective error:", "you must",
               "provide 'objective' to tell LightWin what it should fit.",
               color='red')
        return False

    l_obj = wtf['objective']
    if not isinstance(l_obj, list):
        printc("settings_manager._test_objective error:", "you must",
               "provide a list of objective, even if there is only one.",
               color='red')
        return False

    implemented = [
        'w_kin', 'phi_abs_array', 'mismatch factor',
        'eps_zdelta', 'beta_zdelta', 'gamma_zdelta', 'alpha_zdelta',
        'M_11', 'M_12', 'M_22', 'M_21']

    if not all([obj in implemented for obj in l_obj]):
        printc("settings_manager._test_objective error:", "at least one",
               "objective was not recognized.", color='red')
        printc("settings_manager._test_objective info:", "to add your",
               "own objective, make sure that 1. it can be returned by",
               "the Accelerator.get() method, 2. it is present in the.",
               "util.d_output dictionaries and 3. it is in the above",
               "'implemented' dict.")
        return False

    return True

def _test_opti_method(wtf):
    """Test the optimisation method."""
    if 'opti method' not in wtf.keys():
        printc("settings_manager._test_opti_method error:", "you must",
               "provide 'opti method' to tell LightWin what optimisation",
               "algorithm it should use.", color='red')
        return False

    implemented = ['least_squares', 'PSO']
    # TODO: specific testing for each method (look at the kwargs)
    if wtf['opti method'] not in implemented:
        printc("settings_manager._test_opti_method error:", "algorithm not",
               "implemented.", color='red')
        return False
    return True

def _test_position(wtf):
    """Test where the objectives are evaluated."""
    if 'position' not in wtf.keys():
        printc("settings_manager._test_opti_method error:", "you must",
               "provide 'position' to tell LightWin where objectives",
               "should be evaluated.", color='red')
        return False

    implemented = [
        'end_mod',      # End of last lattice with a compensating cavity
        '1_mod_after',  # End of one lattice after last lattice with a compensating cavity
        'both']         # Both lattices
    if wtf['position'] not in implemented:
        printc("settings_manager._test_opti_method error:", "position not",
               "implemented.", color='red')
        return False
    return True
