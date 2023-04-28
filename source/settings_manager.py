#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:36:08 2023.

@author: placais

Handle simulation parameters. In particular:
    - which cavities are broken?
    - how should they be fixed?
"""
import logging
import os
import json


class ConfigManager():
    """Surpisingly, a manager for your config."""

    def __init__(self, config_path: str, project_path: str) -> None:
        """Reads a config file."""
        self.config = json.load(open(config_path, encoding='utf-8'))
        self.project_path = project_path

    def generate_wtf(self) -> list | dict:
        """Generate a proper wtf (list of) dict."""
        wtf = None
        return wtf

    def test_config(self) -> None:
        """Run all the config dic tests, and save the config if ok."""
        self._test_wtf(self.config['l_wtf'])

        save_path = os.path.join(self.project_path, 'config.json')
        with open(save_path, encoding='utf-8') as file:
            json.dump(self.config, file)


    def _test_wtf(self, l_wtf: list | dict) -> None:
        """Test the 'what_to_fit' dictionaries."""
        if isinstance(l_wtf, dict):
            logging.info("A single wtf dict was provided and will be used "
                         + "for all the faults.")
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

            assert 'phi_s fit' in wtf.keys() and isinstance(
                wtf['phi_s fit'], bool)

    def generate_list_of_faults(self):
        """Generate a proper (list of) faults."""
        failed = None
        return failed


# =============================================================================
# Testing of wtf strategy
# =============================================================================
def _test_strategy(wtf: dict) -> bool:
    """Specific test for the key 'strategy' of what_to_fit."""
    if 'strategy' not in wtf.keys():
        logging.error("You must provide 'strategy' to tell LightWin how "
                      + "compensating cavities are chosen.")
        return False

    # You must provide the number of compensating cavities per faulty
    # cavity. Nearby broken cavities are automatically gathered and fixed
    #  together.
    if wtf['strategy'] == 'k out of n':
        return _test_strategy_k_out_of_n(wtf)

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
        return _test_strategy_manual(wtf)

    # You must provide the number of compensating lattices per faulty
    # cavity. Close broken cavities are gathered and fixed together.
    if wtf['strategy'] == 'l neighboring lattices':
        return _test_strategy_l_neighboring_lattices(wtf)

    logging.error("The 'strategy' key did not match any authorized value "
                  + f"({wtf['strategy']}).")
    return False


def _test_strategy_k_out_of_n(wtf: dict) -> bool:
    """Even more specific test for k out of n strategy."""
    if 'k' not in wtf.keys():
        logging.error("You must provide k, the number of compensating "
                      + "cavities per failed cavity.")
        return False

    if not isinstance(wtf['k'], int):
        logging.error("k must be an integer.")
        return False

    return True


def _test_strategy_manual(wtf: dict) -> bool:
    """Even more specific test for manual strategy."""
    test = 'manual list' in wtf.keys() and isinstance(wtf['manual list'], list)
    if not test:
        logging.error("You must provide a list of lists of compensating "
                      + "cavities corresponding to each list of failed "
                      + "cavities.")
        return False
    logging.info("You must insure that all the elements in manual list are "
                 + "cavities.")
    return True


def _test_strategy_l_neighboring_lattices(wtf: dict) -> bool:
    """Even more specific test for l neighboring lattices strategy."""
    if not ('l' in wtf.keys() and isinstance(wtf['l'], int)):
        logging.error("You must provide l, the number of compensating "
                      + "lattices.")
        return False
    return True


# =============================================================================
# Testing of wtf objective
# =============================================================================
def _test_objective(wtf: dict) -> bool:
    """Specific test for the key 'objective' of what_to_fit."""
    if 'objective' not in wtf.keys():
        logging.error("You must provide 'objective' to tell LightWin what it "
                      + "should fit.")
        return False

    l_obj = wtf['objective']
    if not isinstance(l_obj, list):
        logging.error("You must provide a list of objective, even if there is "
                      + "only one.")
        return False

    implemented = [
        'w_kin', 'phi_abs_array', 'mismatch factor',
        'eps_zdelta', 'beta_zdelta', 'gamma_zdelta', 'alpha_zdelta',
        'M_11', 'M_12', 'M_22', 'M_21']

    if not all(obj in implemented for obj in l_obj):
        logging.error("At least one objective was not recognized.")
        logging.info("""To add your own objective, make sure that:
                     1. it can be returned by the Accelerator.get() method;
                     2. it is present in the util.d_output dictionaries;
                     3. it is in the above 'implemented' dict.""")
        return False

    if 'scale_objective' in wtf.keys():
        if not isinstance(len(wtf['scale objective']), list) \
           or len(wtf['scale_objective']) != len(wtf['objective']):
            logging.error("If you want to scale the objectives by a factor, "
                          + "you must provide a list of scale factors (one "
                          + "scale factor per objective.")
            return False

    return True


# =============================================================================
# Testing of wtf optimisation method
# =============================================================================
def _test_opti_method(wtf: dict) -> bool:
    """Test the optimisation method."""
    if 'opti method' not in wtf.keys():
        logging.error("You must provide 'opti method' to tell LightWin what "
                      + "optimisation algorithm it should use.")
        return False

    implemented = ['least_squares', 'PSO']
    # TODO: specific testing for each method (look at the kwargs)
    if wtf['opti method'] not in implemented:
        logging.error("Algorithm not implemented.")
        return False
    return True


# =============================================================================
# Testing of wtf position
# =============================================================================
def _test_position(wtf: dict) -> bool:
    """Test where the objectives are evaluated."""
    if 'position' not in wtf.keys():
        logging.error("You must provide 'position' to tell LightWin where "
                      + "objectives should be evaluated.")
        return False

    implemented = [
        # End of last lattice with a compensating cavity
        'end_mod',
        # End of one lattice after last lattice with a compensating cavity
        '1_mod_after',
        # Both lattices
        'both']
    if wtf['position'] not in implemented:
        logging.error("Position not implemented.")
        return False
    return True
