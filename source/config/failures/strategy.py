#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:29:15 2023.

@author: placais

In this module we define the testing functions for the optimisation strategy.
The documentation for each strategy is in the dedicated module
:mod:`failures.strategy`.

"""
import logging
import configparser


def test_strategy(c_wtf: configparser.SectionProxy) -> bool:
    """Specific test for the key 'strategy' of what_to_fit."""
    if 'strategy' not in c_wtf.keys():
        logging.error("You must provide 'strategy' to tell LightWin how "
                      + "compensating cavities are chosen.")
        return False

    tests = {'k out of n': _k_out_of_n,
             'manual': _manual,
             'l neighboring lattices': _l_neighboring_lattices,
             'global': _global,
             'global downstream': _global_downstream,
             }

    key = c_wtf['strategy']
    if key not in tests:
        logging.error("The 'strategy' key did not match any authorized value "
                      + f"({c_wtf['strategy']}).")
        return False

    return tests[key](c_wtf)


def _k_out_of_n(c_wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for k out of n strategy."""
    if 'k' not in c_wtf.keys():
        logging.error("You must provide k, the number of compensating cavities"
                      " per failed cavity.")
        return False

    try:
        c_wtf.getint('k')
    except ValueError:
        logging.error("k must be an integer.")
        return False

    return True


def _manual(c_wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for manual strategy."""
    if 'manual list' not in c_wtf.keys():
        logging.error("You must provide a list of lists of compensating "
                      + "cavities corresponding to each list of failed "
                      + "cavities.")
        return False

    scenarios = c_wtf.getgroupedfaults('failed')
    groupcomp = c_wtf.getgroupedfaults('manual list')
    if len(scenarios) != len(groupcomp):
        logging.error("Discrepancy between the number of FaultScenarios and "
                      + "the number of corresponding list of compensating "
                      + "cavities. In other words: 'failed' and 'manual list' "
                      + "entries must have the same number of lines.")
        return False

    for scen, comp in zip(scenarios, groupcomp):
        if len(scen) != len(comp):
            logging.error("In a FaultScenario, discrepancy between the number "
                          + "of fault groups and group of compensating "
                          + "cavities. In other words: 'failed' and 'manual "
                          + "list' entries must have the same number of "
                          + "pipes on every line.")
            return False

    return True


def _l_neighboring_lattices(c_wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for l neighboring lattices strategy."""
    if 'l' not in c_wtf.keys():
        logging.error("You must provide l, the number of compensating "
                      + "lattices.")
        return False

    try:
        c_wtf.getint('l')
    except ValueError:
        logging.error("l must be an integer.")
        return False

    if c_wtf.getint('l') % 2 != 0:
        logging.error("l must be even.")
        return False

    return True


def _global(c_wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for global strategy."""
    logging.warning("Option still under implementation.")
    logging.warning("As for now, field amplitudes are always modified during "
                    + "the fit. If you want the 'classic' global compensation,"
                    + " you should manually set the bounds for k_e to a very "
                    + "low value in optimisation/variables.py.")

    if 'position' not in c_wtf.keys():
        logging.error("You must provide 'position' to tell LightWin where "
                      + "objectives should be evaluated.")
        return False

    if 'end of linac' not in c_wtf.getliststr('position'):
        logging.warning("With global methods, objectives should be evaluated "
                        + "at the end of the linac. LW will run anyway and "
                        + "'position' key will not be modified.")

    return True


def _global_downstream(c_wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for global downstream strategy."""
    return _global(c_wtf)
