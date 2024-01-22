#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this module we define a function to test validity of optimisation algorithm.

Optimisation algorithms are defined in the :mod:`optimisation.algorithms`
subpackage.

.. todo::
    Specific test for every optimisation method? For now, just trust the user.

"""
import logging
import configparser


def test_optimisation_algorithm(c_wtf: configparser.SectionProxy) -> bool:
    """Test the optimisation method."""
    if 'optimisation_algorithm' not in c_wtf.keys():
        logging.error("You must provide 'optimisation_algorithm' to tell "
                      "LightWin what optimisation algorithm it should use.")
        return False

    implemented = ('least_squares',
                   'least_squares_penalty',
                   'nsga',
                   'downhill_simplex',
                   'nelder_mead',
                   'differential_evolution',
                   'explorator',
                   'experimental')
    # TODO: specific testing for each method (look at the kwargs)
    if c_wtf['optimisation_algorithm'] not in implemented:
        logging.error("Algorithm not implemented.")
        return False
    return True
