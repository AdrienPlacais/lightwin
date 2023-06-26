#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:20:14 2023.

@author: placais

Everything related to the plots.
"""
import logging
import configparser


def test(c_plots: configparser.SectionProxy) -> None:
    """Test that the provided plots can be done."""
    passed = True

    implemented = ['energy', 'phase', 'cav', 'emittance', 'twiss', 'envelopes',
                   'transfer matrices']
    to_update_or_fix = ['twiss', 'envelopes', 'transfer matrices']

    for key in c_plots.keys():
        if key not in implemented:
            logging.error(f"The plot {key} is not implemented. Authorized "
                          f"values are: {implemented}.")
            passed = False

        if key in to_update_or_fix:
            logging.warning(f"The plot {key} does not work as it should. "
                            "Sorry!")

    if not passed:
        raise IOError("Error treating the plots parameters.")

    logging.info(f"files parameters {c_plots.name} tested with success.")


def config_to_dict(c_plots: configparser.SectionProxy) -> dict:
    """Save plots info into a dict."""
    plots = {}
    for key in c_plots.keys():
        plots[key] = c_plots.getboolean(key)
    return plots
