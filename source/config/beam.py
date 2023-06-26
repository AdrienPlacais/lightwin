#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:50:26 2023.

@author: placais

All the functions related to the `beam` key of the config file.
"""
import logging
import configparser
import numpy as np

from constants import c


# =============================================================================
# Front end
# =============================================================================
def test(c_beam: configparser.SectionProxy) -> None:
    """Test the the beam parameters."""
    passed = True

    # Test that all mandatory keys are here
    mandatory = ["LINAC", "E_REST_MEV", "Q_ADIM", "E_MEV",
                 "F_BUNCH_MHZ", "I_MILLI_A", "SIGMA_ZDELTA"]
    for key in mandatory:
        if key not in c_beam.keys():
            logging.error(f"{key} is mandatory and missing.")
            passed = False

    # Test the values of the keys in beam
    if np.abs(c_beam.getfloat("I_MILLI_A")) > 1e-10:
        logging.warning("You asked LW a beam current different from "
                        + "0mA. Space-charge, transverse dynamics are "
                        + "not implemented yet, so this parameter "
                        + "will be ignored.")

    if not passed:
        raise IOError("Wrong value in c_beam.")

    logging.info(f"beam parameters {c_beam.name} tested with success.")


def config_to_dict(c_beam: configparser.SectionProxy) -> dict:
    """Convert beam configparser into a dict."""
    beam = {}
    # Special getters
    getter = {
        'E_REST_MEV': c_beam.getfloat,
        'Q_ADIM': c_beam.getfloat,
        'E_MEV': c_beam.getfloat,
        'F_BUNCH_MHZ': c_beam.getfloat,
        'I_MILLI_A': c_beam.getfloat,
        'SIGMA_ZDELTA': c_beam.getmatrixfloat,
    }

    for key in c_beam.keys():
        key = key.upper()
        if key in getter:
            beam[key] = getter[key](key)
            continue

        beam[key] = c_beam.get(key)

    # Add some useful keys
    beam["INV_E_REST_MEV"] = 1. / beam["E_REST_MEV"]
    beam["GAMMA_INIT"] = 1. + beam["E_MEV"] / beam["E_REST_MEV"]
    beam["OMEGA_0_BUNCH"] = 2e6 * np.pi * beam["F_BUNCH_MHZ"]
    beam["LAMBDA_BUNCH"] = c / beam["F_BUNCH_MHZ"]
    beam["Q_OVER_M"] = beam["Q_ADIM"] * beam["INV_E_REST_MEV"]
    beam["M_OVER_Q"] = 1. / beam["Q_OVER_M"]

    return beam
