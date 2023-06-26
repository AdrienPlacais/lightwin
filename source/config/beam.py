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
    mandatory = ["linac", "e_rest_mev", "q_adim", "e_mev",
                 "f_bunch_mhz", "i_milli_a", "sigma_zdelta"]
    for key in mandatory:
        if key not in c_beam.keys():
            logging.error(f"{key} is mandatory and missing.")
            passed = False

    # Test the values of the keys in beam
    if np.abs(c_beam.getfloat("i_milli_a")) > 1e-10:
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
        'e_rest_mev': c_beam.getfloat,
        'q_adim': c_beam.getfloat,
        'e_mev': c_beam.getfloat,
        'f_bunch_mhz': c_beam.getfloat,
        'i_milli_a': c_beam.getfloat,
        'sigma_zdelta': c_beam.getmatrixfloat,
    }

    for key in c_beam.keys():
        if key in getter:
            beam[key] = getter[key](key)
            continue

        beam[key] = c_beam.get(key)

    # Add some useful keys
    beam["inv_e_rest_mev"] = 1. / beam["e_rest_mev"]
    beam["gamma_init"] = 1. + beam["e_mev"] / beam["e_rest_mev"]
    beam["omega_0_bunch"] = 2e6 * np.pi * beam["f_bunch_mhz"]
    beam["lambda_bunch"] = c / beam["f_bunch_mhz"]
    beam["q_over_m"] = beam["q_adim"] * beam["inv_e_rest_mev"]
    beam["m_over_q"] = 1. / beam["q_over_m"]

    return beam
