#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:09:07 2023.

@author: placais

.. todo::
    Not sure if this is still used?

"""
import logging
import configparser


def test_position(c_wtf: configparser.SectionProxy) -> bool:
    """Test where the objectives are evaluated."""
    logging.warning("Position key still exists but is doublon with "
                    "objective_preset and design_space_preset. Will be "
                    "necessary to refactor.")
    if 'position' not in c_wtf.keys():
        logging.error("You must provide 'position' to tell LightWin where "
                      + "objectives should be evaluated.")
        return False

    positions = c_wtf.getliststr('position')
    implemented = (
        # End of last lattice with a compensating or failed cavity
        'end of last altered lattice',
        # One lattice after last lattice with a compensating/failed cavity
        'one lattice after last altered lattice',
        # End of last lattice with a failed cavity
        'end of last failed lattice',
        # One lattice after last lattice with a failed cavity
        'one lattice after last failed lattice',
        # End of linac
        'end of linac',
    )
    if not all(pos in implemented for pos in positions):
        logging.error("At least one position was not recognized. Allowed "
                      + f"values are: {implemented}.")
        return False
    return True
