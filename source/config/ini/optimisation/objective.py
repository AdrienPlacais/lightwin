#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this module we set a function to check validity of ``objective_preset`` key.

The documentation relative to every objective preset is in
:mod:`optimisation.objective.factory`.

"""
import logging
import configparser


objective_presets = ('simple_ADS',
                     'sync_phase_as_objective_ADS',
                     'experimental'
                     )  #:


def test_objective_preset(c_wtf: configparser.SectionProxy) -> bool:
    """Specific test for the key 'objective_preset' of what_to_fit."""
    if 'objective_preset' not in c_wtf.keys():
        logging.error("You must provide 'objective_preset' to tell LightWin"
                      " what it should fit.")
        return False

    objective_preset = c_wtf.get('objective_preset')
    if objective_preset not in objective_presets:
        logging.error(f"Objective preset {objective_preset} was not "
                      "recognized. Check that is is implemented in "
                      "optimisation.objective.factory and that you added it "
                      "to the list of implemented in config.wtf.")
        return False
    return True