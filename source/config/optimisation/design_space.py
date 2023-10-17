#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we set the function to test validity of the ``design_space_preset`` key.

The documentation relative to every preset is in
:mod:`optimisation.design_space.factory`. This is also where you should add you
own presets.

"""
import logging
import configparser


def test_design_space_preset(c_wtf: configparser.SectionProxy) -> bool:
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
