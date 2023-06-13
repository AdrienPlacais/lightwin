#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:30:58 2023.

@author: placais
"""
from typing import Any

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.envelope_1d import Envelope1D


def init_calc_with(beam_calculator_parameters: dict[str, Any]
                   ) -> BeamCalculator:
    """
    Take the appropriate beam calculator and set it up.

    Parameters
    ----------
    beam_calculator_parameters : dict
        Holds beam calculator parameters, as returned by the config_manager.

    Returns
    -------
    beam_calculator : BeamCalculator
        The solver that will compute propagation of the beam in the
        accelerator, set up according to beam_calculator_parameters.

    """
    tool = beam_calculator_parameters.pop('TOOL')
    calculators = {
        'LightWin': Envelope1D,
    }

    beam_calculator = calculators[tool](**beam_calculator_parameters)
    return beam_calculator
