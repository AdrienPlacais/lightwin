#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds a factory to create the :class:`.BeamCalculator`."""
from typing import Any

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.envelope_1d.envelope_1d import Envelope1D
from beam_calculation.tracewin.tracewin import TraceWin
from beam_calculation.envelope_3d.envelope_3d import Envelope3D


def create_beam_calculator_objects(
        *beam_calculators_parameters: dict[str, Any] | None,
) -> tuple[BeamCalculator | None]:
    """
    Take the appropriate beam calculators and set them up.

    Parameters
    ----------
    *beam_calculator_parameters : dict | None
        Tuple holding beam calculator parameters, as returned by the
        `config_manager`.

    Returns
    -------
    beam_calculators : tuple[BeamCalculator | None]
        The solvers that will compute propagation of the beam in the
        accelerator, set up according to `beam_calculator_parameters`.

    """
    out_folder = 'beam_calculation'
    beam_calculators = []

    for beam_calculator_parameters in beam_calculators_parameters:
        if beam_calculator_parameters is None:
            beam_calculators.append(None)
            continue

        tool = beam_calculator_parameters['tool']
        keys_not_handled = ('tool', 'simulation type')
        clean_parameters = {key: val
                            for key, val in beam_calculator_parameters.items()
                            if key not in keys_not_handled}

        calculators = {
            'Envelope1D': Envelope1D,
            'TraceWin': TraceWin,
            'Envelope3D': Envelope3D,
        }

        beam_calculators.append(calculators[tool](out_folder=out_folder,
                                                  **clean_parameters))
        out_folder += '_post'
    return tuple(beam_calculators)
