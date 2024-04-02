#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define various functions to compute the synchronous phase."""
import cmath
import math
import logging

# from beam_calculation.simulation_output.simulation_output import \
#     SimulationOutput


def phi_s_legacy(integrated_field: complex | float) -> dict[str, float]:
    """
    Compute the synchronous phase with its historical definition.

    Parameters
    ----------
    integrated_field : complex
        Complex electric field felt by the synchronous particle.

    Returns
    -------
    phi_s : float
        Synchronous phase of the cavity.

    """
    polar_itg = cmath.polar(integrated_field)
    cav_params = {'v_cav_mv': polar_itg[0],
                  'phi_s': polar_itg[1]}
    return cav_params


def phi_s_lagniel(simulation_output: object) -> float:
    """
    Compute synchronous phase with new model proposed by JM Lagniel.

    See  Longitudinal beam dynamics at high accelerating fields, what changes?
    ROSCOFF 2021.

    Parameters
    ----------
    transf_mat_21 : float
        (2, 1) component of the field map transfer matrix.
    delta_w_kin : float
        Energy gained by the synchronous particle in the cavity.

    Returns
    -------
    phi_s : float
        Corrected synchronous phase of the cavity.

    """
    logging.error("phi_s_lagniel not implemented")
    transf_mat_21 = simulation_output.transf_mat_21
    delta_w_kin = simulation_output.delta_w_kin
    return transf_mat_21 / delta_w_kin


def phi_s_from_tracewin_file(simulation_output: object) -> float:
    """Get the synchronous phase from a TraceWin output file.

    It is up to you to edit the ``tracewin.ini`` file in order to have the
    synchronous phase that you want.

    """
    logging.error("phi_s_tracewin not implemented")
    filepath = simulation_output.filepath
    del filepath
    return -math.pi / 4.


SYNCHRONOUS_PHASE_FUNCTIONS = {
    'legacy': phi_s_legacy,
    'historical': phi_s_legacy,
    'lagniel': phi_s_lagniel,
    'tracewin': phi_s_from_tracewin_file,
}  #:
