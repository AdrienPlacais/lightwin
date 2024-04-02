#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define functions to switch between the various phases.

Mainly used by :class:`.CavitySettings`.

"""
import logging
import math
from typing import Callable

from scipy.optimize import minimize_scalar

from beam_calculation.parameters.element_parameters import \
    ElementBeamCalculatorParameters


def diff_angle(phi_1: float, phi_2: float) -> float:
    """Compute smallest difference between two angles."""
    delta_phi = math.atan2(math.sin(phi_2 - phi_1),
                           math.cos(phi_2 - phi_1)
                           )
    return delta_phi


# =============================================================================
# Conversion between different phases
# =============================================================================
def phi_0_abs_to_phi_s(phi_0_abs: float,
                       phi_s_func: Callable,
                       *args,
                       **kwargs) -> float:
    """Compute the objective synchronous phase to match ``phi_0_abs``."""
    logging.debug("phi_0_abs to phi_s...")
    return -1


def phi_0_abs_to_rel(phi_0_abs: float, phi_rf: float) -> float:
    """Compute relative entry phase from absolute."""
    logging.debug("phi_0_abs to phi_0_rel")
    phi_0_rel = (phi_0_abs + phi_rf) % (2. * math.pi)
    return phi_0_rel


def phi_0_rel_to_abs(phi_0_rel: float, phi_rf: float) -> float:
    """Compute relative entry phase from absolute."""
    logging.debug("Computing rel to abs...")
    phi_0_abs = (phi_0_rel - phi_rf) % (2. * math.pi)
    return phi_0_abs


def phi_0_rel_to_phi_s(phi_0_rel: float,
                       phi_s_func: Callable,
                       *args,
                       **kwargs) -> float:
    """Compute the objective synchronous phase to match ``phi_0_rel``."""
    logging.debug("phi_0_rel to phi_s...")
    return -1


def phi_s_to_phi_0_rel(phi_s: float,
                       beam_calculator_func: Callable,
                       phi_s_func: Callable | None,
                       ) -> float:
    r"""Compute relative entry phase from synchronous phase.

    .. todo::
        Should I square the result in ``wrapper_synch`` or ``minimize_scalar``
        takes care of it?

    Parameters
    ----------
    phi_s : float
        Synchronous phase for which we want to find corresponding
        :math:`\phi_{0, rel}`.
    phi_s_func : Callable[[SimulationOutput], float] | None, optional
        Function that takes in the :class:`.SimulationOutput` returned by
        ``beam_calculator_func`` and return the corresponding :math:`\phi_s`.
        The default is None, in which case we simply ``get`` the synchronous
        phase that shall already by in the :class:`.SimulationOutput`.
    beam_calculator_func : Callable[[Any], SimulationOutput]
        Function that takes in ``phi_0_rel`` and returns a
        :class:`.SimulationOutput`. You should use ``functools.partial`` to
        pre-set the constant arguments and keyword arguments such as ``w_kin``.

    Returns
    -------
    float
        The :math:`\phi_{0, rel}` permitting to obtain the input
        :math:`\phi_s`.

    """
    logging.debug("phi_s to phi_0_rel")

    if phi_s_func is None:
        phi_s_func = lambda simulation_output: simulation_output.get('phi_s')

    def wrapper_synch(phi_0_rel: float) -> float:
        simulation_output = beam_calculator_func(phi_0_rel)
        phi_s_from_results = phi_s_func(simulation_output)
        diff = diff_angle(phi_s, phi_s_from_results)
        return diff**2

    res = minimize_scalar(wrapper_synch, bounds=(0, 2. * math.pi))
    if not res.success:
        logging.error('Synch phase not found')

    phi_0_rel = res.x
    return phi_0_rel


def phi_bunch_to_phi_rf(phi_bunch: float,
                        rf_over_bunch_frequencies: float,
                        ) -> float:
    """Convert the bunch phase to a rf phase."""
    return phi_bunch * rf_over_bunch_frequencies
