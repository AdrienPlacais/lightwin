#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a function to generate a :class:`.BeamParameters` for TraceWin."""
from typing import Callable
import logging

import numpy as np

from core.elements.element import Element
from core.beam_parameters.beam_parameters import BeamParameters
from core.transfer_matrix import TransferMatrix

from util import converters


def beam_parameters_factory(
        element_to_index: Callable[[str | Element, str | None], int | slice],
        transfer_matrix: TransferMatrix,
        sigma_in: np.ndarray,
        multipart: bool,
        **results,
        ) -> BeamParameters:
    z_abs, gamma_kin, beta_kin = _extract_beam_parameters_data(results)

    beam_parameters = BeamParameters(z_abs=z_abs,
                                     gamma_kin=gamma_kin,
                                     beta_kin=beta_kin,
                                     element_to_index=element_to_index,
                                     sigma_in=sigma_in,
                                     n_points=z_abs.shape[0])

    phase_space_names = _get_name_of_phase_spaces(multipart)
    beam_parameters.create_phase_spaces(*(phase_space_names))

    _generate_phase_spaces_from_sigma(beam_parameters,
                                      results,
                                      gamma_kin,
                                      beta_kin,
                                      *('x', 'y', 'zdelta'))
    _generate_phiw_z_t_phase_spaces_from_existing_ones(beam_parameters)
    if multipart:
        _generate_99_phase_spaces_with_eps_only(beam_parameters,
                                                results)

    return beam_parameters


def _extract_beam_parameters_data(
        results: dict[str, np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get data from results with the same name as in the reste of the code.

    Parameters
    ----------
    results : dict[str, np.ndarray]
        Results dictionary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Absolute position, Lorentz gamma, Lorentz beta.

    .. todo::
        ``function(result, *('z_abs', 'gamma_kin', 'beta_kin'))`` with some
        kind of dictionary? Would be cleaner and could be used in several
        places.

    """
    z_abs = results['z(m)']
    if z_abs[0] > 1e-10:
        logging.warning("Inconsistency with EnvelopeiD, first element of array"
                        "is not the very start of the linac.")

    gamma_kin = results['gamma']
    beta_kin = converters.energy(gamma_kin, 'gamma to beta')
    assert isinstance(beta_kin, np.ndarray)

    return z_abs, gamma_kin, beta_kin


def _get_name_of_phase_spaces(multipart: bool) -> tuple[str, ...]:
    """Determine phase space names, according to multipart simul. or not."""
    if multipart:
        return ('x', 'y', 'z', 't', 'zdelta', 'phiw', 'x99', 'y99', 'phiw99')
    return ('x', 'y', 'z', 't', 'zdelta', 'phiw')


def _generate_phase_spaces_from_sigma(beam_parameters: BeamParameters,
                                      results: dict[str, np.ndarray],
                                      gamma_kin: np.ndarray,
                                      beta_kin: np.ndarray,
                                      *phase_space_names: str) -> None:
    for phase_space_name in phase_space_names:
        args = _extract_phase_space_data_for_sigma(results,
                                                   phase_space_name)
        phase_space = beam_parameters.get(phase_space_name)
        phase_space.reconstruct_full_sigma_matrix(*args,
                                                  eps_is_normalized=True,
                                                  gamma_kin=gamma_kin,
                                                  beta_kin=beta_kin)


def _extract_phase_space_data_for_sigma(
        results: dict[str, np.ndarray],
        phase_space_name: str
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_space_to_keys = {
        'x': ('SizeX', "sxx'", 'ex'),
        'y': ('SizeY', "syy'", 'ey'),
        'zdelta': ('SizeZ', "szdp", 'ezdp'),
    }
    keys = phase_space_to_keys[phase_space_name]
    sigma_00 = results[keys[0]]**2
    sigma_01 = results[keys[1]]
    eps_normalized = results[keys[2]]
    return sigma_00, sigma_01, eps_normalized


def _generate_phiw_z_t_phase_spaces_from_existing_ones(
        beam_parameters: BeamParameters,
        ) -> None:
    r"""Generate :math:`[\phi-W]`, :math:`[z-z']` and :math:`[t]`.

    Parameters
    ----------
    beam_parameters : BeamParameters
        Object holding already existing phase spaces.

    """
    beam_parameters.init_other_phase_spaces_from_zdelta(*('phiw', 'z'))
    beam_parameters.t.init_from_averaging_x_and_y(
        beam_parameters.x,
        beam_parameters.y
    )


def _generate_99_phase_spaces_with_eps_only(beam_parameters: BeamParameters,
                                            results: dict[str, np.ndarray],
                                            ) -> None:
    r"""Generate :math:`99%` RMS in :math:`[x-x'] [y-y'] [z-\delta]`.

    Parameters
    ----------
    beam_parameters : BeamParameters
        Object with already created phase spaces.
    results : dict[str, np.ndarray]
        Holds results from TraceWin.

    """
    eps_phiw99 = results['ep99']
    eps_x99 = results['ex99']
    eps_y99 = results['ey99']
    beam_parameters.create_phase_spaces('phiw99', 'x99', 'y99')
    beam_parameters.init_99percent_phase_spaces(eps_phiw99, eps_x99, eps_y99)
