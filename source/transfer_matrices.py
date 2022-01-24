#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021

@author: placais

File holding all the longitudinal transfer sub-matrices. Units are taken
exactly as in TraceWin, i.e. first line is z (m) and second line is dp/p.
"""

import numpy as np
from constants import c, q_adim, E_rest_MeV
import helper
import solver
import elements
import particle


# =============================================================================
# Transfer matrices
# =============================================================================
def dummy():
    """Return a dummy transfer matrix."""
    r_zz = np.full((2, 2), np.NaN)
    return r_zz


def z_drift_element(element, synch):
    """
    Compute the longitudinal sub-matrix of a drift.

    On a more general point of view, this is the longitudinal transfer sub-
    matrix of every non-accelerating element.

    Parameters
    ----------
    element:
        Length and gamma are extracted from this object.

    FIXME: I think there are better options...
    """
    assert isinstance(synch, particle.Particle), 'A Particle should be ' \
        + 'given if element_or_length is an Element.'
    n_steps = element.solver_transf_mat.n_steps
    delta_s = element.length_m / n_steps
    r_zz = np.full((n_steps, 2, 2), np.NaN)

    for i in range(n_steps):
        r_zz[i, :, :] = z_drift_length(delta_s, synch.energy['gamma'])

        synch.advance_position(delta_s)
        synch.set_energy(0., delta_e=True)
        delta_phi = synch.omega0['ref'] * delta_s / (synch.energy['beta']
                                                     * c)
        synch.advance_phi(delta_phi)

    return r_zz


def z_drift_length(delta_s, gamma):
    """
    Compute the longitudinal sub-matrix of a drift.

    Parameters
    ----------
    delta_s:
        Length of the drift.
    gamma: float
        Lorentz velocity factor.
    """
    r_zz = np.array(([1., delta_s*gamma**-2],
                     [0., 1.]))
    return r_zz


def z_field_map_electric_field(cavity, synch):
    """
    Compute the z transfer submatrix of an accelerating cavity.

    In the process, it also computes the variation of energy, the synchronous
    phase, the accelerating field.
    """
    assert isinstance(cavity, elements.FieldMap)
    assert isinstance(synch, particle.Particle)
    acc_field = cavity.acc_field
    solver_param = cavity.solver_transf_mat
    synch.enter_cavity(acc_field.omega0_rf)

# =============================================================================
# Initialisation
# =============================================================================
    # gamma at entrance, middle and exit of cavity
    gamma = {'in': synch.energy['gamma'],
             'middle': None,
             'out': None}
    # Variation of synch part parameters (not necessary for z, as it is always
    # dz)
    delta = {'e_mev': None,
             'phi': None}

    # Initialize gamma and synch:
    if solver_param.method == 'leapfrog':
        solver.init_leapfrog_cavity(acc_field, gamma, solver_param.d_z, synch)

    elif solver_param.method == 'RK':
        du_dz = solver.init_rk4_cavity(acc_field, gamma, synch)

    # We loop until reaching the end of the cavity
    cavity.transfer_matrix = np.zeros((solver_param.n_steps + 1, 2, 2))
    cavity.transfer_matrix[0, :, :] = np.eye(2)

# =============================================================================
# Loop over cavity
# =============================================================================
    for i in range(1, solver_param.n_steps + 1):
        gamma['in'] = gamma['out']

        # form cos + j * sin
        cavity.f_e = q_adim * acc_field.e_func(synch.z['rel'],
                                               synch.phi['rel']) \
            * (1. + np.tan(synch.phi['rel']) * 1j)

        if solver_param.method == 'leapfrog':
            delta['e_mev'] = q_adim \
                * acc_field.e_func(synch.z['rel'], synch.phi['rel']) \
                * solver_param.d_z

        elif solver_param.method == 'RK':
            u_rk = np.array(([synch.energy['kin_mev'],
                              synch.phi['rel']]))
            temp = solver.rk4(u_rk, du_dz, synch.z['rel'],
                              solver_param.d_z)
            delta['e_mev'] = temp[0]
            delta['phi'] = temp[1]

        synch.set_energy(delta['e_mev'], delta_e=True)
        gamma['out'] = synch.energy['gamma']

        # Warning, the gamma and beta in synch object are at the exit of the
        # cavity. We recompute the gamma and beta in the middle of the cavity.
        gamma['middle'] = (gamma['out'] + gamma['in']) * .5
        beta_middle = helper.gamma_to_beta(gamma['middle'])

        # Compute transfer matrix using thin lens approximation
        cavity.transfer_matrix[i, :, :] = z_thin_lens(acc_field,
                                                      solver_param.d_z, gamma,
                                                      beta_middle, synch)

        if solver_param.method == 'leapfrog':
            delta['phi'] = acc_field.n_cell * synch.omega0['bunch'] \
                * solver_param.d_z / (helper.gamma_to_beta(gamma['out']) * c)

        synch.advance_phi(delta['phi'])
        synch.advance_position(solver_param.d_z)

    synch.exit_cavity()


def z_thin_lens(acc_field, d_z, gamma, beta_middle, synch,
                flag_correction_determinant=True):
    """
    Compute the longitudinal transfer matrix of a thin slice of cavity.

    The slice is assimilated as a 'drift-gap-drift' (thin lens approximation).

    Parameters
    ----------
    acc_field: namedtuple
        Holds electric field function and important parameters of the electric
        field.
    d_z: real
        Spatial step in m.
    gamma: dict
        Lorentz factor of synchronous particle at entrance, middle and exit of
        the drift-gap-drift.
    beta_middle:
        Lorentz factor at middle of cavity (beta_s in TW doc).
    synch: dict
        Sych_part dict.
    flag_correction_determinant: boolean, optional
        To activate/deactivate the correction of the determinant (absent from
        TraceWin documentation).

    Return
    ------
    m_z: np.array((2, 2))
        Longitudinal transfer matrix of the drift-gap-drift.
    """
    assert isinstance(gamma, dict)

# =============================================================================
#   In
# =============================================================================
    transf_mat = z_drift_length(.5 * d_z, gamma['in'])

# =============================================================================
#   Mid
# =============================================================================
    # We place ourselves at the middle of the gap:
    z_k = synch.z['rel'] + .5 * d_z
    delta_phi_half_step = .5 * d_z * acc_field.omega0_rf / (beta_middle * c)
    phi_k = synch.phi['rel'] + delta_phi_half_step

    # Transfer matrix components
    k_0 = q_adim * d_z / (gamma['middle'] * beta_middle**2 * E_rest_MeV)
    k_1 = k_0 * acc_field.de_dt_func(z_k, phi_k, beta_middle)
    k_2 = 1. - (2. - beta_middle**2) * k_0 * acc_field.e_func(z_k, phi_k)

    # Correction to ensure det < 1
    if flag_correction_determinant:
        k_3 = (1. - k_0 * acc_field.e_func(z_k, phi_k))  \
                / (1. - k_0 * (2. - beta_middle**2)
                   * acc_field.e_func(z_k, phi_k))
        transf_mat = np.array(([k_3, 0.], [k_1, k_2])) @ transf_mat

    else:
        transf_mat = np.array(([1., 0.], [k_1, k_2])) @ transf_mat

# =============================================================================
#   Out
# =============================================================================
    transf_mat = z_drift_length(.5 * d_z, gamma['out']) @ transf_mat
    return transf_mat


def not_an_element():
    """Return identity matrix."""
    r_zz = np.eye(2, 2)
    return r_zz
