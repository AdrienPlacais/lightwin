#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021

@author: placais

File holding all the longitudinal transfer sub-matrices. Units are taken
exactly as in TraceWin, i.e. first line is z (m) and second line is dp/p.
"""

import numpy as np
from constants import c, q_adim, E_rest_MeV, FLAG_PHI_ABS
import helper
import solver
import elements
import particle


# =============================================================================
# Transfer matrices
# =============================================================================
def z_drift_element(elt, synch):
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
    n_steps = elt.tmat['solver_param']['n_steps']
    delta_s = elt.length_m / n_steps
    r_zz = np.full((n_steps, 2, 2), np.NaN)

    for i in range(n_steps):
        idx_abs = elt.idx['in'] + i
        r_zz[i, :, :] = z_drift_length(delta_s,
                                       synch.energy['gamma_array'][idx_abs])

        synch.advance_position(delta_s, idx=idx_abs+1)
        synch.set_energy(0., idx=idx_abs+1, delta_e=True)
        delta_phi = synch.omega0['ref'] * delta_s \
            / (synch.energy['beta_array'][idx_abs] * c)
        synch.advance_phi(delta_phi, idx=idx_abs+1)

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
    idx_in = cavity.idx['in']
    method, n_steps, d_z = cavity.tmat['solver_param'].values()

    acc_f = cavity.acc_field
    synch.enter_cavity(acc_f, flag_cav_comp=cavity.info['compensate'],
                       idx_in=idx_in)

    # TODO: put this in Particle?
    phi = {
        True: lambda sync: sync.phi['abs_rf'],
        False: lambda sync: sync.phi['rel'],
        }

# =============================================================================
# Initialisation
# =============================================================================
    # gamma at entrance, middle and exit of cavity
    gamma = {'in': synch.energy['gamma_array'][idx_in],
             'middle': None, 'out': None}
    # Variation of synch part parameters (not necessary for z, as it is always
    # dz)
    delta = {'e_mev': None, 'phi_rf': None}

    # Initialize gamma and synch:
    if method == 'leapfrog':
        solver.init_leapfrog_cavity(cavity, gamma, d_z, synch)

    elif method == 'RK':
        du_dz = solver.init_rk4_cavity(cavity, gamma, synch)

    # We loop until reaching the end of the cavity
    transfer_matrix = np.zeros((n_steps, 2, 2))

# =============================================================================
# Loop over cavity
# =============================================================================
    for i in range(n_steps):
        idx_abs = i + idx_in
        gamma['in'] = gamma['out']

        acc_f.update_itg_field(synch.z['rel'], phi[FLAG_PHI_ABS](synch), d_z)

        if method == 'leapfrog':
            print('Warning, absolute phase not tested with leapfrog.')
            delta['e_mev'] = q_adim \
                * acc_f.e_func(synch.z['rel'], phi[FLAG_PHI_ABS](synch)) * d_z

        elif method == 'RK':
            phi_rf = phi[FLAG_PHI_ABS](synch)
            u_rk = np.array(([synch.energy['kin_array_mev'][idx_abs],
                              phi_rf]))
            temp = solver.rk4(u_rk, du_dz, synch.z['rel'], d_z)
            delta['e_mev'] = temp[0]
            delta['phi_rf'] = temp[1]

        synch.set_energy(delta['e_mev'], idx=idx_abs + 1, delta_e=True)
        gamma['out'] = synch.energy['gamma_array'][idx_abs + 1]

        # Warning, the gamma and beta in synch object are at the exit of the
        # cavity. We recompute the gamma and beta in the middle of the cavity.
        gamma['middle'] = (gamma['out'] + gamma['in']) * .5
        beta_middle = helper.gamma_to_beta(gamma['middle'])

        # Compute transfer matrix using thin lens approximation
        transfer_matrix[i, :, :] = z_thin_lens(cavity, d_z, gamma, beta_middle,
                                               synch, phi[FLAG_PHI_ABS](synch))

        if method == 'leapfrog':
            delta['phi_rf'] = acc_f.n_cell * synch.omega0['bunch'] * d_z / (
                helper.gamma_to_beta(gamma['out']) * c)

        synch.advance_phi(delta['phi_rf'], idx=idx_abs + 1, flag_rf=True)
        synch.advance_position(d_z, idx=idx_abs + 1)

    synch.exit_cavity(cavity.idx)

    return transfer_matrix


def z_thin_lens(cavity, d_z, gamma, beta_middle, synch, phi_rf,
                flag_correction_determinant=True):
    """
    Compute the longitudinal transfer matrix of a thin slice of cavity.

    The slice is assimilated as a 'drift-gap-drift' (thin lens approximation).

    Parameters
    ----------
    cavity : Element
        Cavity where the particle is.
    d_z : float
        Longitudinal spatial step in m.
    gamma : dict
        Holds Lorentz mass factor at entrance, middle and exit of the cavity.
    beta_middle : float
        Lorentz speed factor at the middle of the cavity (beta_s in TW doc).
    synch : Particle
        Particle under study.
    phi_rf : float
        Phase of the particle, expressed as phi_rf = omega_0_rf * t.
    flag_correction_determinant : boolean, optional
        Determines if the rouine enforces Det(transf_mat) < 1. The default is
        True.
    flag_phi_abs : boolean, optional
        Determines if the phase should be calculated with absolute value or
        with relative. The default is False.
    """
    assert isinstance(gamma, dict)
    acc_f = cavity.acc_field

# =============================================================================
#   In
# =============================================================================
    transf_mat = z_drift_length(.5 * d_z, gamma['in'])

# =============================================================================
#   Mid
# =============================================================================
    # We place ourselves at the middle of the gap:
    z_k = synch.z['rel'] + .5 * d_z
    delta_phi_half_step = .5 * d_z * acc_f.omega0_rf / (beta_middle * c)
    phi_rf_k = phi_rf + delta_phi_half_step
    # TODO : also update phi_k (abs/rel)

    # Transfer matrix components
    k_0 = q_adim * d_z / (gamma['middle'] * beta_middle**2 * E_rest_MeV)
    k_1 = k_0 * acc_f.de_dt_func(z_k, phi_rf_k, beta_middle)
    k_2 = 1. - (2. - beta_middle**2) * k_0 * acc_f.e_func(z_k, phi_rf_k)

    # Correction to ensure det < 1
    if flag_correction_determinant:
        k_3 = (1. - k_0 * acc_f.e_func(z_k, phi_rf_k))  \
            / (1. - k_0 * (2. - beta_middle**2)
               * acc_f.e_func(z_k, phi_rf_k))
        transf_mat = np.array(([k_3, 0.], [k_1, k_2])) @ transf_mat

    else:
        transf_mat = np.array(([1., 0.], [k_1, k_2])) @ transf_mat

# =============================================================================
#   Out
# =============================================================================
    transf_mat = z_drift_length(.5 * d_z, gamma['out']) @ transf_mat
    return transf_mat
