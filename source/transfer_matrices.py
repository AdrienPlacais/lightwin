#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021

@author: placais

File holding all the longitudinal transfer sub-matrices. Units are taken
exactly as in TraceWin, i.e. first line is z (m) and second line is dp/p.
"""

import numpy as np
from constants import c, q_adim, m_MeV
import helper
import solver
import elements


# =============================================================================
# Transfer matrices
# =============================================================================
def dummy():
    """Return a dummy transfer matrix."""
    r_zz = np.full((2, 2), np.NaN)
    return r_zz


def z_drift(delta_s, gamma):
    """
    Compute the longitudinal sub-matrix of a drift.

    On a more general point of view, this is the longitudinal transfer sub-
    matrix of every non-accelerating element.

    Parameters
    ----------
    delta_s: float
        Drift length (m).
    gamma: float
        Lorentz factor.

    Returns
    -------
    r_zz: np.array
        Transfer longitudinal sub-matrix.
    """
    r_zz = np.array(([1., delta_s*gamma**-2],
                     [0., 1.]))
    return r_zz


def z_field_map_electric_field(cavity, rf_field, solver_param):
    """
    Compute the z transfer submatrix of an accelerating cavity.

    In the process, it also computes the variation of energy, the synchronous
    phase, the accelerating field.

    Parameters
    ----------
    cavity: FieldMap (Element) object
        Cavity of which you need the transfer matrix.
    rf_field: namedtuple
        Holds electric field function and important parameters of the electric
        field.
    solver_param: namedtuple
        Holds solver parameters.

    Returns
    -------
    f_e: complex
        To compute synchronous phase and acceleration in cavity.
    """
    assert isinstance(cavity, elements.FieldMap)
    assert isinstance(rf_field, elements.rf_field)
    assert isinstance(solver_param, elements.solver_param)

# =============================================================================
# Initialisation
# =============================================================================
    # Replace original two elements array
    cavity.energy_array_mev = np.array(([cavity.energy_array_mev[0]]))
    cavity.pos_m = np.linspace(0., cavity.length_m, solver_param.n_steps + 1)

    # gamma at entrance, middle and exit of cavity
    gamma = {'in': helper.mev_to_gamma(cavity.energy_array_mev[0], m_MeV),
             'synch': 0.,
             'out': 0.}
    # Synch part parameters
    synch_part = {'e_mev': 0.,
                  'phi': 0.,
                  'z': 0.}
    # Variation of synch part parameters (not necessqry for z, as it is always
    # dz)
    delta = {'e_mev': 0.,
             'phi': 0.}

    # Initialize gamma and synch_part:
    if solver_param.method == 'leapfrog':
        solver.init_leapfrog_cavity(rf_field, cavity.energy_array_mev[0],
                                    gamma, solver_param.d_z, synch_part)

    elif solver_param.method == 'RK':
        du_dz = solver.init_rk4_cavity(rf_field, cavity.energy_array_mev[0],
                                       gamma, synch_part)

    # We loop until reaching the end of the cavity
    cavity.transfer_matrix = np.zeros((solver_param.n_steps + 1, 2, 2))
    cavity.transfer_matrix[0, :, :] = np.eye(2)

# =============================================================================
# Loop over cavity
# =============================================================================
    for i in range(1, solver_param.n_steps + 1):
        gamma['in'] = gamma['out']

        f_e = q_adim * rf_field.ez_func(synch_part['z'])[()] \
            * (np.cos(synch_part['phi']) + np.sin(synch_part['phi']) * 1j)

        if solver_param.method == 'leapfrog':
            delta['e_mev'] = q_adim * rf_field.ez_func(synch_part['z'])[()] \
                * np.cos(synch_part['phi']) * solver_param.d_z

        elif solver_param.method == 'RK':
            u_rk = np.array(([synch_part['e_mev'], synch_part['phi']]))
            temp = solver.rk4(u_rk, du_dz, synch_part['z'], solver_param.d_z)
            delta['e_mev'] = temp[0]
            delta['phi'] = temp[1]

        synch_part['e_mev'] += delta['e_mev']
        gamma['out'] = gamma['in'] + delta['e_mev'] / m_MeV
        gamma['synch'] = (gamma['out'] + gamma['in']) * .5
        beta_s = helper.gamma_to_beta(gamma['synch'])

        # Compute transfer matrix using thin lens approximation
        cavity.transfer_matrix[i, :, :] = z_thin_lens(rf_field,
                                                      solver_param.d_z, gamma,
                                                      beta_s, synch_part)

        if solver_param.method == 'leapfrog':
            delta['phi'] = rf_field.omega_0 * solver_param.d_z \
                / (helper.gamma_to_beta(gamma['out']) * c)
        synch_part['phi'] += delta['phi']
        synch_part['z'] += solver_param.d_z

        cavity.energy_array_mev = np.hstack((cavity.energy_array_mev,
                                             synch_part['e_mev']))

    return f_e


def z_thin_lens(rf_field, d_z, gamma, beta_s, synch_part,
                flag_correction_determinant=True):
    """
    Compute the longitudinal transfer matrix of a thin slice of cavity.

    The slice is assimilated as a 'drift-gap-drift' (thin lens approximation).

    Parameters
    ----------
    rf_field: namedtuple
        Holds electric field function and important parameters of the electric
        field.
    d_z: real
        Spatial step in m.
    gamma: dict
        Lorentz factor of synchronous particle at entrance, middle and exit of
        the drift-gap-drift.
    beta_s:
        Lorentz factor of synchronous particle.
    synch_part: dict
        Sych_part dict.
    flag_correction_determinant: boolean, optional
        To activate/deactivate the correction of the determinant (absent from
        TraceWin documentation).

    Return
    ------
    m_z: np.array((2, 2))
        Longitudinal transfer matrix of the drift-gap-drift.
    """
    assert isinstance(synch_part, dict)
    assert isinstance(gamma, dict)

# =============================================================================
#   In
# =============================================================================
    transf_mat = z_drift(.5 * d_z, gamma['in'])

# =============================================================================
#   Mid
# =============================================================================
    # We place ourselves at the middle of the gap:
    z_k = synch_part['z'] + .5 * d_z
    delta_phi_half_step = .5 * d_z * rf_field.omega_0 / (beta_s * c)
    phi_k = synch_part['phi'] + delta_phi_half_step

    # Electric field and it's derivatives
    e_z = rf_field.ez_func(z_k)[()]
    dez_dt = e_z * np.sin(phi_k) * rf_field.omega_0 / (beta_s * c)

    # Transfer matrix components
    k_0 = q_adim * d_z / (gamma['synch'] * beta_s**2 * m_MeV)
    k_1 = k_0 * dez_dt
    k_2 = 1. - (2. - beta_s**2) * k_0 * e_z * np.cos(phi_k)

    # Correction to ensure det < 1
    if flag_correction_determinant:
        k_3 = (1. - k_0 * e_z * np.cos(phi_k))  \
                / (1. - k_0 * (2. - beta_s**2) * e_z * np.cos(phi_k))
        transf_mat = np.array(([k_3, 0.], [k_1, k_2])) @ transf_mat

    else:
        transf_mat = np.array(([1., 0.], [k_1, k_2])) @ transf_mat

# =============================================================================
#   Out
# =============================================================================
    transf_mat = z_drift(.5 * d_z, gamma['out']) @ transf_mat
    return transf_mat


def not_an_element():
    """Return identity matrix."""
    r_zz = np.eye(2, 2)
    return r_zz


# def z_sinus_cavity(l_m, e_0_mev, f_mhz, eot, theta_s, N):
#     """
#     Compute the z transfer submatrix of a sinus cavity.

#     In the process, it also computes the variation of energy, the synchronous
#     phase, the accelerating field.

#     Parameters
#     ----------
#     l_m: float
#         Cavity length in m.
#     e_0_mev: float
#         Energy at the cavity's entrance in MeV.
#     f_mhz: float
#         Frequency of the cavity in MHz.
#     eot: float
#         Mean electric field of the cavity in V/m.
#     theta_s: float
#         Phase of the synchronous particle at the entrance of the cavity,
#         in deg and relative to the RF phase.
#     N: int
#         Number of cells in the cavity.

#     Returns
#     -------
#     r_zz: np.array
#         Transfer matrix of the cavity.
#     e_mev: float
#         Energy of the particle beam when it goes out of the cavity.
#     """
#     flag_plot = False

#     # Set useful parameters
#     omega_0 = 2e6 * np.pi * f_mhz
#     gamma_0 = 1. + e_0_mev / m_MeV
#     beta_0 = np.sqrt(1. - gamma_0**-2)
#     lambda_rf = 1e-6 * c / f_mhz
#     beta_c = 2. * l_m / (N * lambda_rf)
#     K = omega_0 / c
#     phi_0 = np.deg2rad(theta_s)
#     e0_mv_m = 2e-6 * eot

#     # =========================================================================
#     # Simulation parameters
#     # =========================================================================
#     # Initial step size calculated with betalambda
#     n = 1000
#     # The n_cells cells cavity is divided in n*n_cells steps of length dz:
#     dz = l_m / (n * N)

#     # =========================================================================
#     # Compute energy gain and synchronous phase
#     # =========================================================================
#     # Position of synch. particle advanced by a half-step
#     z_s = .5 * dz
#     # Time corresponding to this position (we consider that speed is constant
#     # during this short half-step)
#     t_s = z_s / (beta_0 * c)
#     phi_s = omega_0 * t_s
#     phi_rf = phi_0 + phi_s

#     e_mev = e_0_mev
#     gamma_out = gamma_0

#     z_pos_array = [0.]
#     energy_array = [e_0_mev]
#     gamma_array = [gamma_0]

#     m_z_list = np.zeros((2, 2, n*N))

#     # Then, we loop until reaching the end of the cavity
#     for i in range(n * N):
#         gamma_in = gamma_out

#         # Compute energy gain
#         e_z = e0_mv_m * np.sin(phi_rf) * np.sin(K * z_s / beta_c)
#         if(i == 0 or i == n * N - 1):
#             delta_e_mev = q_adim * e_z * dz * .5
#             # During first iteration, we go from z = 0 to z = dz / 2
#             # During last, z = z_max - delta_z/2 to z_max
#         else:
#             delta_e_mev = q_adim * e_z * dz
#         e_mev += delta_e_mev
#         gamma_out = 1. + e_mev / m_MeV
#         # beta_out = np.sqrt(1. - gamma_out**-2)

#         # We take gamma and beta at the middle of current cell
#         gamma_s = (gamma_out + gamma_in) * .5
#         beta_s = np.sqrt(1. - gamma_s**-2)

#         # Compute transfer matrix using thin lens approximation
#         k_0 = q_adim * e0_mv_m * np.sin(K * z_s / beta_c)  \
#             / (gamma_s * beta_s**2 * m_MeV) * dz
#         k_1 = -k_0 * K * np.cos(phi_rf) / beta_s
#         k_2 = 1. - k_0 * np.sin(phi_rf)

#         m_in = z_drift(.5 * dz, gamma_in)
#         m_mid = np.array(([1., 0.],
#                           [k_1, k_2]))
#         m_out = z_drift(.5 * dz, gamma_out)

#         # Compute m_in * m_mid * m_out * m_t
#         m_z = m_in @ m_mid @ m_out
#         m_z_list[:, :, i] = np.copy(m_z)

#         # Next step
#         if i < n * N - 1:
#             z_s += dz
#             t_s += dz / (beta_s * c)
#             # Error is lower with beta_s than with beta_out
#         else:
#             # This values will not be used again, so this has no influence on
#             # the results
#             z_s += .5 * dz
#             t_s += .5 * dz / (beta_s * c)
#         phi_rf = omega_0 * t_s + phi_0

#         # Save data
#         z_pos_array.append(z_s)
#         energy_array.append(e_mev)
#         gamma_array.append(gamma_out)

#     # =========================================================================
#     # End of loop
#     # =========================================================================
#     z_pos_array = np.array(z_pos_array)
#     energy_array = np.array(energy_array)
#     gamma_array = np.array(gamma_array)
#     r_zz = helper.right_recursive_matrix_product(m_z_list,
#                                                  idx_min=0,
#                                                  idx_max=n * N - 1)

#     e_out_mev = energy_array[-1]

#     if flag_plot:
#         if plt.fignum_exists(22):
#             fig = plt.figure(22)
#             ax = fig.axes[0]
#         else:
#             fig = plt.figure(22)
#             ax = fig.add_subplot(111)
#         iter_array = np.linspace(0, n*N-1, n*N)
#         ax.plot(iter_array, m_z_list[0, 0, :], label='M_11')
#         ax.plot(iter_array, m_z_list[0, 1, :], label='M_12')
#         ax.plot(iter_array, m_z_list[1, 0, :], label='M_21')
#         ax.plot(iter_array, m_z_list[1, 1, :], label='M_22')
#         ax.grid(True)
#         ax.legend()

#     return r_zz, e_out_mev
