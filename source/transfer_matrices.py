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


def z_field_map_electric_field(cavity, rf_field, method='RK'):
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
    methode: opt, str
        Solving method. 'RK' or 'leapfrog'.

    Returns
    -------
    m_z_list: np.array(n_iter+1, 2, 2)
        Array holding the transfer matrices of the drift-gap-drift slices.
    energy_array: np.array
        Energy as a function of z_array.
    z_array: np.array
        Evolution of z.
    f_e: complex
        To compute synchronous phase and acceleration in cavity.
    """
    assert isinstance(cavity, elements.FieldMap)
    assert isinstance(rf_field, elements.rf_field)

    # Set useful parameters
    energy_array = np.array(([cavity.energy_array_mev[0]]))

    # The n_cells cells cavity is divided in n*n_cells steps of length d_z:
    n_steps = 100 * rf_field.n_cell
    d_z = cavity.length_m / n_steps
    z_array = np.linspace(0., cavity.length_m, n_steps + 1)

    gamma = {'in': helper.mev_to_gamma(energy_array[0], m_MeV),
             'synch': 0.,
             'out': 0.}

    # =========================================================================
    # Compute energy gain and synchronous phase
    # =========================================================================
    if method == 'leapfrog':
        z_s, t_s, e_mev, gamma = \
            solver.init_leapfrog_cavity(rf_field, energy_array[0], gamma,
                                        d_z)

    elif method == 'RK':
        z_s, t_s, e_mev, du_dz, gamma = \
            solver.init_rk4_cavity(rf_field, energy_array[0], gamma)

    phi_rf = rf_field.omega_0 * t_s + rf_field.phi_0

    # We loop until reaching the end of the cavity
    m_z_list = np.zeros((n_steps + 1, 2, 2))
    m_z_list[0, :, :] = np.eye(2)

    # Dict to hold variations of energy and phase
    delta = {'e_mev': 0.,
             'phi': 0.}

    for i in range(1, n_steps + 1):
        gamma['in'] = gamma['out']

        f_e = q_adim * rf_field.ez_func(z_s)[()] * (np.cos(phi_rf) +
                                                    np.sin(phi_rf) * 1j)

        if method == 'leapfrog':
            delta['e_mev'] = q_adim * rf_field.ez_func(z_s)[()] \
                * np.cos(phi_rf) * d_z

        elif method == 'RK':
            u_rk = np.array(([e_mev, phi_rf]))
            temp = solver.rk4(u_rk, du_dz, z_s, d_z)
            delta['e_mev'] = temp[0]
            delta['phi'] = temp[1]

        e_mev += delta['e_mev']
        gamma['out'] = gamma['in'] + delta['e_mev'] / m_MeV
        gamma['synch'] = (gamma['out'] + gamma['in']) * .5
        beta_s = helper.gamma_to_beta(gamma['synch'])

        # Compute transfer matrix using thin lens approximation
        m_z_list[i, :, :] = z_thin_lens(rf_field, d_z, gamma, beta_s,
                                        phi_rf, z_s)

        if method == 'leapfrog':
            delta['phi'] = rf_field.omega_0 * d_z \
                / (helper.gamma_to_beta(gamma['out']) * c)
        phi_rf += delta['phi']
        z_s += d_z

        energy_array = np.hstack((energy_array, e_mev))

    return m_z_list[1:, :, :], energy_array, z_array, f_e


def z_thin_lens(rf_field, d_z, gamma, beta_s, phi_rf, z_s,
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
    phi_RF: real
        Phase.
    z_s: real
        Synchronous position.
    flag_correction_determinant: boolean, optional
        To activate/deactivate the correction of the determinant (absent from
        TraceWin documentation).

    Return
    ------
    m_z: np.array((2, 2))
        Longitudinal transfer matrix of the drift-gap-drift.
    """
    # We place ourselves at the middle of the gap:
    z_k = z_s + .5 * d_z
    delta_phi_half_step = (d_z * rf_field.omega_0) \
        / (2. * beta_s * c)
    phi_k = phi_rf + delta_phi_half_step

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
        m_mid = np.array(([k_3, 0.], [k_1, k_2]))

    else:
        m_mid = np.array(([1., 0.], [k_1, k_2]))

    m_in = z_drift(.5 * d_z, gamma['in'])
    m_out = z_drift(.5 * d_z, gamma['out'])
    m_z = m_out @ m_mid @ m_in
    return m_z


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
