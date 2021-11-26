#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021

@author: placais

File holding all the longitudinal transfer sub-matrices. Units are taken
exactly as in TraceWin, i.e. first line is z (m) and second line is dp/p.
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from constants import c, q_adim, m_MeV
import helper
import solver


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


def z_field_map_electric_field(e_0_mev, f_mhz, fz_array, k_e, theta_i,
                               n_cells, n_z, z_max):
    """
    Compute the z transfer submatrix of an accelerating cavity.

    In the process, it also computes the variation of energy, the synchronous
    phase, the accelerating field.

    Parameters
    ----------
    e_0_mev: float
        Energy at the cavity's entrance in MeV.
    f_mhz: float
        Frequency of the cavity in MHz.
    fz_array: numpy.array
        Array of electric field values in MV/m.
    k_e: float
        Electric field multiplication factor.
    theta_i: float
        Phase of particles at the cavity's entrance in deg.
    n_cells: int
        Number of cells in the cavity.
    n_z: int
        Number of points (minus one) in the fz_array.
    z_max: float
        Relative position of the cavity's exit in m.

    Returns
    -------
    r_zz: np.array
        Transfer matrix of the cavity.
    e_mev: float
        Energy of the particle beam when it goes out of the cavity.
    ----------
    """
    # method = 'leapfrog'
    method = 'RK'

    if e_0_mev == 16.6:
        print('Method: ', method)

    # Set useful parameters
    omega_0 = 2e6 * np.pi * f_mhz
    gamma_0 = helper.mev_to_gamma(e_0_mev, m_MeV)
    beta_0 = helper.gamma_to_beta(gamma_0)
    phi_0 = np.deg2rad(theta_i)

    # Local coordinates of the cavity:
    z_cavity_array = np.linspace(0., z_max, n_z + 1)

    # ez and its derivative functions:
    kind = 'linear'
    fill_value = 0.
    fz_scaled = k_e * fz_array
    ez_func = interp1d(z_cavity_array, fz_scaled, bounds_error=False,
                       kind=kind, fill_value=fill_value, assume_sorted=True)

    # The n_cells cells cavity is divided in n*n_cells steps of length dz:
    n = 100
    dz = z_max / (n * n_cells)

    # =========================================================================
    # Compute energy gain and synchronous phase
    # =========================================================================
    if method == 'leapfrog':
        z_s, t_s, e_mev, gamma_out = \
            solver.init_leapfrog_cavity(phi_0, beta_0, e_0_mev,
                                        gamma_0, ez_func, dz)

    elif method == 'RK':
        z_s, t_s, e_mev, gamma_out, du_dz = \
            solver.init_rk4_cavity(omega_0, beta_0, e_0_mev,
                                   gamma_0, ez_func)

    phi_rf = omega_0 * t_s + phi_0

    # Used to compute Vcav and phis. First component is real part, second is
    # imaginary part.
    # TODO possible to make this better...
    f_e = [0., 0.]

    # We loop until reaching the end of the cavity
    n_iter = n * n_cells + 1
    m_z_list = np.zeros((n_iter, 2, 2))
    m_z_list[0, :, :] = np.eye(2)
    z_array = np.array(([0.]))
    energy_array = np.array(([e_0_mev]))
    gamma_array = [gamma_out]

    for i in range(1, n_iter):
        gamma_in = gamma_out

        f_e[0] += q_adim * ez_func(z_s)[()] * np.cos(phi_rf)
        f_e[1] += q_adim * ez_func(z_s)[()] * np.sin(phi_rf)

        if method == 'leapfrog':
            delta_e_mev = q_adim * ez_func(z_s)[()] * np.cos(phi_rf) * dz

        elif method == 'RK':
            u = np.array(([e_mev, phi_rf]))
            delta_u = solver.rk4(u, du_dz, z_s, dz)
            delta_e_mev = delta_u[0]
            delta_phi = delta_u[1]

        e_mev += delta_e_mev
        gamma_out = gamma_in + delta_e_mev / m_MeV
        beta_out = helper.gamma_to_beta(gamma_out)

        gamma_s = (gamma_out + gamma_in) * .5
        beta_s = helper.gamma_to_beta(gamma_s)

        # Compute transfer matrix using thin lens approximation
        if method == 'leapfrog':
            phi_k = phi_rf
            z_k = z_s

        elif method == 'RK':
            z_k = z_s + .5 * dz
            delta_phi_half_step = (dz * omega_0) / (2. * beta_s * c)
            phi_k = phi_rf + delta_phi_half_step

        de_dz = ez_func(z_k)[()] * np.sin(phi_k) * omega_0 / (beta_s * c)
        gamma_array = [gamma_in, gamma_s, gamma_out]
        m_z_list[i, :, :] = z_thin_lens(ez_func(z_k)[()], de_dz, dz,
                                        gamma_array, beta_s, phi_k, omega_0)

        if method == 'leapfrog':
            z_s += dz
            t_s += dz / (beta_out * c)
            phi_rf = omega_0 * t_s + phi_0

        elif method == 'RK':
            z_s += dz
            phi_rf += delta_phi

        energy_array = np.hstack((energy_array, e_mev))
        z_array = np.hstack((z_array, z_s))
        gamma_array.append(gamma_out)

    # =========================================================================
    # End of loop
    # =========================================================================
    gamma_array = np.array(gamma_array)

    # Synchronous phase
    # phi_s = np.arctan(f_e[1] / f_e[0])
    # phi_s_deg = np.rad2deg(phi_s)
    # V_cav_MV = np.abs((e_0_mev - MT_and_energy_evolution[-1, 0, 1])
    #                   / np.cos(phi_s))

    return m_z_list[1:, :, :], energy_array, z_array


def z_thin_lens(ez, dez_dt, dz, gamma_array, beta_s, phi, omega_0,
                flag_correction_determinant=True):
    """
    Compute the longitudinal transfer matrix of a thin slice of cavity.

    The slice is assimilated as a 'drift-gap-drift' (thin lens approximation).

    Parameters
    ----------
    ez: real
        z-electric field at the center of the gap.
    dez_dt: real
        Time derivative of the z electric field at the center of the gap.
    gamma_array: list(3)
        Lorentz factor of synchronous particle at entrance, middle and exit of
        the drift-gap-drift.
    beta_s:
        Lorentz factor of synchronous particle.
    phi: real
        Synchronous phase.
    omega_0: real
        RF pulsation of cavity.
    flag_correction_determinant: boolean, optional
        To activate/deactivate the correction of the determinant (absent from
        TraceWin documentation).

    Return
    ------
    m_z: np.array((2, 2))
        Longitudinal transfer matrix of the drift-gap-drift.
    """
    k_0 = q_adim * dz / (gamma_array[1] * beta_s**2 * m_MeV)
    k_1 = k_0 * dez_dt
    k_2 = 1. - (2. - beta_s**2) * k_0 * ez * np.cos(phi)

    # Correction to ensure det < 1
    if flag_correction_determinant:
        k_3 = (1. - k_0 * ez * np.cos(phi))  \
                / (1. - k_0 * (2. - beta_s**2) * ez * np.cos(phi))
        m_mid = np.array(([k_3, 0.], [k_1, k_2]))

    else:
        m_mid = np.array(([1., 0.], [k_1, k_2]))

    m_in = z_drift(.5 * dz, gamma_array[0])
    m_out = z_drift(.5 * dz, gamma_array[1])

    # Compute m_out * m_mid * m_in * m_t
    m_z = m_out @ m_mid @ m_in
    return m_z


def not_an_element():
    """Return identity matrix."""
    r_zz = np.eye(2, 2)
    return r_zz


def z_sinus_cavity(l_m, e_0_mev, f_mhz, eot, theta_s, N):
    """
    Compute the z transfer submatrix of a sinus cavity.

    In the process, it also computes the variation of energy, the synchronous
    phase, the accelerating field.

    Parameters
    ----------
    l_m: float
        Cavity length in m.
    e_0_mev: float
        Energy at the cavity's entrance in MeV.
    f_mhz: float
        Frequency of the cavity in MHz.
    eot: float
        Mean electric field of the cavity in V/m.
    theta_s: float
        Phase of the synchronous particle at the entrance of the cavity,
        in deg and relative to the RF phase.
    N: int
        Number of cells in the cavity.

    Returns
    -------
    r_zz: np.array
        Transfer matrix of the cavity.
    e_mev: float
        Energy of the particle beam when it goes out of the cavity.
    """
    flag_plot = False

    # Set useful parameters
    omega_0 = 2e6 * np.pi * f_mhz
    gamma_0 = 1. + e_0_mev / m_MeV
    beta_0 = np.sqrt(1. - gamma_0**-2)
    lambda_rf = 1e-6 * c / f_mhz
    beta_c = 2. * l_m / (N * lambda_rf)
    K = omega_0 / c
    phi_0 = np.deg2rad(theta_s)
    e0_mv_m = 2e-6 * eot

    # =========================================================================
    # Simulation parameters
    # =========================================================================
    # Initial step size calculated with betalambda
    n = 1000
    # The n_cells cells cavity is divided in n*n_cells steps of length dz:
    dz = l_m / (n * N)

    # =========================================================================
    # Compute energy gain and synchronous phase
    # =========================================================================
    # Position of synch. particle advanced by a half-step
    z_s = .5 * dz
    # Time corresponding to this position (we consider that speed is constant
    # during this short half-step)
    t_s = z_s / (beta_0 * c)
    phi_s = omega_0 * t_s
    phi_rf = phi_0 + phi_s

    e_mev = e_0_mev
    gamma_out = gamma_0

    z_pos_array = [0.]
    energy_array = [e_0_mev]
    gamma_array = [gamma_0]

    m_z_list = np.zeros((2, 2, n*N))

    # Then, we loop until reaching the end of the cavity
    for i in range(n * N):
        gamma_in = gamma_out

        # Compute energy gain
        e_z = e0_mv_m * np.sin(phi_rf) * np.sin(K * z_s / beta_c)
        if(i == 0 or i == n * N - 1):
            delta_e_mev = q_adim * e_z * dz * .5
            # During first iteration, we go from z = 0 to z = dz / 2
            # During last, z = z_max - delta_z/2 to z_max
        else:
            delta_e_mev = q_adim * e_z * dz
        e_mev += delta_e_mev
        gamma_out = 1. + e_mev / m_MeV
        # beta_out = np.sqrt(1. - gamma_out**-2)

        # We take gamma and beta at the middle of current cell
        gamma_s = (gamma_out + gamma_in) * .5
        beta_s = np.sqrt(1. - gamma_s**-2)

        # Compute transfer matrix using thin lens approximation
        k_0 = q_adim * e0_mv_m * np.sin(K * z_s / beta_c)  \
            / (gamma_s * beta_s**2 * m_MeV) * dz
        k_1 = -k_0 * K * np.cos(phi_rf) / beta_s
        k_2 = 1. - k_0 * np.sin(phi_rf)

        m_in = z_drift(.5 * dz, gamma_in)
        m_mid = np.array(([1., 0.],
                          [k_1, k_2]))
        m_out = z_drift(.5 * dz, gamma_out)

        # Compute m_in * m_mid * m_out * m_t
        m_z = m_in @ m_mid @ m_out
        m_z_list[:, :, i] = np.copy(m_z)

        # Next step
        if i < n * N - 1:
            z_s += dz
            t_s += dz / (beta_s * c)
            # Error is lower with beta_s than with beta_out
        else:
            # This values will not be used again, so this has no influence on
            # the results
            z_s += .5 * dz
            t_s += .5 * dz / (beta_s * c)
        phi_rf = omega_0 * t_s + phi_0

        # Save data
        z_pos_array.append(z_s)
        energy_array.append(e_mev)
        gamma_array.append(gamma_out)

    # =========================================================================
    # End of loop
    # =========================================================================
    z_pos_array = np.array(z_pos_array)
    energy_array = np.array(energy_array)
    gamma_array = np.array(gamma_array)
    r_zz = helper.right_recursive_matrix_product(m_z_list,
                                                 idx_min=0,
                                                 idx_max=n * N - 1)

    e_out_mev = energy_array[-1]

    if flag_plot:
        if plt.fignum_exists(22):
            fig = plt.figure(22)
            ax = fig.axes[0]
        else:
            fig = plt.figure(22)
            ax = fig.add_subplot(111)
        iter_array = np.linspace(0, n*N-1, n*N)
        ax.plot(iter_array, m_z_list[0, 0, :], label='M_11')
        ax.plot(iter_array, m_z_list[0, 1, :], label='M_12')
        ax.plot(iter_array, m_z_list[1, 0, :], label='M_21')
        ax.plot(iter_array, m_z_list[1, 1, :], label='M_22')
        ax.grid(True)
        ax.legend()

    return r_zz, e_out_mev
