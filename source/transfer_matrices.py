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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import helper


# =============================================================================
# Transfer matrices
# =============================================================================
def dummy(Delta_s, gamma):
    """Return a dummy transfer matrix."""
    R_zz = np.full((2, 2), np.NaN)
    return R_zz


def z_drift(Delta_s, gamma):
    """
    Compute the longitudinal sub-matrix of a drift.

    On a more general point of view, this is the longitudinal transfer sub-
    matrix of every non-accelerating element.

    Parameters
    ----------
    Delta_s: float
        Drift length (m).
    gamma: float
        Lorentz factor.

    Returns
    -------
    R_zz: np.array
        Transfer longitudinal sub-matrix.
    """
    R_zz = np.array(([1., Delta_s*gamma**-2],
                     [0., 1.]))
    return R_zz


def z_field_map_electric_field(E_0_MeV, f_MHz, Fz_array, k_e, theta_i,
                               N_cells, nz, zmax):
    """
    Compute the z transfer submatrix of an accelerating cavity.

    In the process, it also computes the variation of energy, the synchronous
    phase, the accelerating field.

    Parameters
    ----------
    E_0_MeV: float
        Energy at the cavity's entrance in MeV.
    f_MHz: float
        Frequency of the cavity in MHz.
    Fz_array: numpy.array
        Array of electric field values in MV/m.
    k_e: float
        Electric field multiplication factor.
    theta_i: float
        Phase of particles at the cavity's entrance in deg.
    N_cells: int
        Number of cells in the cavity.
    nz: int
        Number of points (minus one) in the Fz_array.
    zmax: float
        Relative position of the cavity's exit in m.

    Returns
    -------
    R_zz: np.array
        Transfer matrix of the cavity.
    E_MeV: float
        Energy of the particle beam when it goes out of the cavity.
    ----------
    """
    #  method = 'classic'
    #  method = 'leapfrog'
    method = 'RK'

    if(E_0_MeV == 16.6):
        print('Method: ', method)

    # Set useful parameters
    omega_0 = 2e6 * np.pi * f_MHz
    gamma_0 = 1. + E_0_MeV / m_MeV
    beta_0 = np.sqrt(1. - gamma_0**-2)
    phi_0 = np.deg2rad(theta_i)

    # Local coordinates of the cavity:
    z_cavity_array = np.linspace(0., zmax, nz + 1)
    dz_cavity = z_cavity_array[1] - z_cavity_array[0]

    # Ez and its derivative functions:
    kind = 'linear'
    fill_value = 0.
    Ez_func = interp1d(z_cavity_array, k_e * Fz_array, bounds_error=False,
                       kind=kind, fill_value=fill_value, assume_sorted=True)
    dE_dz_array = np.gradient(k_e * Fz_array, dz_cavity)
    dE_dz_func = interp1d(z_cavity_array, dE_dz_array, bounds_error=False,
                          kind=kind, fill_value=fill_value, assume_sorted=True)

    # =========================================================================
    # Simulation parameters
    # =========================================================================
    # The N_cells cells cavity is divided in n*N_cells steps of length dz:
    n = 1000
    dz = zmax / (n * N_cells)

    # =========================================================================
    # Compute energy gain and synchronous phase
    # =========================================================================
    if(method == 'classic'):
        z_s = .5 * dz
        t_s = z_s / (beta_0 * c)
        E_MeV = E_0_MeV
        gamma_out = gamma_0

    elif(method == 'leapfrog'):
        # Leapfrog method:
        #   pos(i+1) = pos(i) + speed(i+0.5) * dt
        #   speed(i+0.5) = speed(i-0.5) * accel(i) * dt
        # Here, dt is not fixed but dz.
        #   z(i+1) += dz
        #   t(i+1) = t(i) + dz / (c beta(i+1/2))
        #       (time and space variables are on whole steps)
        #   beta calculated from W(i+1/2) = W(i-1/2) + qE(i)dz
        #       (speed/energy are on half steps)
        z_s = 0.
        t_s = z_s / (beta_0 * c)
        # Rewind energy
        E_MeV = E_0_MeV - q_adim * Ez_func(z_s)[()] * np.cos(phi_0) * .5 * dz
        gamma_out = 1. + E_MeV / m_MeV

    elif(method == 'RK'):
        z_s = 0.
        t_s = z_s / (beta_0 * c)
        E_MeV = E_0_MeV
        gamma_out = gamma_0

        def du_dz(z, u):
            """
            Compute derivative of system energy/time

            Parameters
            ----------
            u: np.array(2)
                First component is E_MeV(i).
                Second component is t(i).

            Return
            ------
            v: np.array(2)
                First component is (E_MeV(i+1) - E_MeV(i)) / dz.
                Second component is (t(i+1) - t(i)) / dz.
            """
            v0 = q_adim * Ez_func(z)[()] * np.cos(omega_0*u[1] + phi_0)

            gamma = 1. + u[0] / m_MeV
            beta = np.sqrt(1. - gamma**-2)
            v1 = 1. / (beta * c)

            v = np.array(([v0, v1]))
            return v

    phi_RF = phi_0 + omega_0 * t_s

    # Used to compute Vcav and phis
    F_E_real = 0.
    F_E_imag = 0.

    energy_array = np.array(([z_s, E_MeV]))
    gamma_array = [gamma_out]

    M_z_list = np.zeros((2, 2, n*N_cells))

    # Then, we loop until reaching the end of the cavity
    for i in range(n * N_cells):
        gamma_in = gamma_out

        F_E_real += q_adim * Ez_func(z_s)[()] * np.cos(phi_RF)
        F_E_imag += q_adim * Ez_func(z_s)[()] * np.sin(phi_RF)

        if(method == 'classic'):
            E_interp = Ez_func(z_s)[()]
            E_r = E_interp * np.cos(phi_RF)

            # Energy and gamma at the exit of current cell
            if(i == 0 or i == n * N_cells - 1):
                delta_E_MeV = q_adim * E_r * dz * .5
                # During first iteration, we go from z = 0 to z = dz / 2
                # During last, z = zmax - delta_z/2 to zmax
            else:
                delta_E_MeV = q_adim * E_r * dz

        elif(method == 'leapfrog'):
            E_interp = Ez_func(z_s)[()]
            E_r = E_interp * np.cos(phi_RF)
            delta_E_MeV = q_adim * E_r * dz

        elif(method == 'RK'):
            u = np.array(([E_MeV, t_s]))
            k_1 = du_dz(z_s,           u)
            k_2 = du_dz(z_s + .5 * dz, u + .5 * dz * k_1)
            k_3 = du_dz(z_s + .5 * dz, u + .5 * dz * k_2)
            k_4 = du_dz(z_s + dz,      u + dz * k_3)
            delta_u = (k_1 + 2.*k_2 + 2.*k_3 + k_4) * dz / 6.
            delta_E_MeV = delta_u[0]
            delta_t = delta_u[1]

        E_MeV += delta_E_MeV
        gamma_out = gamma_in + delta_E_MeV / m_MeV
        beta_out = np.sqrt(1. - gamma_out**-2)

        gamma_s = (gamma_out + gamma_in) * .5
        beta_s = np.sqrt(1. - gamma_s**-2)

        # Compute transfer matrix using thin lens approximation
        if(method == 'classic'):
            K_0 = q_adim * np.cos(phi_RF) * dz / (gamma_s * beta_s**2 * m_MeV)
            K_1 = dE_dz_func(z_s)[()] * K_0
            K_2 = 1. - (2. - beta_s**2) * Ez_func(z_s) * K_0

        elif(method == 'leapfrog'):
            K_0 = q_adim * np.cos(phi_RF) * dz / (gamma_s * beta_s**2 * m_MeV)
            K_1 = dE_dz_func(z_s)[()] * K_0
            K_2 = 1. - (2. - beta_s**2) * Ez_func(z_s) * K_0

        elif(method == 'RK'):
            z_essai = z_s + .5*dz
            delta_t_essai = dz / (2. * beta_s * c)
            phi_essai = (t_s + delta_t_essai) * omega_0 + phi_0
            K_0 = q_adim * dz / (gamma_s * beta_s**2 * m_MeV)
            dv_of_temp = Ez_func(z_essai) * np.sin(phi_essai) * omega_0 \
                / (beta_s * c)
            dv_of_spat = dE_dz_func(z_essai)[()] * np.cos(phi_essai)
            # Best for top line and more logicl: dv_of_spat
            # Best for bottom line: dv_of_temp
            K_1 = K_0 * (dv_of_spat)
            K_2 = 1. - K_0 * (2. - beta_s**2)   \
                * Ez_func(z_essai) * np.cos(phi_essai)

        M_in = z_drift(.5 * dz, gamma_in)
        M_mid = np.array(([1., 0.],
                          [K_1, K_2]))
        M_out = z_drift(.5 * dz, gamma_out)

        # Compute M_out * M_mid * M_in * M_t
        M_z = M_out @ M_mid @ M_in
        M_z_list[:, :, i] = np.copy(M_z)

        if(method == 'classic'):
            # Next step
            z_s += dz
            t_s += dz / (beta_out * c)

        elif(method == 'leapfrog'):
            z_s += dz
            t_s += dz / (beta_out * c)

        elif(method == 'RK'):
            z_s += dz
            t_s += delta_t

        phi_RF = omega_0 * t_s + phi_0

        # Save data
        energy_array = np.vstack((energy_array, np.array(([z_s, E_MeV]))))
        gamma_array.append(gamma_out)

    # =========================================================================
    # End of loop
    # =========================================================================
    gamma_array = np.array(gamma_array)

    MT_and_energy_evolution = np.full((n * N_cells, 3, 2), np.NaN)
    for i in range(n*N_cells):
        # First line: position and energy
        MT_and_energy_evolution[i, 0, :] = energy_array[i, :]
        # Second and third line: transfer matrix components of slices
        # (a recursive matrix product shall be performed in order to obtain
        # real transfer matrix components)
        MT_and_energy_evolution[i, 1:, :] = M_z_list[:, :, i]

    # Global MT of element
    R_zz = helper.right_recursive_matrix_product(M_z_list,
                                                 idx_min=0,
                                                 idx_max=n * N_cells - 1)

    # Synchronous phase
    phi_s = np.arctan(F_E_imag / F_E_real)
    phi_s_deg = np.rad2deg(phi_s)
    V_cav_MV = np.abs((E_0_MeV - MT_and_energy_evolution[-1, 0, 1])
                      / np.cos(phi_s))

    return R_zz, MT_and_energy_evolution, V_cav_MV, phi_s_deg


def not_an_element(Delta_s, gamma):
    """Return identity matrix."""
    R_zz = np.eye(2, 2)
    return R_zz


def z_sinus_cavity(L_m, E_0_MeV, f_MHz, EoT, theta_s, N):
    """
    Compute the z transfer submatrix of a sinus cavity.

    In the process, it also computes the variation of energy, the synchronous
    phase, the accelerating field.

    Parameters
    ----------
    L_m: float
        Cavity length in m.
    E_0_MeV: float
        Energy at the cavity's entrance in MeV.
    f_MHz: float
        Frequency of the cavity in MHz.
    EoT: float
        Mean electric field of the cavity in V/m.
    theta_s: float
        Phase of the synchronous particle at the entrance of the cavity,
        in deg and relative to the RF phase.
    N: int
        Number of cells in the cavity.

    Returns
    -------
    R_zz: np.array
        Transfer matrix of the cavity.
    E_MeV: float
        Energy of the particle beam when it goes out of the cavity.
    """
    flag_plot = False

    # Set useful parameters
    omega_0 = 2e6 * np.pi * f_MHz
    gamma_0 = 1. + E_0_MeV / m_MeV
    beta_0 = np.sqrt(1. - gamma_0**-2)
    lambda_RF = 1e-6 * c / f_MHz
    beta_c = 2. * L_m / (N * lambda_RF)
    K = omega_0 / c
    phi_0 = np.deg2rad(theta_s)
    E0_MV_m = 2e-6 * EoT

    # =========================================================================
    # Simulation parameters
    # =========================================================================
    # Initial step size calculated with betalambda
    n = 1000
    # The N_cells cells cavity is divided in n*N_cells steps of length dz:
    dz = L_m / (n * N)

    # =========================================================================
    # Compute energy gain and synchronous phase
    # =========================================================================
    # Position of synch. particle advanced by a half-step
    z_s = .5 * dz
    # Time corresponding to this position (we consider that speed is constant
    # during this short half-step)
    t_s = z_s / (beta_0 * c)
    phi_s = omega_0 * t_s
    phi_RF = phi_0 + phi_s

    E_MeV = E_0_MeV
    gamma_out = gamma_0

    z_pos_array = [0.]
    energy_array = [E_0_MeV]
    gamma_array = [gamma_0]

    M_z_list = np.zeros((2, 2, n*N))

    # Then, we loop until reaching the end of the cavity
    for i in range(n * N):
        gamma_in = gamma_out

        # Compute energy gain
        E_z = E0_MV_m * np.sin(phi_RF) * np.sin(K * z_s / beta_c)
        if(i == 0 or i == n * N - 1):
            delta_E_MeV = q_adim * E_z * dz * .5
            # During first iteration, we go from z = 0 to z = dz / 2
            # During last, z = zmax - delta_z/2 to zmax
        else:
            delta_E_MeV = q_adim * E_z * dz
        E_MeV += delta_E_MeV
        gamma_out = 1. + E_MeV / m_MeV
        beta_out = np.sqrt(1. - gamma_out**-2)

        # We take gamma and beta at the middle of current cell
        gamma_s = (gamma_out + gamma_in) * .5
        beta_s = np.sqrt(1. - gamma_s**-2)

        # Compute transfer matrix using thin lens approximation
        K_0 = q_adim * E0_MV_m * np.sin(K * z_s / beta_c)  \
            / (gamma_s * beta_s**2 * m_MeV) * dz
        K_1 = -K_0 * K * np.cos(phi_RF) / beta_s
        K_2 = 1. - K_0 * np.sin(phi_RF)

        M_in = z_drift(.5 * dz, gamma_in)
        M_mid = np.array(([1., 0.],
                          [K_1, K_2]))
        M_out = z_drift(.5 * dz, gamma_out)

        # Compute M_in * M_mid * M_out * M_t
        M_z = M_in @ M_mid @ M_out
        M_z_list[:, :, i] = np.copy(M_z)

        # Next step
        if(i < n * N - 1):
            z_s += dz
            t_s += dz / (beta_s * c) # Error is lower with beta_s than with beta_out
        else:
            # This values will not be used again, so this has no influence on
            # the results
            z_s += .5 * dz
            t_s += .5 * dz / (beta_s * c)
        phi_RF = omega_0 * t_s + phi_0

        # Save data
        z_pos_array.append(z_s)
        energy_array.append(E_MeV)
        gamma_array.append(gamma_out)

    # =========================================================================
    # End of loop
    # =========================================================================
    z_pos_array = np.array(z_pos_array)
    energy_array = np.array(energy_array)
    gamma_array = np.array(gamma_array)
    R_zz = helper.right_recursive_matrix_product(M_z_list,
                                                idx_min=0,
                                                idx_max=n * N - 1)

    E_out_MeV = energy_array[-1]

    if(flag_plot):
        if(plt.fignum_exists(22)):
            fig = plt.figure(22)
            ax = fig.axes[0]
        else:
            fig = plt.figure(22)
            ax = fig.add_subplot(111)
        iter_array = np.linspace(0, n*N-1, n*N)
        ax.plot(iter_array, M_z_list[0, 0, :], label='M_11')
        ax.plot(iter_array, M_z_list[0, 1, :], label='M_12')
        ax.plot(iter_array, M_z_list[1, 0, :], label='M_21')
        ax.plot(iter_array, M_z_list[1, 1, :], label='M_22')
        ax.grid(True)
        ax.legend()

    return R_zz, E_out_MeV
