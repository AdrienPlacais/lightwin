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
    R_zz = np.array(([1., Delta_s/gamma**2],
                     [0., 1.]))
    return R_zz


def z_field_map_electric_field(E_0_MeV, f_MHz, Fz_array, k_e, theta_i,
                               N_cell, nz, zmax):
    """
    Compute the transfer matrix of an accelerating cavity.

    In the process, it also computes the variation of energy, the synchronous
    phase, the accelerating field.

    Parameters
    ----------
    E_0_MeV: float
        Energy at the cavity's entrance in MeV.
    f_MHz: float
        Frequency of the cavity in MHz.
    Fz_array: numpy.array
        Array of electric field values in V/m.
    k_e: float
        Electric field multiplication factor.
    theta_i: float
        Phase of particles at the cavity's entrance in deg.
    N_cell: int
        Number of cells in the cavity.
    nz: int
        Number of points (minus one) in the Fz_array.
    zmax: float
        Relative position of the cavity's exit in mm.

    Returns
    -------
    R_zz: np.array
        Transfer matrix of the cavity.
    E_MeV: float
        Energy of the particle beam when it goes out of the cavity.
    ----------
    """
    # Set useful parameters
    omega_0 = 2e6 * np.pi * f_MHz
    beta_0 = np.sqrt((1. + E_0_MeV / m_MeV)**2 - 1.) / (1. + E_0_MeV / m_MeV)
    gamma_0 = 1. / np.sqrt(1. - beta_0**2)
    phi_RF = np.deg2rad(theta_i)

    # Local coordinates of the cavity:
    z_cavity_array = np.linspace(0., zmax, nz + 1)
    dz = z_cavity_array[1] - z_cavity_array[0]

    Ez_func = interp1d(z_cavity_array, k_e * Fz_array)
    dE_dz_array = np.diff(k_e * Fz_array) / dz

    # Interpolate dE/dz on the middle of every cell
    dE_dz_func = interp1d((z_cavity_array[:-1] + z_cavity_array[1:]) * 0.5,
                          dE_dz_array,
                          fill_value='extrapolate')
    # We use fill_value=extrapolate to define the function on [0, .5*dz] and
    # [zmax-.5*dz, zmax].

    # =========================================================================
    # Simulation parameters
    # =========================================================================
    # Initial step size calculated with betalambda
    factor = 1000.
    lambda_RF = 1e-6 * c / f_MHz
    # Spatial step
    step = beta_0 * lambda_RF / (2. * N_cell * factor)

    # =========================================================================
    # Compute energy gain and synchronous phase
    # =========================================================================
    # Step size
    idx_max = int(np.floor(zmax / step))

    # Init synchronous particle
    E_out_MeV = E_0_MeV
    gamma_out = gamma_0
    z = .5 * step   # Middle of first spatial step

    E_r = 0.
    E_i = 0.

    t = step / (2. * beta_0 * c)

    # TODO: check if script is faster with numpy arrays
    z_pos_array = [0.]
    energy_array = [E_0_MeV]
    beta_array = [beta_0]
    gamma_array = [gamma_0]

    R_zz = np.eye(2, 2)

    # Loop over the spatial steps
    for i in range(idx_max):
        E_in_MeV = E_out_MeV
        gamma_in = gamma_out

        # Compute energy gain
        E_interp = Ez_func(z)[()]
        acceleration = q_adim * E_interp * np.cos(phi_RF)
        E_r += acceleration
        E_i += q_adim * E_interp * np.sin(phi_RF)

        # Energy and gamma at the exit of current cell
        E_out_MeV = acceleration * step + E_in_MeV
        gamma_out = gamma_in + acceleration * step / m_MeV

        # Synchronous gamma
        gamma_s = (gamma_out + gamma_in) * .5
        beta_s = np.sqrt(1. - 1. / gamma_s**2)
        beta_out = np.sqrt(1. - 1. / gamma_out**2)

        # Synchronous phase
        phi_s = np.arctan(E_i / E_r)
        phi_s_deg = np.rad2deg(phi_s)

        # Compute transfer matrix
        K_0 = q_adim * np.cos(phi_RF) * step / (gamma_s * beta_s**2 * m_MeV)
        K_1 = dE_dz_func(z) * K_0
        K_2 = 1. - (2. - beta_s**2) * Ez_func(z) * K_0

        M_in = z_drift(.5 * step, gamma_in)
        M_mid = np.array(([1., 0.],
                          [K_1, K_2]))
        M_out = z_drift(.5 * step, gamma_out)

        # Compute M_in * M_mid * M_out * M_t
        R_zz = np.matmul(np.matmul(np.matmul(M_in, M_mid), M_out), R_zz)

        # Next step
        z += step
        t += step / (beta_s * c)
        phi_RF += step * omega_0 / (beta_s * c)

        # Save data
        z_pos_array.append(z)
        energy_array.append(E_out_MeV)
        beta_array.append(beta_out)   # TODO: check this!
        gamma_array.append(gamma_out)     # TODO: check this!

    # =========================================================================
    # End of loop
    # =========================================================================
    z_pos_array = np.array(z_pos_array)
    energy_array = np.array(energy_array)
    beta_array = np.array(beta_array)
    gamma_array = np.array(gamma_array)

    E_MeV = energy_array[-1]

    # TODO: save and/or output V_cav and phi_s
    V_cav_MV = np.abs((E_MeV - E_0_MeV) / np.cos(phi_s))
    # phi_s_deg = phi_s_deg
    print('V_cav = ', V_cav_MV, 'MV')
    print('phi_s = ', phi_s_deg, 'deg')

    return R_zz, E_MeV

    def not_an_element(Delta_s, gamma):
        """Return identity matrix."""
        R_zz = np.eye(2, 2)
        return R_zz
