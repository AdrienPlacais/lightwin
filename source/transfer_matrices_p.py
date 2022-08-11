#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021.

@author: placais

File holding all the longitudinal transfer sub-matrices. Units are taken
exactly as in TraceWin, i.e. first line is z (m) and second line is dp/p.

TODO check du_dz outside of field_map function
TODO reimplement itg_field
"""

import numpy as np
from constants import c, q_adim, E_rest_MeV, inv_E_rest_MeV, OMEGA_0_BUNCH, \
    E_MEV


# =============================================================================
# Electric field functions
# =============================================================================
def e_func(z, e_spat, phi, phi_0):
    """
    Give the electric field at position z and phase phi.

    The field is normalized and should be multiplied by k_e.
    """
    return e_spat(z) * np.cos(phi + phi_0)


def de_dt_func(z, e_spat, phi, phi_0):
    """
    Give the first time derivative of the electric field at (z, phi).

    The field is normalized and should be multiplied by
    k_e * omega0_rf * delta_z / c
    """
    return e_spat(z) * np.sin(phi + phi_0)


# =============================================================================
# Motion integration functions
# =============================================================================
def rk4(u, du, x, dx):
    """
    4-th order Runge-Kutta integration.

    This function calculates the variation of u between x and x+dx.
    Warning: this is a slightly modified version of the RK. The k_i are
    proportional to delta_u instead of du_dz.

    Parameters
    ----------
    u : np.array
        Holds the value of the function to integrate in x.
    du_dx : function
        Gives the variation of u components with x.
    x : real
        Where u is known.
    dx : real
        Integration step.

    Return
    ------
    delta_u : real
        Variation of u between x and x+dx.
    """
    half_dx = .5 * dx
    k_1 = du(x, u)
    k_2 = du(x + half_dx, u + .5 * k_1)
    k_3 = du(x + half_dx, u + .5 * k_2)
    k_4 = du(x + dx, u + k_3)
    delta_u = (k_1 + 2. * k_2 + 2. * k_3 + k_4) / 6.
    return delta_u


# =============================================================================
# Transfer matrices
# =============================================================================
def z_drift(delta_s, gamma_in, n_steps=1):
    """Calculate the transfer matrix of a drift."""
    gamma_in_min2 = gamma_in**-2
    r_zz = np.full((n_steps, 2, 2), np.array([[1., delta_s * gamma_in_min2],
                                              [0., 1.]]))
    beta_in = np.sqrt(1. - gamma_in_min2)
    delta_phi = OMEGA_0_BUNCH * delta_s / (beta_in * c)

    # Two possibilites: second one is faster
    # l_gamman = [gamma for i in range(n_steps)]
    # l_phi_rel = [(i+1)*delta_phi for i in range(n_steps)]
    # gamma_phi = np.empty((n_steps, 2))
    # gamma_phi[:, 0] = l_W_kin
    # gamma_phi[:, 1] = l_phi_rel
    gamma_phi = np.empty((n_steps, 2))
    gamma_phi[:, 0] = gamma_in
    gamma_phi[:, 1] = np.arange(0., n_steps) * delta_phi + delta_phi
    return r_zz, gamma_phi, None


def z_field_map_rk4(d_z, gamma_in, n_steps, omega0_rf, k_e, phi_0_rel, e_spat):
    """Calculate the transfer matrix of a FIELD_MAP using Runge-Kutta."""
    z_rel = 0.
    itg_field = 0.
    half_dz = .5 * d_z

    r_zz = np.empty((n_steps, 2, 2))
    gamma_phi = np.empty((n_steps + 1, 2))
    gamma_phi[0, 0] = gamma_in
    gamma_phi[0, 1] = 0.

    # Constants to speed up calculation
    delta_phi_norm = omega0_rf * d_z / c
    delta_gamma_norm = q_adim * d_z * inv_E_rest_MeV
    k_k = delta_gamma_norm * k_e

    # Define the motion function to integrate
    def du(z, u):
        """
        Compute variation of energy and phase.

        Parameters
        ----------
        z : real
            Position where variation is calculated.
        u : np.array
            First component is gamma. Second is phase in rad.

        Return
        ------
        v : np.array
            First component is delta gamma / delta z in MeV / m.
            Second is delta phase / delta_z in rad / m.
        """
        v0 = k_k * e_func(z, e_spat, u[1], phi_0_rel)
        beta = np.sqrt(1. - u[0]**-2)
        v1 = delta_phi_norm / beta
        return np.array([v0, v1])

    for i in range(n_steps):
        # Compute gamma and phase changes
        delta_gamma_phi = rk4(u=gamma_phi[i, :], du=du,
                              x=z_rel, dx=d_z)

        # Update
        gamma_phi[i + 1, :] = gamma_phi[i, :] + delta_gamma_phi

        # Update itg_field. Used to compute V_cav and phi_s.
        # itg_field += e_func(k_e, z_rel, e_spat, w_phi[i, 1], phi_0_rel) \
            # * (1. + 1j * np.tan(w_phi[i, 1] + phi_0_rel)) * d_z

        # Compute gamma and phi at the middle of the thin lense
        gamma_phi_middle = gamma_phi[i, :] + .5 * delta_gamma_phi

        # To speed up (corresponds to the gamma_variation at the middle of the
        # thin lense at cos(phi + phi_0) = 1
        delta_gamma_middle_max = k_k * e_spat(z_rel + half_dz)

        # r_zz[i, :, :] = z_thin_lense(z_rel, d_z, half_dz, w_phi[i:i + 2, :],
                                     # gamma_middle, beta_middle, omega0_rf,
                                     # k_e, phi_0_rel, e_spat)
        # Compute thin lense transfer matrix
        r_zz[i, :, :] = z_thin_lense2(
            gamma_phi[i, 0], gamma_phi[i + 1, 0], gamma_phi_middle,
            half_dz, delta_gamma_middle_max, phi_0_rel, omega0_rf)

        z_rel += d_z

    return r_zz, gamma_phi[1:, :], itg_field


def z_field_map_leapfrog(d_z, w_kin_in, n_steps, omega0_rf, k_e, phi_0_rel,
                         e_spat):
    """
    Calculate the transfer matrix of a FIELD_MAP using leapfrog.

    This method is less precise than RK4. However, it is much faster.

    Classic leapfrog method:
        speed(i+0.5) = speed(i-0.5) + accel(i) * dt
        pos(i+1)     = pos(i)       + speed(i+0.5) * dt

    Here, dt is not fixed but dz.
        z(i+1) += dz
        t(i+1) = t(i) + dz / (c beta(i+1/2))
    (time and space variables are on whole steps)
        beta calculated from W(i+1/2) = W(i-1/2) + qE(i)dz
    (speed/energy is on half steps)
    """
    z_rel = 0.
    itg_field = 0.
    half_dz = .5 * d_z

    r_zz = np.empty((n_steps, 2, 2))
    w_phi = np.empty((n_steps + 1, 2))
    w_phi[0, 1] = 0.
    # Rewind energy from i=0 to i=-0.5 if we are at the first cavity:
    if w_kin_in == E_MEV:
        w_phi[0, 0] = w_kin_in - q_adim * e_func(
            k_e, z_rel, e_spat, w_phi[0, 1], phi_0_rel) * half_dz
    else:
        w_phi[0, 0] = w_kin_in

    for i in range(n_steps):
        # Compute forces at step i
        delta_w = q_adim * e_func(k_e, z_rel, e_spat, w_phi[i, 1], phi_0_rel) \
            * d_z
        # Compute energy at step i + 0.5
        w_phi[i + 1, 0] = w_phi[i, 0] + delta_w
        gamma = 1. + w_phi[i + 1, 0] * inv_E_rest_MeV
        beta = np.sqrt(1. - gamma**-2)

        # Compute phase at step i + 1
        delta_phi = omega0_rf * d_z / (beta * c)
        w_phi[i + 1, 1] = w_phi[i, 1] + delta_phi

        # Update
        itg_field += e_func(k_e, z_rel, e_spat, w_phi[i, 1], phi_0_rel) \
            * (1. + 1j * np.tan(w_phi[i, 1] + phi_0_rel)) * d_z

        # We already are at the step i + 0.5, so gamma_middle and beta_middle
        # are the same as gamma and beta
        r_zz[i, :, :] = z_thin_lense(z_rel, d_z, half_dz, w_phi[i:i + 2, :],
                                     gamma, beta, omega0_rf, k_e, phi_0_rel,
                                     e_spat)
        # Strictly speaking, kinetic energy should be advanced of half a step
        # However, this does not really change the results

        z_rel += d_z

    return r_zz, w_phi[1:, :], itg_field


def z_thin_lense(z_rel, d_z, half_dz, w_phi, gamma_middle, beta_middle,
                 omega0_rf, k_e, phi_0, e_spat):
    """
    Thin lense approximation: drift-acceleration-drift.

    Parameters
    ----------
    z_rel : float
        Relative position in m.
    d_z : float
        Spatial step in m.
    half_dz : float
        Half a spatial step in m.
    phi_rel : float
        Relative phase in rad.
    w_phi : np.ndarray
        First colum is w_kin (MeV) and second is phase (rad). Only two lines:
        first is step i (entrance of first drift), second is step i + 1 (exit
        of second drift).
        w_phi = [[w_kin_in,  phi_rel_in],
                 [w_kin_out, dummy]]
    gamma_middle : float
        Lorentz mass factor at the middle of the accelerating gap.
    beta_middle : float
        Lorentz velocity factor at the middle of the accelerating gap.
    """
    # In
    r_zz = z_drift(half_dz, w_phi[0, 0])[0][0]

    # Middle
    z_k = z_rel + half_dz
    delta_phi_half_step = half_dz * omega0_rf / (beta_middle * c)
    phi_k = w_phi[0, 1] + delta_phi_half_step

    # Transfer matrix components
    k_0 = q_adim * d_z / (gamma_middle * beta_middle**2 * E_rest_MeV)
    factor = omega0_rf / (beta_middle * c)
    k_1 = k_0 * de_dt_func(k_e, z_k, e_spat, phi_k, phi_0, factor)
    e_func_k = e_func(k_e, z_k, e_spat, phi_k, phi_0)
    k_2 = 1. - (2. - beta_middle**2) * k_0 * e_func_k

    # Correction to ensure det < 1
    k_3 = (1. - k_0 * e_func_k) / (1. - k_0 * (2. - beta_middle**2) * e_func_k)

    # @ is matrix product
    r_zz = np.array(([k_3, 0.], [k_1, k_2])) @ r_zz

    # Out
    tmp = z_drift(half_dz, w_phi[1, 0])[0][0]
    r_zz = tmp @ r_zz

    return r_zz


def z_thin_lense2(gamma_in, gamma_out, gamma_phi_m, half_dz,
                  delta_gamma_m_max, phi_0, omega0_rf):
    # Used for tm components
    beta_m = np.sqrt(1. - gamma_phi_m[0]**-2)
    k_speed1 = delta_gamma_m_max / (gamma_phi_m[0] * beta_m**2)
    k_speed2 = k_speed1 * np.cos(gamma_phi_m[1] + phi_0)

    # Thin lense transfer matrices components
    k_1 = k_speed1 * omega0_rf / (beta_m * c) * np.sin(gamma_phi_m[1] + phi_0)
    k_2 = 1. - (2. - beta_m**2) * k_speed2
    k_3 = (1. - k_speed2) / k_2

    # Middle transfer matrix components
    k_1 = k_speed1 * omega0_rf / (beta_m * c) * np.sin(gamma_phi_m[1] + phi_0)
    k_2 = 1. - (2. - beta_m**2) * k_speed2
    k_3 = (1. - k_speed2) / k_2

    # Faster than matmul or matprod_22
    r_zz_array = z_drift(half_dz, gamma_out)[0][0] \
                 @ (np.array(([k_3, 0.], [k_1, k_2])) \
                    @ z_drift(half_dz, gamma_in)[0][0])
    return r_zz_array
