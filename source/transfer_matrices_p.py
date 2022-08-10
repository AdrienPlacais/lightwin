#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021.

@author: placais

File holding all the longitudinal transfer sub-matrices. Units are taken
exactly as in TraceWin, i.e. first line is z (m) and second line is dp/p.
"""

import numpy as np
from constants import c, q_adim, E_rest_MeV, inv_E_rest_MeV, OMEGA_0_BUNCH, \
    E_MEV


# =============================================================================
# Electric field functions
# =============================================================================
def e_func(k_e, z, e_spat, phi, phi_0):
    """Electric field."""
    return k_e * e_spat(z) * np.cos(phi + phi_0)


def de_dt_func(k_e, z, e_spat, phi, phi_0, factor):
    """Time-derivative of electric field."""
    return factor * k_e * e_spat(z) * np.sin(phi + phi_0)


# =============================================================================
# Motion integration functions
# =============================================================================
def rk4(u, du_dx, x, dx):
    """
    4-th order Runge-Kutta integration.

    This function calculates the variation of u between x and x+dx.

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
    k_1 = du_dx(x, u)
    k_2 = du_dx(x + half_dx, u + half_dx * k_1)
    k_3 = du_dx(x + half_dx, u + half_dx * k_2)
    k_4 = du_dx(x + dx, u + dx * k_3)
    delta_u = (k_1 + 2. * k_2 + 2. * k_3 + k_4) * dx / 6.
    return delta_u


# =============================================================================
# Transfer matrices
# =============================================================================
def z_drift(delta_s, w_kin_in, n_steps=1):
    """Calculate the transfer matrix of a drift."""
    gamma_in_min2 = (1. + w_kin_in * inv_E_rest_MeV)**-2
    r_zz = np.full((n_steps, 2, 2), np.array([[1., delta_s * gamma_in_min2],
                                              [0., 1.]]))
    beta_in = np.sqrt(1. - gamma_in_min2)
    delta_phi = OMEGA_0_BUNCH * delta_s / (beta_in * c)

    # Two possibilites: second one is faster
    # l_W_kin = [w_kin_in for i in range(n_steps)]
    # l_phi_rel = [(i+1)*delta_phi for i in range(n_steps)]
    # w_phi = np.empty((n_steps, 2))
    # w_phi[:, 0] = l_W_kin
    # w_phi[:, 1] = l_phi_rel
    w_phi = np.empty((n_steps, 2))
    w_phi[:, 0] = w_kin_in
    w_phi[:, 1] = np.arange(0., n_steps) * delta_phi + delta_phi
    return r_zz, w_phi, None


def z_field_map_rk4(d_z, w_kin_in, n_steps, omega0_rf, k_e, phi_0_rel, e_spat):
    """Calculate the transfer matrix of a FIELD_MAP using Runge-Kutta."""
    z_rel = 0.
    itg_field = 0.
    half_dz = .5 * d_z

    r_zz = np.empty((n_steps, 2, 2))
    w_phi = np.empty((n_steps + 1, 2))
    w_phi[0, 0] = w_kin_in
    w_phi[0, 1] = 0.
    l_gamma = [1. + w_kin_in * inv_E_rest_MeV]

    # Define the motion function to integrate
    def du_dz(z, u):
        """
        Compute variation of energy and phase.

        Parameters
        ----------
        z : real
            Position where variation is calculated.
        u : np.array
            First component is energy in MeV. Second is phase in rad.

        Return
        ------
        v : np.array
            First component is delta energy / delta z in MeV / m.
            Second is delta phase / delta_z in rad / m.
        """
        v0 = q_adim * e_func(k_e, z, e_spat, u[1], phi_0_rel)
        gamma_float = 1. + u[0] * inv_E_rest_MeV
        beta = np.sqrt(1. - gamma_float**-2)
        v1 = omega0_rf / (beta * c)
        return np.array([v0, v1])

    for i in range(n_steps):
        # Compute energy and phase changes
        delta_w_phi = rk4(w_phi[i, :], du_dz, z_rel, d_z)
        # Update itg_field. Used to compute V_cav and phi_s.
        itg_field += e_func(k_e, z_rel, e_spat, w_phi[i, 1], phi_0_rel) \
            * (1. + 1j * np.tan(w_phi[i, 1] + phi_0_rel)) * d_z

        w_phi[i + 1, :] = w_phi[i, :] + delta_w_phi
        l_gamma.append(1. + w_phi[i + 1, 0] * inv_E_rest_MeV)

        gamma_middle = .5 * (l_gamma[-1] + l_gamma[-2])
        beta_middle = np.sqrt(1. - gamma_middle**-2)

        r_zz[i, :, :] = z_thin_lense(z_rel, d_z, half_dz, w_phi[i:i + 2, :],
                                     gamma_middle, beta_middle, omega0_rf,
                                     k_e, phi_0_rel, e_spat)
        z_rel += d_z

    return r_zz, w_phi[1:, :], itg_field


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
