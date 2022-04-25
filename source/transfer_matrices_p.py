#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021.

@author: placais

File holding all the longitudinal transfer sub-matrices. Units are taken
exactly as in TraceWin, i.e. first line is z (m) and second line is dp/p.
"""

import numpy as np
from constants import c, q_adim, E_rest_MeV, inv_E_rest_MeV, OMEGA_0_BUNCH


# =============================================================================
# Transfer matrices
# =============================================================================
def z_drift(delta_s, W_kin_in, n_steps=1):
    """Calculate the transfer matrix of a drift."""
    gamma_in_min2 = (1. + W_kin_in * inv_E_rest_MeV)**-2
    r_zz = np.full((n_steps, 2, 2), np.array([[1., delta_s * gamma_in_min2],
                                              [0., 1.]]))
    beta_in = np.sqrt(1. - gamma_in_min2)
    delta_phi = OMEGA_0_BUNCH * delta_s / (beta_in * c)

    # Two possibilites: second one is the best
    # l_W_kin = [W_kin_in for i in range(n_steps)]
    # l_phi_rel = [(i+1)*delta_phi for i in range(n_steps)]
    # w_phi = np.empty((n_steps, 2))
    # w_phi[:, 0] = l_W_kin
    # w_phi[:, 1] = l_phi_rel
    w_phi = np.empty((n_steps, 2))
    w_phi[:, 0] = W_kin_in
    w_phi[:, 1] = np.arange(0., n_steps) * delta_phi + delta_phi
    return r_zz, w_phi, None


def e_func(k_e, z, e_spat, phi, phi_0):
    """Electric field."""
    return k_e * e_spat(z) * np.cos(phi + phi_0)


def de_dt_func(k_e, z, e_spat, phi, phi_0, factor):
    """Time-derivative of electric field."""
    return factor * k_e * e_spat(z) * np.sin(phi + phi_0)


def rk4(u, du_dx, x, dx):
    """
    4-th order Runge-Kutta integration.

    This function calculates the variation of u between x and x+dx.

    Parameters
    ----------
    u: np.array
        Holds the value of the function to integrate in x.
    df_dx: function
        Gives the variation of u components with x.
    x: real
        Where u is known.
    dx: real
        Integration step.

    Return
    ------
    delta_u: real
        Variation of u between x and x+dx.
    """
    half_dx = .5 * dx
    k_1 = du_dx(x, u)
    k_2 = du_dx(x + half_dx, u + half_dx * k_1)
    k_3 = du_dx(x + half_dx, u + half_dx * k_2)
    k_4 = du_dx(x + dx, u + dx * k_3)
    delta_u = (k_1 + 2. * k_2 + 2. * k_3 + k_4) * dx / 6.
    return delta_u


def z_field_map(d_z, W_kin_in, n_steps, omega0_rf, k_e, phi_0_rel, e_spat):
    """Calculate the transfer matrix of a FIELD_MAP."""
    z_rel = 0.
    itg_field = 0.
    half_d_z = .5 * d_z

    r_zz = np.empty((n_steps, 2, 2))
    w_phi = np.empty((n_steps + 1, 2))
    w_phi[0, 0] = W_kin_in
    w_phi[0, 1] = 0.
    l_gamma = [1. + W_kin_in * inv_E_rest_MeV]
    l_beta = [np.sqrt(1. - l_gamma[0]**-2)]

    def du_dz(z, u):
        v0 = q_adim * e_func(k_e, z, e_spat, u[1], phi_0_rel)
        gamma_float = 1. + u[0] * inv_E_rest_MeV
        beta = np.sqrt(1. - gamma_float**-2)
        v1 = omega0_rf / (beta * c)
        return np.array([v0, v1])

    for i in range(n_steps):
        # Compute energy and phase changes
        delta_w_phi = rk4(w_phi[i, :], du_dz, z_rel, d_z)

        # Update
        itg_field += e_func(k_e, z_rel, e_spat, w_phi[i, 1], phi_0_rel) \
            * (1. + 1j * np.tan(w_phi[i, 1] + phi_0_rel)) * d_z

        w_phi[i + 1, :] = w_phi[i, :] + delta_w_phi
        l_gamma.append(1. + w_phi[i + 1, 0] * inv_E_rest_MeV)
        l_beta.append(np.sqrt(1. - l_gamma[-1]**-2))

        gamma_middle = .5 * (l_gamma[-1] + l_gamma[-2])
        beta_middle = np.sqrt(1. - gamma_middle**-2)

        r_zz[i, :, :] = z_thin_lense(d_z, half_d_z, w_phi[i, 0], gamma_middle,
                                     w_phi[i + 1, 0], beta_middle, z_rel,
                                     w_phi[i, 1], omega0_rf, k_e, phi_0_rel,
                                     e_spat)

        z_rel += d_z

    return r_zz, w_phi[1:, :], itg_field


def z_thin_lense(d_z, half_dz, W_kin_in, gamma_middle, W_kin_out,
                 beta_middle, z_rel, phi_rel, omega0_rf, norm, phi_0,
                 e_spat):
    """Thin lense approximation: drift-acceleration-drift."""
    # In
    r_zz = z_drift(half_dz, W_kin_in)[0][0]

    # Middle
    z_k = z_rel + half_dz
    delta_phi_half_step = half_dz * omega0_rf / (beta_middle * c)
    phi_k = phi_rel + delta_phi_half_step

    # Transfer matrix components
    k_0 = q_adim * d_z / (gamma_middle * beta_middle**2 * E_rest_MeV)
    factor = omega0_rf / (beta_middle * c)
    k_1 = k_0 * de_dt_func(norm, z_k, e_spat, phi_k, phi_0, factor)
    e_func_k = e_func(norm, z_k, e_spat, phi_k, phi_0)
    k_2 = 1. - (2. - beta_middle**2) * k_0 * e_func_k

    # Correction to ensure det < 1
    k_3 = (1. - k_0 * e_func_k) / (1. - k_0 * (2. - beta_middle**2) * e_func_k)

    r_zz = np.array(([k_3, 0.], [k_1, k_2])) @ r_zz

    # Out
    tmp = z_drift(half_dz, W_kin_out)[0][0]
    r_zz = tmp @ r_zz

    return r_zz
