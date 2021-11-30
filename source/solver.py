#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:41:00 2021

@author: placais
"""

import numpy as np
import helper
from constants import c, q_adim, m_MeV


# =============================================================================
# RK4
# =============================================================================
def init_rk4_cavity(rf_field, e_0_mev, gamma, synch_part):
    """Init RK4 methods to compute transfer matrix of a cavity."""
    synch_part['z'] = 0.
    beta_0 = helper.gamma_to_beta(gamma['in'])
    synch_part['phi'] = rf_field.omega_0 * synch_part['z'] / (beta_0 * c) \
        + rf_field.phi_0
    synch_part['e_mev'] = e_0_mev
    gamma['out'] = gamma['in']

    def du_dz(z, u):
        """
        Compute derivative of system energy/time.

        Parameters
        ----------
        u: np.array(2)
            First component is e_mev(i).
            Second component is phi(i).

        Return
        ------
        v: np.array(2)
            First component is (e_mev(i+1) - e_mev(i)) / dz.
            Second component is (phi(i+1) - phi(i)) / dz.
        """
        v0 = q_adim * rf_field.ez_func(z)[()] * np.cos(u[1])

        gamma_float = 1. + u[0] / m_MeV
        beta = np.sqrt(1. - gamma_float**-2)
        v1 = rf_field.omega_0 / (beta * c)
        return np.array(([v0, v1]))
    return du_dz


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
    delta_u = (k_1 + 2.*k_2 + 2.*k_3 + k_4) * dx / 6.
    return delta_u


# =============================================================================
# Leapfrog
# =============================================================================
def init_leapfrog_cavity(rf_field, e_0_mev, gamma, dz, synch_part):
    """Init leapfrog method to compute transfer matrix of cavity."""
    # Leapfrog method:
    #   pos(i+1) = pos(i) + speed(i+0.5) * dt
    #   speed(i+0.5) = speed(i-0.5) * accel(i) * dt
    # Here, dt is not fixed but dz.
    #   z(i+1) += dz
    #   t(i+1) = t(i) + dz / (c beta(i+1/2))
    #       (time and space variables are on whole steps)
    #   beta calculated from W(i+1/2) = W(i-1/2) + qE(i)dz
    #       (speed/energy are on half steps)
    synch_part['z'] = 0.
    beta_0 = helper.gamma_to_beta(gamma['in'])
    synch_part['phi'] = rf_field.omega_0 * synch_part['z'] / (beta_0 * c) \
        + rf_field.phi_0
    # Rewind energy
    synch_part['e_mev'] = e_0_mev - q_adim \
        * rf_field.ez_func(synch_part['z'])[()] \
        * np.cos(rf_field.phi_0) * .5 * dz
    gamma['out'] = helper.mev_to_gamma(e_0_mev, m_MeV)
