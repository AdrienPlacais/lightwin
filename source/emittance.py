#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:42:41 2021

@author: placais
"""

import numpy as np
import helper
import accelerator
from constants import m_MeV, c


def mm_mrad_to_deg_mev(emit_z_z_prime, f_mhz):
    """
    Convert emittance.

    Convert z_z_prime emittance (pi.mm.mrad) to longitudinal PW emittance
    (pi.deg.MeV).
    """
    lambda_um = c / f_mhz
    emit_pw = 360. * m_MeV / lambda_um * emit_z_z_prime
    return emit_pw


def transform_mt(transfer_matrix, n_points):
    """Change form of the transfer matrix."""
    transformed = np.full((n_points, 3, 3), np.NaN)
    for i in range(n_points):
        C = transfer_matrix[i, 0, 0]
        C_prime = transfer_matrix[i, 1, 0]
        S = transfer_matrix[i, 0, 1]
        S_prime = transfer_matrix[i, 1, 1]

        transformed[i, :, :] =  np.array((
            [C**2,      -2.*C*S,                S**2],
            [-C*C_prime, C_prime*S + C*S_prime, -S*S_prime],
            [C_prime**2, -2.*C_prime*S_prime,   S_prime**2]))
    return transformed


def transport_twiss_parameters(linac, alpha_z0, beta_z0):
    """Transport Twiss parameters."""
    assert isinstance(linac, accelerator.Accelerator)
    transfer_matrix = linac.transfer_matrix_cumul
    n_points = transfer_matrix.shape[0]

    transformed = transform_mt(transfer_matrix, n_points)

    twiss = np.full((n_points, 3), np.NaN)
    twiss[0, :] = np.array(([beta_z0, alpha_z0, (1. + alpha_z0**2) / beta_z0]))

    for i in range(1, n_points):
        twiss[i, :] = transformed[i, :, :] @ twiss[0, :]
    return twiss


def plot_twiss(linac, twiss):
    """Plot Twiss parameters."""
    fig, ax = helper.create_fig_if_not_exist(33, [111])
    ax = ax[0]
    z_pos = linac.get_from_elements('pos_m', 'abs')
    ax.plot(z_pos, twiss[:, 1], label=r'$\alpha_z$')
    ax.plot(z_pos, twiss[:, 0], label=r'$\beta_z$')
    ax.plot(z_pos, twiss[:, 2], label=r'$\gamma_z$')
    ax.set_xlabel('s [m]')
    ax.set_ylabel('Twiss parameters')
    ax.grid(True)
    ax.legend()


def plot_phase_spaces(twiss):
    """Plot ellipsoid."""
    fig, ax = helper.create_fig_if_not_exist(34, [111])
    ax = ax[0]

    ax.set_xlabel('z [m]')
    ax.set_ylabel("z' [m]")
    ax.grid(True)


def beam_unnormalized_beam_emittance(w, w_prime):
    """Compute the beam rms unnormalized emittance (pi.m.rad)."""
    emitt_w = beam_rms_size(w)**2 * beam_rms_size(w_prime)**2
    emitt_w -= compute_mean(w * w_prime)**2
    emitt_w = np.sqrt(emitt_w)
    return emitt_w


def beam_unnormalized_effective_emittance(w, w_prime):
    """Compute the beam rms effective emittance (?)."""
    return 5. * beam_unnormalized_beam_emittance(w, w_prime)


def twiss_parameters(w, w_prime):
    """Compute the Twiss parameters."""
    emitt_w = beam_unnormalized_beam_emittance(w, w_prime)
    alpha_w = -beam_rms_correlation(w, w_prime) / emitt_w
    beta_w = beam_rms_size(w)**2 / emitt_w
    gamma_w = beam_rms_size(w_prime)**2 / emitt_w
    return alpha_w, beta_w, gamma_w


def compute_mean(w):
    """Compute the mean value of a property over the beam at location s <w>."""
    return np.NaN


def beam_rms_size(w):
    """Compute the beam rms size ~w."""
    return np.sqrt(compute_mean((w - compute_mean(w))**2))


def beam_rms_correlation(w, v):
    """Compute the beam rms correlation bar(wv)."""
    return compute_mean((w - compute_mean(w)) * (v - compute_mean(v)))
