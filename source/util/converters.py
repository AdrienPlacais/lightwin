#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:27:54 2023

@author: placais

All functions to change units.
"""
import logging
import numpy as np
from constants import E_REST_MEV, LAMBDA_BUNCH, Q_OVER_M, M_OVER_Q, E_REST_MEV,\
    OMEGA_0_BUNCH, c


def position(pos_in : float | np.ndarray, beta: float | np.ndarray, key: str,
             omega: float=OMEGA_0_BUNCH) -> float | np.ndarray:
    """Phase/position converters."""
    d_convert = {
        "z to phi": lambda pos, bet: -omega * pos / (bet * c),
        "phi to z": lambda pos, bet: -pos * bet * c / omega,
    }
    return d_convert[key](pos_in, beta)


def energy(energy_in: float | np.ndarray, key: str, q_over_m: float=Q_OVER_M,
           m_over_q: float=M_OVER_Q, e_rest: float=E_REST_MEV
           ) -> float | np.ndarray:
    """Convert energy or Lorentz factor into another related quantity."""
    d_convert = {
        "v to kin": lambda x: 0.5 * m_over_q * x**2 * 1e-6,
        "kin to v": lambda x: np.sqrt(2e6 * q_over_m * x),
        "kin to gamma": lambda x: 1. + x / e_rest,
        "gamma to kin": lambda x: e_rest * (x - 1.),
        "beta to gamma": lambda x: 1. / np.sqrt(1. - x**2),
        "gamma to beta": lambda x: np.sqrt(1. - x**-2),
        "kin to beta": lambda x: np.sqrt(1. - (e_rest / (x + e_rest)**2)),
        "beta to kin": lambda x: None,
        "kin to p": lambda x: np.sqrt((x + e_rest)**2 - e_rest**2),
        "p to kin": lambda x: np.sqrt(x**2 + e_rest**2) - e_rest,
        "gamma to p": lambda x: x * np.sqrt(1. - x**-2) * e_rest,
        "beta to p": lambda x: x / np.sqrt(1. - x**2) * e_rest,
    }
    return d_convert[key](energy_in)


def longitudinal(long_in: float | np.ndarray, ene: float | np.ndarray,
                 key: str, e_rest: float=E_REST_MEV) -> float | np.ndarray:
    """Convert energies between longitudinal phase spaces."""
    d_convert = {
        "zprime gamma to zdelta": lambda zp, gam: zp * gam**-2 * 1e-1,
        "zprime kin to zdelta": lambda zp, kin:
            zp * (1. + kin / e_rest)**-2 * 1e-1,
    }
    return d_convert[key](long_in, ene)


# TODO may be possible to save some operations by using lambda func?
def emittance(eps_orig: float | np.ndarray, gamma: float | np.ndarray,
              key: str, lam: float | np.ndarray=LAMBDA_BUNCH,
              e_0: float | np.ndarray=E_REST_MEV) -> float | np.ndarray:
    """Convert emittance from a phase space to another."""
    beta = np.sqrt(1. - gamma**-2)

    # Lighten the dict
    gamma2 = gamma**2
    k_1 = 360. * e_0 * (gamma * beta) / lam
    k_2 = k_1 * gamma2

    # Dict of emittances conversion constants
    d_convert = {
        "w to z": 1. / k_2,
        "z to w": k_2,
        "w to zdelta": 1e-6 / k_1,
        "zdelta to w": 1e6 * k_1,
        "z to zdelta": 1e-6 * gamma2,
        "zdelta to z": 1e6 / gamma2,
    }
    eps_new = eps_orig * d_convert[key]
    return eps_new


def twiss(twiss_orig: np.ndarray, gamma: float | np.ndarray, key: str,
          lam: float | np.ndarray=LAMBDA_BUNCH,
          e_0: float | np.ndarray=E_REST_MEV) -> np.ndarray:
    """Convert Twiss array from a phase space to another."""
    beta = np.sqrt(1. - gamma**-2)

    # Lighten the dict
    k_1 = e_0 * (gamma * beta) * lam / 360.
    k_2 = k_1 * beta**2
    k_3 = k_2 * gamma**2

    # Dict of emittances conversion constants
    d_convert = {
        "w to z": [-1., 1e-6 * k_3],
        "z to w": [-1., 1e6 / k_3],
        "w to zdelta": [-1., 1e-5 * k_2],
        "zdelta to w": [-1., 1e5 / k_2],
        "z to zdelta": [1., 1e1 * gamma**-2],
        "zdelta to z": [1., 1e-1 * gamma**2],
    }
    factors = d_convert[key]

    # New array of Twiss parameters in the desired phase space
    twiss_new = np.empty(twiss_orig.shape)
    twiss_new[:, 0] = twiss_orig[:, 0] * factors[0]
    twiss_new[:, 1] = twiss_orig[:, 1] * factors[1]
    twiss_new[:, 2] = twiss_orig[:, 2] / factors[1]
    return twiss_new


