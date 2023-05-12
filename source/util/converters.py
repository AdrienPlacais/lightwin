#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:27:54 2023

@author: placais

All functions to change units.
"""
import numpy as np
from constants import E_REST_MEV, LAMBDA_BUNCH


# TODO may be possible to save some operations by using lambda func?
def emittance(eps_orig: float | np.ndarray, str_convert: str,
              gamma: float | np.ndarray, beta: float | np.ndarray=None,
              lam: float | np.ndarray=LAMBDA_BUNCH,
              e_0: float | np.ndarray=E_REST_MEV) -> float | np.ndarray:
    """Convert emittance from a phase space to another."""
    if beta is None:
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
    eps_new = eps_orig * d_convert[str_convert]
    return eps_new


def twiss(twiss_orig: np.ndarray, str_convert: str, gamma: float | np.ndarray,
          beta: float | np.ndarray=None, lam: float | np.ndarray=LAMBDA_BUNCH,
          e_0: float | np.ndarray=E_REST_MEV) -> np.ndarray:
    """Convert Twiss array from a phase space to another."""
    if beta is None:
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
    factors = d_convert[str_convert]

    # New array of Twiss parameters in the desired phase space
    twiss_new = np.empty(twiss_orig.shape)
    twiss_new[:, 0] = twiss_orig[:, 0] * factors[0]
    twiss_new[:, 1] = twiss_orig[:, 1] * factors[1]
    twiss_new[:, 2] = twiss_orig[:, 2] / factors[1]
    return twiss_new


