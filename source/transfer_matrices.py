#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021

@author: placais

File holding all the longitudinal transfer sub-matrices. Units are taken
exactly as in TraceWin, i.e. first line is z (m) and second line is dp/p.
"""

import numpy as np


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


def not_an_element():
    """Return identity matrix."""
    R_zz = np.eye(2, 2)
    return R_zz
