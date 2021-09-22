#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021

@author: placais
"""

import numpy as np


# =============================================================================
# Physical constants
# =============================================================================
c = 2.99792458e8


# =============================================================================
# Transfer matrices
# =============================================================================
def transfer_matrix_z_drift(Delta_s, gamma):
    """
    Compute the longitudinal sub-matrix of a drift.

    Parameters
    ----------
    Delta_s: float
        Drift length (mm).
    gamma: float
        lala

    Returns
    -------
    R_zz: np.array
        Transfer longitudinal sub-matrix.
    """
    R_zz = np.array([1., Delta_s/gamma**2],
                    [0., 1.])
    return R_zz
