#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:15:48 2021

@author: placais
"""

import numpy as np


def printc(message, color='red', opt_message=''):
    """Print colored messages."""
    if color in ('red', 'r', 'warning'):
        escape_code = '\x1B[31m'

    if color in ('blue', 'b', 'message'):
        escape_code = '\x1b[34m'

    if color in ('green', 'g', 'results'):
        escape_code = '\x1b[32m'

    if color in ('magenta', 'm', 'error'):
        escape_code = '\x1b[35m'

    if color in ('cyan', 'c', 'info'):
        escape_code = '\x1b[36m'

    normal_code = '\x1b[0m'

    print(escape_code + message + normal_code + opt_message)


def load_electric_field_1D(path):
    """
    Load a 1D electric field (.edz extension).

    Parameters
    ----------
    path: string
        The path to the .edz file to load.

    Returns
    -------
    Fz: np.array
        Array of electric field in MV/m.
    zmax: float
        z position of the filemap end.
    Norm: float
        Norm of the electric field.

    Currently not returned
    ----------------------
    nz: int
        Number of points in the array minus one.
    """
    i = 0
    k = 0

    with open(path) as file:
        for line in file:
            if(i == 0):
                line_splitted = line.split(' ')

                # Sometimes the separator is a tab and not a space:
                if(len(line_splitted) < 2):
                    line_splitted = line.split('\t')

                nz = int(line_splitted[0])
                # Sometimes there are several spaces or tabs between numbers
                zmax = float(line_splitted[-1])
                Fz = np.full((nz + 1), np.NaN)

            elif(i == 1):
                Norm = float(line)

            else:
                Fz[k] = float(line)
                k += 1

            i += 1

    return nz, zmax, Norm, Fz


def v_to_MeV(v, m_over_q):
    """Convert m/s to MeV."""
    E_eV = 0.5 * m_over_q * v**2
    return E_eV * 1e-6


def MeV_to_v(E_MeV, q_over_m):
    """Convert MeV to m/s."""
    v = np.sqrt(2. * q_over_m * 1e6 * E_MeV)
    return v


def recursive_matrix_product(M, idx_min, idx_max):
    """
    Compute the matrix product along the last array. For transfer matrices.

    Parameters
    ----------
    M: dim 3 np.array
        Array of the form (2, 2, n).
    idx_min: int
        First index to consider.
    idx_max: int
        Last index to consider.
    """
    M_out = np.eye(2)

    for i in range(idx_min, idx_max + 1):
        M_out = M_out @ M[:, :, i]

    return M_out
