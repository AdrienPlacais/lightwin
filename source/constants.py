#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:51:33 2021.

@author: placais
"""
import numpy as np


# =============================================================================
# Physical constants
# =============================================================================
c = 2.99792458e8

# =============================================================================
# Proton
# =============================================================================
E_rest_MeV = 938.27203
inv_E_rest_MeV = 1. / E_rest_MeV
m_kg = 1.672649e-27
q_adim = 1.
q_C = 1.602176565e-19
q_over_m = q_C / m_kg
m_over_q = m_kg / q_C

# =============================================================================
# Project constants
# =============================================================================
project_folder = ''

# Input Twiss parameters
ALPHA_Z = 0.1387
BETA_Z = 20.6512404  # mm/pi%
BETA_W = 71.472671   # deg/pi.MeV

SIGMA_ZDELTA = np.array(([2.9511603e-06, -1.9823111e-07],
                         [-1.9823111e-07, 7.0530641e-07]))
# =============================================================================
# Simulation constants -- user interface
# =============================================================================
# To determine if the phases in the cavities are absolute or relative.
# If True, cavities keep their absolute phi_0 (!! relative phi_0 may be changed
# though !!).
# If False, cavities keep their relative phi_0; all cavities after the first
# modified cavity change their status to 'rephased'.
FLAG_PHI_ABS = True

# Fit performed over phi_s?
FLAG_PHI_S_FIT = False

# Method to integrate the motion. leapfrog or RK (RK4)
# METHOD = 'leapfrog'
METHOD = 'RK'

# Number of spatial steps per RF cavity cell
if 'leapfrog' in METHOD:
    N_STEPS_PER_CELL = 40
elif 'RK' in METHOD:
    N_STEPS_PER_CELL = 20
# With jm, the electric field are not interpolated. We evaluate electric field
# at every position it is known
elif 'jm' in METHOD:
    N_STEPS_PER_CELL = None

# To determine if transfer_matrices_c (Cython) should be used instead of _p
# (pure Python). _c is ~2 to 4 times faster than _p.
# Warning, you may have to relaod the kernel to force iPython to take the
# change in FLAG_CYTHON into account.
FLAG_CYTHON = True
if FLAG_CYTHON:
    METHOD += '_c'
else:
    METHOD += '_p'

E_MEV = 16.6
GAMMA_INIT = 1. + E_MEV / E_rest_MeV
F_BUNCH_MHZ = 176.1
OMEGA_0_BUNCH = 2e6 * np.pi * F_BUNCH_MHZ
LAMBDA_BUNCH = c / F_BUNCH_MHZ

# Optimisation method: least_squares or PSO
OPTI_METHOD = 'least_squares'

WHAT_TO_FIT = {
    # =========================================================================
    #     How compensating cavities are chosen?
    # =========================================================================
    # 'strategy': 'manual',
    'strategy': 'neighbors',
    # =========================================================================
    #     What should we fit?
    # =========================================================================
    # 'objective': ['energy'],
    # 'objective': ['phase'],
    'objective': 'energy_phase',
    # 'objective': ['transf_mat'],
    # 'objective': ['all'],
    # =========================================================================
    #     Where should we evaluate objective?
    # =========================================================================
    'position': 'end_mod',
    # 'position': '1_mod_after',
    # 'position': 'both',
}

# =============================================================================
# Simulation constants -- end of user interface
# =============================================================================
DICT_STR_PHI = {True: 'abs', False: 'rel'}
DICT_STR_PHI_RF = {True: 'abs_rf', False: 'rel'}
DICT_STR_PHI_0 = {True: 'phi_0_abs', False: 'phi_0_rel'}

STR_PHI_ABS = DICT_STR_PHI[FLAG_PHI_ABS]
STR_PHI_ABS_RF = DICT_STR_PHI_RF[FLAG_PHI_ABS]
STR_PHI_0_ABS = DICT_STR_PHI_0[FLAG_PHI_ABS]
