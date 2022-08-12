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

# =============================================================================
# Simulation constants -- user interface
# =============================================================================
# To determine if the phases in the cavities are absolute or relative.
# If they are relative, the linac is implicitely rephased when some cavities
# are faulty (global compensation).
FLAG_PHI_ABS = True
# Fit performed over phi_s?
FLAG_PHI_S_FIT = True

# Method to integrate the motion. leapfrog or RK (RK4)
METHOD = 'leapfrog'
# METHOD = 'RK'

# Number of spatial steps per RF cavity cell
if 'leapfrog' in METHOD:
    N_STEPS_PER_CELL = 20
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

# Optimisation method: least_squares or PSO
OPTI_METHOD = 'least_squares'

WHAT_TO_FIT = {
    # =========================================================================
    #     How compensatong cavities are chosen?
    # =========================================================================
    # 'strategy': 'manual',
    'strategy': 'neighbors',
    # =========================================================================
    #     What should we fit?
    # =========================================================================
    # 'objective': 'energy',
    # 'objective': 'phase',
    # 'objective': 'energy_phase',
    # 'objective': 'transf_mat',
    'objective': 'all',
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
