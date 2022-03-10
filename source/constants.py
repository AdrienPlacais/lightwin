#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:51:33 2021

@author: placais
"""

# =============================================================================
# Physical constants
# =============================================================================
c = 2.99792458e8

# =============================================================================
# Proton
# =============================================================================
E_rest_MeV = 938.27203
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
# Simulation constants
# =============================================================================
# To determine if the phases in the cavities are absolute or relative.
# If they are relative, the linac is implicitely rephased when some cavities
# are faulty.
# It is more relatable to use absolute phases when studying error compensation.
FLAG_PHI_ABS = True
DICT_STR_PHI = {True: 'abs', False: 'rel'}
STR_PHI_ABS = DICT_STR_PHI[FLAG_PHI_ABS]
