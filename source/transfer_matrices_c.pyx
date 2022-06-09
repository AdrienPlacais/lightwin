#!/usr/bin/env python3
#cython: language_level=3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021.

@author: placais

This file holds the same functions as transfer_matrices_p, but in Cython.

Cython needs to be compiled to work. Check the instrutions in setup.py.

Currently adapted to MYRRHA only:
    electric fields are hard-coded;
    energy at linac entrance is hard-coded.
"""
import cython
from libc.math cimport sin, cos, sqrt, tan, floor
import numpy as np
cimport numpy as np
np.import_array()

# Must be changed to double if C float is replaced by double
DTYPE = np.float64
#  ctypedef np.float64_t DTYPE_t
# DTYPE = double
ctypedef double DTYPE_t

cdef DTYPE_t c_cdef = 2.99792458e8
cdef DTYPE_t E_rest_MeV_cdef = 938.27203
cdef DTYPE_t inv_E_rest_MeV_cdef = 0.0010657889908537506
cdef DTYPE_t OMEGA_0_BUNCH_cdef = 1106468932.594325
cdef DTYPE_t q_adim_cdef = 1.
cdef DTYPE_t E_MEV_cdef = 16.6

cdef int N_POINTS_SIMPLE_SPOKE = 207
cdef DTYPE_t INV_DZ_SIMPLE_SPOKE = N_POINTS_SIMPLE_SPOKE / 0.415160
cdef DTYPE_t[:] E_Z_SIMPLE_SPOKE

cdef int N_POINTS_SPOKE_ESS = 255
cdef DTYPE_t INV_DZ_SPOKE_ESS = N_POINTS_SPOKE_ESS / 0.636
cdef DTYPE_t[:] E_Z_SPOKE_ESS

cdef int N_POINTS_BETA065 = 525
cdef DTYPE_t INV_DZ_BETA065 = N_POINTS_BETA065 / 1.050
cdef DTYPE_t[:] E_Z_BETA065

cpdef init_arrays():
    """Initialize electric fields for efficiency."""
    global E_Z_SIMPLE_SPOKE
    global E_Z_SPOKE_ESS
    global E_Z_BETA065

    E_Z_SIMPLE_SPOKE = np.array([-0.00171217, -0.00171885, -0.00181223, -0.00192758, -0.0021238, -0.00236368, -0.00268261, -0.00307873, -0.00355268, -0.00415657, -0.00483353, -0.0057229, -0.00667139, -0.00795504, -0.00939794, -0.0111118, -0.0131418, -0.0155562, -0.0184837, -0.0216921, -0.0259918, -0.030679, -0.0364727, -0.0433374, -0.0513632, -0.0608689, -0.0720973, -0.0853878, -0.10138, -0.11897, -0.142345, -0.16798, -0.198994, -0.235924, -0.276936, -0.329464, -0.386502, -0.457077, -0.538241, -0.6317, -0.739509, -0.86339, -1.00509, -1.16187, -1.34439, -1.54263, -1.76185, -1.99874, -2.24952, -2.5112, -2.7784, -3.04639, -3.30987, -3.56348, -3.8025, -4.02216, -4.22121, -4.39836, -4.55322, -4.68618, -4.79821, -4.89072, -4.96498, -5.0221, -5.06518, -5.09593, -5.11591, -5.12654, -5.12906, -5.12457, -5.11387, -5.09748, -5.07642, -5.0511, -5.02178, -4.98859, -4.9515, -4.91038, -4.86482, -4.81405, -4.75781, -4.69534, -4.62573, -4.54793, -4.46076, -4.36289, -4.25268, -4.12764, -3.98733, -3.83045, -3.6559, -3.46296, -3.2514, -3.02155, -2.77421, -2.51027, -2.23344, -1.94679, -1.65304, -1.35469, -1.054, -0.752562, -0.451266, -0.150358, 0.150358, 0.451266, 0.752562, 1.054, 1.35469, 1.65304, 1.94679, 2.23344, 2.51027, 2.77421, 3.02155, 3.2514, 3.46296, 3.6559, 3.83045, 3.98733, 4.12764, 4.25268, 4.36289, 4.46076, 4.54793, 4.62573, 4.69534, 4.75781, 4.81405, 4.86482, 4.91038, 4.9515, 4.98859, 5.02178, 5.0511, 5.07642, 5.09748, 5.11387, 5.12457, 5.12906, 5.12654, 5.11591, 5.09593, 5.06518, 5.0221, 4.96498, 4.89072, 4.79821, 4.68618, 4.55322, 4.39836, 4.22121, 4.02216, 3.8025, 3.56348, 3.30987, 3.04639, 2.7784, 2.5112, 2.24952, 1.99874, 1.76185, 1.54263, 1.34439, 1.16187, 1.00509, 0.86339, 0.739509, 0.6317, 0.538241, 0.457077, 0.386502, 0.329464, 0.276936, 0.235924, 0.198994, 0.16798, 0.142345, 0.11897, 0.10138, 0.0853878, 0.0720973, 0.0608689, 0.0513632, 0.0433374, 0.0364727, 0.030679, 0.0259918, 0.0216921, 0.0184837, 0.0155562, 0.0131418, 0.0111118, 0.00939794, 0.00795504, 0.00667139, 0.0057229, 0.00483353, 0.00415657, 0.00355268, 0.00307873, 0.00268261, 0.00236368, 0.0021238, 0.00192758, 0.00181223, 0.00171885, 0.00171217], dtype=DTYPE)

    E_Z_SPOKE_ESS = np.array([0.0156, 0.0156, 0.0156, 0.0182, 0.0212, 0.0241, 0.0273, 0.0343, 0.0413, 0.0482, 0.0601, 0.0745, 0.0889, 0.108, 0.135, 0.163, 0.194, 0.244, 0.294, 0.352, 0.432, 0.511, 0.607, 0.733, 0.858, 0.991, 1.16, 1.32, 1.49, 1.66, 1.82, 1.95, 2.07, 2.20, 2.29, 2.36, 2.43, 2.50, 2.57, 2.60, 2.62, 2.65, 2.67, 2.70, 2.72, 2.73, 2.74, 2.75, 2.76, 2.77, 2.78, 2.79, 2.80, 2.81, 2.82, 2.83, 2.84, 2.85, 2.86, 2.87, 2.87, 2.88, 2.88, 2.88, 2.87, 2.86, 2.84, 2.81, 2.78, 2.71, 2.64, 2.57, 2.46, 2.32, 2.18, 2.03, 1.82, 1.62, 1.40, 1.18, 0.964, 0.768, 0.575, 0.383, 0.196, 0.0178, -0.160, -0.338, -0.538, -0.739, -0.940, -1.16, -1.39, -1.63, -1.85, -2.07, -2.29, -2.44, -2.59, -2.74, -2.83, -2.91, -2.99, -3.05, -3.08, -3.11, -3.14, -3.14, -3.15, -3.15, -3.15, -3.14, -3.14, -3.13, -3.12, -3.11, -3.10, -3.08, -3.07, -3.06, -3.06, -3.05, -3.04, -3.03, -3.03, -3.02, -3.02, -3.02, -3.02, -3.02, -3.02, -3.02, -3.03, -3.03, -3.04, -3.05, -3.06, -3.06, -3.07, -3.08, -3.10, -3.11, -3.12, -3.13, -3.14, -3.14, -3.15, -3.15, -3.15, -3.14, -3.14, -3.11, -3.08, -3.05, -2.99, -2.91, -2.83, -2.74, -2.59, -2.44, -2.29, -2.07, -1.85, -1.63, -1.39, -1.16, -0.940, -0.739, -0.538, -0.338, -0.160, 0.0178, 0.196, 0.383, 0.575, 0.767, 0.964, 1.18, 1.40, 1.62, 1.82, 2.03, 2.18, 2.32, 2.46, 2.57, 2.64, 2.71, 2.78, 2.81, 2.83, 2.86, 2.87, 2.87, 2.88, 2.88, 2.87, 2.87, 2.86, 2.85, 2.84, 2.83, 2.82, 2.81, 2.80, 2.79, 2.78, 2.77, 2.76, 2.75, 2.74, 2.73, 2.72, 2.69, 2.67, 2.65, 2.62, 2.60, 2.57, 2.50, 2.43, 2.36, 2.28, 2.19, 2.07, 1.95, 1.82, 1.66, 1.49, 1.32, 1.16, 0.990, 0.858, 0.732, 0.606, 0.510, 0.431, 0.352, 0.294, 0.244, 0.194, 0.163, 0.135, 0.108, 0.0888, 0.0744, 0.0601, 0.0482, 0.0412, 0.0343, 0.0273, 0.0241, 0.0212, 0.0182, 0.0156, 0.0156], dtype=DTYPE)

    E_Z_BETA065 = np.array([0.00258854, 0.00259463, 0.00263518, 0.00269423, 0.00277581, 0.00288892, 0.00301471, 0.00318152, 0.00336413, 0.00357576, 0.00381777, 0.00407594, 0.00438014, 0.00469931, 0.00505583, 0.00544215, 0.00585721, 0.00630347, 0.00678256, 0.00729416, 0.00783987, 0.00842269, 0.00903125, 0.00970093, 0.0103928, 0.0111384, 0.0119288, 0.0127536, 0.013655, 0.0145983, 0.0155993, 0.0166719, 0.0178122, 0.0190172, 0.02033, 0.0216998, 0.0231852, 0.0247701, 0.0264321, 0.0282672, 0.0301674, 0.0322539, 0.034461, 0.0368007, 0.0393627, 0.0420143, 0.0449585, 0.0480424, 0.0513453, 0.0549271, 0.0586321, 0.0627866, 0.0670944, 0.0717541, 0.0767559, 0.0819797, 0.087778, 0.0937842, 0.100344, 0.107312, 0.114658, 0.122728, 0.131078, 0.140281, 0.149956, 0.160249, 0.171435, 0.182995, 0.195846, 0.209212, 0.223555, 0.238969, 0.254961, 0.272688, 0.291017, 0.310831, 0.331878, 0.353886, 0.37797, 0.402818, 0.429824, 0.458166, 0.487983, 0.520153, 0.553258, 0.589307, 0.626673, 0.666113, 0.708007, 0.750989, 0.797611, 0.845361, 0.895704, 0.948287, 1.00233, 1.05966, 1.11805, 1.17915, 1.24189, 1.30615, 1.3727, 1.44005, 1.50942, 1.57961, 1.65079, 1.72267, 1.79502, 1.8676, 1.94012, 2.01254, 2.08412, 2.15524, 2.22516, 2.2938, 2.36149, 2.42665, 2.4908, 2.55234, 2.61187, 2.66966, 2.72394, 2.77687, 2.82623, 2.87332, 2.91815, 2.95914, 2.99862, 3.03404, 3.06723, 3.09784, 3.12467, 3.14994, 3.17087, 3.18972, 3.20576, 3.21812, 3.22889, 3.23498, 3.23888, 3.23963, 3.23726, 3.23178, 3.22304, 3.21081, 3.1961, 3.17575, 3.15334, 3.12608, 3.09481, 3.06057, 3.01956, 2.97597, 2.92593, 2.87106, 2.81204, 2.74463, 2.67392, 2.5945, 2.50937, 2.41851, 2.31765, 2.21279, 2.09699, 1.97511, 1.84621, 1.70708, 1.56381, 1.40934, 1.25001, 1.08412, 0.910975, 0.734877, 0.551895, 0.366885, 0.179319, -0.0093763, -0.198086, -0.385694, -0.570774, -0.753844, -0.930065, -1.10334, -1.26938, -1.42887, -1.5835, -1.72692, -1.86621, -1.99524, -2.11723, -2.23314, -2.33804, -2.43893, -2.52976, -2.61482, -2.69411, -2.76459, -2.83174, -2.89037, -2.94475, -2.99421, -3.03704, -3.07721, -3.11036, -3.14038, -3.16618, -3.18681, -3.20523, -3.21748, -3.22696, -3.2325, -3.2343, -3.23241, -3.22678, -3.2172, -3.20486, -3.18634, -3.16561, -3.1397, -3.10957, -3.07631, -3.03601, -2.99305, -2.94344, -2.8889, -2.83011, -2.76278, -2.6921, -2.61259, -2.5273, -2.43622, -2.33505, -2.22985, -2.11361, -1.99127, -1.86185, -1.72214, -1.57827, -1.42314, -1.26312, -1.09652, -0.922645, -0.745801, -0.562056, -0.37628, -0.187948, 0.0015063, 0.190964, 0.379305, 0.565095, 0.748859, 0.92573, 1.09964, 1.26627, 1.42632, 1.58149, 1.7254, 1.86516, 1.99462, 2.11701, 2.23329, 2.33853, 2.43974, 2.53085, 2.61618, 2.69572, 2.76642, 2.83378, 2.8926, 2.94716, 2.99678, 3.03975, 3.08007, 3.11333, 3.14347, 3.16938, 3.1901, 3.20861, 3.22094, 3.2305, 3.23612, 3.23798, 3.23616, 3.23058, 3.22106, 3.20878, 3.19031, 3.16963, 3.14376, 3.11368, 3.08046, 3.04021, 2.9973, 2.94774, 2.89325, 2.83451, 2.76724, 2.69663, 2.61719, 2.53197, 2.44097, 2.33989, 2.23479, 2.11866, 1.99644, 1.86715, 1.72759, 1.58388, 1.42894, 1.26913, 1.10275, 0.929111, 0.752519, 0.569057, 0.383576, 0.195557, 0.00643069, -0.182685, -0.370673, -0.556101, -0.739497, -0.915995, -1.08953, -1.25579, -1.41548, -1.57028, -1.71385, -1.85326, -1.9824, -2.10448, -2.22046, -2.32542, -2.42637, -2.51724, -2.60234, -2.68167, -2.75218, -2.81936, -2.87802, -2.93243, -2.98193, -3.0248, -3.06502, -3.09821, -3.12829, -3.15416, -3.17485, -3.19335, -3.20569, -3.21527, -3.22093, -3.22285, -3.2211, -3.21561, -3.20621, -3.19404, -3.17573, -3.15522, -3.12956, -3.09969, -3.06672, -3.02674, -2.98413, -2.9349, -2.88078, -2.82244, -2.75561, -2.68547, -2.60655, -2.5219, -2.4315, -2.33109, -2.22669, -2.11134, -1.98994, -1.86152, -1.72291, -1.58018, -1.42631, -1.26759, -1.10236, -0.92993, -0.754568, -0.572389, -0.388207, -0.201505, -0.0137001, 0.1741, 0.360788, 0.54495, 0.727106, 0.90245, 1.07487, 1.2401, 1.39886, 1.5528, 1.69566, 1.83443, 1.96308, 2.0848, 2.20052, 2.30544, 2.40641, 2.49753, 2.58301, 2.66287, 2.73415, 2.8022, 2.862, 2.91773, 2.96873, 3.0134, 3.05555, 3.09107, 3.12367, 3.15233, 3.17624, 3.19813, 3.21444, 3.2283, 3.23868, 3.2458, 3.24977, 3.25058, 3.24812, 3.24339, 3.23372, 3.22232, 3.20689, 3.1883, 3.16736, 3.14131, 3.11338, 3.08079, 3.04489, 3.00611, 2.96181, 2.9154, 2.86351, 2.8082, 2.74946, 2.68508, 2.6185, 2.54611, 2.47074, 2.39192, 2.30853, 2.22336, 2.1338, 2.04266, 1.9494, 1.85447, 1.75895, 1.66291, 1.56732, 1.47259, 1.37937, 1.28818, 1.19955, 1.11399, 1.03118, 0.953694, 0.878135, 0.80876, 0.742285, 0.68008, 0.622784, 0.567644, 0.519145, 0.472474, 0.430344, 0.391444, 0.355025, 0.322984, 0.292342, 0.265562, 0.240509, 0.217686, 0.197358, 0.178018, 0.161584, 0.14593, 0.132031, 0.119424, 0.107687, 0.0975826, 0.0879623, 0.0796234, 0.0719029, 0.0648836, 0.0587103, 0.0528521, 0.0478885, 0.043194, 0.0390233, 0.0352732, 0.0317704, 0.0287851, 0.0259443, 0.0234773, 0.0212098, 0.0191413, 0.0173381, 0.0156279, 0.0141773, 0.0128153, 0.0116029, 0.0105236, 0.00951232, 0.00866295, 0.00785669, 0.00715976, 0.00652827, 0.00595423, 0.00546679, 0.00500828, 0.00463035, 0.00428439, 0.0039857, 0.00373562, 0.003507, 0.0033404, 0.00319071, 0.00308307, 0.00300628, 0.00295327, 0.00294558], dtype=DTYPE)


# =============================================================================
# Helpers
# =============================================================================
cdef DTYPE_t interp(DTYPE_t z, DTYPE_t[:] e_z, DTYPE_t inv_dz, int n_points):
    """Interpolation function."""
    cdef int i
    cdef DTYPE_t delta_e_z, slope, offset
    cdef out

    if z < 0. or z > (n_points - 1) / inv_dz:
        out = 0.

    else:
        i =  int(floor(z * inv_dz))
        if i < n_points - 1:
            # Faster with array of delta electric field?
            delta_e_z = e_z[i + 1] - e_z[i]
            slope = delta_e_z * inv_dz
            offset = e_z[i] - i * delta_e_z
            out = slope * z + offset
        else:
            out = e_z[n_points]

    return out


cdef DTYPE_t e_func(DTYPE_t k_e, DTYPE_t z, DTYPE_t[:] e_z, DTYPE_t inv_dz,
                    int n_points, DTYPE_t phi, DTYPE_t phi_0):
    """Give the electric field at position z and phase phi."""
    return k_e * interp(z, e_z, inv_dz, n_points) * cos(phi + phi_0)


cdef DTYPE_t de_dt_func(DTYPE_t k_e, DTYPE_t z, DTYPE_t[:] e_z, DTYPE_t inv_dz,
                        int n_points, DTYPE_t phi, DTYPE_t phi_0,
                        DTYPE_t factor):
    """Give the first time derivative of the electric field at (z, phi)."""
    return factor * k_e * interp(z, e_z, inv_dz, n_points) * sin(phi + phi_0)


cdef rk4(DTYPE_t[:] u, DTYPE_t x, DTYPE_t dx, DTYPE_t k_e, DTYPE_t[:] e_z,
         DTYPE_t inv_dz, int n_points, DTYPE_t phi_0_rel, DTYPE_t omega0_rf):
    """Integrate the motion over the space step dx."""
    # Variables:
    cdef DTYPE_t half_dx = .5 * dx
    cdef Py_ssize_t i

    # Memory views:
    delta_u_array = np.empty(2, dtype=DTYPE)
    cdef DTYPE_t[:] delta_u = delta_u_array

    k_i_array = np.zeros((4, 2), dtype=DTYPE)
    cdef DTYPE_t[:, :] k_i = k_i_array

    du_dz_i_array = np.zeros((2), dtype=DTYPE)
    cdef DTYPE_t[:] du_dz_i = du_dz_i_array

    tmp_array = np.zeros((2), dtype=DTYPE)
    cdef DTYPE_t[:] tmp = tmp_array

    # Equiv of k_1 = du_dx(x, u):
    du_dz_i = du_dz(x, u, k_e, e_z, inv_dz, n_points, phi_0_rel, omega0_rf)
    k_i[0, 0] = du_dz_i[0]
    k_i[0, 1] = du_dz_i[1]

    # Equiv of
    # k_2 = du_dx(x + half_dx, u + half_dx * k_1)
    # k_3 = du_dx(x + half_dx, u + half_dx * k_2)
    for i in [1, 2]:
        # Compute tmp = u + half_dx * k_1,2
        tmp[0] = u[0] + half_dx * k_i[i - 1, 0]
        tmp[1] = u[1] + half_dx * k_i[i - 1, 1]
        du_dz_i = du_dz(x + half_dx, tmp, k_e, e_z, inv_dz, n_points,
                        phi_0_rel, omega0_rf)
        k_i[i, 0] = du_dz_i[0]
        k_i[i, 1] = du_dz_i[1]

    # Compute u + dx * k_3
    tmp[0] = u[0] + dx * k_i[2, 0]
    tmp[1] = u[1] + dx * k_i[2, 1]
    # Equiv of k_4 = du_dx(x + dx, u + dx * k_3)
    du_dz_i = du_dz(x + dx, tmp, k_e, e_z, inv_dz, n_points, phi_0_rel,
                    omega0_rf)
    k_i[3, 0] = du_dz_i[0]
    k_i[3, 1] = du_dz_i[1]

    # Equiv of delta_u = (k_1 + 2. * k_2 + 2. * k_3 + k_4) * dx / 6.
    delta_u[0] = (k_i[0, 0] + 2. * k_i[1, 0] + 2. * k_i[2, 0] + k_i[3, 0]) \
            * dx / 6.
    delta_u[1] = (k_i[0, 1] + 2. * k_i[1, 1] + 2. * k_i[2, 1] + k_i[3, 1]) \
            * dx / 6.
    return delta_u_array


cdef du_dz(DTYPE_t z, DTYPE_t[:] u, DTYPE_t k_e, DTYPE_t[:] e_z, DTYPE_t inv_dz,
           int n_points, DTYPE_t phi_0_rel, DTYPE_t omega0_rf):
    """First derivative of the motion."""
    # Variables:
    cdef DTYPE_t gamma, beta

    # Memory views:
    v_array = np.empty(2, dtype=DTYPE)
    cdef DTYPE_t[:] v = v_array

    v[0] = q_adim_cdef * e_func(k_e, z, e_z, inv_dz, n_points, u[1], phi_0_rel)
    gamma = 1. + u[0] * inv_E_rest_MeV_cdef
    beta = sqrt(1. - gamma**-2)
    v[1] = omega0_rf / (beta * c_cdef)
    return v_array


# Kept for legacy, but @ is faster than this
# cdef matprod_22(DTYPE_t[:, :] m_1, DTYPE_t[:, :] m_2):
    # out = np.array((
        # [m_1[0, 0] * m_2[0, 0] + m_1[0, 1] * m_2[1, 0],
         # m_1[0, 0] * m_2[0, 1] + m_1[0, 1] * m_2[1, 1]],
        # [m_1[1, 0] * m_2[0, 0] + m_1[1, 1] * m_2[1, 0],
         # m_1[1, 0] * m_2[0, 1] + m_1[1, 1] * m_2[1, 1]]), dtype=DTYPE)
    # return out


# =============================================================================
# Transfer matrices
# =============================================================================
cpdef z_drift(DTYPE_t delta_s, DTYPE_t w_kin_in, np.int64_t n_steps=1):
    """Calculate the transfer matrix of a drift."""
    # Variables:
    cdef DTYPE_t gamma_in_min2, beta_in, delta_phi
    cdef Py_ssize_t i

    # Memory views:
    w_phi_array = np.empty([n_steps, 2], dtype=DTYPE)
    cdef DTYPE_t[:, :] w_phi = w_phi_array

    gamma_in_min2 = (1. + w_kin_in * inv_E_rest_MeV_cdef)**-2

    cdef np.ndarray[DTYPE_t, ndim=3] r_zz_array = np.full(
        [n_steps, 2, 2],
        np.array([[1., delta_s * gamma_in_min2],
                  [0., 1.]], dtype=DTYPE),
        dtype=DTYPE)

    beta_in = sqrt(1. - gamma_in_min2)
    delta_phi = OMEGA_0_BUNCH_cdef * delta_s / (beta_in * c_cdef)
    for i in range(n_steps):
        w_phi[i, 0] = w_kin_in
        w_phi[i, 1] = (i + 1) * delta_phi
    return r_zz_array, w_phi_array, None


def z_field_map_rk4(DTYPE_t d_z, DTYPE_t w_kin_in, np.int64_t n_steps,
                    DTYPE_t omega0_rf, DTYPE_t k_e, DTYPE_t phi_0_rel,
                    np.int64_t section_idx):
    """Calculate the transfer matrix of a field map using Runge-Kutta."""
    # Variables:
    cdef DTYPE_t z_rel = 0.
    cdef complex itg_field = 0.
    cdef DTYPE_t half_d_z = .5 * d_z
    cdef DTYPE_t gamma_next
    cdef DTYPE_t gamma = 1. + w_kin_in * inv_E_rest_MeV_cdef
    cdef DTYPE_t beta = sqrt(1. - gamma**-2)
    cdef np.int64_t i
    cdef DTYPE_t tmp

    # Arrays:
    cdef np.ndarray[DTYPE_t, ndim=3] r_zz_array = np.empty([n_steps, 2, 2],
                                                           dtype=DTYPE)

    # Memory views:
    w_phi_array = np.empty((n_steps + 1, 2), dtype=DTYPE)
    delta_w_phi_array = np.zeros((2), dtype=DTYPE)
    cdef DTYPE_t[:, :] w_phi = w_phi_array
    cdef DTYPE_t[:] delta_w_phi = delta_w_phi_array
    cdef DTYPE_t[:] e_z
    cdef DTYPE_t inv_dz
    cdef int n_points

    w_phi[0, 0] = w_kin_in
    w_phi[0, 1] = 0.

    if section_idx == 0:
        e_z = E_Z_SIMPLE_SPOKE
        inv_dz = INV_DZ_SIMPLE_SPOKE
    elif section_idx == 1:
        e_z = E_Z_SPOKE_ESS
        inv_dz = INV_DZ_SPOKE_ESS
    elif section_idx == 2:
        e_z = E_Z_BETA065
        inv_dz = INV_DZ_BETA065
    else:
        raise IOError('wrong section_idx in z_field_map')
    n_points = e_z.shape[0]

    for i in range(n_steps):
        # Compute energy and phase changes
        delta_w_phi = rk4(w_phi[i, :], z_rel, d_z, k_e, e_z, inv_dz,
                          n_points, phi_0_rel, omega0_rf)

        # Update
        itg_field += e_func(k_e, z_rel, e_z, inv_dz, n_points, w_phi[i, 1],
                            phi_0_rel) \
            * (1. + 1j * tan(w_phi[i, 1] + phi_0_rel)) * d_z

        w_phi[i + 1, 0] = w_phi[i, 0] + delta_w_phi[0]
        w_phi[i + 1, 1] = w_phi[i, 1] + delta_w_phi[1]
        gamma_next = 1. + w_phi[i + 1, 0] * inv_E_rest_MeV_cdef

        gamma_middle = .5 * (gamma + gamma_next)
        beta_middle = sqrt(1. - gamma_middle**-2)

        r_zz_array[i, :, :] = z_thin_lense(
            d_z, half_d_z, w_phi[i, 0], gamma_middle, w_phi[i + 1, 0],
            beta_middle, z_rel, w_phi[i, 1], omega0_rf, k_e, phi_0_rel, e_z,
            inv_dz, n_points)
        z_rel += d_z
        gamma = gamma_next

    return r_zz_array, w_phi_array[1:, :], itg_field


def z_field_map_leapfrog(DTYPE_t d_z, DTYPE_t w_kin_in, np.int64_t n_steps,
                         DTYPE_t omega0_rf, DTYPE_t k_e, DTYPE_t phi_0_rel,
                         np.int64_t section_idx):
    """Calculate the transfer matrix of a field map using leapfrog."""
    # Variables:
    cdef DTYPE_t z_rel = 0.
    cdef complex itg_field = 0.
    cdef DTYPE_t half_d_z = .5 * d_z
    cdef DTYPE_t gamma_next, beta_next
    cdef DTYPE_t gamma = 1. + w_kin_in * inv_E_rest_MeV_cdef
    cdef DTYPE_t beta = sqrt(1. - gamma**-2)
    cdef np.int64_t i
    cdef DTYPE_t tmp
    cdef DTYPE_t delta_w, delta_phi

    # Arrays:
    cdef np.ndarray[DTYPE_t, ndim=3] r_zz_array = np.empty([n_steps, 2, 2],
                                                           dtype=DTYPE)

    # Memory views:
    w_phi_array = np.empty((n_steps + 1, 2), dtype=DTYPE)
    cdef DTYPE_t[:, :] w_phi = w_phi_array
    cdef DTYPE_t[:] e_z
    cdef DTYPE_t inv_dz
    cdef int n_points

    if section_idx == 0:
        e_z = E_Z_SIMPLE_SPOKE
        inv_dz = INV_DZ_SIMPLE_SPOKE
    elif section_idx == 1:
        e_z = E_Z_SPOKE_ESS
        inv_dz = INV_DZ_SPOKE_ESS
    elif section_idx == 2:
        e_z = E_Z_BETA065
        inv_dz = INV_DZ_BETA065
    else:
        raise IOError('wrong section_idx in z_field_map')
    n_points = e_z.shape[0]

    w_phi[0, 1] = 0.
    # Rewind energy from i=0 to i=-0.5 if we are at the first cavity:
    if w_kin_in == E_MEV_cdef:
        w_phi[0, 0] = w_kin_in - q_adim_cdef * e_func(
            k_e, z_rel, e_z, inv_dz, n_points, w_phi[0, 1], phi_0_rel) \
            * half_d_z
    else:
        w_phi[0, 0] = w_kin_in

    for i in range(n_steps):
        # Compute forces at step i
        delta_w = q_adim_cdef * e_func(k_e, z_rel, e_z, inv_dz, n_points,
                                       w_phi[i, 1], phi_0_rel) * d_z
        # New energy at i + 0.5
        w_phi[i + 1, 0] = w_phi[i, 0] + delta_w
        gamma = 1. + w_phi[i + 1, 0] * inv_E_rest_MeV_cdef
        beta = sqrt(1. - gamma**-2)

        # Compute phase at steo i + 1
        delta_phi = omega0_rf * d_z / (beta * c_cdef)
        w_phi[i + 1, 1] = w_phi[i, 1] + delta_phi

        # Update
        itg_field += e_func(k_e, z_rel, e_z, inv_dz, n_points, w_phi[i, 1],
                            phi_0_rel) \
            * (1. + 1j * tan(w_phi[i, 1] + phi_0_rel)) * d_z

        # We already are at the step i + 0.5, so gamma_middle and beta_middle
        # are the same as gamma and beta
        r_zz_array[i, :, :] = z_thin_lense(
            d_z, half_d_z, w_phi[i, 0], gamma, w_phi[i + 1, 0],
            beta, z_rel, w_phi[i, 1], omega0_rf, k_e, phi_0_rel, e_z,
            inv_dz, n_points)

        z_rel += d_z

    return r_zz_array, w_phi_array[1:, :], itg_field

def z_field_map_jm(DTYPE_t d_z, DTYPE_t w_kin_in, np.int64_t n_steps,
                   DTYPE_t omega0_rf, DTYPE_t k_e, DTYPE_t phi_0_rel,
                   np.int64_t section_idx):
    """Calculate the transfer matrix of a field map using Lagniel's algo."""
    # Particle energy
    cdef DTYPE_t gamma = 1. + w_kin_in * inv_E_rest_MeV_cdef
    cdef DTYPE_t gamma2 = gamma * gamma
    cdef DTYPE_t beta = sqrt(1. - 1. / gamma2)

    # Particle acceleration
    cdef DTYPE_t delta_w, delta_phi, delta_gamma
    cdef DTYPE_t[:] e_z

    # Particle phase
    cdef DTYPE_t phi_abs, phi_rel

    # Constants to speed up calculation
    cdef DTYPE_t phi_cav = omega0_rf * d_z / c_cdef
    cdef DTYPE_t delta_gamma_cav = q_adim_cdef * d_z * inv_E_rest_MeV_cdef
    cdef DTYPE_t kk = delta_gamma_cav * k_e

    # Integers
    cdef int n_points
    cdef np.int64_t i

    # Transfer matrix
    cdef DTYPE_t s1_11, s1_12, s1_21, s1_22
    cdef DTYPE_t        s2_12, s2_21, s2_22
    cdef np.ndarray[DTYPE_t, ndim=3] r_zz

    # V_cav and phi_s
    cdef complex itg_field = 0.
    cdef DTYPE_t sum_sin, sum_cos

    cdef DTYPE_t gamma_out, beta_out, w_kin_out, phi_out

    # kk already calculated

    # Particle enter cavity at absolute phase=phi_0
    # (ie at relative phase=0)
    phi_abs = phi_0_rel
    # To have phi particle = 0 at first integration step.
    phi_rel = -phi_cav / beta

    # Define electric field
    if section_idx == 0:
        e_z = E_Z_SIMPLE_SPOKE
        # inv_dz = INV_DZ_SIMPLE_SPOKE
    elif section_idx == 1:
        e_z = E_Z_SPOKE_ESS
        # inv_dz = INV_DZ_SPOKE_ESS
    elif section_idx == 2:
        e_z = E_Z_BETA065
        # inv_dz = INV_DZ_BETA065
    else:
        raise IOError('wrong section_idx in z_field_map')
    n_points = e_z.shape[0]

    r_zz = np.empty([n_points + 1, 2, 2])
    s1_11 = 1.
    s1_12 = 0.
    s1_21 = 0.
    s1_22 = 1.

    sum_sin = 0.
    sum_cos = 0.

    for i in range(n_points):
        # Update phi
        delta_phi = phi_cav / beta
        phi_rel = phi_rel + delta_phi

        # Acceleration without the cos term
        k_gam = kk * e_z[i]
        # Update absolute phase
        phi_abs = phi_0_rel + phi_rel

        # N.B. dphi already divided by beta
        s2_12 = -delta_phi / (beta * beta * gamma2 * gamma)
        s2_21 = -k_gam * sin(phi_abs)
        s2_22 = 1.0 + (s2_12 * s2_21)

        r_zz[i, 0, 0] = s1_11 + (s1_21 * s2_12)            # no dim
        r_zz[i, 0, 1] = s1_12 + (s1_22 * s2_12)            # d_rad/d_gamma
        r_zz[i, 1, 0] = (s1_11 * s2_21) + (s1_21 * s2_22)  # d_gamma/d_rad
        r_zz[i, 1, 1] = (s1_12 * s2_21) + (s1_22 * s2_22)  # no dim

        s1_11 = r_zz[i, 0, 0]
        s1_12 = r_zz[i, 0, 1]
        s1_21 = r_zz[i, 1, 0]
        s1_22 = r_zz[i, 1, 1]

        delta_gamma = k_gam * cos(phi_abs)
        gamma = gamma + delta_gamma
        gamma2 = gamma * gamma

        sum_sin = sum_sin - s2_21
        sum_cos = sum_cos + delta_gamma

        # Warning! beta not updated...

    gamma_out = gamma
    beta_out = beta_out
    w_kin_out = (gamma - 1.) * E_rest_MeV_cdef
    phi = phi + phi_cav / (2. * beta)   # half step end cav
    phi_out = phi
    itg_field = sum_cos + 1j * sum_sin

    return r_zz_array, w_phi_array[1:, :], itg_field

cdef z_thin_lense(DTYPE_t d_z, DTYPE_t half_dz, DTYPE_t w_kin_in,
                  DTYPE_t gamma_middle, DTYPE_t w_kin_out, DTYPE_t beta_middle,
                  DTYPE_t z_rel, DTYPE_t phi_rel, DTYPE_t omega0_rf,
                  DTYPE_t norm, DTYPE_t phi_0,
                  DTYPE_t[:] e_z, DTYPE_t inv_dz, int n_points):
    """Calculate the transfer matrix of a drift-gap-drift."""
    # Variables:
    cdef DTYPE_t z_k, phi_k, k_0, k_1, k_2, k_3
    cdef DTYPE_t e_func_k

    # Arrays:
    # cdef np.ndarray[DTYPE_t, ndim=2] r_zz = np.zeros([2, 2], dtype=DTYPE)
    # Faster to not declare the type of r_zz, maybe because it is already
    # given by z_drift function

    z_k = z_rel + half_dz
    phi_k = phi_rel + half_dz * omega0_rf / (beta_middle * c_cdef)

    # Transfer matrix components of middle (accelerating part)
    e_func_k = e_func(norm, z_k, e_z, inv_dz, n_points, phi_k, phi_0)

    k_0 = q_adim_cdef * d_z / (gamma_middle * beta_middle**2 * E_rest_MeV_cdef)
    k_1 = k_0 * de_dt_func(norm, z_k, e_z, inv_dz, n_points, phi_k, phi_0,
                           omega0_rf / (beta_middle * c_cdef))
    k_2 = 1. - (2. - beta_middle**2) * k_0 * e_func_k
    # Correction to ensure det < 1
    k_3 = (1. - k_0 * e_func_k) / (1. - k_0 * (2. - beta_middle**2) * e_func_k)

    # Matrix product: end @ (middle @ in)
    # r_zz_array = matprod_22(z_drift(half_dz, w_kin_out)[0][0],
                            # matprod_22(np.array(([k_3, 0.], [k_1, k_2]),
                            # dtype=DTYPE),
                                       # z_drift(half_dz, w_kin_in)[0][0])
                           # )
    # Faster than matmul or matprod_22
    r_zz_array = z_drift(half_dz, w_kin_out)[0][0] \
                 @ (np.array(([k_3, 0.], [k_1, k_2]), dtype=DTYPE) \
                    @ z_drift(half_dz, w_kin_in)[0][0])

    return r_zz_array
