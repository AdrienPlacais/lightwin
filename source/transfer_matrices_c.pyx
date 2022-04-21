#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:34 2021.

@author: placais

File holding all the longitudinal transfer sub-matrices. Units are taken
exactly as in TraceWin, i.e. first line is z (m) and second line is dp/p.
"""
import cython
from libc.stdio cimport printf
from libc.math cimport sin, cos, sqrt, tan
import numpy as np
cimport numpy as np
np.import_array()

# Must be changed to double if C float is replaced by double
DTYPE = np.float64

cdef float c_cdef = 2.99792458e8
cdef float E_rest_MeV_cdef = 938.27203
cdef float inv_E_rest_MeV_cdef = 0.0010657889908537506
cdef float OMEGA_0_BUNCH_cdef = 1106468932.594325
cdef float q_adim_cdef = 1.

# https://stackoverflow.com/questions/14124049/is-there-any-type-for-function-
# in-cython
#  ctypedef float (*f_type)(float)

# =============================================================================
# Transfer matrices
# =============================================================================
cpdef z_drift(float delta_s, float W_kin_in, Py_ssize_t n_steps=1):
    cdef float gamma_in_min2, beta_in, delta_phi
    cdef np.ndarray r_zz = np.empty([n_steps, 2, 2], dtype = DTYPE)
    cdef np.ndarray w_phi = np.empty([n_steps, 2], dtype = DTYPE)
    cdef Py_ssize_t i

    gamma_in_min2 = (1. + W_kin_in * inv_E_rest_MeV_cdef)**-2
    r_zz = np.full((n_steps, 2, 2), np.array([[1., delta_s * gamma_in_min2],
                                              [0., 1.]]))

    beta_in = sqrt(1. - gamma_in_min2)
    delta_phi = OMEGA_0_BUNCH_cdef * delta_s / (beta_in * c_cdef)
    # Two possibilites: which one is the best?
    # l_W_kin = [W_kin_in for i in range(n_steps)]
    # l_phi_rel = [(i+1)*delta_phi for i in range(n_steps)]
    # w_phi = np.empty((n_steps, 2))
    # w_phi[:, 0] = l_W_kin
    # w_phi[:, 1] = l_phi_rel
    w_phi[:, 0] = W_kin_in
    w_phi[:, 1] = np.arange(0., n_steps) * delta_phi + delta_phi
    return r_zz, w_phi, None


cdef float e_func(float k_e, float z, e_spat, float phi, float phi_0):
    return k_e * e_spat(z) * cos(phi + phi_0)


cdef float de_dt_func(float k_e, float z, e_spat, float phi, float phi_0,
                           float factor):
    return factor * k_e * e_spat(z) * sin(phi + phi_0)


# TODO: types of u and du_dx
# cdef rk4(float[:] u, du_dx, float x, float dx):
cdef rk4(double [:] u, du_dx, float x, float dx):
    cdef float half_dx = .5 * dx
    cdef np.ndarray k_1 = np.zeros([2], dtype = DTYPE)
    cdef np.ndarray k_2 = np.zeros([2], dtype = DTYPE)
    cdef np.ndarray k_3 = np.zeros([2], dtype = DTYPE)
    cdef np.ndarray k_4 = np.zeros([2], dtype = DTYPE)
    cdef np.ndarray delta_u = np.zeros([2], dtype = DTYPE)
    k_1 = du_dx(x, u)
    k_2 = du_dx(x + half_dx, u + half_dx * k_1)
    k_3 = du_dx(x + half_dx, u + half_dx * k_2)
    k_4 = du_dx(x + dx, u + dx * k_3)
    delta_u = (k_1 + 2.*k_2 + 2.*k_3 + k_4) * dx / 6.
    return delta_u


# TODO cpdef, type e_spat
def z_field_map(float d_z, float W_kin_in, Py_ssize_t n_steps, float omega0_rf,
                float k_e, float phi_0_rel, e_spat):
    cdef float z_rel = 0.
    cdef complex itg_field = 0.
    cdef float half_d_z = .5 * d_z

    cdef list r_zz = []
    cdef np.ndarray W_phi = np.empty([n_steps + 1, 2], dtype = DTYPE)
    cdef np.ndarray delta_W_phi = np.zeros([2], dtype = DTYPE)
    cdef list l_gamma = [1. + W_kin_in * inv_E_rest_MeV_cdef]
    cdef list l_beta = [sqrt(1. - l_gamma[0]**-2)]

    cdef Py_ssize_t i
    cdef float tmp
    W_phi[0, 0] = W_kin_in
    W_phi[0, 1] = 0.

    # u is defined as a MEMORYVIEW for more efficient access
    # def du_dz(float z, float[:] u):
    def du_dz(float z, double[:] u):
        cdef float gamma_float, beta
        cdef np.ndarray v = np.empty([2], dtype = DTYPE)

        v[0] = q_adim_cdef * e_func(k_e, z, e_spat, u[1], phi_0_rel)
        gamma_float = 1. + u[0] * inv_E_rest_MeV_cdef
        beta = sqrt(1. - gamma_float**-2)
        v[1] = omega0_rf / (beta * c_cdef)
        return v

    for i in range(n_steps):
        # Compute energy and phase changes
        delta_W_phi = rk4(W_phi[i, :], du_dz, z_rel, d_z)

        # Update
        itg_field += e_func(k_e, z_rel, e_spat, W_phi[i, 1], phi_0_rel) \
            * (1. + 1j * tan(W_phi[i, 1] + phi_0_rel)) * d_z

        W_phi[i + 1, :] = W_phi[i, :] + delta_W_phi
        l_gamma.append(1. + W_phi[i+1, 0] * inv_E_rest_MeV_cdef)
        l_beta.append(sqrt(1. - l_gamma[-1]**-2))

        gamma_middle = .5 * (l_gamma[-1] + l_gamma[-2])
        beta_middle = sqrt(1. - gamma_middle**-2)

        r_zz.append(z_thin_lense(d_z, half_d_z, W_phi[i, 0], gamma_middle,
                                 W_phi[i+1, 0], beta_middle, z_rel,
                                 W_phi[i, 1], omega0_rf, k_e, phi_0_rel,
                                 e_spat))
        z_rel += d_z

    return np.array(r_zz), W_phi[1:, :], itg_field


cdef z_thin_lense(float d_z, float half_dz, float W_kin_in, float gamma_middle,
                  float W_kin_out, float beta_middle, float z_rel,
                  float phi_rel, float omega0_rf, float norm, float phi_0,
                 e_spat):
    cdef float z_k, delta_phi_half_step, phi_k, k_0, k_1, k_2, k_3, factor
    cdef float e_func_k
    cdef np.ndarray r_zz = np.zeros([2, 2], dtype = DTYPE)
    cdef np.ndarray tmp = np.zeros([2, 2], dtype = DTYPE)
    # In
    r_zz = z_drift(half_dz, W_kin_in)[0][0]

    # Middle
    z_k = z_rel + half_dz
    delta_phi_half_step = half_dz * omega0_rf / (beta_middle * c_cdef)
    phi_k = phi_rel + delta_phi_half_step

    # Transfer matrix components
    k_0 = q_adim_cdef * d_z / (gamma_middle * beta_middle**2 * E_rest_MeV_cdef)
    factor = omega0_rf / (beta_middle * c_cdef)
    k_1 = k_0 * de_dt_func(norm, z_k, e_spat, phi_k, phi_0, factor)
    e_func_k = e_func(norm, z_k, e_spat, phi_k, phi_0)
    k_2 = 1. - (2. - beta_middle**2) * k_0 * e_func_k

    # Correction to ensure det < 1
    k_3 = (1. - k_0 * e_func_k) / (1. - k_0 * (2. - beta_middle**2) * e_func_k)

    r_zz = np.array(([k_3, 0.], [k_1, k_2])) @ r_zz

    # Out
    tmp = z_drift(half_dz, W_kin_out)[0][0]
    r_zz = tmp @ r_zz

    return r_zz
