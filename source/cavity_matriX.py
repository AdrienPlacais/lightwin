#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:10:33 2021

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt
from elements import select_and_load_field_map_file
from essais import electric_field
from scipy.interpolate import interp1d
from transfer_matrices import z_drift
from helper import MeV_to_v, v_to_MeV

plt.rc('axes.formatter', useoffset=False)

# =============================================================================
# Physical constants
# =============================================================================
c = 2.99792458e8
m_MeV = 938.27202900
m_kg = 1.672649e-27
q_adim = 1.
q_C = 1.602e-19
q_over_m = q_C / m_kg
m_over_q = m_kg / q_C

debug_plot = True

# =============================================================================
# Beam parameters
# =============================================================================
f_Hz = 352.2e6
omega_0 = 2. * np.pi * f_Hz

N_cell = 500

phi_RG_deg = 142.089    # LINAC.theta_i[36]
phi_RF = np.deg2rad(phi_RG_deg)

E_0_MeV = 18.793905

# Beta, manually calculated and imported from TW
beta_TW = 0.1972014
beta_0 = np.sqrt((1. + E_0_MeV / m_MeV)**2 - 1.) / (1. + E_0_MeV / m_MeV)
gamma_0 = 1. / np.sqrt(1. - beta_0**2)

# Coef on the field
k = 1.68927 * 1.000092819734090     # LINAC.k_e[36] * ?
# Load electric field
nz, zmax, Norm, Fz_array = select_and_load_field_map_file(
    '/home/placais/TraceWin/work_field_map/work_field_map.dat',
    100., 0.0, 'Simple_Spoke_1D')
Fz_array *= k

z_cavity_array = np.linspace(0., zmax, nz + 1)      # z cav local coordinates
Fz_func = interp1d(z_cavity_array, Fz_array)    # interpolate field

cavity_field = electric_field(Norm, Fz_func, omega_0, phi_RF)

dz = z_cavity_array[1] - z_cavity_array[0]
dE_dz_array = np.diff(Fz_array) / dz
# Interpolate dE/dz on the middle of every cell
dE_dz_func = interp1d((z_cavity_array[:-1] + z_cavity_array[1:]) * 0.5,
                      dE_dz_array,
                      fill_value='extrapolate')
# We use fill_value=extrapolate to define the function on [0, .5*dz] and
# [zmax-.5*dz, zmax].

# =============================================================================
# Simulation parameters
# =============================================================================
# Initial step size calculated with betalambda
factor = 2.
lambda_RF = c / f_Hz
# Spatial step
step = beta_0 * lambda_RF / (2. * N_cell * factor)

# =============================================================================
# Compute energy gain and synchronous phase
# =============================================================================
# Step size
idx_max = int(np.floor(zmax / step))

# Init synchronous particle
E_out_MeV = E_0_MeV
gamma_out = gamma_0
z = .5 * step   # Middle of first spatial step

E_r = 0.
E_i = 0.

t = step / (2. * beta_0 * c)

z_pos_array = [0.]
energy_array = [E_0_MeV]
beta_array = [beta_0]
gamma_array = [gamma_0]

Mt = np.eye(2)

# Loop over the spatial steps
for i in range(idx_max):
    E_in_MeV = E_out_MeV
    gamma_in = gamma_out

    # Compute energy gain
    E_interp = Fz_func(z)[()]
    acceleration = q_adim * E_interp * np.cos(phi_RF)
    E_r += acceleration
    E_i += q_adim * E_interp * np.sin(phi_RF)

    # Energy and gamma at the exit of current cell
    E_out_MeV = acceleration * step + E_in_MeV
    gamma_out = gamma_in + acceleration * step / m_MeV

    # Synchronous gamma
    gamma_s = (gamma_out + gamma_in) * .5
    beta_s = np.sqrt(1. - 1. / gamma_s**2)
    beta_out = np.sqrt(1. - 1. / gamma_out**2)

    # Synchronous phase
    phi_s = np.arctan(E_i / E_r)
    phi_s_deg = np.rad2deg(phi_s)

    # Compute transfer matrix
    K_0 = q_adim * np.cos(phi_RF) * step / (gamma_s * beta_s**2 * m_MeV)
    K_1 = dE_dz_func(z) * K_0
    K_2 = 1. - (2. - beta_s**2) * Fz_func(z) * K_0

    M_in = z_drift(.5 * step, gamma_in)
    M_mid = np.array(([1., 0.],
                      [K_1, K_2]))
    M_out = z_drift(.5 * step, gamma_out)

    # Compute M_in * M_mid * M_out * M_t
    Mt = np.matmul(np.matmul(np.matmul(M_in, M_mid), M_out), Mt)

    # Next step
    z += step
    t += step / (beta_s * c)
    phi_RF += step * omega_0 / (beta_s * c)

    # Save data
    z_pos_array.append(z)
    energy_array.append(E_out_MeV)
    beta_array.append(beta_out)   # TODO: check this!
    gamma_array.append(gamma_out)     # TODO: check this!

# =============================================================================
# End of loop
# =============================================================================
z_pos_array = np.array(z_pos_array)
energy_array = np.array(energy_array)
beta_array = np.array(beta_array)
gamma_array = np.array(gamma_array)

Mt_TW = np.array(([0.90142688, 0.38549366],
                  [-0.34169225, 0.95204762]))
E_MeV_TW = 19.173416
phi_s_deg_TW = -41.583201
V_cav_MV_TW = 0.50737183

E_MeV = energy_array[-1]
err_E = 100. * (E_MeV_TW - E_MeV) / E_MeV_TW
err_phi_s = 100. * (phi_s_deg_TW - phi_s_deg) / phi_s_deg_TW
V_cav_MV = np.abs((E_MeV - E_0_MeV) / np.cos(phi_s))
err_V_cav = 100. * (V_cav_MV_TW - V_cav_MV) / V_cav_MV_TW
err_Mt = 100. * np.divide(Mt_TW - Mt, Mt_TW)

np.set_printoptions(precision=4)
print('-----------------------------------------------------')
print('                 TW          here      Rel. error (%)')
print('E_out (MeV): ', '{:.6f}'.format(E_MeV_TW), '  ',
      '{:.6f}'.format(E_MeV), '  ', '{:+.5f}'.format(err_E))
print('phi_s (deg): ', '{:+.5f}'.format(phi_s_deg_TW), '  ',
      '{:+.5f}'.format(phi_s_deg), '  ', '{:+.5f}'.format(err_phi_s))
print('V_cav (MV):   ', '{:.6f}'.format(V_cav_MV_TW), '    ',
      '{:.6f}'.format(V_cav_MV), ' ', '{:+.5f}'.format(err_V_cav))
print('-----------------------------------------------------')
print('MT TW:')
print(Mt_TW)
print('MT here:')
print(Mt)
print('Matrix of rel. error, element-wise (%):')
print(err_Mt)


if(debug_plot):
    if(not plt.fignum_exists((10))):
        fig = plt.figure(10)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

    else:
        fig = plt.figure(10)
        ax1, ax2 = fig.get_axes()

    ax1.plot(z_pos_array*1e3, energy_array, label='Fred')
    ax1.set_ylabel('Energy [MeV]')
    ax1.grid(True)
    ax1.legend()

    # ax2.plot(t_z, z*1e3, label=phi_0)
    ax2.set_ylabel('Position [mm]')
    ax2.set_xlabel('Time [ns]')
    ax2.grid(True)
    fig.show()