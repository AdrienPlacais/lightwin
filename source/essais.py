#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:05:38 2021

@author: placais
"""

import numpy as np
from elements import select_and_load_field_map_file
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt

# =============================================================================
# Physical constants
# =============================================================================
c = 2.99792458e8
m = 1.672649e-27
q = 1.602e-19
q_over_m = q / m
m_over_q = m / q


# =============================================================================
# Classes
# =============================================================================
class electric_field():
    """Class to hold the electric field properties."""

    def __init__(self, Norm, Fz_func, omega_0, phi_0):
        """Initialize class."""
        self.Norm = Norm
        self.Fz_func = Fz_func
        self.omega_0 = omega_0
        self.phi_0 = phi_0

    def electric_acceleration(self, z, t):
        """
        Acceleration created by electric field.

        TODO: More arguments such as Norm, etc.

        Parameters
        ----------
        z: float
            Position of the particle in m.
        t: float
            Time in s.
        """
        a = q_over_m * self.Norm * self.Fz_func(z) * np.cos(
            self.omega_0 * t + self.phi_0)
        return a


# =============================================================================
# Functions
# =============================================================================
def compute_acceleration_with_PIC(elec_f, v_0, dt, z_0=0., t_0=0.):
    """
    Use a PIC method to compute acceleration in cavity.

    TODO: Verification of time step dt.
    TODO: Output time? Phase? Position?

    Parameters
    ----------
    elec_f: electric_field object
        Electric field properties.
    v_0: float
        Initial velocity in m/s.
    dt: float
        Size of the time step.
    z_0: float, optional
        Initial position in m. Default: 0 (entry of cavity).
    t_0: float, optional
        Time of particle entrance in the cavity.
    """
    # Init some arrays and indices:
    z = [z_0]
    i = 0
    t = t_0

    # Initial acceleration
    a = elec_f.electric_acceleration(z_0, t_0 - dt)
    # Rewind velocity of half a step
    v = [float(v_0 - 0.5 * a * dt)]

    debug_plot = False
    debug_output_info = False

    # Particle tracking loop
    while(True):
        i += 1
        a = elec_f.electric_acceleration(z[-1], t)
        v.append(v[-1] + a * dt)
        z.append(z[-1] + v[-1] * dt)
        t += dt

        if(z[-1] > zmax):
            if(debug_output_info):
                print("Particle went through cavity.")
            i += 1
            break

        elif(t > t_max):
            if(debug_output_info):
                print("Particle took too many time.")
            i += 1
            break

        elif(z[-1] < 0.):
            if(debug_output_info):
                print("Particle reflected by cavity.")
            i += 1
            break

    if(debug_output_info):
        print("Number of steps:", i)

    z = np.array(z)
    v = np.array(v)
    beta = v / c

    if(debug_plot):
        E_MeV = (0.5 * m_over_q * v**2)*1e-6

        t_z = np.linspace(0., i * dt, i) * 1e9
        t_v = np.linspace(-.5 * dt, t_z[-1] - 0.5 * dt, i) * 1e9

        if(not plt.fignum_exists((10))):
            fig = plt.figure(10)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

        else:
            fig = plt.figure(10)
            ax1, ax2 = fig.get_axes()

        ax1.plot(t_v, E_MeV)
        ax1.set_ylabel('Energy [MeV]')
        ax1.grid(True)

        ax2.plot(t_z, z*1e3, label=phi_0)
        ax2.set_ylabel('Position [mm]')
        ax2.set_xlabel('Time [ns]')
        ax2.grid(True)
        ax2.legend()
        fig.show()

    return beta


# =============================================================================
# Inputs
# =============================================================================
nz, zmax, Norm, Fz_array = select_and_load_field_map_file(
    '/home/placais/TraceWin/work_field_map/work_field_map.dat',
    100., 0.0, 'Simple_Spoke_1D')


f_MHz = 352.2
omega_0 = 2. * np.pi * f_MHz * 1e6

z_array = np.linspace(0., zmax, nz + 1)         # z cav local coordinates
Fz_func = interp1d(z_array, Fz_array)           # interpolate field


phi_0 = np.deg2rad(142.089)     # LINAC.theta_i[36]
t_0 = 0.
E_MeV = 18.793905
v_0 = np.sqrt(2. * E_MeV * 1e6 * q / m)
dt = 1e-12
t = 0.
z = [0.]
t_max = 1.

# LINAC.k_e[36] = 1.68927
cavity_field = electric_field(Norm * 1.68927, Fz_func, omega_0, phi_0)

beta = compute_acceleration_with_PIC(cavity_field, v_0, dt)
gamma = 1. / np.sqrt(1. - beta**2)
