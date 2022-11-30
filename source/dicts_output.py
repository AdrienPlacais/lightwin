#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:01:01 2022.

@author: placais
"""

# NB: all phi are in radians. The reason why they all have unit [deg] is that
# they are always converted to degrees when outputed.
d_markdown = {
    # Accelerator
    'M_11': r"$M_{11}$",
    'M_12': r"$M_{12}$",
    'M_21': r"$M_{21}$",
    'M_22': r"$M_{22}$",
    'eps_zdelta': r"$\epsilon_{z\delta}$ [$\pi$.m.rad]",
    'eps_z': r"$\epsilon_{zz'}$ [mm/$\pi$.mrad]",
    'eps_w': r"$\epsilon_{\phi W}$ [deg/$\pi$.MeV]",
    'alpha_zdelta': r"$\alpha_{z\delta}$ [1]",
    'alpha_z': r"$\alpha_{zz'}$ [1]",
    'alpha_w': r"$\alpha_{\phi W}$ [1]",
    'beta_zdelta': r"$\beta_{z\delta}$ [mm/$\pi$.%]",
    'beta_z': r"$\beta_{zz'}$ [mm/$\pi$.mrad]",
    'beta_w': r"$\beta_{\phi W}$ [deg/$\pi$.MeV]",
    'gamma_zdelta': r"$\gamma_{z\delta}$ [$\pi$/mm.rad]",
    'gamma_z': r"$\gamma_{zz'}$ [$\pi$/mm.mrad]",
    'gamma_w': r"$\gamma_{\phi W}$ [$\pi$/deg.MeV]",
    'envelope_pos_zdelta': r"$\sigma_z$ [m]",
    'envelope_pos_z': r"$\sigma_z$ [mm]",
    'envelope_pos_w': r"$\sigma_\phi$ [deg]",
    'envelope_energy_zdelta': r"$\sigma_\delta$ [rad]",
    'envelope_energy_z': r"$\sigma_{z'}$ [mrad]",
    'envelope_energy_w': r"$\sigma_\phi$ [MeV]",
    'mismatch factor': r"$M$",
    # Element
    'elt number': "Element number",
    # RfField
    'v_cav_mv': "Acc. field [MV]",
    'phi_s': "Synch. phase [deg]",
    'k_e': r"$k_e$ [1]",
    'phi_0_abs': r"$\phi_{0, abs}$ [deg]",
    'phi_0_rel': r"$\phi_{0, rel}$ [deg]",
    # Particle
    'z_abs': "Synch. position [m]",
    'w_kin': "Beam energy [MeV]",
    'w_kin_err': "Error",
    'phi_abs_array': "Beam phase [deg]",
    'phi_abs_array_err': "Error",
    'beta': r"Synch. $\beta$ [1]",
    'beta_err': r"Abs. $\beta$ error [1]",
    # Misc
    'struct': "Structure",
    # ListOfElements
}

d_plot_kwargs = {
    'z_abs': {'marker': None},
    'element number': {'marker': None},
    'w_kin': {'marker': None},
    'w_kin_err': {'marker': None},
    'phi_abs_array': {'marker': None},
    'phi_abs_array_err': {'marker': None},
    'beta': {'marker': None},
    'beta_err': {'marker': None},
    'struct': {'marker': None},
    'v_cav_mv': {'marker': 'o'},
    'phi_s': {'marker': 'o'},
    'k_e': {'marker': 'o'},
    'eps_zdelta': {'marker': None},
    'eps_z': {'marker': None},
    'eps_w': {'marker': None},
    'alpha_zdelta': {'marker': None},
    'alpha_z': {'marker': None},
    'alpha_w': {'marker': None},
    'beta_zdelta': {'marker': None},
    'beta_z': {'marker': None},
    'beta_w': {'marker': None},
    'gamma_zdelta': {'marker': None},
    'gamma_z': {'marker': None},
    'gamma_w': {'marker': None},
    'envelope_pos_zdelta': {'marker': None},
    'envelope_pos_z': {'marker': None},
    'envelope_pos_w': {'marker': None},
    'envelope_energy_zdelta': {'marker': None},
    'envelope_energy_z': {'marker': None},
    'envelope_energy_w': {'marker': None},
    'mismatch factor': {'marker': None},
}
