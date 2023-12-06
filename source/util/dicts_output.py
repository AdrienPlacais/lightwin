#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Here we define functions to make proper outputs of various quantities."""
# NB: all phi are in radians. The reason why they all have unit [deg] is that
# they are always converted to degrees when outputed.
markdown = {
    # Accelerator
    'M_11': r"$M_{11}$",
    'M_12': r"$M_{12}$",
    'M_21': r"$M_{21}$",
    'M_22': r"$M_{22}$",
    # Beam Parameters
    'eps_zdelta': r"$\epsilon_{z\delta}$ [$\pi$.mm.%]",
    'alpha_zdelta': r"$\alpha_{z\delta}$ [1]",
    'beta_zdelta': r"$\beta_{z\delta}$ [mm/$\pi$.%]",
    'gamma_zdelta': r"$\gamma_{z\delta}$ [%/$\pi$.mm]",
    'envelope_pos_zdelta': r"$\sigma_z$ @ $1\sigma$ [mm]",
    'envelope_energy_zdelta': r"$\sigma_\delta$ @ $1\sigma$ [%]",

    'eps_z': r"$\epsilon_{zz'}$ [$\pi$.mm.mrad]",
    'alpha_z': r"$\alpha_{zz'}$ [1]",
    'beta_z': r"$\beta_{zz'}$ [mm/$\pi$.mrad]",
    'gamma_z': r"$\gamma_{zz'}$ [mrad/$\pi$.mm]",
    'envelope_pos_z': r"$\sigma_z$ @ $1\sigma$ [mm]",
    'envelope_energy_z': r"$\sigma_{z'}$ @ $1\sigma$ [mrad]",

    'eps_phiw': r"$\epsilon_{\phi W}$ [$\pi$.deg.MeV]",
    'alpha_phiw': r"$\alpha_{\phi W}$ [1]",
    'beta_phiw': r"$\beta_{\phi W}$ [deg/$\pi$.MeV]",
    'gamma_phiw': r"$\gamma_{\phi W}$ [MeV/$\pi$.deg]",
    'envelope_pos_phiw': r"Norm. $\sigma_\phi$ @ $1\sigma$ [deg]",
    'envelope_energy_phiw': r"Norm. $\sigma_\phi$ @ $1\sigma$ [MeV]",

    'mismatch_factor': r"$M$",
    'mismatch_factor_x': r"$M_{xx'}$",
    'mismatch_factor_y': r"$M_{yy'}$",
    'mismatch_factor_t': r"$M_t$",
    'mismatch_factor_zdelta': r"$M_{z\delta}$",

    # Element
    'elt_idx': "Element index",
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
    'phi_abs': "Beam phase [deg]",
    'beta': r"Synch. $\beta$ [1]",
    # Misc
    'struct': "Structure",
    'err_simple': "Error",
    'err_abs': "Abs. error",
    'err_rel': "Rel. error",
    'err_log': "log of rel. error",
    # ListOfElements
    # TraceWin
    'pow_lost': "Lost power [W]",
    'eps_x': r"Norm. RMS $\epsilon_{xx'}$ [$\pi$mm mrad]",
    'eps_y': r"Norm. RMS $\epsilon_{yy'}$ [$\pi$mm mrad]",
    'eps_t': r"Norm. RMS $\epsilon_t$ [$\pi$mm mrad]",
    'eps_phiw99': r"Norm. 99% $\epsilon_{zz'}$ [$\pi$mm mrad]",
    'eps_x99': r"Norm. 99% $\epsilon_{xx'}$ [$\pi$mm mrad]",
    'eps_y99': r"Norm. 99% $\epsilon_{yy'}$ [$\pi$mm mrad]",

    'optimisation_time': "Optimisation time",
}

plot_kwargs = {
    # Accelerator
    'eps_zdelta': {'marker': None},
    'eps_z': {'marker': None},
    'eps_phiw': {'marker': None},
    'alpha_zdelta': {'marker': None},
    'alpha_z': {'marker': None},
    'alpha_phiw': {'marker': None},
    'beta_zdelta': {'marker': None},
    'beta_z': {'marker': None},
    'beta_phiw': {'marker': None},
    'gamma_zdelta': {'marker': None},
    'gamma_z': {'marker': None},
    'gamma_phiw': {'marker': None},
    'envelope_pos_zdelta': {'marker': None},
    'envelope_pos_z': {'marker': None},
    'envelope_pos_phiw': {'marker': None},
    'envelope_energy_zdelta': {'marker': None},
    'envelope_energy_z': {'marker': None},
    'envelope_energy_phiw': {'marker': None},
    'mismatch_factor_zdelta': {'marker': None},
    # Element
    'elt_idx': {'marker': None},
    # RfField
    'v_cav_mv': {'marker': 'o'},
    'phi_s': {'marker': 'o'},
    'k_e': {'marker': 'o'},
    # Particle
    'z_abs': {'marker': None},
    'w_kin': {'marker': None},
    'phi_abs': {'marker': None},
    'beta': {'marker': None},
    # Misc
    'struct': {'marker': None},
    'err_abs': {'marker': None},
    'err_rel': {'marker': None},
    'err_log': {'marker': None},
    # ListOfElements
}
