#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:44:00 2021

@author: placais
"""
import numpy as np
import pandas as pd
import helper
from constants import E_rest_MeV, c


class Particle():
    """
    Class to hold the position, energy, etc of a particle.

    Phase is defined as:
        phi = omega_0_bunch * t
    while in electric_field it is:
        phi = omega_0_rf * t
    """

    def __init__(self, z, e_mev, omega0_bunch, n_steps=1, synchronous=False):
        self.synchronous = synchronous
        self.flag_phi_abs = True

        self.z = {
            'rel': z,           # Position from the start of the element
            'abs_array': np.full((n_steps + 1), np.NaN),
            }
        self.z['abs_array'][0] = z

        self.omega0 = {
            'ref': omega0_bunch,        # The one we use
            'bunch': omega0_bunch,      # Should match 'ref' outside cavities
            'rf': 2. * omega0_bunch,    # Should match 'ref' inside cavities
            'lambda_array': np.full((n_steps + 1), np.NaN),
            }

        self.energy = {
            'kin_array_mev': np.full((n_steps + 1), np.NaN),
            'gamma_array': np.full((n_steps + 1), np.NaN),
            'beta_array': np.full((n_steps + 1), np.NaN),
            'p_array_mev': np.full((n_steps + 1), np.NaN),
            }
        self.set_energy(e_mev, idx=0, delta_e=False)

        # Dict used to navigate between phi_rf = omega_rf * t and
        # phi = omega_bunch * t (which is the default in absence of any other
        # precision)
        self.frac_omega = {
            'rf_to_bunch': 1.,
            'bunch_to_rf': 1.,
            }
        self.phi = {
            'rel': None,
            'abs': None,
            'abs_rf': None,
            'abs_array': np.full((n_steps + 1), np.NaN),
            }
        self._init_phi(idx=0)

        self.phase_space = {
            # z_abs-s_abs or z_rel-s_rel
            'z_array': np.full((n_steps + 1), np.NaN),
            'delta_array': np.full((n_steps + 1), np.NaN),  # (p - p_s) / p_s
            'both_array': np.full((n_steps + 1), np.NaN),
            'phi_array_rad': np.full((n_steps + 1), np.NaN),
            }

    def set_energy(self, e_mev, idx=np.NaN, delta_e=False):
        """
        Update the energy dict.

        Parameters
        ----------
        e_mev: float
            New energy in MeV.
        idx: int, opt
            Index of the the energy concerned. If NaN, e_mev replaces the first
            NaN element of kin_array_mev.
        delta_e: bool, opt
            If True, energy is increased by e_mev. If False, energy is set to
            e_mev.
        """
        if np.isnan(idx):
            idx = np.where(np.isnan(self.energy['kin_array_mev']))[0][0]

        if delta_e:
            self.energy['kin_array_mev'][idx] = \
                self.energy['kin_array_mev'][idx-1] + e_mev
        else:
            self.energy['kin_array_mev'][idx] = e_mev

        gamma = helper.kin_to_gamma(self.energy['kin_array_mev'][idx],
                                    E_rest_MeV)
        beta = helper.gamma_to_beta(gamma)
        p_mev = helper.gamma_and_beta_to_p(gamma, beta, E_rest_MeV)

        self.energy['gamma_array'][idx] = gamma
        self.energy['beta_array'][idx] = beta
        self.energy['p_array_mev'][idx] = p_mev
        self.omega0['lambda_array'][idx] = 2. * np.pi * c / self.omega0['ref']

    def advance_position(self, delta_pos, idx=np.NaN):
        """
        Advance particle by delta_pos.

        Parameters
        ----------
        delta_pos: float
            Difference of position in m.
        idx: int, opt
            Index of the the energy concerned. If NaN, the new position is at
            the first NaN element of abs_array.
        """
        if np.isnan(idx):
            idx = np.where(np.isnan(self.z['abs_array']))[0][0]
        self.z['rel'] += delta_pos
        self.z['abs_array'][idx] = self.z['abs_array'][idx-1] + delta_pos

    def _init_phi(self, idx=0):
        """Init phi by taking z_rel and beta."""
        phi_abs = helper.z_to_phi(self.z['abs_array'][idx],
                                  self.energy['beta_array'][idx],
                                  self.omega0['bunch'])
        self.phi['abs'] = phi_abs
        self.phi['abs_rf'] = phi_abs * self.frac_omega['bunch_to_rf']
        self.phi['rel'] = phi_abs
        self.phi['abs_array'][idx] = phi_abs
        df = {
            'phi_abs_array': [self.phi['abs_array'][idx]],
            'phi_abs': [self.phi['abs']],
            'phi_abs_rf': [self.phi['abs_rf']],
            'phi_rel': [self.phi['rel']],
            }
        self.df = pd.DataFrame(df)

    def advance_phi(self, delta_phi, idx=np.NaN, flag_rf=False):
        """
        Increase relative and absolute phase by delta_phi.

        Parameters
        ----------
        delta_phi : float
            Phase increment, expressed as phi = omega0_bunch * time. Shouldn't
            be mistaken with phi = omega0_rf * bunch.
        idx : integer, optional
            Index of the new phase in abs_array. By default, the first NaN
            element of the array is replaced. Thus, idx has to be given only
            when recomputing the transfer matrices.
        flag_rf : boolean, optional
            If False, delta_phi = omega_0_bunch * delta_t. Otherwise,
            delta_phi = omega_0_rf * delta_t. Default is False.
        """
        if np.isnan(idx):
            idx = np.where(np.isnan(self.phi['abs_array']))[0][0]

        if flag_rf:
            self.phi['abs_rf'] += delta_phi
            self.phi['rel'] += delta_phi
            delta_phi *= self.frac_omega['rf_to_bunch']
        else:
            self.phi['abs_rf'] += delta_phi
            self.phi['rel'] += delta_phi

        self.phi['abs'] += delta_phi
        self.phi['abs_array'][idx] = self.phi['abs_array'][idx-1] + delta_phi

        new_row = {
            'phi_abs_array': self.phi['abs_array'][idx],
            'phi_abs': self.phi['abs'],
            'phi_abs_rf': self.phi['abs_rf'],
            'phi_rel': self.phi['rel'],
            }
        new_row = pd.Series(data=new_row)
        self.df = self.df.append(new_row, ignore_index=True)

    def compute_phase_space_tot(self, synch):
        """
        Compute phase-space array.

        synch_particle is an instance of Particle corresponding to the
        synchronous particle.
        """
        self.phase_space['phi_array_rad'] = self.phi['abs_array'] \
            - synch.phi['abs_array']
        # Warning, according to doc lambda is RF wavelength... which does not
        # make any sense outside of cavities.
        self.phase_space['z_array'] = helper.phi_to_z(
            self.phase_space['phi_array_rad'],
            self.energy['beta_array'],
            self.omega0['bunch'])

        self.phase_space['delta_array'] = (self.energy['p_array_mev']
                                           - synch.energy['p_array_mev']) \
            / synch.energy['p_array_mev']

        self.phase_space['both_array'] = np.vstack(
            (self.phase_space['z_array'],
             self.phase_space['delta_array'])
            )
        self.phase_space['both_array'] = np.swapaxes(
            self.phase_space['both_array'], 0, 1)

    def enter_cavity(self, acc_field):
        """Change the omega0 at the entrance."""
        self.omega0['ref'] = acc_field.omega0_rf
        self.omega0['rf'] = acc_field.omega0_rf
        self.frac_omega['bunch_to_rf'] =\
            self.omega0['rf'] / self.omega0['bunch']
        self.frac_omega['rf_to_bunch'] = 1. / self.frac_omega['bunch_to_rf']

        # Convert the relative initial phase of the particle into an absolute
        # initial phase
        if self.synchronous:
            acc_field.phi_0_rel_to_abs(self.phi['abs_rf'])
        else:
            assert acc_field.phi_0['abs'] is not None, 'The absolute ' + \
                'entrance phase of the cavity shall be calculated with the' + \
                ' synchronous particle prior to any non-synch calculation.'

    def exit_cavity(self, index):
        """Reset frac_omega."""
        self.omega0['ref'] = self.omega0['bunch']
        self.frac_omega['rf_to_bunch'] = 1.
        self.frac_omega['bunch_to_rf'] = 1.


def create_rand_particles(e_0_mev, omega0_bunch):
    """Create two random particles."""
    delta_z = 1e-4
    delta_E = 1e-4

    rand_1 = Particle(-1.42801442802603928417e-04,
                      1.66094219207764304258e+01,
                      omega0_bunch)
    rand_2 = Particle(2.21221539793564048182e-03,
                      1.65923664093018210508e+01,
                      omega0_bunch)

    # rand_1 = Particle(
    #     random.uniform(0., delta_z * .5),
    #     random.uniform(e_0_mev,  e_0_mev + delta_E * .5),
    #     omega0_bunch)

    # rand_2 = Particle(
    #     random.uniform(-delta_z * .5, 0.),
    #     random.uniform(e_0_mev - delta_E * .5, e_0_mev),
    #     omega0_bunch)

    return rand_1, rand_2
