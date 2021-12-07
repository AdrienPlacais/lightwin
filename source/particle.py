#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:44:00 2021

@author: placais
"""

import numpy as np
import helper
from constants import m_MeV, c, m_kg, q_over_m


class Particle():
    """Class to hold the position, energy, etc of a particle."""

    def __init__(self, z, e_mev, omega0_bunch):
        self.z = {
            'abs': z,           # Position from the start of the line
            'rel': z,           # Position from the start of the element
            'abs_array': [],
            }

        self.energy = {
            'e_mev': None,
            'gamma': None,
            'beta': None,
            'p': None,
            'e_array_mev': [],
            'gamma_array': [],
            }
        self.set_energy(e_mev, delta_e=False)

        self.phi = {
            'abs': None,
            'rel': None,
            'abs_deg': None,
            }
        self.init_phi(omega0_bunch)

        self.phase_space = {
            'z': None,      # z_abs - s_abs or z_rel - s_rel
            'delta': None   # (p - p_s) / p_s
            }

    def set_energy(self, e_mev, delta_e=False):
        """
        Update the energy dict.

        If delta_e is True energy is increased by e_mev.
        If False, energy is set to e_mev.
        """
        if delta_e:
            self.energy['e_mev'] += e_mev
        else:
            self.energy['e_mev'] = e_mev

        self.energy['gamma'] = helper.mev_to_gamma(self.energy['e_mev'], m_MeV)
        self.energy['beta'] = helper.gamma_to_beta(self.energy['gamma'])
        self.energy['p'] = helper.mev_and_gamma_to_p(self.energy['e_mev'],
                                                     self.energy['gamma'],
                                                     m_kg, q_over_m)

        self.energy['e_array_mev'].append(self.energy['e_mev'])
        self.energy['gamma_array'].append(self.energy['gamma'])

    def advance_position(self, delta_pos):
        """Advance particle by delt_pos."""
        self.z['abs'] += delta_pos
        self.z['rel'] += delta_pos
        self.z['abs_array'].append(self.z['abs'])

    def init_phi(self, omega0_bunch):
        """Init phi by taking z_rel and beta."""
        print('Init. Old phi: ', self.phi['abs_deg'])
        self.phi['abs'] = omega0_bunch * self.z['rel'] / (self.energy['beta']
                                                          * c)
        self.phi['rel'] = self.phi['abs']
        self.phi['abs_deg'] = np.rad2deg(self.phi['abs'])
        print('New phi: ', self.phi['abs_deg'])
        print('')

    def advance_phi(self, delta_phi):
        """Increase relative and absolute phase by delta_phi."""
        print('Old phi: ', self.phi['abs_deg'])
        self.phi['abs'] += delta_phi
        self.phi['rel'] += delta_phi
        self.phi['abs_deg'] += np.rad2deg(delta_phi)
        print('New phi: ', self.phi['abs_deg'])
        print('')

    def compute_phase_space(self, synch_particle):
        """
        Compute phase-space array.

        synch_particle is an instance of Particle corresponding to the
        synchronous particle.
        """
        self.phase_space['z'] = self.z['rel'] - synch_particle.z['rel']
        self.phase_space['delta'] = (self.energy['p']
                                     - synch_particle.energy['p']) \
            / synch_particle.energy['p']
