#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:44:00 2021

@author: placais
"""

import helper
from constants import m_MeV, c


class Particle():
    """Class to hold the position, energy, etc of a particle."""

    def __init__(self, z, e_mev, omega_0):
        self.z = {
            'abs': z,    # Position from the start of the line
            'rel': 0.,      # Position from the start of the element
            }
        # TODO not sure how to handle z

        self.energy = {
            'e_mev': None,
            'gamma': None,
            'beta': None,
            'e_array_mev': [],
            'gamma_array': [],
            }
        self.set_energy(e_mev, delta_e=False)

        self.phi = None
        self.set_phi(omega_0)

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

        self.energy['e_array_mev'].append(self.energy['e_mev'])
        self.energy['gamma_array'].append(self.energy['gamma'])

    def advance_position(self, delta_pos):
        """Advance particle by delt_pos."""
        self.z['abs'] += delta_pos
        self.z['rel'] += delta_pos

    def set_phi(self, omega_0):
        """Update phi by taking z_rel and beta."""
        self.phi = omega_0 * self.z['rel'] / (self.energy['beta'] * c)
