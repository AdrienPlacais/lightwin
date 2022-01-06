#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:44:00 2021

@author: placais
"""
import numpy as np
import helper
from constants import m_MeV, m_kg, c


class Particle():
    """Class to hold the position, energy, etc of a particle."""

    def __init__(self, z, e_mev, omega0_bunch, synchronous=False):
        print('part init to: ', z, e_mev)
        self.synchronous = synchronous
        self.z = {
            'abs': z,           # Position from the start of the line
            'rel': z,           # Position from the start of the element
            'abs_array': [z],
            }

        self.omega0 = {
            'ref': omega0_bunch,        # The one we use
            'bunch': omega0_bunch,      # Should match 'ref' outside cavities
            'rf': None,                 # Should match 'ref' inside cavities
            'lambda_array': [],
            }

        self.energy = {
            'e_mev': None,
            'gamma': None,
            'beta': None,
            'p': None,
            'e_array_mev': [],
            'gamma_array': [],
            'beta_array': [],
            'p_array': [],
            }
        self.set_energy(e_mev, delta_e=False)

        self.phi = {
            'abs': None,
            'rel': None,
            'abs_deg': None,
            'abs_array': [],
            # Used to keep the delta phi on the whole cavity:
            'idx_cav_entry': None,
            }
        self._init_phi()

        self.phase_space = {
            'z_array': [],      # z_abs - s_abs or z_rel - s_rel
            'delta_array': [],  # (p - p_s) / p_s
            'both_array': [],
            'phi_array_rad': [],
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
        self.energy['p'] = helper.gamma_and_beta_to_p(self.energy['gamma'],
                                                      self.energy['beta'],
                                                      m_kg)
        self.energy['e_array_mev'].append(self.energy['e_mev'])
        self.energy['gamma_array'].append(self.energy['gamma'])
        self.energy['beta_array'].append(self.energy['beta'])
        self.energy['p_array'].append(self.energy['p'])

        self.omega0['lambda_array'].append(2. * np.pi * c / self.omega0['ref'])

    def advance_position(self, delta_pos):
        """Advance particle by delt_pos."""
        self.z['abs'] += delta_pos
        self.z['rel'] += delta_pos
        self.z['abs_array'].append(self.z['abs'])

    def _init_phi(self):
        """Init phi by taking z_rel and beta."""
        self.phi['abs'] = helper.z_to_phi(self.z['rel'],
                                          self.energy['beta'],
                                          self.omega0['bunch'])
        self.phi['rel'] = self.phi['abs']
        self.phi['abs_deg'] = np.rad2deg(self.phi['abs'])
        self.phi['abs_array'].append(self.phi['abs'])

    def advance_phi(self, delta_phi):
        """Increase relative and absolute phase by delta_phi."""
        self.phi['abs'] += delta_phi
        self.phi['rel'] += delta_phi
        self.phi['abs_deg'] += np.rad2deg(delta_phi)
        self.phi['abs_array'].append(self.phi['abs'])

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

        self.phase_space['delta_array'] = (self.energy['p_array']
                                           - synch.energy['p_array']) \
            / synch.energy['p_array']

        self.phase_space['both_array'] = np.vstack(
            (self.phase_space['z_array'],
             self.phase_space['delta_array'])
            )
        self.phase_space['both_array'] = np.swapaxes(
            self.phase_space['both_array'], 0, 1)

    def enter_cavity(self, omega0_rf):
        """Change the omega0 and save the phase at the entrance."""
        self.phi['idx_cav_entry'] = len(self.phi['abs_array'])
        self.omega0['ref'] = omega0_rf
        self.omega0['rf'] = omega0_rf

    def exit_cavity(self):
        """Recompute phi with the proper omega0, reset omega0."""
        # Helpers
        idx_entry = self.phi['idx_cav_entry']
        idx_exit = len(self.phi['abs_array'])
        frac_omega = self.omega0['bunch'] / self.omega0['rf']

        # Set proper phi
        for i in range(idx_entry, idx_exit):
            delta_phi = self.phi['abs_array'][i] \
                - self.phi['abs_array'][idx_entry - 1]
            self.phi['abs_array'][i] = self.phi['abs_array'][idx_entry - 1] \
                + delta_phi * frac_omega
        self.phi['abs'] = self.phi['abs_array'][-1]
        self.phi['abs_deg'] = np.rad2deg(self.phi['abs'])

        # Reset proper omega
        self.omega0['ref'] = self.omega0['bunch']

        # Remove unsused variables
        self.phi['idx_cav_entry'] = None

    def list_to_array(self):
        """
        Get all object attributes that are lists and convert it into np.arrays.

        When the size of the array is not known, we use lists rather than
        np.arrays as 'append' is more efficient than 'np.hstack' (no data
        copy).
        However, np.arrays are better for math operations, so when all data
        has been computed, we convert all lists into np.arrays.
        """
        list_of_attrib = dir(self)
        for attrib_name in list_of_attrib:
            attrib = getattr(self, attrib_name)

            if isinstance(attrib, list):
                attrib = np.array(attrib)

            elif isinstance(attrib, dict):
                for key in attrib.keys():
                    if isinstance(attrib[key], list):
                        attrib[key] = np.array(attrib[key])


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
