#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:44:00 2021.

@author: placais
"""
import numpy as np
import pandas as pd
import helper
from constants import E_REST_MEV, OMEGA_0_BUNCH


class Particle():
    """
    Class to hold the position, energy, etc of a particle.

    Phase is defined as:
        phi = omega_0_bunch * t
    while in electric_field it is:
        phi = omega_0_rf * t
    """

    def __init__(self, z, e_mev, n_steps=1, synchronous=False,
                 reference=True):
        self.info = {
            # Is this particle the generator?
            'synchronous': synchronous,
            # Is it in a reference (non-faulty) linac?
            'reference': reference,
            # Are the phases absolute? Or relative?
            'fixed': False,
            # FIXME: still used. Can be replaced by cav status?
            # I think it could be a good thing as to prepare the global + local
            # compensation
        }

        self.z = {
            'rel': z,           # Position from the start of the element
            'abs_array': np.full((n_steps + 1), np.NaN),
        }
        self.z['abs_array'][0] = z
        self.energy = {
            'kin_array_mev': np.full((n_steps + 1), np.NaN),
            'gamma_array': np.full((n_steps + 1), np.NaN),
            'beta_array': np.full((n_steps + 1), np.NaN),
            'p_array_mev': np.full((n_steps + 1), np.NaN),
        }
        self.set_energy(e_mev, idx=0, delta_e=False)

        self.phi = {
            'rel': None,
            'abs': None,
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
        if not self.info["synchronous"]:
            print("__init__ non-synch Particle: the absolute position of",
                  "a non synch particle is not initialized.")

    def init_abs_z(self, list_of_elements):
        """Create the array of absolute positions."""
        assert self.info["synchronous"], """This routine only works for the
        synch particle I think."""
        # Get all positions
        z_abs = [elt.pos_m["abs"]
                 for elt in list_of_elements]
        # Concatenate list of arrays into unique array
        z_abs = np.concatenate(z_abs)
        # Remove duplicates (last pos_m["abs"] of an element == first of
        # following element)
        z_abs = np.unique(z_abs)
        self.z["abs_array"] = z_abs

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
                self.energy['kin_array_mev'][idx - 1] + e_mev
        else:
            self.energy['kin_array_mev'][idx] = e_mev

        gamma = helper.kin_to_gamma(self.energy['kin_array_mev'][idx],
                                    E_REST_MEV)
        beta = helper.gamma_to_beta(gamma)
        p_mev = helper.gamma_and_beta_to_p(gamma, beta, E_REST_MEV)

        self.energy['gamma_array'][idx] = gamma
        self.energy['beta_array'][idx] = beta
        self.energy['p_array_mev'][idx] = p_mev

    def _init_phi(self, idx=0):
        """Init phi by taking z_rel and beta."""
        phi_abs = helper.z_to_phi(self.z['abs_array'][idx],
                                  self.energy['beta_array'][idx],
                                  OMEGA_0_BUNCH)
        self.phi['abs'] = phi_abs
        self.phi['abs_array'][idx] = phi_abs
        self.phi['rel'] = phi_abs
        self.df = pd.DataFrame({
            'phi_abs_array': [self.phi['abs_array'][idx]],
            'phi_abs': [self.phi['abs']],
            'phi_rel': [self.phi['rel']],
        })

    def advance_phi(self, delta_phi, idx=np.NaN, flag_rf=False):
        """
        Increase relative and absolute phase by delta_phi.

        Parameters
        ----------
        delta_phi : float
            Phase increment.
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

        self.phi['rel'] += delta_phi

        if flag_rf:
            self.phi['abs_rf'] += delta_phi
            delta_phi *= self.frac_omega['rf_to_bunch']

        self.phi['abs'] += delta_phi
        self.phi['abs_array'][idx] = self.phi['abs_array'][idx - 1] + delta_phi

    def keep_energy_and_phase(self, results, idx_range):
        """Assign the energy and phase data to synch after MT calculation."""
        w_kin = np.array(results["w_kin"])
        self.energy['kin_array_mev'][idx_range] = w_kin
        self.energy['gamma_array'][idx_range] = helper.kin_to_gamma(w_kin)
        self.energy['beta_array'][idx_range] = helper.kin_to_beta(w_kin)
        self.phi['abs_array'][idx_range] = np.array(results["phi_abs"])


def create_rand_particles(e_0_mev):
    """Create two random particles."""
    delta_z = 1e-4
    delta_E = 1e-4

    rand_1 = Particle(-1.42801442802603928417e-04,
                      1.66094219207764304258e+01,)
    rand_2 = Particle(2.21221539793564048182e-03,
                      1.65923664093018210508e+01,)

    # rand_1 = Particle(
    #     random.uniform(0., delta_z * .5),
    #     random.uniform(e_0_mev,  e_0_mev + delta_E * .5),
    #     omega0_bunch)

    # rand_2 = Particle(
    #     random.uniform(-delta_z * .5, 0.),
    #     random.uniform(e_0_mev - delta_E * .5, e_0_mev),
    #     omega0_bunch)

    return rand_1, rand_2
