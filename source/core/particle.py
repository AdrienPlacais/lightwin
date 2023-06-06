#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:44:00 2021.

@author: placais
"""
from typing import Any
import logging
import numpy as np
import pandas as pd
from util.helper import recursive_items, recursive_getter
import util.converters as convert


class Particle:
    """
    Class to hold the position, energy, etc of a particle.

    Phase is defined as:
        phi = omega_0_bunch * t
    while in electric_field it is:
        phi = omega_0_rf * t
    """

    def __init__(self, z_0: float, e_mev: float, n_steps: int = 1,
                 synchronous: bool = False, reference: bool = True) -> None:
        self.part_info = {
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

        self.pos = {
            'z_rel': z_0,           # Position from the start of the element
            'z_abs': np.full((n_steps + 1), np.NaN),
        }
        self.pos['z_abs'][0] = z_0
        self.energy = {
            'w_kin': np.full((n_steps + 1), np.NaN),
            'gamma': np.full((n_steps + 1), np.NaN),
            'beta': np.full((n_steps + 1), np.NaN),
            'p': np.full((n_steps + 1), np.NaN),  # Necessary? TODO
        }
        self.set_energy(e_mev, idx=0, delta_e=False)

        self.phi = {
            'phi_rel': None,
            'phi_abs': None,
            'phi_abs_rf': None,
            'phi_abs_array': np.full((n_steps + 1), np.NaN),
        }
        self._init_phi(idx=0)

        if not self.part_info["synchronous"]:
            logging.warning("The absolute position of a non synchronous "
                            + "particle is not initialized.")

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: tuple[str], to_deg: bool = False, **kwargs: dict
            ) -> tuple[Any]:
        """Shorthand to get attributes."""
        val = {}
        for key in keys:
            val[key] = []

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)

            if val[key] is not None and to_deg and 'phi' in key:
                val[key] = np.rad2deg(val[key])

        # Convert to list
        out = [val[key] for key in keys]

        if len(out) == 1:
            return out[0]
        # implicit else:
        return tuple(out)

    def init_abs_z(self, abs_z_array: np.ndarray) -> None:
        """Create the array of absolute positions."""
        assert self.part_info["synchronous"], """This routine only works for
        the synch particle I think."""
        self.pos['z_abs'] = abs_z_array

    def set_energy(self, e_mev: float, idx: int | np.NaN = np.NaN,
                   delta_e: bool = False) -> None:
        """
        Update the energy dict.

        Parameters
        ----------
        e_mev : float
            New energy in MeV.
        idx : int | np.NaN, opt
            Index of the the energy concerned. If NaN, e_mev replaces the first
            NaN element of w_kin. The default is np.NaN
        delta_e : bool, opt
            If True, energy is increased by e_mev. If False, energy is set to
            e_mev. The default is False.
        """
        if np.isnan(idx):
            idx = np.where(np.isnan(self.energy['w_kin']))[0][0]

        if delta_e:
            self.energy['w_kin'][idx] = \
                self.energy['w_kin'][idx - 1] + e_mev
        else:
            self.energy['w_kin'][idx] = e_mev

        gamma = convert.energy(self.energy['w_kin'][idx], "kin to gamma")
        beta = convert.energy(gamma, "gamma to beta")
        p_mev = convert.energy(gamma, "gamma to p")

        self.energy['gamma'][idx] = gamma
        self.energy['beta'][idx] = beta
        self.energy['p'][idx] = p_mev

    def _init_phi(self, idx: int = 0) -> None:
        """Init phi by taking z_rel and beta."""
        phi_abs = convert.position(self.pos['z_abs'][idx],
                                   self.energy['beta'][idx], "z to phi")
        self.phi['phi_abs'] = phi_abs
        self.phi['phi_abs_array'][idx] = phi_abs
        self.phi['phi_rel'] = phi_abs
        self.df = pd.DataFrame({
            'phi_abs_array': [self.phi['phi_abs_array'][idx]],
            'phi_abs': [self.phi['phi_abs']],
            'phi_rel': [self.phi['phi_rel']],
        })

    # FIXME still used?
    def advance_phi(self, delta_phi: float, idx: int | np.Nan = np.NaN,
                    flag_rf: bool = False) -> None:
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
            delta_phi = omega_0_rf * delta_t. The default is False.
        """
        if np.isnan(idx):
            idx = np.where(np.isnan(self.phi['phi_abs_array']))[0][0]

        self.phi['phi_rel'] += delta_phi

        if flag_rf:
            self.phi['phi_abs_rf'] += delta_phi
            delta_phi *= self.frac_omega['rf_to_bunch']

        self.phi['phi_abs'] += delta_phi
        self.phi['phi_abs_array'][idx] = self.phi['phi_abs_array'][idx - 1] \
            + delta_phi

    def keep_energy_and_phase(self, results: dict[str, np.ndarray],
                              idx_range: range) -> None:
        """Assign the energy and phase data to synch after MT calculation."""
        w_kin = np.array(results["w_kin"])
        self.energy['w_kin'][idx_range] = w_kin
        self.energy['gamma'][idx_range] = convert.energy(w_kin, "kin to gamma")
        self.energy['beta'][idx_range] = convert.energy(w_kin, "kin to beta")
        self.phi['phi_abs_array'][idx_range] = np.array(results[
            "phi_abs_array"])


# def create_rand_particles(e_0_mev):
#     """Create two random particles."""
#     delta_z = 1e-4
#     delta_E = 1e-4

#     rand_1 = Particle(-1.42801442802603928417e-04,
#                       1.66094219207764304258e+01,)
#     rand_2 = Particle(2.21221539793564048182e-03,
#                       1.65923664093018210508e+01,)

#     # rand_1 = Particle(
#     #     random.uniform(0., delta_z * .5),
#     #     random.uniform(e_0_mev,  e_0_mev + delta_E * .5),
#     #     omega0_bunch)

#     # rand_2 = Particle(
#     #     random.uniform(-delta_z * .5, 0.),
#     #     random.uniform(e_0_mev - delta_E * .5, e_0_mev),
#     #     omega0_bunch)

#     return rand_1, rand_2
