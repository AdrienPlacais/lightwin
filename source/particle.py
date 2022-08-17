#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:44:00 2021.

@author: placais
"""
import numpy as np
import pandas as pd
import helper
from constants import E_rest_MeV, c, FLAG_PHI_ABS, OMEGA_0_BUNCH


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

        self.omega0 = {
            'ref': OMEGA_0_BUNCH,        # The one we use
            # FIXME: default set f_rf = 2*f_bunch is dirty
            'rf': 2. * OMEGA_0_BUNCH,    # Should match 'ref' inside cavities
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
                self.energy['kin_array_mev'][idx - 1] + e_mev
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
        self.z['abs_array'][idx] = self.z['abs_array'][idx - 1] + delta_pos

    def _init_phi(self, idx=0):
        """Init phi by taking z_rel and beta."""
        phi_abs = helper.z_to_phi(self.z['abs_array'][idx],
                                  self.energy['beta_array'][idx],
                                  OMEGA_0_BUNCH)
        self.phi['abs'] = phi_abs
        self.phi['abs_rf'] = phi_abs * self.frac_omega['bunch_to_rf']
        self.phi['abs_array'][idx] = phi_abs
        self.phi['rel'] = phi_abs
        self.df = pd.DataFrame({
            'phi_abs_array': [self.phi['abs_array'][idx]],
            'phi_abs': [self.phi['abs']],
            'phi_abs_rf': [self.phi['abs_rf']],
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

    def set_abs_phi(self, new_phi, idx=np.NaN, flag_rf=False):
        """
        Set new absolute phase.

        Parameters
        ----------
        new_phi : float
            New phase.
        idx : integer, optional
            Index of the new phase in abs_array. By default, the first NaN
            element of the array is replaced. Thus, idx has to be given only
            when recomputing the transfer matrices.
        flag_rf : boolean, optional
            If False, new_phi = omega_0_bunch * t. Otherwise,
            phi = omega_0_rf * t. Default is False.
        """
        if np.isnan(idx):
            idx = np.where(np.isnan(self.phi['abs_array']))[0][0]

        if flag_rf:
            self.phi['abs_rf'] = new_phi
            new_phi *= self.frac_omega['rf_to_bunch']

        self.phi['abs'] = new_phi
        self.phi['abs_array'][idx] = new_phi

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
            self.energy['beta_array'], OMEGA_0_BUNCH)

        self.phase_space['delta_array'] = \
            (self.energy['p_array_mev'] - synch.energy['p_array_mev']) / \
            synch.energy['p_array_mev']

        self.phase_space['both_array'] = np.vstack(
            (self.phase_space['z_array'],
             self.phase_space['delta_array'])
        )
        self.phase_space['both_array'] = np.swapaxes(
            self.phase_space['both_array'], 0, 1)

    def set_omega_rf(self, new_omega, phi_bunch_abs):
        """
        Define rf pulsation and convert absolute bunch phase into abs rf phase.

        Parameters
        ----------
        new_omega : float
            RF pulsation.
        phi_bunch_abs : float
            Particle phase as omega_bunch * t.

        Return
        ------
        phi_rf_abs : float
            Particle phase as omega_rf * t.
        """
        self.omega0['rf'] = new_omega
        self.omega0['ref'] = new_omega
        self.frac_omega['rf_to_bunch'] = OMEGA_0_BUNCH / new_omega
        self.frac_omega['bunch_to_rf'] = new_omega / OMEGA_0_BUNCH
        phi_rf_abs = phi_bunch_abs * self.frac_omega['bunch_to_rf']
        return phi_rf_abs

    def enter_cavity(self, acc_field, cav_status='nominal', idx_in=np.NaN,
                     nominal_phi_0_rel=None):
        """
        Change the omega0 at the entrance and compute abs. entry phase.

        acc_field : RfField
            Accelerating field in the current cavity.
        cav_status: str, optional
            Describe the working condition of the cavity. Default is 'nominal'.
        """
        if np.isnan(idx_in):
            idx_in = np.where(np.isnan(self.phi['abs_array']))[0][0] - 1
        bla = self.set_omega_rf(acc_field.omega0_rf)

        self.phi['abs'] = self.phi['abs_array'][idx_in]
        self.phi['abs_rf'] = self.phi['abs'] * self.frac_omega['bunch_to_rf']

        if self.info['synchronous']:
            # Ref linac: we compute every missing phi_0
            if self.info['reference']:
                acc_field.convert_phi_0(
                    self.phi['abs_rf'],
                    abs_to_rel=acc_field.absolute_phase_flag
                )
            else:
                # Phases should have been imported from reference linac
                if cav_status == 'nominal':
                    # We already have the phi0's from the reference linac.
                    # We recompute the relative or absolute one according to
                    # FLAG_PHI_ABS
                    acc_field.convert_phi_0(self.phi['abs_rf'],
                                            abs_to_rel=FLAG_PHI_ABS)
                elif cav_status == 'rephased':
                    # We must keep the relative phase equal to reference linac
                    acc_field.rephase_cavity(self.phi['abs_rf'])

                elif cav_status == 'fault':
                    # Useless, as we used drift functions when there is a fault
                    print('prout particle.enter_cavity')

                elif cav_status == 'compensate':
                    # The phi0's are set by the fitting algorithm. We compute
                    # the missing (abs or rel) value of phi0 for the sake of
                    # completeness, but it won't be used to calculate the
                    # matrix
                    acc_field.convert_phi_0(self.phi['abs_rf'],
                                            abs_to_rel=FLAG_PHI_ABS)

        else:
            print('Warning enter_cavity! Not sure what will happen with a',
                  'non synchronous particle.')

    def exit_cavity(self):
        """Reset frac_omega."""
        self._set_omega_rf(OMEGA_0_BUNCH)
        self.phi['abs_rf'] = None

    def transfer_data(self, elt, w_kin, phi_abs):
        """Assign the energy and phase data to synch after MT calculation."""
        r_idx_elt = range(elt.idx['s_in'] + 1, elt.idx['s_out'] + 1)
        idx_elt_prec = r_idx_elt[0] - 1

        ene = self.energy
        ene['kin_array_mev'][r_idx_elt] = w_kin
        ene['gamma_array'][r_idx_elt] = \
            helper.kin_to_gamma(w_kin, E_rest_MeV)
        ene['beta_array'][r_idx_elt] = \
            helper.gamma_to_beta(ene['gamma_array'][r_idx_elt])

        self.z['abs_array'][r_idx_elt] = \
            self.z['abs_array'][idx_elt_prec] + elt.pos_m['rel'][1:]

        self.phi['abs_array'][r_idx_elt] = phi_abs


def convert_phi_0_p(phi_in, phi_rf_abs, abs_to_rel):
    """Calculate the missing phi_0 (relative or absolute)."""
    if abs_to_rel:
        phi_out = np.mod(phi_in + phi_rf_abs, 2. * np.pi)
    else:
        phi_out = np.mod(phi_in - phi_rf_abs, 2. * np.pi)
    return phi_out



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
