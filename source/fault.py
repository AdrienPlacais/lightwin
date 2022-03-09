#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:51:15 2022

@author: placais

Module holding all the fault-related functions.
brok_lin: holds for "broken_linac", the linac with faults.
ref_lin: holds for "reference_linac", the ideal linac brok_lin should tend to.
"""
import numpy as np
from scipy.optimize import minimize, least_squares
import debug


dict_phase = {
    True: lambda elt: elt.acc_field.phi_0['abs'],
    False: lambda elt: elt.acc_field.phi_0['rel']
    }


class FaultScenario():
    """A class to hold all fault related data."""

    def __init__(self, ref_linac, broken_linac):
        self.ref_lin = ref_linac
        self.brok_lin = broken_linac

        assert ref_linac.synch.info['reference'] is True
        assert broken_linac.synch.info['reference'] is False

        self.fail_list = []
        self.comp_list = {
            'only_cav': [],
            'all_elts': []}

        self.what_to_fit = {
            'strategy': None,   # How are selected the compensating cavities?
            'objective': None,  # What do we want to fit?
            'position': None,   # Where are we measuring 'objective'?
            }

        self.info = {}

        if not broken_linac.synch.info['abs_phases']:
            print('Warning, the phases in the broken linac are relative.',
                  'It may be more relatable to use absolute phases, as',
                  'it would avoid the implicit rephasing of the linac at',
                  'each cavity.\n')

    def break_at(self, fail_idx):
        """
        Break cavities at indices fail_idx.

        All faulty cavities are added to fail_list.
        """
        for idx in fail_idx:
            cav = self.brok_lin.elements['list'][idx]
            assert cav.info['name'] == 'FIELD_MAP', 'Error, the element at ' \
                + 'position ' + str(idx) + ' is not a FIELD_MAP.'
            cav.fail()
            self.fail_list.append(cav)

    def transfer_phi0_from_ref_to_broken(self):
        """
        Transfer the entry phases from ref linac to broken.

        If the absolute initial phases are not kept between reference and
        broken linac, it comes down to rephasing the linac. This is what we
        want to avoid when FLAG_PHI_ABS = True.
        """
        ref_cavities = self.ref_lin.elements_of('FIELD_MAP')
        brok_cavities = self.brok_lin.elements_of('FIELD_MAP')
        assert len(ref_cavities) == len(brok_cavities)

        # Transfer both relative and absolute phase flags
        for i, ref_cav in enumerate(ref_cavities):
            ref_acc_f = ref_cav.acc_field
            brok_acc_f = brok_cavities[i].acc_field

            brok_acc_f.phi_0['rel'] = ref_acc_f.phi_0['rel']
            brok_acc_f.phi_0['abs'] = ref_acc_f.phi_0['abs']

    def _select_modules_with_failed_cav(self):
        """Look for modules with at least one failed cavity."""
        modules = self.brok_lin.elements['list_lattice']
        modules_with_fail = []
        for module in modules:
            cav_module = self.brok_lin.elements_of(nature='FIELD_MAP',
                                                   sub_list=module)
            for cav in cav_module:
                if cav.info['failed']:
                    modules_with_fail.append(module)
                    break
        print('There are', len(modules_with_fail), 'module(s) with at',
              'least one failed cavity.')
        return modules_with_fail

    def _select_comp_modules(self, modules_with_fail):
        """Give failed modules and their neighbors."""
        modules = self.brok_lin.elements['list_lattice']
        neighbor_modules = []
        for module in modules_with_fail:
            idx = modules.index(module)
            if idx > 0:
                neighbor_modules.append(modules[idx-1])
            if idx < len(modules) - 1:
                neighbor_modules.append(modules[idx+1])
        # We return all modules that could help to compensation, ie neighbors
        # as well as faulty modules
        return neighbor_modules + modules_with_fail

    def _select_comp_cav(self, comp_modules):
        """Give cavities in comp_modules that still work."""
        comp_cav = []
        for module in comp_modules:
            # All cavities in comp_module
            cavities_modules = self.brok_lin.elements_of(nature='FIELD_MAP',
                                                         sub_list=module)
            # Only working cavities in comp_module
            for cav in cavities_modules:
                if not cav.info['failed']:
                    comp_cav.append(cav)
        # Sort them in the proper order
        comp_cav = sorted(comp_cav, key=lambda elt: elt.idx['in'])
        return comp_cav

    def _select_compensating_cavities(self, what_to_fit, manual_list):
        """
        Select the cavities that will be used for compensation.

        All compensating cavities are added to comp_list['only_cav'], and their
        info['compensate'] flag is switched to True.
        comp_list['all_elts'] also contains drifts, quads, etc of the
        compensating modules.
        """
        if what_to_fit['strategy'] == 'manual':
            self.comp_list['only_cav'] = [self.brok_lin.elements['list'][idx]
                                          for idx in manual_list]

        elif self.what_to_fit['strategy'] == 'neighbors':
            modules_with_fail = self._select_modules_with_failed_cav()
            comp_modules = self._select_comp_modules(modules_with_fail)
            self.comp_list['only_cav'] = self._select_comp_cav(comp_modules)

        # Change info of all the compensating cavities
        for cav in self.comp_list['only_cav']:
            cav.info['compensate'] = True

        # Portion of linac with compensating cavities, as well as drifts and
        # quads
        complete_modules = []
        elts = self.brok_lin.elements['list']
        for i in range(elts.index(self.comp_list['only_cav'][0]),
                       elts.index(self.comp_list['only_cav'][-1])+1):
            complete_modules.append(elts[i])
        self.comp_list['all_elts'] = complete_modules

    def fix(self, method, what_to_fit, manual_list=None):
        """
        Try to compensate the faulty cavities.

        Parameters
        ----------
        method : str
            Tells which algorithm should be used to compute transfer matrices.
        what_to_fit : dict
            Holds the strategies of optimisation.
        manual_list : list, optional
            List of the indices of the cavities that compensate the fault when
            'strategy' == 'manual'. The default is None.
        """
        debug_fit = True
        debug_cav = False
        self.what_to_fit = what_to_fit
        print("Starting fit with parameters:", what_to_fit)

        self._select_compensating_cavities(what_to_fit, manual_list)
        n_cav = len(self.comp_list['only_cav'])
        for linac in [self.ref_lin, self.brok_lin]:
            self.info[linac.name + ' cav'] = debug.output_cavities(linac,
                                                                   debug_cav)

        # Set the fit variables
        initial_guesses, bounds = self._set_fit_parameters()
        fun_objective, idx_objective = self._select_objective(
            self.what_to_fit['position'],
            self.what_to_fit['objective'])

        # TODO: set constraints on the synch phase
        def wrapper(prop_array):
            # Unpack
            for i in range(n_cav):
                cav = self.comp_list['only_cav'][i]
                acc_f = cav.acc_field
                # FIXME
                # if acc_f.absolute_phase_flag:
                if self.brok_lin.synch.info['abs_phases']:
                    acc_f.phi_0['abs'] = prop_array[i]

                else:
                    acc_f.phi_0['rel'] = prop_array[i]

                acc_f.norm = prop_array[i+n_cav]

            # Update transfer matrices
            self.brok_lin.compute_transfer_matrices(method,
                                                    self.comp_list['all_elts'])

            obj = np.abs(fun_objective(self.ref_lin, idx_objective)
                         - fun_objective(self.brok_lin, idx_objective))
            return obj

        dict_fitter = {
            'energy': [minimize, initial_guesses, bounds],
            'phase': [minimize, initial_guesses, bounds],
            'energy_phase': [least_squares, initial_guesses,
                             (bounds[:, 0], bounds[:, 1])],
            'transfer_matrix': [least_squares, initial_guesses,
                                (bounds[:, 0], bounds[:, 1])],
            'all': [least_squares, initial_guesses,
                    (bounds[:, 0], bounds[:, 1])],
            }  # minimize and least_squares do not take the same bounds format
        fitter = dict_fitter[what_to_fit['objective']]
        sol = fitter[0](wrapper, x0=fitter[1], bounds=fitter[2])
        # TODO check methods
        # TODO check Jacobian
        # TODO check x_scale

        for i in range(n_cav):
            cav = self.comp_list['only_cav'][i]
            acc_f = cav.acc_field
            # FIXME
            if self.brok_lin.synch.info['abs_phases']:
                acc_f.phi_0['abs'] = sol.x[i]
            else:
                acc_f.phi_0['rel'] = sol.x[i]

            acc_f.norm = sol.x[i+n_cav]

        # When fit is complete, also recompute last elements
        self.brok_lin.synch.info['fixed'] = True
        if sol.success:
            self.brok_lin.name = 'Fixed'
        else:
            self.brok_lin.name = 'Poorly fixed'

        self.brok_lin.compute_transfer_matrices(method)
        self.info[self.brok_lin.name + ' cav'] = \
            debug.output_cavities(self.brok_lin, debug_cav)

        print('\n', sol)
        self.info['sol'] = sol
        self.info['fit'] = debug.output_fit(self, initial_guesses, bounds,
                                            debug_fit)

    def _set_fit_parameters(self):
        """
        Set initial conditions and boundaries for the fit.

        In the returned arrays, first half of components are initial phases
        phi_0, while second half of components are norms.

        Returns
        -------
        initial_guess: np.array
            Initial guess for the initial phase and norm of the compensating
            cavities.
        bounds: np.array of bounds
            Array of (min, max) bounds for the electric fields of the
            compensating cavities.
        """
        initial_guess = []
        bounds = []
        abs_flag = self.ref_lin.synch.info['abs_phases']

        # Handle phase
        limits_phase = (0., 2.*np.pi)
        for elt in self.comp_list['only_cav']:
            initial_guess.append(dict_phase[abs_flag](elt))
            bounds.append(limits_phase)

        # Handle norm
        limits_norm = {
            'relative': [0.6, 1.5],    # [60%, 150%] of norm
            'absolute': [1., np.inf]   # ridiculous limits for the norm
            }
        for elt in self.comp_list['only_cav']:
            norm = elt.acc_field.norm
            initial_guess.append(norm)
            down = max(limits_norm['relative'][0] * norm,
                       limits_norm['absolute'][0])
            upp = min(limits_norm['relative'][1] * norm,
                      limits_norm['absolute'][1])
            bounds.append((down, upp))

        initial_guess = np.array(initial_guess)
        bounds = np.array(bounds)
        return initial_guess, bounds

    def _select_objective(self, position_str, objective_str):
        """Select the objective to fit."""
        # Where do you want to verify that the objective is matched?
        dict_position = {
            'end_of_last_comp_cav':
                self.comp_list['only_cav'][-1].idx['out'] - 1,
            'one_module_after_last_comp_cav': np.NaN,   # TODO
            }

        # What do you want to match?
        dict_objective = {
            'energy': lambda linac, idx:
                linac.synch.energy['kin_array_mev'][idx],
            'phase': lambda linac, idx:
                linac.synch.phi['abs_array'][idx],
            'energy_phase': lambda linac, idx: np.array(
                [linac.synch.energy['kin_array_mev'][idx],
                 linac.synch.phi['abs_array'][idx]]),
            'transfer_matrix': lambda linac, idx:
                linac.transf_mat['cumul'][idx, :, :].flatten(),
            'all': lambda linac, idx:
                np.hstack((
                    np.array(
                        [linac.synch.energy['kin_array_mev'][idx],
                         linac.synch.phi['abs_array'][idx]]),
                    linac.transf_mat['cumul'][idx, :, :].flatten()
                    ))
            }

        idx_pos = dict_position[position_str]
        fun_objective = dict_objective[objective_str]
        return fun_objective, idx_pos
