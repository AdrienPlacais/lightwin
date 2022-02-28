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


class FaultScenario():
    """A class to hold all fault related data."""

    def __init__(self, ref_linac, broken_linac):
        self.ref_lin = ref_linac
        self.brok_lin = broken_linac

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

    def break_at(self, fail_idx):
        """
        Break cavities at indices fail_idx.

        All faulty cavities are added to fail_list.
        """
        for idx in fail_idx:
            cav = self.brok_lin.list_of_elements[idx]
            assert cav.name == 'FIELD_MAP', 'Error, the element at ' + \
                'position ' + str(idx) + ' is not a FIELD_MAP.'
            cav.fail()
            self.fail_list.append(cav)

    def _select_compensating_cavities(self, what_to_fit, manual_list):
        """
        Select the cavities that will be used for compensation.

        All compensating cavities are added to comp_list['only_cav'], and their
        status['compensate'] flag is switched to True.
        comp_list['all_elts'] also contains drifts, quads, etc of the
        compensating modules.
        """
        # FIXME
        # self.cavities = list(filter(lambda elt: elt.name == 'FIELD_MAP',
        #                             self.brok_lin.list_of_elements))

        # self.working = list(filter(lambda cav: not cav.status['failed'],
        #                            self.cavities))
        self.comp_list['only_cav'] = [self.brok_lin.list_of_elements[idx] for
                                      idx in manual_list]

        for cav in self.comp_list['only_cav']:
            cav.status['compensate'] = True

        # Portion of linac with compensating cavities, as well as drifts and
        # quads
        complete_modules = []
        elts = self.brok_lin.list_of_elements
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
        self.what_to_fit = what_to_fit
        print("Starting fit with parameters:", what_to_fit)

        self._select_compensating_cavities(what_to_fit, manual_list)
        n_cav = len(self.comp_list['only_cav'])
        for linac in [self.ref_lin, self.brok_lin]:
            self.info[linac.name + ' cav'] = debug.output_cavities(linac,
                                                                   debug_fit)

        # Set the fit variables
        initial_guesses, bounds = self._set_fit_parameters()
        fun_objective, idx_objective = self._select_objective(
            self.what_to_fit['position'],
            self.what_to_fit['objective'])

        # TODO: set constraints on the synch phase
        def wrapper(prop_array):
            # Unpack
            for i in range(n_cav):
                acc_f = self.comp_list['only_cav'][i].acc_field
                acc_f.phi_0 = prop_array[i]
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
            }  # minimize and least_squares do not take the same bounds format
        fitter = dict_fitter[what_to_fit['objective']]

        sol = fitter[0](wrapper, x0=fitter[1], bounds=fitter[2], verbose=2)
        # TODO check if some boundaries are reached
        # TODO check methods
        # TODO check Jacobian
        # TODO check x_scale

        for i in range(n_cav):
            self.comp_list['only_cav'][i].acc_field.phi_0 = sol.x[i]
            self.comp_list['only_cav'][i].acc_field.norm = sol.x[i+n_cav]

        # When fit is complete, also recompute last elements
        self.brok_lin.compute_transfer_matrices(method)
        if sol.success:
            self.brok_lin.name = 'Fixed'
        else:
            self.brok_lin.name = 'Poorly fixed'

        self.info[self.brok_lin.name + ' cav'] = \
            debug.output_cavities(self.brok_lin, debug_fit)
        print(sol)
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

        relative_limits = {
            'phi_0_down': 0.6, 'phi_0_up': 1.5,  # [60%, 150%] of phi_0
            'norm_down': 0.6, 'norm_up': 1.5,   # [60%, 150%] of norm
            }
        absolute_limits = {
            'phi_0_down': np.deg2rad(100.), 'phi_0_up': np.deg2rad(170.),
            'norm_down': 1., 'norm_up': 2.1,
            }
        dict_prop = {
            'phi_0': lambda acc_f: acc_f.phi_0,
            'norm': lambda acc_f: acc_f.norm
            }

        def _set_boundaries(prop_str, prop):
            """Take the most constraining boundaries between abs and rel."""
            down_lim = max((relative_limits[prop_str + '_down'] * prop,
                            absolute_limits[prop_str + '_down']))
            up_lim = min((relative_limits[prop_str + '_up'] * prop,
                          absolute_limits[prop_str + '_up']))
            return (down_lim, up_lim)

        for prop_str in ['phi_0', 'norm']:
            for elt in self.comp_list['only_cav']:
                prop = dict_prop[prop_str](elt.acc_field)
                initial_guess.append(prop)
                bnds = _set_boundaries(prop_str, prop)
                bounds.append(bnds)

        initial_guess = np.array(initial_guess)
        bounds = np.array(bounds)
        return initial_guess, bounds

    def _set_constraints(self):
        """Add constraints on the synchronous phase."""
        # TODO Finish this function
        limit_phi_s_down = -90.

        def phi_s_min(cav):
            return cav.acc_field.phi_s_deg - limit_phi_s_down

        percent_phi_s_max = 1.1

        def phi_s_max(cav):
            idx = self.brok_lin.where_is(cav)
            equiv = self.ref_lin.list_of_elements[idx]
            maxi = percent_phi_s_max * equiv.acc_field.phi_s_deg
            return maxi - cav.acc_field.phi_s_deg

        constraints = []
        for cav in self.comp_list['only_cav']:
            constraints.append({'type': 'ineq', 'fun': phi_s_min(cav)})
            constraints.append({'type': 'ineq', 'fun': phi_s_max(cav)})
        return constraints

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
            }

        idx_pos = dict_position[position_str]
        fun_objective = dict_objective[objective_str]
        return fun_objective, idx_pos
