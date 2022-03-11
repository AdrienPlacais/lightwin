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
from constants import FLAG_PHI_ABS, STR_PHI_ABS
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

        if not FLAG_PHI_ABS:
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
            cav.update_status('failed')
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
            for str_phi_abs in ['rel', 'abs']:
                brok_acc_f.phi_0[str_phi_abs] = ref_acc_f.phi_0[str_phi_abs]

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

    def _select_compensating_cavities(self, what_to_fit, manual_list):
        """
        Select the cavities that will be used for compensation.

        All compensating cavities are added to comp_list['only_cav'], and their
        info['compensate'] flag is switched to True.
        comp_list['all_elts'] also contains drifts, quads, etc of the
        compensating modules.
        """
        self.what_to_fit = what_to_fit
        if self.what_to_fit['strategy'] == 'manual':
            self.comp_list['only_cav'] = [self.brok_lin.elements['list'][idx]
                                          for idx in manual_list]

        elif self.what_to_fit['strategy'] == 'neighbors':
            modules_with_fail = [
                module
                for module in self.brok_lin.elements['list_lattice']
                for elt in module
                if elt._info['status'] == 'failed'
                ]
            # TODO: replace this with a coomprehension list
            comp_modules = self._select_comp_modules(modules_with_fail)

            self.comp_list['only_cav'] = [
                     cav
                     for module in comp_modules
                     for cav in module
                     if cav._info['name'] == 'FIELD_MAP'
                     and cav._info['status'] != 'failed'
                     ]

        self.comp_list['only_cav'] = sorted(self.comp_list['only_cav'],
                                            key=lambda elt: elt.idx['in'])

        # Change info of all the compensating cavities
        for cav in self.comp_list['only_cav']:
            cav.update_status('compensate')

        # We take everything between first and last compensating cavities
        self.comp_list['all_elts'] = []
        elts = self.brok_lin.elements['list']
        for i in range(elts.index(self.comp_list['only_cav'][0]),
                       elts.index(self.comp_list['only_cav'][-1])+1):
            self.comp_list['all_elts'].append(elts[i])
        # TODO : better with a comprehension list?

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
        debugs = {
            'fit': True,
            'cav': False,
            }
        self.what_to_fit = what_to_fit
        print("Starting fit with parameters:", self.what_to_fit)

        self._select_compensating_cavities(self.what_to_fit, manual_list)

        for linac in [self.ref_lin, self.brok_lin]:
            self.info[linac.name + ' cav'] = \
                debug.output_cavities(linac, debugs['cav'])

        # Set the fit variables
        initial_guesses, bounds = self._set_fit_parameters()
        fun_objective, idx_objective = self._select_objective(
            self.what_to_fit['position'],
            self.what_to_fit['objective'])

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
        fitter = dict_fitter[self.what_to_fit['objective']]
        sol = fitter[0](wrapper, x0=fitter[1], bounds=fitter[2],
                        args=(self, method, fun_objective, idx_objective))
        # TODO check methods
        # TODO check Jacobian
        # TODO check x_scale

        for i, cav in enumerate(self.comp_list['only_cav']):
            cav.acc_field.phi_0[STR_PHI_ABS] = sol.x[i]
            cav.acc_field.norm = sol.x[i + len(self.comp_list['only_cav'])]

        # When fit is complete, also recompute last elements
        self.brok_lin.synch.info['status'] = 'fixed'
        if sol.success:
            self.brok_lin.name = 'Fixed'
        else:
            self.brok_lin.name = 'Poorly fixed'

        self.brok_lin.compute_transfer_matrices(method)
        self.info[self.brok_lin.name + ' cav'] = \
            debug.output_cavities(self.brok_lin, debugs['cav'])

        print('\n', sol)
        self.info['sol'] = sol
        self.info['fit'] = debug.output_fit(self, initial_guesses, bounds,
                                            debugs['fit'])

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

        # Handle phase
        limits_phase = (0., 2.*np.pi)
        for elt in self.comp_list['only_cav']:
            initial_guess.append(dict_phase[FLAG_PHI_ABS](elt))
            bounds.append(limits_phase)

        # Handle norm
        limits_norm = {
            'relative': [0.9, 1.3],    # [90%, 130%] of norm
            'absolute': [1., np.inf]   # ridiculous abs limits
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

        def fun_multi_objective(linac, idx_list):
            fun_simple = dict_objective[objective_str]
            obj = fun_simple(linac, idx_list[0])
            for idx in idx_list[1:]:
                obj = np.hstack((obj, fun_simple(linac, idx)))
            return obj

        idx_pos_list = [idx_pos]
        for idx in idx_pos_list:
            elt = self.brok_lin.where_is_this_index(idx)
            print('\nWe try to match at synch index:', idx, 'which is',
                  elt._info, ', the', self.brok_lin.where_is(elt, nature=True),
                  'th of his kind.\n')
        return fun_multi_objective, idx_pos_list


# TODO: set constraints on the synch phase
def wrapper(prop_array, fault_sce, method, fun_objective, idx_objective):
    """Fit function."""
    # Unpack
    for i, cav in enumerate(fault_sce.comp_list['only_cav']):
        acc_f = cav.acc_field
        acc_f.phi_0[STR_PHI_ABS] = prop_array[i]

        acc_f.norm = prop_array[i+len(fault_sce.comp_list['only_cav'])]

    # Update transfer matrices
    fault_sce.brok_lin.compute_transfer_matrices(
        method, fault_sce.comp_list['all_elts'])

    obj = np.abs(fun_objective(fault_sce.ref_lin, idx_objective)
                 - fun_objective(fault_sce.brok_lin, idx_objective))
    return obj
