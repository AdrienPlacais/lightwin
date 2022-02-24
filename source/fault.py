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
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import debug


class FaultScenario():
    """A class to hold all fault related data."""

    def __init__(self, ref_linac, broken_linac):
        self.ref_lin = ref_linac
        self.brok_lin = broken_linac

        self.fail_list = []
        self.comp_list = {
            'cav': [],
            'all_elts': []}

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

    def _select_compensating_cavities(self, strategy, manual_list):
        """
        Select the cavities that will be used for compensation.

        All compensating cavities are added to comp_list['cav'], and their
        status['compensate'] flag is switched to True.
        comp_list['all_elts'] also contains drifts, quads, etc of the
        compensating modules.
        """
        # FIXME
        # self.cavities = list(filter(lambda elt: elt.name == 'FIELD_MAP',
        #                             self.brok_lin.list_of_elements))

        # self.working = list(filter(lambda cav: not cav.status['failed'],
        #                            self.cavities))
        self.comp_list['cav'] = [self.brok_lin.list_of_elements[idx] for idx in
                                 manual_list]

        for cav in self.comp_list['cav']:
            cav.status['compensate'] = True

        # Portion of linac with compensating cavities, as well as drifts and
        # quads
        complete_modules = []
        elts = self.brok_lin.list_of_elements
        for i in range(elts.index(self.comp_list['cav'][0]),
                       elts.index(self.comp_list['cav'][-1])+1):
            complete_modules.append(elts[i])
        self.comp_list['all_elts'] = complete_modules

    def fix(self, strategy, objective, manual_list=None):
        """Fix cavities."""
        self.objective = objective
        method = 'RK'   # FIXME
        debug_plot = False
        debug_cav_info = False

        self._select_compensating_cavities(strategy, manual_list)
        n_cav = len(self.comp_list)
        if debug_cav_info:
            for linac in [self.ref_lin, self.brok_lin]:
                debug.output_cavities(linac)

        if debug_plot:
            fig = plt.figure(42)
            ax = fig.add_subplot(111)
            ax.grid(True)
            idx_max = self.brok_lin.list_of_elements[-1].idx['out']
            xxx = np.linspace(0, idx_max, idx_max + 1)

        # Set the fit variables
        initial_guesses, bounds = self._set_fit_parameters()

        # TODO: set constraints on the synch phase

        if(False):
            def wrapper(x):
                # Unpack
                for i in range(n_cav):
                    self.comp_list[i].acc_field.norm = x[2*i]
                    self.comp_list[i].acc_field.phi_0 = x[2*i+1]

                # Update transfer matrices
                self.brok_lin.compute_transfer_matrices(method, self.comp_list['all_elts'])
                if debug_plot:
                    yyy = self.brok_lin.synch.energy['kin_array_mev']
                    ax.plot(xxx, yyy)
                    plt.show()
                return self._qty_to_fit()
            sol = minimize(wrapper, x0=initial_guesses, bounds=bounds)

            for i in range(n_cav):
                self.comp_list[i].acc_field.norm = sol.x[2*i]
                self.comp_list[i].acc_field.phi_0 = sol.x[2*i+1]
            print(sol)

            # When fit is complete, also recompute last elements
            self.brok_lin.compute_transfer_matrices(method)

            if sol.success:
                self.brok_lin.name = 'Fixed'
            else:
                self.brok_lin.name = 'Poorly fixed'

            if debug_cav_info:
                debug.output_cavities(self.brok_lin)
        self.brok_lin.name = 'tmp'

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
            'norm_down': 0.6, 'norm_up': 1.5,   # [60%, 150%] of norm
            'phi_0_down': 0.6, 'phi_0_up': 1.5,  # [60%, 150%] of phi_0
            }
        absolute_limits = {
            'norm_down': 1., 'norm_up': 2.1,
            'phi_0_down': np.deg2rad(110.), 'phi_0_up': np.deg2rad(170.),
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
            for elt in self.comp_list['cav']:
                prop = dict_prop[prop_str](elt.acc_field)
                initial_guess.append(prop)
                bnds = _set_boundaries(prop_str, prop)
                bounds.append(bnds)

        initial_guess = np.array(initial_guess)
        bounds = np.array(bounds)
        return initial_guess, bounds

    def _qty_to_fit(self):
        """
        Return the quantity to minimize for the fit.

        It computes the absolute difference of the quantity given by
        self.objective between the reference linac and the broken one.
        """
        idx = self.comp_list[-1].idx['out'] - 1

        def res(linac):
            dict_objective = {
                'energy': linac.synch.energy['kin_array_mev'][idx],
                'phase': linac.synch.phi['abs_array'][idx],
                'energy_phase': np.array(
                    [linac.synch.energy['kin_array_mev'][idx],
                     linac.synch.phi['abs_array'][idx]]),
                'transfer_matrix': linac.transf_mat['cumul'][idx, :, :],
                }
            return dict_objective[self.objective]
        return np.abs(res(self.ref_lin) - res(self.brok_lin))
