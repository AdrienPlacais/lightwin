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

        self.fail = []
        self.comp = []

        self.solver = {
            'min_ke': 1.6,
            'max_ke': 2.5,
            'min_phi0': np.deg2rad(110.),
            'max_phi0': np.deg2rad(170.),
            }

    def break_at(self, fail_idx):
        """Break cavities at indices fail_idx."""
        for idx in fail_idx:
            cav = self.brok_lin.list_of_elements[idx]
            assert cav.name == 'FIELD_MAP', 'Error, the element at ' + \
                'position ' + str(idx) + ' is not a FIELD_MAP.'
            cav.fail()
            self.fail.append(cav)

    def fix(self, strategy, objective, manual_list=None):
        """Fix cavities."""
        # Select which cavities will be used to compensate the fault
        self._select_compensating(strategy, manual_list)
        for cav in self.comp:
            cav.status['compensate'] = True

        # Output cavities info
        for linac in [self.ref_lin, self.brok_lin]:
            debug.output_cavities(linac)

        self.objective = objective
        method = 'RK'
        debug_plot = False

        if debug_plot:
            fig = plt.figure(42)
            ax = fig.add_subplot(111)
            ax.grid(True)
            xxx = np.linspace(0, 1631, 1632)

        # Set the fit variables
        n_cav = len(self.comp)
        x0 = np.full((2 * n_cav), np.NaN)
        bounds = np.full((2 * n_cav, 2), np.NaN)

        for i in range(n_cav):
            x0[2*i] = self.comp[i].acc_field.norm
            x0[2*i+1] = self.comp[i].acc_field.phi_0
            bounds[2*i, :] = np.array([self.solver['min_ke'],
                                       self.solver['max_ke']])
            bounds[2*i+1, :] = np.array([self.solver['min_phi0'],
                                         self.solver['max_phi0']])
        # @TODO: not elegant

        # Portion of linac with compensating cavities, as well as drifts and
        # quads
        complete_modules = []
        elts = self.brok_lin.list_of_elements
        for i in range(elts.index(self.comp[0]), elts.index(self.comp[-1])+1):
            complete_modules.append(elts[i])

        def wrapper(x):
            # Unpack
            for i in range(n_cav):
                self.comp[i].acc_field.norm = x[2*i]
                self.comp[i].acc_field.phi_0 = x[2*i+1]

            # Update transfer matrices
            self.brok_lin.compute_transfer_matrices(method, complete_modules)
            if debug_plot:
                yyy = self.brok_lin.synch.energy['kin_array_mev']
                ax.plot(xxx, yyy)
                plt.show()
            return self._qty_to_fit()
        sol = minimize(wrapper, x0=x0, bounds=bounds)

        for i in range(n_cav):
            self.comp[i].acc_field.norm = sol.x[2*i]
            self.comp[i].acc_field.phi_0 = sol.x[2*i+1]
        print(sol)

        # When fit is complete, also recompute last elements
        self.brok_lin.compute_transfer_matrices(method)

        if sol.success:
            self.brok_lin.name = 'Fixed'
        else:
            self.brok_lin.name = 'Poorly fixed'
        debug.output_cavities(self.brok_lin)

    def _select_compensating(self, strategy, manual_list):
        """Select the cavities that will be used for compensation."""
        # self.cavities = list(filter(lambda elt: elt.name == 'FIELD_MAP',
        #                             self.brok_lin.list_of_elements))

        # self.working = list(filter(lambda cav: not cav.status['failed'],
        #                            self.cavities))
        self.comp = [self.brok_lin.list_of_elements[idx] for idx in
                     manual_list]

    def _qty_to_fit(self):
        """
        Return the quantity to minimize for the fit.

        It computes the absolute difference of the quantity given by
        self.objective between the reference linac and the broken one.
        """
        idx = self.comp[-1].idx['out'] - 1

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
