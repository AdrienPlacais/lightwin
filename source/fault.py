#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:51:15 2022.

@author: placais

Module holding all the individual Fault functions.
If several close cavities fail, they are regrouped in the same Fault object and
are fixed together.

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

n_comp_latt_per_fault = 2
debugs = {
    'fit_complete': False,
    'fit_compact': True,
    'fit_progression': True,
    'cav': True,
    'verbose': 0,
    }


class Fault():
    """A class to hold one or several close Faults."""

    def __init__(self, ref_lin, brok_lin, fail_idx):
        self.ref_lin = ref_lin
        self.brok_lin = brok_lin
        self.fail = {'l_cav': [], 'l_idx': fail_idx}
        self.comp = {'l_cav': [], 'l_all_elts': None, 'l_recompute': None}
        self.info = {'sol': None, 'initial_guesses': None, 'bounds': None,
                     'jac': None}

    def select_neighboring_cavities(self):
        """
        Select the cavities neighboring the failed one(s).

        More precisely:
        Select the lattices with comp cav, extract cavities from it.

        As for now, n_comp_latt_per_fault is the number of compensating
        lattices per faulty cavity. This number is however too high for
        MYRRHA's high beta section.

        # TODO: get this function out of the Class?
        Would be better for consistency w/ manual list
        Required arguments:
            l_lattices from brok_lin.elements['l_sections'] list of lattices
            list of elements of brok_lin
            index in lattice reference
            self.fail['l_idx'] indexes of failed cavities

        Return
        ------
        l_comp_cav : list
            List of the cavities (_Element object) used for compensation.
        """
        comp_lattices_idx = []
        l_lattices = [lattice
                      for section in self.brok_lin.elements['l_sections']
                      for lattice in section
                      ]
        # Get lattices neighboring each faulty cavity
        # FIXME: too many lattices for faults in Section 3
        for idx in self.fail['l_idx']:
            failed_cav = self.brok_lin.elements['list'][idx]
            idx_lattice = failed_cav.idx['lattice'][0]
            for shift in [-1, +1]:
                idx = idx_lattice + shift
                while ((idx in comp_lattices_idx)
                       and (idx in range(0, len(l_lattices)))):
                    idx += shift
                # FIXME: dirty hack
                if abs(idx - idx_lattice) < 3:
                    comp_lattices_idx.append(idx)

                # Also add the lattice with the fault
                if idx_lattice not in comp_lattices_idx:
                    comp_lattices_idx.append(idx_lattice)

        comp_lattices_idx.sort()

        # List of compensating (+ broken) cavitites
        l_comp_cav = [cav
                      for idx in comp_lattices_idx
                      for cav in l_lattices[idx]
                      if cav.info['nature'] == 'FIELD_MAP'
                      ]
        return l_comp_cav

    def prepare_cavities(self, l_comp_cav):
        """
        Prepare the optimisation process.

        In particular, give status 'compensate' and 'broken' to proper
        cavities. Define the full lattices incorporating the compensating and
        faulty cavities.
        """
        # Break proper cavities
        for idx in self.fail['l_idx']:
            cav = self.brok_lin.elements['list'][idx]
            cav.update_status('failed')
            if cav in l_comp_cav:
                l_comp_cav.remove(cav)
            self.fail['l_cav'].append(cav)

        # Assign compensating cavities
        for cav in l_comp_cav:
            if cav.info['status'] != 'nominal':
                print('warning check fault.update_status_cavities: ',
                      'several faults want the same compensating cavity!')
            cav.update_status('compensate')
            self.comp['l_cav'].append(cav)

        # List of all elements of the compensating zone
        l_lattices = [lattice
                      for section in self.brok_lin.elements['l_sections']
                      for lattice in section
                      ]
        self.comp['l_all_elts'] = [elt
                                   for lattice in l_lattices
                                   for elt in lattice
                                   if any([cav in lattice
                                           for cav in self.comp['l_cav']])
                                   ]

    def _select_cavities_to_rephase(self):
        """
        Change the status of some cavities to 'rephased'.

        If the calculation is in relative phase, all cavities that are after
        the the first failed one are rephased.
        Even in the case of an absolute phase calculation, cavities in the
        HEBT are rephased.
        """
        # We get first failed cav index
        ffc_idx = min([
            fail_cav.idx['elements']
            for fail_cav in self.fail_list
            ])
        after_ffc = self.brok_lin.elements['list'][ffc_idx:]

        cav_to_rephase = [cav
                          for cav in after_ffc
                          if (cav.info['nature'] == 'FIELD_MAP'
                              and cav.info['status'] == 'nominal')
                          and (cav.info['zone'] == 'HEBT'
                               or not FLAG_PHI_ABS)
                          ]
        for cav in cav_to_rephase:
            cav.update_status('rephased')

    def _select_comp_modules(self, modules_with_fail):
        """Give failed modules and their neighbors."""
        modules = self.brok_lin.elements['l_lattices']
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

    def fix_single(self, what_to_fit):
        """
        Try to compensate the faulty cavities.

        Parameters
        ----------
        what_to_fit : dict
            Holds the strategies of optimisation.
        manual_list : list, optional
            List of the indices of the cavities that compensate the fault when
            'strategy' == 'manual'. The default is None.
        """
        self.what_to_fit = what_to_fit
        print("Starting fit with parameters:", self.what_to_fit)

        # Set the fit variables
        initial_guesses, bounds, x_scales = \
            self._set_fit_parameters(what_to_fit['fit_over_phi_s'])
        self.info['initial_guesses'] = initial_guesses
        self.info['bounds'] = bounds

        fun_objective, idx_objective, l_elements = self._select_objective(
            self.what_to_fit['position'],
            self.what_to_fit['objective'])
        self.comp['l_recompute'] = l_elements

        dict_fitter = {
            'energy':
                [minimize, initial_guesses, bounds],
            'phase':
                [minimize, initial_guesses, bounds],
            'energy_phase':
                [least_squares, initial_guesses, (bounds[:, 0], bounds[:, 1])],
            'transfer_matrix':
                [least_squares, initial_guesses, (bounds[:, 0], bounds[:, 1])],
            'all':
                [least_squares, initial_guesses, (bounds[:, 0], bounds[:, 1])],
            }  # minimize and least_squares do not take the same bounds format
        fitter = dict_fitter[self.what_to_fit['objective']]

        global count
        count = 0
        sol = fitter[0](fun=wrapper, x0=fitter[1], bounds=fitter[2],
                        args=(self, fun_objective, idx_objective,
                              what_to_fit),
                        jac='2-point',  # Default
                        # 'trf' not ideal as jac is not sparse.
                        # 'dogbox' may have difficulties with rank-defficient
                        # jac.
                        method='dogbox',
                        ftol=1e-8, gtol=1e-8,   # Default
                        xtol=1e-8,      # Solver is sometimes 'lazy' and ends
                        # with xtol termination condition, while settings are
                        # clearly not optimized
                        # x_scale='jac',    # TODO
                        # loss=linear,      # TODO
                        # f_scale=1.0,      # TODO
                        # diff_step=None,   # TODO
                        # tr_solver=None, tr_options={},   # TODO
                        # jac_sparsity=None,    # TODO
                        verbose=debugs['verbose'],
                        )

        if debugs['fit_progression']:
            debug.output_fit_progress(count, sol.fun, final=True)

        print('\nmessage:', sol.message, '\nnfev:', sol.nfev, '\tnjev:',
              sol.njev, '\noptimality:', sol.optimality, '\nstatus:',
              sol.status, '\tsuccess:', sol.success, '\nx:', sol.x, '\n\n')
        self.info['sol'] = sol
        self.info['jac'] = sol.jac

        return sol

    def _set_fit_parameters(self, flag_synch=False):
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
        x_scales = []

        typical_phase_var = np.deg2rad(10.)
        typical_norm_var = .1

        # Handle phase
        if flag_synch:
            limits_phase = (-np.pi/2., 0.)
            rel_limit_phase_up = .4    # +40% over nominal synch phase
        else:
            # limits_phase = (-np.inf, np.inf)
            # These bounds seems more logical but disturb the optimisation
            limits_phase = (0., 8.*np.pi)

        for elt in self.comp['l_cav']:
            if flag_synch:
                equiv_cav = self.ref_lin.elements['list'][elt.idx['element']]
                equiv_phi_s = equiv_cav.acc_field.cav_params['phi_s_rad']
                initial_guess.append(equiv_phi_s)
                lim_down = limits_phase[0]
                lim_up = min(limits_phase[1],
                             equiv_phi_s * (1. - rel_limit_phase_up))
                lim_phase = (lim_down, lim_up)

                bounds.append(lim_phase)
            else:
                # initial_guess.append(dict_phase[FLAG_PHI_ABS](elt))
                initial_guess.append(0.)
                bounds.append(limits_phase)
            x_scales.append(typical_phase_var)

        # Handle norm
        limits_norm = {
            'relative': [0.5, 1.3],    # [50%, 130%] of norm
            'absolute': [1., np.inf]   # ridiculous abs limits
            }   # TODO: personnalize limits according to zone, technology
        limits_norm_up = {
            'low beta': 1.3 * 3.03726,
            'medium beta': 1.3 * 4.45899,
            'high beta': 1.3 * 6.67386,
            }
        for elt in self.comp['l_cav']:
            norm = elt.acc_field.norm
            down = max(limits_norm['relative'][0] * norm,
                       limits_norm['absolute'][0])
            upp = limits_norm_up[elt.info['zone']]

            initial_guess.append(norm)
            bounds.append((down, upp))
            x_scales.append(typical_norm_var)

        initial_guess = np.array(initial_guess)
        bounds = np.array(bounds)
        return initial_guess, bounds, x_scales

    def _select_objective(self, str_position, str_objective):
        """
        Select the objective to fit.

        Parameters
        ----------
        str_position : string
            Indicates where the objective should be matched.
        str_objective : string
            Indicates what should be fitted.

        Return
        ------
        fun_multi_objective : function
            Takes linac and a list of indices into argument, returns a list
            of the physical quantities defined by str_objective at the
            positions defined by the list of indices.
        l_idx_pos : list of int
            Indices where the objectives should be matched. Expressed as
            indexes for the synchronous particle.
        l_elements : list of _Element
            Fraction of the linac that will be recomputed.
        """
        # Which lattices' transfer matrices will be required?
        d_lattices = {
            'end_of_last_comp_cav': lambda l_cav:
                self.brok_lin.elements['l_lattices']
                [l_cav[0].idx['lattice'][0]:l_cav[-1].idx['lattice'][0] + 1],
            'one_module_after_last_comp_cav': lambda l_cav:
                self.brok_lin.elements['l_lattices']
                [l_cav[0].idx['lattice'][0]:l_cav[-1].idx['lattice'][0] + 2],
            'both': lambda l_cav:
                self.brok_lin.elements['l_lattices']
                [l_cav[0].idx['lattice'][0]:l_cav[-1].idx['lattice'][0] + 2],
            }

        # Where do you want to verify that the objective is matched?
        d_position = {
            'end_of_last_comp_cav': lambda lattices:
                [lattices[-1][-1].idx['s_out']],
            'one_module_after_last_comp_cav': lambda lattices:
                [lattices[-1][-1].idx['s_out']],
            'both': lambda lattices: [lattices[-2][-1].idx['s_out'],
                                      lattices[-1][-1].idx['s_out']],
            }

        # What do you want to match?
        d_objective = {
            'energy': lambda linac, idx:
                [linac.synch.energy['kin_array_mev'][idx]],
            'phase': lambda linac, idx:
                [linac.synch.phi['abs_array'][idx]],
            'transfer_matrix': lambda linac, idx:
                list(linac.transf_mat['cumul'][idx, :, :].flatten()),
                }
        d_objective['energy_phase'] = lambda linac, idx: \
            d_objective['energy'](linac, idx) \
            + d_objective['phase'](linac, idx)
        d_objective['all'] = lambda linac, idx: \
            d_objective['energy_phase'](linac, idx) \
            + d_objective['transfer_matrix'](linac, idx)

        l_lattices = d_lattices[str_position](self.comp['l_cav'])
        l_elements = [elt
                      for lattice in l_lattices
                      for elt in lattice
                      ]
        l_idx_pos = d_position[str_position](l_lattices)
        fun_simple = d_objective[str_objective]

        def fun_multi_objective(linac, l_idx):
            obj = fun_simple(linac, l_idx[0])
            for idx in l_idx[1:]:
                obj = obj + fun_simple(linac, idx)
            return np.array(obj)

        for idx in l_idx_pos:
            elt = self.brok_lin.where_is_this_index(idx)
            print('\nWe try to match at synch index:', idx, 'which is',
                  elt.info, elt.idx, ".")
        return fun_multi_objective, l_idx_pos, l_elements


def wrapper(prop_array, fault, fun_objective, idx_objective,
            what_to_fit):
    """
    Fit function.

    TODO: should not modify the acc_f objects, and instead transfer the norm
    and phi_0.
    """
    global count
    # Unpack
    for i, cav in enumerate(fault.comp['l_cav']):
        acc_f = cav.acc_field
        if what_to_fit['fit_over_phi_s']:
            acc_f.phi_s_rad_objective = prop_array[i]
        else:
            acc_f.phi_0[STR_PHI_ABS] = prop_array[i]
        acc_f.norm = prop_array[i+len(fault.comp['l_cav'])]

    # Update transfer matrices
    fault.brok_lin.compute_transfer_matrices(
        fault.comp['l_recompute'], what_to_fit['fit_over_phi_s'])

    obj = fun_objective(fault.ref_lin, idx_objective) \
        - fun_objective(fault.brok_lin, idx_objective)

    if debugs['fit_progression'] and count % 20 == 0:
        debug.output_fit_progress(count, obj, what_to_fit)
    count += 1

    return obj
