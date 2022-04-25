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
from pymoo.core.problem import Problem
from constants import FLAG_PHI_ABS, FLAG_PHI_S_FIT, OPT_METHOD
import debug


dict_phase = {
    True: lambda elt: elt.acc_field.phi_0['abs'],
    False: lambda elt: elt.acc_field.phi_0['rel']
}

N_COMP_LATT_PER_FAULT = 2
debugs = {
    'fit_complete': False,
    'fit_compact': False,
    'fit_progression': False,
    'cav': True,
    'verbose': 1,
}


class Fault():
    """A class to hold one or several close Faults."""

    def __init__(self, ref_lin, brok_lin, fail_idx):
        self.ref_lin = ref_lin
        self.brok_lin = brok_lin
        self.fail = {'l_cav': [], 'l_idx': fail_idx}
        self.comp = {'l_cav': [], 'l_all_elts': None, 'l_recompute': None,
                     'n_cav': None}
        self.info = {'sol': None, 'initial_guesses': None, 'bounds': None,
                     'jac': None}

    def select_neighboring_cavities(self):
        """
        Select the cavities neighboring the failed one(s).

        More precisely:
        Select the lattices with comp cav, extract cavities from it.

        As for now, N_COMP_LATT_PER_FAULT is the number of compensating
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
        self.comp['n_cav'] = len(self.comp['l_cav'])

        # List of all elements of the compensating zone
        l_lattices = [lattice
                      for section in self.brok_lin.elements['l_sections']
                      for lattice in section
                      ]

        self.comp['l_all_elts'] = [elt
                                   for lattice in l_lattices
                                   for elt in lattice
                                   if any((cav in lattice
                                           for cav in self.comp['l_cav']))
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

        cav_to_rephase = [
            cav for cav in after_ffc
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
                neighbor_modules.append(modules[idx - 1])
            if idx < len(modules) - 1:
                neighbor_modules.append(modules[idx + 1])
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
        """
        self.what_to_fit = what_to_fit

        # Set the fit variables
        initial_guesses, bounds = self._set_fit_parameters()

        fun_objective, idx_objective, idx_objective2, l_elts = \
            self._select_objective(self.what_to_fit['position'],
                                   self.what_to_fit['objective'])
        # Save some
        self.info['initial_guesses'] = initial_guesses
        self.info['bounds'] = bounds
        self.comp['l_recompute'] = l_elts

        args = (self, fun_objective, idx_objective, idx_objective2,
                what_to_fit)

        if OPT_METHOD == 'classic':
            sol = self._proper_fix_classic_opt(initial_guesses, bounds, args)

        elif OPT_METHOD == 'PSO':
            sol = self._proper_fix_pso(initial_guesses, bounds, args)

        return sol

    def _proper_fix_classic_opt(self, init_guess, bounds, args):
        """Fix with classic optimisation."""
        global COUNT
        COUNT = 0

        if init_guess.shape[0] == 1:
            solver = minimize
        else:
            solver = least_squares
            bounds = (bounds[:, 0], bounds[:, 1])

        sol = solver(fun=wrapper, x0=init_guess, bounds=bounds, args=args,
                     jac='2-point',  # Default
                     # 'trf' not ideal as jac is not sparse.
                     # 'dogbox' may have difficulties with rank-defficient jac.
                     method='dogbox',
                     ftol=1e-8, gtol=1e-8,   # Default
                     # Solver is sometimes 'lazy' and ends with xtol
                     # termination condition, while settings are clearly not
                     #  optimized
                     xtol=1e-8,
                     # x_scale='jac',    # TODO
                     # loss=linear,      # TODO
                     # f_scale=1.0,      # TODO
                     # diff_step=None,   # TODO
                     # tr_solver=None, tr_options={},   # TODO
                     # jac_sparsity=None,    # TODO
                     verbose=debugs['verbose'],
                     )

        if debugs['fit_progression']:
            debug.output_fit_progress(COUNT, sol.fun, final=True)

        print('\nmessage:', sol.message, '\nnfev:', sol.nfev, '\tnjev:',
              sol.njev, '\noptimality:', sol.optimality, '\nstatus:',
              sol.status, '\tsuccess:', sol.success, '\nx:', sol.x, '\n\n')
        self.info['sol'] = sol
        self.info['jac'] = sol.jac

        return sol

    def _proper_fix_pso(self, init_guess, bounds, args):
        """Fix with multi-PSO algorithm."""

        class Compensate(Problem):
            """Class holding PSO."""

            def __init__(self):
                # args = (self, fun_objective, idx_objective, idx_objective2,
                #         what_to_fit)
                super().__init__(n_var=init_guess.shape[0], n_obj=len(args[2]),
                                 n_constr=0,
                                 xl=bounds[:, 0], xu=bounds[:, 1])

            def _evaluate(self, var, out, *args, **kwargs):
                out["F"] = wrapper(args)
        sol = None
        return sol

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
        if FLAG_PHI_S_FIT:
            limits_phase = (-np.pi / 2., 0.)
            rel_limit_phase_up = .4    # +40% over nominal synch phase
        else:
            limits_phase = (0., 8. * np.pi)

        for elt in self.comp['l_cav']:
            if FLAG_PHI_S_FIT:
                equiv_phi_s = self.ref_lin.elements['list'][
                    elt.idx['element']].acc_field.cav_params['phi_s_rad']
                initial_guess.append(equiv_phi_s)
                lim_down = limits_phase[0]
                lim_up = min(limits_phase[1],
                             equiv_phi_s * (1. - rel_limit_phase_up))
                lim_phase = (lim_down, lim_up)

                bounds.append(lim_phase)
            else:
                initial_guess.append(0.)
                bounds.append(limits_phase)

        # Handle norm
        limits_norm = {
            # Down norms
            'relative': 0.5,    # [50%, 130%] of norm
            'absolute': 1.,  # ridiculous abs limits
            # Up norms according to technology:
            'low beta': 1.3 * 3.03726,
            'medium beta': 1.3 * 4.45899,
            'high beta': 1.3 * 6.67386,
        }
        for elt in self.comp['l_cav']:
            norm = elt.acc_field.norm
            lim_down = max(limits_norm['relative'] * norm,
                           limits_norm['absolute'])
            lim_up = limits_norm[elt.info['zone']]

            initial_guess.append(norm)
            bounds.append((lim_down, lim_up))

        initial_guess = np.array(initial_guess)
        bounds = np.array(bounds)
        print('initial_guess:\n', initial_guess, '\nbounds:\n', bounds)
        return initial_guess, bounds

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
        # Which lattices' data are necessary?
        d_lattices = {
            'end_mod': lambda l_cav: self.brok_lin.elements['l_lattices']
            [l_cav[0].idx['lattice'][0]:l_cav[-1].idx['lattice'][0] + 1],
            '1_mod_after': lambda l_cav: self.brok_lin.elements['l_lattices']
            [l_cav[0].idx['lattice'][0]:l_cav[-1].idx['lattice'][0] + 2],
            'both': lambda l_cav: self.brok_lin.elements['l_lattices']
            [l_cav[0].idx['lattice'][0]:l_cav[-1].idx['lattice'][0] + 2],
        }
        l_lattices = d_lattices[str_position](self.comp['l_cav'])
        l_elements = [elt
                      for lattice in l_lattices
                      for elt in lattice]

        # Where do you want to verify that the objective is matched?
        d_pos = {
            'end_mod': lambda lattices: [lattices[-1][-1].idx['s_out']],
            '1_mod_after': lambda lattices: [lattices[-1][-1].idx['s_out']],
            'both': lambda lattices: [lattices[-2][-1].idx['s_out'],
                                      lattices[-1][-1].idx['s_out']],
        }
        l_idx_ref = d_pos[str_position](l_lattices)
        shift_s_idx_brok = self.comp['l_all_elts'][0].idx['s_in']
        l_idx_brok = [idx - shift_s_idx_brok
                      for idx in l_idx_ref]

        # What do you want to match?
        d_obj_ref = {
            'energy': lambda ref_lin, idx:
                [ref_lin.synch.energy['kin_array_mev'][idx]],
            'phase': lambda ref_lin, idx:
                [ref_lin.synch.phi['abs_array'][idx]],
            'transf_mat': lambda ref_lin, idx:
                list(ref_lin.transf_mat['cumul'][idx].flatten()),
        }
        d_obj_ref['energy_phase'] = lambda ref_lin, idx: \
            d_obj_ref['energy'](ref_lin, idx) \
            + d_obj_ref['phase'](ref_lin, idx)
        d_obj_ref['all'] = lambda ref_lin, idx: \
            d_obj_ref['energy_phase'](ref_lin, idx) \
            + d_obj_ref['transf_mat'](ref_lin, idx)

        fun_ref = d_obj_ref[str_objective]

        d_obj_brok = {
            'energy': lambda calc, idx: [calc['W_kin'][idx]],
            'phase': lambda calc, idx: [calc['phi_abs'][idx]],
            'transf_mat': lambda calc, idx: list(calc['r_zz'][idx].flatten())
        }
        d_obj_brok['energy_phase'] = lambda calc, idx: \
            d_obj_brok['energy'](calc, idx) + d_obj_brok['phase'](calc, idx)
        d_obj_brok['all'] = lambda calc, idx: \
            d_obj_brok['energy_phase'](calc, idx) + \
            d_obj_brok['transf_mat'](calc, idx)

        fun_brok = d_obj_brok[str_objective]

        def fun_multi_obj(ref_lin, calc, l_idx_ref, l_idx_brok,
                          flag_out=False):
            obj_ref = fun_ref(ref_lin, l_idx_ref[0])
            obj_brok = fun_brok(calc, l_idx_brok[0])

            for idx1, idx2 in zip(l_idx_ref[1:], l_idx_brok[1:]):
                obj_ref += fun_ref(ref_lin, idx1)
                obj_brok += fun_brok(calc, idx2)
            # print('ref', obj_ref)
            # print('brk', obj_brok)
            return np.abs(np.array(obj_ref) - np.array(obj_brok))

        for idx in l_idx_ref:
            elt = self.brok_lin.where_is_this_index(idx)
            print('\nWe try to match at synch index:', idx, 'which is',
                  elt.info, elt.idx, ".")
        return fun_multi_obj, l_idx_ref, l_idx_brok, l_elements


def wrapper(prop_array, fault, fun_multi_obj, idx_ref, idx_brok, what_to_fit):
    """Fit function."""
    global COUNT

    d_fits = {
        'flag': True,
        'l_phi': prop_array[:fault.comp['n_cav']].tolist(),
        'l_norm': prop_array[fault.comp['n_cav']:].tolist()
    }

    # Update transfer matrices
    keys = ('r_zz', 'W_kin', 'phi_abs')
    values = fault.brok_lin.compute_transfer_matrices(
        fault.comp['l_recompute'], d_fits=d_fits, flag_transfer_data=False)
    calc = dict(zip(keys, values))
    obj = fun_multi_obj(fault.ref_lin, calc, idx_ref, idx_brok)

    if debugs['fit_progression'] and COUNT % 20 == 0:
        debug.output_fit_progress(COUNT, obj, what_to_fit)
    COUNT += 1

    return obj
