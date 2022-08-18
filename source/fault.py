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

FIXME: is _select_comp_modules still in use?
"""
import numpy as np
from scipy.optimize import minimize, least_squares
import PSO as pso
from constants import FLAG_PHI_ABS, FLAG_PHI_S_FIT, OPTI_METHOD, WHAT_TO_FIT
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
        self.comp = {'l_cav': [], 'l_all_elts': [], 'l_recompute': None,
                     'n_cav': None}
        self.info = {'sol': None, 'initial_guesses': None, 'bounds': None,
                     'jac': None, 'l_obj_label': [], 'l_prop_label': [],
                     'resume': None}
        self.count = None

        # We directly break the proper cavities
        self._set_broken_cavities()

    def fix_single(self):
        """Try to compensate the faulty cavities."""
        # Set the fit variables
        initial_guesses, bounds, phi_s_limits, l_prop_label \
            = self._set_fit_parameters()
        l_elts, d_idx = self._select_zone_to_recompute(WHAT_TO_FIT['position'])

        fun_residual = _select_objective(WHAT_TO_FIT['objective'])
        l_obj_label = _set_labels(WHAT_TO_FIT['objective'])

        # Save some data for debug and output purposes
        self.info['initial_guesses'] = initial_guesses
        self.info['bounds'] = bounds
        self.info['l_prop_label'] = l_prop_label
        self.info['l_obj_label'] = l_obj_label
        self.comp['l_recompute'] = l_elts

        wrapper_args = (self, fun_residual, d_idx)

        self.count = 0
        if OPTI_METHOD == 'least_squares':
            flag_success, opti_sol = self._proper_fix_lsq_opt(
                initial_guesses, bounds, wrapper_args)

        elif OPTI_METHOD == 'PSO':
            flag_success, opti_sol = self._proper_fix_pso(
                initial_guesses, bounds, wrapper_args, phi_s_limits)

        return flag_success, opti_sol

    def _proper_fix_lsq_opt(self, init_guess, bounds, wrapper_args):
        """
        Fix with classic least_squares optimisation.

        The least_squares algorithm does not allow to add constraint functions.
        In particular, if you want to control the synchronous phase, you should
        directly optimise phi_s (FLAG_PHI_S_FIT == True) or use PSO algorithm
        (OPTI_METHOD == 'PSO').
        """
        if init_guess.shape[0] == 1:
            solver = minimize
            # TODO: recheck
            kwargs = {}
        else:
            solver = least_squares
            bounds = (bounds[:, 0], bounds[:, 1])
            kwargs = {'jac': '2-point',     # Default
                      # 'trf' not ideal as jac is not sparse. 'dogbox' may have
                      # difficulties with rank-defficient jacobian.
                      'method': 'dogbox',
                      'ftol': 1e-8, 'gtol': 1e-8,   # Default
                      # Solver is sometimes 'lazy' and ends with xtol
                      # termination condition, while settings are clearly not
                      #  optimized
                      'xtol': 1e-8,
                      # TODO: check these args
                      'x_scale': 'jac', 'loss': 'linear', 'f_scale': 1.0,
                      'diff_step': None, 'tr_solver': None, 'tr_options': {},
                      'jac_sparsity': None,
                      'verbose': debugs['verbose']
                      }
        sol = solver(fun=wrapper, x0=init_guess, bounds=bounds,
                     args=wrapper_args, **kwargs)

        if debugs['fit_progression']:
            debug.output_fit_progress(self.count, sol.fun, final=True)

        print('\nmessage:', sol.message, '\nnfev:', sol.nfev, '\tnjev:',
              sol.njev, '\noptimality:', sol.optimality, '\nstatus:',
              sol.status, '\tsuccess:', sol.success, '\nx:', sol.x, '\n\n')
        self.info['sol'] = sol
        self.info['jac'] = sol.jac

        return sol.success, sol.x

    def _proper_fix_pso(self, init_guess, bounds, wrapper_args,
                        phi_s_limits=None):
        """Fix with multi-PSO algorithm."""
        n_obj = 6  # FIXME
        if FLAG_PHI_S_FIT:
            n_constr = 0
        else:
            assert phi_s_limits is not None
            n_constr = 2 * phi_s_limits.shape[0]

        problem = pso.MyProblem(wrapper_pso, init_guess.shape[0], n_obj,
                                n_constr,
                                bounds, wrapper_args, phi_s_limits)
        res = pso.perform_pso(problem)

        weights = pso.set_weights(WHAT_TO_FIT['objective'])
        opti_sol, approx_ideal, approx_nadir = pso.mcdm(res, weights,
                                                        self.info)

        if pso.flag_convergence_history:
            pso.convergence_history(res.history, approx_ideal, approx_nadir)
        if pso.flag_convergence_callback:
            pso.convergence_callback(res.algorithm.callback,
                                     self.info['l_obj_label'])

        return True, opti_sol

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

    def _set_broken_cavities(self):
        """Break the cavities to break."""
        # Break proper cavities
        for idx in self.fail['l_idx']:
            cav = self.brok_lin.elements['list'][idx]
            cav.update_status('failed')
            self.fail['l_cav'].append(cav)

    def set_compensating_cavities(self, strategy, l_comp_idx=None):
        """Define list of elts in compensating zone, update status of cav."""
        # Create a list of cavities that will compensate
        if strategy == 'neighbors':
            l_comp_cav = self.select_neighboring_cavities()
        elif strategy == 'manual':
            assert len(l_comp_idx) > 0, "A list of compensating cavities" \
                    + "is required with WHAT_TO_FIT['strategy'] == 'manual'."
            l_comp_cav = [self.brok_lin.elements['list'][idx]
                          for idx in l_comp_idx]

        # Remove broke cavities, check if some compensating cavities already
        # compensate another fault, update status of comp cav
        for cav in l_comp_cav:
            if cav.info['status'] != 'nominal':
                if cav.info['status'] == 'failed':
                    l_comp_cav.remove(cav)
                    continue

                print("Warning fault.set_compensating_cavities:")
                print("several faults want the same compensating cavity!")

            cav.update_status('compensate (in progress)')
            self.comp['l_cav'].append(cav)

        self.comp['n_cav'] = len(self.comp['l_cav'])

        # Also create a list of all the elements in the compensating lattices
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

    def _set_fit_parameters(self):
        """
        Set initial conditions and boundaries for the fit.

        In the returned arrays, first half of components are initial phases
        phi_0, while second half of components are norms.

        Returns
        -------
        initial_guess : np.array
            Initial guess for the initial phase and norm of the compensating
            cavities.
        bounds : np.array of tuples
            Array of (min, max) bounds for the electric fields of the
            compensating cavities.
        phi_s_limits : np.array of tuples
            Contains upper and lower synchronous phase limits for each cavity.
            Used to define constraints in PSO.
        """
        # Useful dicts
        d_getter = {'norm': lambda cav: cav.acc_field.norm,
                    'phi_0_rel': lambda cav: cav.acc_field.phi_0['rel'],
                    'phi_0_abs': lambda cav: cav.acc_field.phi_0['abs'],
                    'phi_s': lambda cav: cav.acc_field.cav_params['phi_s_rad']}
        d_init_g = {'norm': lambda ref_value: ref_value,
                    'phi_0_rel': lambda ref_value: 0.,
                    'phi_0_abs': lambda ref_value: 0.,
                    'phi_s': lambda ref_value: ref_value}
        d_tech_n = {'low beta': 1.3 * 3.03726,
                    'medium beta': 1.3 * 4.45899,
                    'high beta': 1.3 * 6.67386}
        d_bounds_abs = {'norm': [1., np.NaN],
                        'phi_0_rel': [0., 4. * np.pi],
                        'phi_0_abs': [0., 4. * np.pi],
                        'phi_s': [-.5 * np.pi, 0.]}
        d_bounds_rel = {'norm': [.5, np.NaN],
                        'phi_0_rel': [np.NaN, np.NaN],
                        'phi_0_abs': [np.NaN, np.NaN],
                        'phi_s': [np.NaN, 1. - .4]}   # phi_s+40%, w/ phi_s<0
        d_prop_label = {'norm': r'$k_e$', 'phi_0_abs': r'$\phi_{0, abs}$',
                        'phi_0_rel': r'$\phi_{0, rel}$',
                        'phi_s': r'$\varphi_s$'}

        # Set a list of properties that will be fitted
        if FLAG_PHI_S_FIT:
            l_prop = ['phi_s']
        else:
            if FLAG_PHI_ABS:
                l_prop = ['phi_0_abs']
            else:
                l_prop = ['phi_0_rel']
        l_prop += ['norm', 'phi_s']
        l_prop_label = []

        # Get initial guess and bounds for every property of l_prop and every
        # compensating cavity
        initial_guess, bounds = [], []
        for prop in l_prop:
            for cav in self.comp['l_cav']:
                equiv_cav = self.ref_lin.elements['list'][cav.idx['element']]
                ref_value = d_getter[prop](equiv_cav)
                b_down = np.nanmax((d_bounds_abs[prop][0],
                                    d_bounds_rel[prop][0] * ref_value))
                if prop == 'norm':
                    b_up = d_tech_n[cav.info['zone']]
                else:
                    b_up = np.nanmin((d_bounds_abs[prop][1],
                                      d_bounds_rel[prop][1] * ref_value))
                bounds.append((b_down, b_up))
                initial_guess.append(d_init_g[prop](ref_value))
                l_prop_label.append(' '.join((cav.info['name'],
                                              d_prop_label[prop])))

        n_cav = len(self.comp['l_cav'])
        initial_guess = np.array(initial_guess[:2 * n_cav])
        phi_s_limits = np.array(bounds[2 * n_cav:])
        bounds = np.array(bounds[:2 * n_cav])

        print('initial_guess:\n', initial_guess, '\nbounds:\n', bounds)
        if OPTI_METHOD == 'PSO' and not FLAG_PHI_ABS:
            print('Additional constraint: phi_s_limits:\n', phi_s_limits)

        return initial_guess, bounds, phi_s_limits, l_prop_label

    def _select_zone_to_recompute(self, str_position):
        """
        Determine zone to recompute and indexes of where objective is checked.

        Parameters
        ----------
        str_position : string
            Indicates where the objective should be matched.

        Return
        ------
        l_elts : list of _Element
            List of elements that should be recomputed.
        d_idx : dict
            Dict holding the lists of indexes (ref and broken) to evaluate the
            objectives at the right spot.
        """
        d_idx = {'l_ref': [], 'l_brok': []}

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
        l_elts = [elt
                  for lattice in l_lattices
                  for elt in lattice]
        # Where do you want to verify that the objective is matched?
        d_pos = {
            'end_mod': lambda lattices: [lattices[-1][-1].idx['s_out']],
            '1_mod_after': lambda lattices: [lattices[-1][-1].idx['s_out']],
            'both': lambda lattices: [lattices[-2][-1].idx['s_out'],
                                      lattices[-1][-1].idx['s_out']],
        }
        d_idx['l_ref'] = d_pos[str_position](l_lattices)
        shift_s_idx_brok = self.comp['l_all_elts'][0].idx['s_in']
        d_idx['l_brok'] = [idx - shift_s_idx_brok
                           for idx in d_idx['l_ref']]

        for idx in d_idx['l_ref']:
            elt = self.brok_lin.where_is_this_index(idx)
            print('\nWe try to match at synch index:', idx, 'which is',
                  elt.info, elt.idx, ".")

        return l_elts, d_idx


def _select_objective(str_objective):
    """
    Select the objective to fit.

    Parameters
    ----------
    str_objective : string
        Indicates what should be fitted.

    Return
    ------
    fun_multi_objective : function
        Return the residuals for each objective at the proper position.
    """
    # Data getters
    d_obj_ref = {
        'energy': lambda ref_lin: ref_lin.synch.energy['kin_array_mev'],
        'phase': lambda ref_lin: ref_lin.synch.phi['abs_array'],
        'transf_mat': lambda ref_lin: np.resize(
            ref_lin.transf_mat['cumul'],
            (ref_lin.transf_mat['cumul'].shape[0], 4))
    }
    d_obj_brok = {'energy': lambda calc: calc['W_kin'],
                  'phase': lambda calc: calc['phi_abs'],
                  'transf_mat': lambda calc: np.resize(
                      calc['r_zz'], (calc['r_zz'].shape[0], 4))}

    d_obj_ref['energy_phase'] = lambda var: np.column_stack(
        (d_obj_ref['energy'](var), d_obj_ref['phase'](var)))
    d_obj_brok['energy_phase'] = lambda var: np.column_stack(
        (d_obj_brok['energy'](var), d_obj_brok['phase'](var)))

    d_obj_ref['all'] = lambda var: np.hstack(
        (d_obj_ref['energy_phase'](var), d_obj_ref['transf_mat'](var)))
    d_obj_brok['all'] = lambda var: np.hstack(
        (d_obj_brok['energy_phase'](var), d_obj_brok['transf_mat'](var)))

    # @FIXME: with this syntax the arguments aimed to go to
    # d_obj_ref are sent to d_obj_brok (ie: Accelerator sent instead of calc)
    # for dic in [d_obj_ref, d_obj_brok]:
    #     dic['energy_phase'] = lambda var: np.column_stack(
    #         (dic['energy'](var), dic['phase'](var)))
    #     dic['all'] = lambda var: np.hstack(
    #         (dic['energy_phase'](var), dic['transf_mat'](var)))

    # Functions returning np.array's filled with desired quantities
    fun_ref = d_obj_ref[str_objective]
    fun_brok = d_obj_brok[str_objective]

    def fun_residual(ref_lin, brok_calc, d_idx):
        """Compute difference between ref_linac and current optimis. param."""
        obj = np.abs(fun_ref(ref_lin)[d_idx['l_ref'], :]
                     - fun_brok(brok_calc)[d_idx['l_brok'], :])
        return obj.flatten()
    return fun_residual


def _set_labels(str_objective):
    """Set strings for better visualisation of the optimisation."""
    d_obj_str = {'energy': [r'$W_{kin}$'],
                 'phase': [r'$\phi$'],
                 'transf_mat': [r'$M_{11}$', r'$M_{12}$',
                                r'$M_{21}$', r'$M_{22}$']}
    d_obj_str['energy_phase'] = d_obj_str['energy'] + d_obj_str['phase']
    d_obj_str['all'] = d_obj_str['energy_phase'] + d_obj_str['transf_mat']
    l_obj_label = d_obj_str[str_objective]
    return l_obj_label


def wrapper(arr_cav_prop, fault, fun_residual, d_idx):
    """Unpack arguments and compute proper residues at proper spot."""
    d_fits = {'flag': True,
              'l_phi': arr_cav_prop[:fault.comp['n_cav']].tolist(),
              'l_norm': arr_cav_prop[fault.comp['n_cav']:].tolist()}
    keys = ('r_zz', 'W_kin', 'phi_abs', 'phi_s_rad')

    # Update transfer matrices
    values = fault.brok_lin.compute_transfer_matrices(
        fault.comp['l_recompute'], d_fits=d_fits, flag_transfer_data=False)
    brok_calc = dict(zip(keys, values))
    obj = fun_residual(fault.ref_lin, brok_calc, d_idx)

    if debugs['fit_progression'] and fault.count % 20 == 0:
        debug.output_fit_progress(fault.count, obj)
    fault.count += 1

    return obj


def wrapper_pso(arr_cav_prop, fault, fun_residual, d_idx):
    """Unpack arguments and compute proper residues at proper spot."""
    d_fits = {'flag': True,
              'l_phi': arr_cav_prop[:fault.comp['n_cav']].tolist(),
              'l_norm': arr_cav_prop[fault.comp['n_cav']:].tolist()}
    keys = ('r_zz', 'W_kin', 'phi_abs', 'phi_s_rad')

    # Update transfer matrices
    values = fault.brok_lin.compute_transfer_matrices(
        fault.comp['l_recompute'], d_fits=d_fits, flag_transfer_data=False)
    brok_calc = dict(zip(keys, values))
    obj = fun_residual(fault.ref_lin, brok_calc, d_idx)

    if debugs['fit_progression'] and fault.count % 20 == 0:
        debug.output_fit_progress(fault.count, obj)
    fault.count += 1

    return obj, brok_calc
