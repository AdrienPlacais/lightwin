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

TODO : try to fit gamma instead of W_kin
TODO : at init of Fault, say self.brok_lin = brok_lin.deepcopy() (or copy)
       Return self.brok_lin at the end of fix_all()
       It could also be called fix_linac...
       AH no in fact, maybe plutôt self.fixed_linac = brok_lin after it is
       broken, end of __init__. And then fix only fixed?
       Or can the breakage be done at the init of the Accelerator?
TODO : _set_design_space could be cleaner
"""
# import multiprocessing
import numpy as np
from scipy.optimize import minimize, least_squares

# from multiprocessing.pool import ThreadPool
# from pymoo.core.problem import StarmapParallelization

from constants import FLAG_PHI_ABS, LINAC
import debug
from helper import printc
from list_of_elements import ListOfElements
import pso
from emittance import mismatch_factor


debugs = {
    'fit_complete': False,
    'fit_compact': False,
    'fit_progression': False,    # Print evolution of error on objectives
    'plot_progression': False,   # Plot evolution of error on objectives
    'cav': False,
    'verbose': 0,
}


class Fault():
    """A class to hold one or several close Faults."""

    def __init__(self, ref_lin, brok_lin, fail_cav, wtf):
        self.ref_lin = ref_lin
        self.brok_lin = brok_lin
        self.wtf = wtf

        self.fail = {'l_cav': fail_cav}
        self.comp = {'l_cav': [], 'l_all_elts': [], 'l_recompute': None,
                     'n_cav': None}

        self.info = {
            'X': [],                # Solution
            'X_0': [],              # Initial guess
            'X_lim': [],            # Bounds
            'l_X_str': [],          # Name of variables for output
            'X_in_real_phase': [],  # See get_x_sol_in_real_phase
            'F': [],                # Final objective values
            'hist_F': [],           # Objective evaluations
            'l_F_str': [],          # Name of objectives for output
            'G': [],                # Constraints
            'resume': None,         # For output
        }
        self.count = None

        # We directly break the proper cavities
        for cav in self.fail['l_cav']:
            cav.update_status('failed')

    def fix_single(self, info_other_sol):
        """Try to compensate the faulty cavities."""
        # Set the variables
        x_0, x_lim, constr, l_x_str = self._set_design_space()
        l_elts, d_idx = self._select_zone_to_recompute(self.wtf['position'])
        idx = l_elts[0].get('s_in')
        self.elts = ListOfElements(
            l_elts,
            w_kin=self.brok_lin.synch.energy['kin_array_mev'][idx],
            phi_abs=self.brok_lin.synch.phi['abs_array'][idx],
            idx_in=idx,
            r_zz_cumul = self.brok_lin.transf_mat['cumul'][idx])
        # FIXME would be better of at Fault initialization

        fun_residual, l_f_str = _select_objective(self.wtf['objective'])

        # Save some data for debug and output purposes
        self.info.update({
            'X_0': x_0,
            'X_lim': x_lim,
            'l_X_str': l_x_str,
            'l_F_str': l_f_str,
            'G': constr,
        })
        # self.comp['l_recompute'] = l_elts

        wrapper_args = (self, fun_residual, d_idx, self.wtf['phi_s fit'])

        self.count = 0
        if self.wtf['opti method'] == 'least_squares':
            success, opti_sol = self._proper_fix_lsq_opt(wrapper_args)

        elif self.wtf['opti method'] == 'PSO':
            success, opti_sol = self._proper_fix_pso(
                wrapper_args, info_other_sol=info_other_sol)

        if debugs['plot_progression']:
            debug.plot_fit_progress(self.info['hist_F'],
                                    l_f_str)
        self.info['X'] = opti_sol['X']
        self.info['F'] = opti_sol['F']

        return success, opti_sol

    def _proper_fix_lsq_opt(self, wrapper_args):
        """
        Fix with classic least_squares optimisation.

        The least_squares algorithm does not allow to add constraint functions.
        In particular, if you want to control the synchronous phase, you should
        directly optimise phi_s (FLAG_PHI_S_FIT == True) or use PSO algorithm
        (self.wtf['opti method'] == 'PSO').
        """
        if self.info['X_0'].shape[0] == 1:
            solver = minimize
            kwargs = {}
        else:
            solver = least_squares
            x_lim = (self.info['X_lim'][:, 0],
                     self.info['X_lim'][:, 1])
            kwargs = {'jac': '2-point',     # Default
                      # 'trf' not ideal as jac is not sparse. 'dogbox' may have
                      # difficulties with rank-defficient jacobian.
                      'method': 'dogbox',
                      'ftol': 1e-8, 'gtol': 1e-8,   # Default
                      # Solver is sometimes 'lazy' and ends with xtol
                      # termination condition, while settings are clearly not
                      #  optimized
                      'xtol': 1e-8,
                      'diff_step': None, 'tr_solver': None, 'tr_options': {},
                      'jac_sparsity': None,
                      'verbose': debugs['verbose']
                      }
        sol = solver(fun=wrapper, x0=self.info['X_0'], bounds=x_lim,
                     args=wrapper_args, **kwargs)

        if debugs['fit_progression']:
            debug.output_fit_progress(self.count, sol.fun,
                                      self.info["l_F_str"], final=True)
        if debugs['plot_progression']:
            self.info["hist_F"].append(sol.fun)

        print(f"""\nmessage: {sol.message}\nnfev: {sol.nfev}\tnjev: {sol.njev}
              \noptimality: {sol.optimality}\nstatus: {sol.status}\n
              success: {sol.success}\nsolution: {sol.x}\n\n""")

        return sol.success, {'X': sol.x.tolist(), 'F': sol.fun.tolist()}

    def _proper_fix_pso(self, wrapper_args, info_other_sol=None):
        """Fix with multi-PSO algorithm."""
        if info_other_sol is None:
            info_other_sol = {'F': None, 'X_in_real_phase': None}

        problem = pso.MyProblem(wrapper_pso, wrapper_args)
        res = pso.perform_pso(problem)

        weights = pso.set_weights(self.wtf['objective'])
        d_opti, d_approx = pso.mcdm(res, weights, self.info,
                                    info_other_sol['F'])

        if pso.SAVE_HISTORY:
            pso.convergence_history(res.history, d_approx,
                                    self.wtf['objective'], info_other_sol['F'])
        if pso.FLAG_CONVERGENCE_CALLBACK:
            pso.convergence_callback(res.algorithm.callback,
                                     self.info['l_F_str'])
        if pso.FLAG_DESIGN_SPACE:
            pso.convergence_design_space(
                res.history, d_opti, lsq_x=info_other_sol['X_in_real_phase'])

        self.info.update({'problem': problem, 'res': res})

        # Here we return the ASF sol
        return True, d_opti['asf']

    def prepare_cavities_for_compensation(self, l_comp_cav):
        """
        Prepare the compensating cavities for the upcoming optimisation.

        In particular, update the status of the compensating cavities to
        'compensate (in progress)', create list of all elements in the
        compensating zone.
        """
        new_status = "compensate (in progress)"

        # Remove broke cavities, check if some compensating cavities already
        # compensate another fault, update status of comp cav
        for cav in l_comp_cav:
            current_status = cav.get('status')
            assert current_status != new_status, "Current cavity already has" \
                + " the status that you asked for. Maybe two faults want the" \
                + " same cavity for their compensation?"

            # If the cavity is broken, we do not want to change it's status
            if current_status == 'failed':
                continue

            if current_status in ["compensate (ok)", "compensate (not ok)"]:
                printc("fault.prepare_cavities_for_compensation warning: ",
                       opt_message="you want to update the status of a"
                       + " cavity that is already used for compensation."
                       + " Check"
                       + "fault_scenario._gather_and_create_fault_objects."
                       + " Maybe two faults want to use the same cavity for"
                       + " compensation?")

            cav.update_status(new_status)
            self.comp['l_cav'].append(cav)

        self.comp['n_cav'] = len(self.comp['l_cav'])

        # Also create a list of all the elements in the compensating lattices
        l_lattices = [lattice
                      for section in self.brok_lin.elements['l_sections']
                      for lattice in section]

        self.comp['l_all_elts'] = [elt for lattice in l_lattices
                                   for elt in lattice
                                   if any((cav in lattice
                                           for cav in self.comp['l_cav']))]

    def update_status(self, flag_success):
        """Update status of the compensating cavities."""
        if flag_success:
            new_status = "compensate (ok)"
        else:
            new_status = "compensate (not ok)"

        # Remove broke cavities, check if some compensating cavities already
        # compensate another fault, update status of comp cav
        for cav in self.comp['l_cav']:
            cav.update_status(new_status)

    def _set_design_space(self):
        """
        Set initial conditions and boundaries for the fit.

        In the returned arrays, first half of components are initial phases
        phi_0, while second half of components are norms.

        Returns
        -------
        x_0 : np.array
            Initial guess for the initial phase and norm of the compensating
            cavities.
        x_lim : np.array of tuples
            Array of (min, max) bounds for the electric fields of the
            compensating cavities.
        phi_s_limits : np.array of tuples
            Contains upper and lower synchronous phase limits for each cavity.
            Used to define constraints in PSO.
        """
        # FIXME find a cleaner way to set these limits, esp. when working with
        # different linacs.
        # Useful dicts
        d_init_g = {'k_e': lambda ref_value: ref_value,
                    'phi_0_rel': lambda ref_value: 0.,
                    'phi_0_abs': lambda ref_value: 0.,
                    'phi_s': lambda ref_value: ref_value}
        # Maximum electric field, set as 30% above the section's max electric
        # field
        d_tech_n = {0: 1.3 * 3.03726,
                    1: 1.3 * 4.45899,
                    2: 1.3 * 6.67386}
        d_x_lim_abs = {'k_e': [0., np.NaN],
                       'phi_0_rel': [0., 4. * np.pi],
                       'phi_0_abs': [0., 4. * np.pi],
                       'phi_s': [-.5 * np.pi, 0.]}
        d_x_lim_rel = {'k_e': [.5, np.NaN],
                       'phi_0_rel': [np.NaN, np.NaN],
                       'phi_0_abs': [np.NaN, np.NaN],
                       'phi_s': [np.NaN, 1. - .4]}   # phi_s+40%, w/ phi_s<0
        d_x_label = {'k_e': r'$k_e$', 'phi_0_abs': r'$\phi_{0, abs}$',
                     'phi_0_rel': r'$\phi_{0, rel}$',
                     'phi_s': r'$\varphi_s$'}

        if LINAC == 'JAEA':
            # In Bruce's paper, the maximum electric field is 20% above the
            # nominal electric field (not a single limit for each section as in
            # MYRRHA)
            d_tech_n = {0: np.NaN}
            d_x_lim_rel['k_e'] = [.5, 1.2]
            d_x_lim_rel['phi_s'] = [np.NaN, 1. - .5]
        # if self.wtf['opti method'] == 'PSO':
            # d_x_lim_abs['phi_0_rel'] = [0., 2. * np.pi]
            # d_x_lim_abs['phi_0_abs'] = [0., 2. * np.pi]

        # Set a list of properties that will be fitted
        if self.wtf['phi_s fit']:
            l_x = ['phi_s']
        else:
            if FLAG_PHI_ABS:
                l_x = ['phi_0_abs']
            else:
                l_x = ['phi_0_rel']
        l_x += ['k_e', 'phi_s']
        l_x_str = []

        # Get initial guess and bounds for every property of l_x and every
        # compensating cavity
        x_0, x_lim = [], []
        for __x in l_x:
            for cav in self.comp['l_cav']:
                equiv_idx = cav.idx['element']
                equiv_cav = self.ref_lin.elts[equiv_idx]
                ref_value = equiv_cav.get(__x)

                b_down = np.nanmax((d_x_lim_abs[__x][0],
                                    d_x_lim_rel[__x][0] * ref_value))
                if __x == 'k_e':
                    b_up = np.nanmin((d_tech_n[cav.idx['section']],
                                      d_x_lim_rel[__x][1] * ref_value))
                else:
                    b_up = np.nanmin((d_x_lim_abs[__x][1],
                                      d_x_lim_rel[__x][1] * ref_value))
                x_lim.append((b_down, b_up))
                x_0.append(d_init_g[__x](ref_value))
                l_x_str.append(' '.join((cav.get('elt_name'), d_x_label[__x])))
        n_cav = len(self.comp['l_cav'])
        x_0 = np.array(x_0[:2 * n_cav])
        phi_s_limits = np.array(x_lim[2 * n_cav:])
        x_lim = np.array(x_lim[:2 * n_cav])

        print(f"Initial_guess:\n{x_0}\nBounds:\n{x_lim}")

        return x_0, x_lim, phi_s_limits, l_x_str

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
            [l_cav[0].idx['lattice']:l_cav[-1].idx['lattice'] + 1],
            '1_mod_after': lambda l_cav: self.brok_lin.elements['l_lattices']
            [l_cav[0].idx['lattice']:l_cav[-1].idx['lattice'] + 2],
            'both': lambda l_cav: self.brok_lin.elements['l_lattices']
            [l_cav[0].idx['lattice']:l_cav[-1].idx['lattice'] + 2],
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
            print(f"\nWe try to match at mesh index {idx}.")
            print(f"Info: {elt.get('elt_info')}.")
            print(f"Full indexes: {elt.get('idx')}.")

        return l_elts, d_idx

    def get_x_sol_in_real_phase(self):
        """
        Get least-square solutions in rel/abs phases instead of synchronous.

        Least-squares fits the synchronous phase, while PSO fits the relative
        or absolute entry phase. We get all in relative/absolute to ease
        comparison between solutions.
        """
        # First half of X array: phase of cavities (relative or synchronous
        # according to the value of wtf['phi_s fit']).
        # Second half is the norms of cavities
        x_in_real_phase = self.info["X"].copy()

        key = 'phi_0_rel'
        if FLAG_PHI_ABS:
            key = 'phi_0_abs'

        for i, cav in enumerate(self.comp['l_cav']):
            x_in_real_phase[i] = cav.acc_field.phi_0[key]
            # second half of the array remains untouched
        self.info['X_in_real_phase'] = x_in_real_phase


def _select_objective(l_objectives):
    """
    Select the objective to fit.

    Parameters
    ----------
    l_objectives : list of strings
        Indicates what should be fitted.

    Return
    ------
    fun_multi_objective : function
        Return the residuals for each objective at the proper position.
    """
    # Get data from reference linac
    d_ref = {
        'energy': lambda ref, i_r: ref.synch.energy['kin_array_mev'][i_r],
        'phase': lambda ref, i_r: ref.synch.phi['abs_array'][i_r],
        'M_ij': lambda ref, i_r: ref.transf_mat['cumul'][i_r],
        'eps': lambda ref, i_r: ref.beam_param["eps"]["zdelta"][i_r],
        'twiss': lambda ref, i_r: ref.beam_param["twiss"]["zdelta"][i_r],
    }
    # Get data from results dictionary
    d_brok = {
        'energy': lambda calc, i_b: calc['w_kin'][i_b],
        'phase': lambda calc, i_b: calc['phi_abs'][i_b],
        'M_ij': lambda calc, i_b: calc['r_zz_cumul'][i_b],
        'eps': lambda calc, i_b: calc["d_zdelta"]["eps"][i_b],
        'twiss': lambda calc, i_b: calc["d_zdelta"]["twiss"][i_b],
    }

    # Dictionary to return objective functions
    d_f = {
        'energy': lambda ref, i_r, calc, i_b:
            d_ref["energy"](ref, i_r) - d_brok['energy'](calc, i_b),
        'phase': lambda ref, i_r, calc, i_b:
            d_ref["phase"](ref, i_r) - d_brok['phase'](calc, i_b),
        'M_11': lambda ref, i_r, calc, i_b:
            d_ref["M_ij"](ref, i_r)[0, 0] - d_brok['M_ij'](calc, i_b)[0, 0],
        'M_12': lambda ref, i_r, calc, i_b:
            d_ref["M_ij"](ref, i_r)[0, 1] - d_brok['M_ij'](calc, i_b)[0, 1],
        'M_21': lambda ref, i_r, calc, i_b:
            d_ref["M_ij"](ref, i_r)[1, 0] - d_brok['M_ij'](calc, i_b)[1, 0],
        'M_22': lambda ref, i_r, calc, i_b:
            d_ref["M_ij"](ref, i_r)[1, 1] - d_brok['M_ij'](calc, i_b)[1, 1],
        'eps': lambda ref, i_r, calc, i_b:
            d_ref["eps"](ref, i_r) - d_brok["eps"](calc, i_b),
        'twiss_alpha': lambda ref, i_r, calc, i_b:
            d_ref["twiss"](ref, i_r)[0] - d_brok["twiss"](calc, i_b)[0],
        'twiss_beta': lambda ref, i_r, calc, i_b:
            d_ref["twiss"](ref, i_r)[1] - d_brok["twiss"](calc, i_b)[1],
        'twiss_gamma': lambda ref, i_r, calc, i_b:
            d_ref["twiss"](ref, i_r)[2] - d_brok["twiss"](calc, i_b)[2],
        "mismatch_factor": lambda ref, i_r, calc, i_b:
            mismatch_factor(d_ref["twiss"](ref, i_r),
                            d_brok["twiss"](calc, i_b))[0],
    }

    def fun_residual(ref_lin, d_results, d_idx):
        """Compute difference between ref_linac and current optimis. param."""
        l_obj = []
        for str_obj in l_objectives:
            for i_r, i_b in zip(d_idx['l_ref'], d_idx['l_brok']):
                args = (ref_lin, i_r, d_results, i_b)
                l_obj.append(d_f[str_obj](*args))
        obj = np.abs(np.array(l_obj))
        return obj

    d_obj_str = {'energy': r'$W_{kin}$',
                 'phase': r'$\phi$',
                 'M_11': r'$M_{11}$',
                 'M_12': r'$M_{12}$',
                 'M_21': r'$M_{21}$',
                 'M_22': r'$M_{22}$',
                 'eps': r'$\epsilon_{z\delta}$',
                 'twiss_alpha': r'$\alpha_{z\delta}$',
                 'twiss_beta': r'$\beta_{z\delta}$',
                 'twiss_gamma': r'$\gamma_{z\delta}$',
                 'mismatch_factor': r'$M$',
                 }
    l_f_str = [d_obj_str[str_obj] for str_obj in l_objectives]

    return fun_residual, l_f_str


def wrapper(arr_cav_prop, fault, fun_residual, d_idx, phi_s_fit):
    """
    Unpack arguments and compute proper residues at proper spot.

    Parameters
    ----------
    arr_cav_prop : np.array
        Holds the norms (first half) and phases (second half) of cavities
    fault : Fault object
        The Fault under study.
    fun_residual : function
        Function returning the residues of the objective function at the proper
        location.
    d_idx : dict
        Dict holding the lists of indexes (ref and broken) to evaluate the
        objectives at the right spot.

    Return
    ------
    arr_f : np.array
        Array of residues on the objectives.
    """
    # Convert phases and norms into a dict for compute_transfer_matrices
    d_fits = {'l_phi': arr_cav_prop[:fault.comp['n_cav']].tolist(),
              'l_k_e': arr_cav_prop[fault.comp['n_cav']:].tolist(),
              'phi_s fit': phi_s_fit}

    # Update transfer matrices
    results = fault.elts.compute_transfer_matrices(d_fits,
                                                   flag_transfer_data=False)
    # d_results = fault.brok_lin.compute_transfer_matrices(
        # fault.comp['l_recompute'], d_fits=d_fits, flag_transfer_data=False)
    arr_f = fun_residual(fault.ref_lin, results, d_idx)

    if debugs['fit_progression'] and fault.count % 20 == 0:
        debug.output_fit_progress(fault.count, arr_f, fault.info["l_F_str"])
    if debugs['plot_progression']:
        fault.info['hist_F'].append(arr_f)
    fault.count += 1

    return arr_f


def wrapper_pso(arr_cav_prop, fault, fun_residual, d_idx):
    """Unpack arguments and compute proper residues at proper spot."""
    d_fits = {'l_phi': arr_cav_prop[:fault.comp['n_cav']].tolist(),
              'l_k_e': arr_cav_prop[fault.comp['n_cav']:].tolist(),
              'phi_s fit': False}

    # Update transfer matrices
    results = fault.elts.compute_transfer_matrices(d_fits,
                                                   flag_transfer_data=False)
    # d_results = fault.brok_lin.compute_transfer_matrices(
        # fault.comp['l_recompute'], d_fits=d_fits, flag_transfer_data=False)
    arr_f = fun_residual(fault.ref_lin, results, d_idx)

    if debugs['fit_progression'] and fault.count % 20 == 0:
        debug.output_fit_progress(fault.count, arr_f, fault.info["l_F_str"])
    if debugs['plot_progression']:
        fault.info['hist_F'].append(arr_f)
    fault.count += 1

    return arr_f, results
