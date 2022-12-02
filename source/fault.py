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
# from multiprocessing.pool import ThreadPool
# import multiprocessing
import numpy as np
from scipy.optimize import minimize, least_squares

# from pymoo.core.problem import StarmapParallelization

from dicts_output import d_markdown
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

    def __init__(self, ref_lin, brok_lin, fail_cav, comp_cav, wtf):
        self.ref_lin = ref_lin
        self.brok_lin = brok_lin
        self.wtf = wtf

        self.elts = None
        self.comp = {'l_cav': [],
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
        for cav in fail_cav:
            cav.update_status('failed')

        # We create the list of compensating cavities. We update their status
        # to 'compensate (in progress)' in fix when the optimisatin
        # process starts
        for cav in comp_cav:
            if cav.get('status') != 'failed':
                self.comp['l_cav'].append(cav)
        self.comp['n_cav'] = len(self.comp['l_cav'])

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

    def fix(self, info_other_sol):
        """Try to compensate the faulty cavities."""
        self._prepare_cavities_for_compensation()

        # Set the variables
        x_0, x_lim, constr, l_x_str = self._set_design_space()
        l_elts, d_idx = self._zone_to_recompute(self.wtf['position'])

        idx = l_elts[0].get('s_in')
        self.elts = ListOfElements(
            l_elts,
            w_kin=self.brok_lin.get('w_kin')[idx],
            phi_abs=self.brok_lin.get('phi_abs_array')[idx],
            idx_in=idx,
            tm_cumul=self.brok_lin.get('tm_cumul')[idx])
        # FIXME would be better of at Fault initialization

        # fun_residual, l_f_str = _select_objective(self.wtf['objective'])
        fun_residual, l_f_str = self._select_objective(self.wtf['objective'],
                                                       **d_idx)

        # Save some data for debug and output purposes
        self.info.update({
            'X_0': x_0,
            'X_lim': x_lim,
            'l_X_str': l_x_str,
            'l_F_str': l_f_str,
            'G': constr,
        })

        wrapper_args = (self, fun_residual, self.wtf['phi_s fit'])

        self.count = 0
        if self.wtf['opti method'] == 'least_squares':
            success, opti_sol = self._proper_fix_lsq_opt(wrapper_args)

        elif self.wtf['opti method'] == 'PSO':
            success, opti_sol = self._proper_fix_pso(
                wrapper_args, info_other_sol=info_other_sol)

        if debugs['plot_progression']:
            debug.plot_fit_progress(self.info['hist_F'], self.info['l_F_str'])
        self.info['X'] = opti_sol['X']
        self.info['F'] = opti_sol['F']

        return success, opti_sol

    def _prepare_cavities_for_compensation(self):
        """
        Prepare the compensating cavities for the upcoming optimisation.

        In particular, update the status of the compensating cavities to
        'compensate (in progress)', create list of all elements in the
        compensating zone.
        """
        new_status = "compensate (in progress)"

        # Remove broke cavities, check if some compensating cavities already
        # compensate another fault, update status of comp cav
        for cav in self.comp['l_cav']:
            current_status = cav.get('status')
            assert current_status != new_status, "Current cavity already has" \
                + " the status that you asked for. Maybe two faults want the" \
                + " same cavity for their compensation?"

            if current_status in ["compensate (ok)", "compensate (not ok)"]:
                printc("fault.prepare_cavities_for_compensation warning: ",
                       opt_message="you want to update the status of a"
                       + " cavity that is already used for compensation."
                       + " Check"
                       + "fault_scenario._gather_and_create_fault_objects."
                       + " Maybe two faults want to use the same cavity for"
                       + " compensation?")

            cav.update_status(new_status)

    def _zone_to_recompute(self, str_position):
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
        # Get indexes at start of first compensating lattice, end of last
        # compensating lattice, end of following lattice.
        idx1, idx2, idx3 = self._indexes_start_end_comp_zone(str_position)

        # We have the list of Elements that will be recomputed during
        # optimisation
        d_elts = {
            'end_mod': self.brok_lin.elts[idx1:idx2 + 1],
            '1_mod_after': self.brok_lin.elts[idx1:idx3 + 1],
            'both': self.brok_lin.elts[idx1:idx3 + 1],
        }
        l_elts = d_elts[str_position]

        # Now get indexes
        s_out = self.ref_lin.get('s_out')
        d_position = {'end_mod': [s_out[idx2]],
                      '1_mod_after': [s_out[idx3]],
                      'both': [s_out[idx2], s_out[idx3]],
                      }
        shift = l_elts[0].idx['s_in']
        d_idx = {'l_ref': d_position[str_position],
                 'l_brok': [idx - shift for idx in d_position[str_position]],
                 }
        for idx in d_idx['l_ref']:
            elt = self.brok_lin.where_is_this_index(idx)
            print(f"\nWe try to match at mesh index {idx}.")
            print(f"Info: {elt.get('elt_info')}.")
            print(f"Full indexes: {elt.get('idx')}.\n")

        return l_elts, d_idx

    def _indexes_start_end_comp_zone(self, str_position):
        """Get indexes delimiting compensation zone."""
        l_comp_cav = self.comp['l_cav']

        # We need the list of compensating cavities to be ordered for this
        # routine to work
        l_idx = [cav.get('s_in', to_numpy=False) for cav in l_comp_cav]
        assert l_idx == sorted(l_idx)

        lattices = self.brok_lin.get('lattice')

        # Lattice of first and last compensating cavity
        lattice1 = l_comp_cav[0].get('lattice')
        lattice2 = l_comp_cav[-1].get('lattice')
        lattice3 = lattice2 + 1
        if lattice2 == lattices[-1]:
            # FIXME set default behavior: fall back on end_mod
            assert str_position not in ['1_mod_after', 'both'], \
                f"str_position={str_position} asks for elements outside" \
                + "of the linac."

        # First elt of first lattice, last elt of last lattice
        idx1 = np.where(lattices == lattice1)[0][0]
        idx2 = np.where(lattices == lattice2)[0][-1]
        idx3 = np.where(lattices == lattice3)[0][-1]
        return idx1, idx2, idx3

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
        sol = solver(fun=wrapper_lsq, x0=self.info['X_0'], bounds=x_lim,
                     args=wrapper_args, **kwargs)

        if debugs['fit_progression']:
            debug.output_fit_progress(self.count, sol.fun,
                                      self.info["l_F_str"], final=True)
        if debugs['plot_progression']:
            self.info["hist_F"].append(sol.fun)

        print(
            f"message: {sol.message}"
            f"nfev: {sol.nfev}\tnjev: {sol.njev}"
            f"optimality: {sol.optimality}"
            f"status: {sol.status}"
            f"success: {sol.success}"
            f"solution: {sol.x}\n\n")

        return sol.success, {'X': sol.x.tolist(), 'F': sol.fun.tolist()}

    def _proper_fix_pso(self, wrapper_args, info_other_sol=None):
        """Fix with multi-PSO algorithm."""
        if info_other_sol is None:
            info_other_sol = {'F': None, 'X_in_real_phase': None}

        # Threading: slower
        # n_threads = 2
        # pool = ThreadPool(n_threads)
        # runner = StarmapParallelization(pool.starmap)

        # Processing:
        # n_proc = 8
        # pool = multiprocessing.Pool(n_proc)
        # runner = StarmapParallelization(pool.starmap)

        problem = pso.MyProblem(wrapper_pso, wrapper_args,
                                # elementwise_runner=runner
                               )
        res = pso.perform_pso(problem)

        # pool.close()

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

        if LINAC == 'JAEA':
            # In Bruce's paper, the maximum electric field is 20% above the
            # nominal electric field (not a single limit for each section as in
            # MYRRHA)
            d_tech_n = {0: np.NaN}
            d_x_lim_rel['k_e'] = [.5, 1.2]
            d_x_lim_rel['phi_s'] = [np.NaN, 1. - .5]

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
                equiv_idx = cav.idx['elt_idx']
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
                l_x_str.append(' '.join((cav.get('elt_name', to_numpy=False),
                                         d_markdown[__x])))
        n_cav = len(self.comp['l_cav'])
        x_0 = np.array(x_0[:2 * n_cav])
        phi_s_limits = np.array(x_lim[2 * n_cav:])
        x_lim = np.array(x_lim[:2 * n_cav])

        print(f"Initial_guess:\n{x_0}"
              f"Bounds:\n{x_lim}")

        return x_0, x_lim, phi_s_limits, l_x_str

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

    def _select_objective(self, l_objectives, **d_idx):
        """Select the objective to fit."""
        # List of indexes for ref_lin object and results dictionary
        idx_ref, idx_brok = d_idx.values()

        # mismatch factor treated differently as it is already calculated from
        # two linacs
        exceptions = ['mismatch factor']

        # List of strings to output the objective names and positions
        l_f_str = [f"idx={i_r}, " + d_markdown[key].replace("[deg]", "[rad]")
                   for i_r in idx_ref
                   for key in l_objectives]

        # We evaluate all the desired objectives
        l_ref = [self.ref_lin.get(key)[i_r]
                 if key not in exceptions
                 else self.ref_lin.get('twiss_zdelta')[i_r]
                 for i_r in idx_ref
                 for key in l_objectives]

        print("\nObjectives:")
        for i, (f_str, ref) in enumerate(zip(l_f_str, l_ref)):
            print(f"{i}: {f_str:>35} | {ref}")

        def fun_residual(results):
            """Compute difference between ref value and results dictionary."""
            i = -1
            residues = []
            for i_b in idx_brok:
                for key in l_objectives:
                    i += 1

                    # mismatch factor
                    if key == 'mismatch factor':
                        residues.append(
                            mismatch_factor(
                                l_ref[i], results['twiss_zdelta'][i_b]
                            )[0]
                        )
                        continue

                    # all other keys
                    residues.append(l_ref[i] - results[key][i_b])
            return np.abs(residues)
        return fun_residual, l_f_str


def wrapper_lsq(arr_cav_prop, fault, fun_residual, phi_s_fit):
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
    results = fault.elts.compute_transfer_matrices(d_fits, transfer_data=False)
    arr_f = fun_residual(results)

    if debugs['fit_progression'] and fault.count % 20 == 0:
        debug.output_fit_progress(fault.count, arr_f, fault.info["l_F_str"])
    if debugs['plot_progression']:
        fault.info['hist_F'].append(arr_f)
    fault.count += 1

    return arr_f


def wrapper_pso(arr_cav_prop, fault, fun_residual):
    """Unpack arguments and compute proper residues at proper spot."""
    d_fits = {'l_phi': arr_cav_prop[:fault.comp['n_cav']].tolist(),
              'l_k_e': arr_cav_prop[fault.comp['n_cav']:].tolist(),
              'phi_s fit': False}

    # Update transfer matrices
    results = fault.elts.compute_transfer_matrices(d_fits, transfer_data=False)
    arr_f = fun_residual(results)

    if debugs['fit_progression'] and fault.count % 20 == 0:
        debug.output_fit_progress(fault.count, arr_f, fault.info["l_F_str"])
    if debugs['plot_progression']:
        fault.info['hist_F'].append(arr_f)
    fault.count += 1

    return arr_f, results
