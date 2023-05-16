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
import logging
from typing import Callable
import numpy as np
from scipy.optimize import minimize, least_squares

import config_manager as con
from core.list_of_elements import ListOfElements, equiv_elt
from core.elements import FieldMap
from core.emittance import mismatch_factor
from core.accelerator import Accelerator
from util import debug
from util.dicts_output import d_markdown
from optimisation import pso
from optimisation.linacs_design_space import initial_value, limits, constraints
import visualization.plot


debugs = {
    'fit_complete': False,
    'fit_compact': False,
    'fit_progression': False,    # Print evolution of error on objectives
    'plot_progression': False,   # Plot evolution of error on objectives
    'cav': True,
    'verbose': 0,
}


class Fault():
    """A class to hold one or several close Faults."""

    def __init__(self, ref_lin: Accelerator, brok_lin: Accelerator,
                 fail_cav: list[FieldMap], comp_cav: list[FieldMap], wtf: dict
                 ) -> None:
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
        self.fail = {'l_cav': fail_cav}

        # We create the list of compensating cavities. We update their status
        # to 'compensate (in progress)' in fix when the optimisatin
        # process starts
        for cav in comp_cav:
            if cav.get('status') != 'failed':
                self.comp['l_cav'].append(cav)
        self.comp['n_cav'] = len(self.comp['l_cav'])

    def update_status(self, flag_success: bool) -> None:
        """Update status of the compensating cavities."""
        if flag_success:
            new_status = "compensate (ok)"
        else:
            new_status = "compensate (not ok)"

        # Remove broke cavities, check if some compensating cavities already
        # compensate another fault, update status of comp cav
        for cav in self.comp['l_cav']:
            cav.update_status(new_status)

    def fix(self, info_other_sol: dict) -> tuple[bool, dict]:
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

        fun_residual, l_f_str = self._select_objective(
            self.wtf['objective'], self.wtf['scale objective'], **d_idx)

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
            visualization.plot.plot_fit_progress(self.info['hist_F'],
                                                 self.info['l_F_str'])
        self.info['X'] = opti_sol['X']
        self.info['F'] = opti_sol['F']
        return success, opti_sol

    def _prepare_cavities_for_compensation(self) -> None:
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
                logging.warning(
                    "You want to update the status of a cavity that is "
                    + "already used for compensation. Check "
                    + "fault_scenario._gather_and_create_fault_objects. "
                    + "Maybe two faults want to use the same cavity for "
                    + "compensation?")

            cav.update_status(new_status)

    def _zone_to_recompute(self, str_position: str) -> (list, dict):
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

        # List of elements in compensation zone
        d_elts = {
            'end_mod': self.brok_lin.elts[idx1:idx2 + 1],
            '1_mod_after': self.brok_lin.elts[idx1:idx3 + 1],
            'both': self.brok_lin.elts[idx1:idx3 + 1],
            'end_linac': self.brok_lin.elts[idx1:],
        }
        l_elts = d_elts[str_position]

        # Where objectives are evaluated
        s_out = self.ref_lin.get('s_out')
        d_position = {'end_mod': [s_out[idx2]],
                      '1_mod_after': [s_out[idx3]],
                      'both': [s_out[idx2], s_out[idx3]],
                      'end_linac': [s_out[-1]],
                      }
        shift = l_elts[0].idx['s_in']
        d_idx = {'l_ref': d_position[str_position],
                 'l_brok': [idx - shift for idx in d_position[str_position]],
                 }
        for idx in d_idx['l_ref']:
            elt = self.brok_lin.where_is_this_s_idx(idx)
            # will return None if idx is the last index of the linac
            if elt is None:
                elt = self.brok_lin.where_is_this_s_idx(idx - 1)
                logging.warning("We return _Element just before.")
            logging.info(f"We try to match at mesh index {idx}.\n"
                         + f"Info: {elt.get('elt_info')}.\n"
                         + f"Full indexes: {elt.get('idx')}.")

        return l_elts, d_idx

    def _indexes_start_end_comp_zone(self, str_position: str) -> tuple[int]:
        """Get indexes delimiting compensation zone."""
        # We get altered cavities (compensating or failed) (unsorted)
        l_altered_cav = self.comp['l_cav'] + self.fail['l_cav']

        # We need the list of altered cavities to be ordered for this routine
        l_idx = [cav.get('s_in', to_numpy=False) for cav in l_altered_cav]
        sort = np.argsort(l_idx)

        l_altered_cav = [l_altered_cav[idx] for idx in sort.tolist()]
        l_idx = [cav.get('s_in', to_numpy=False) for cav in l_altered_cav]
        assert l_idx == sorted(l_idx)
        del l_idx, sort
        # Now the list of altered cavities is sorted!

        lattices = self.brok_lin.get('lattice')

        # Lattice of first and last compensating cavity
        # First elt of first lattice
        lattice1 = l_altered_cav[0].get('lattice')
        idx1 = np.where(lattices == lattice1)[0][0]

        # last elt of last lattice
        lattice2 = l_altered_cav[-1].get('lattice')
        idx2 = np.where(lattices == lattice2)[0][-1]

        if lattice2 == lattices[-1]:
            # FIXME set default behavior: fall back on end_mod
            assert str_position not in ['1_mod_after', 'both'], \
                f"str_position={str_position} asks for elements outside" \
                + "of the linac."
            return idx1, idx2, idx2

        # One lattice after (used for 1_mod_after and both)
        lattice3 = lattice2 + 1
        idx3 = np.where(lattices == lattice3)[0][-1]
        return idx1, idx2, idx3

    def _proper_fix_lsq_opt(self, wrapper_args) -> tuple[bool, dict]:
        """
        Fix with classic least_squares optimisation.

        The least_squares algorithm does not allow to add constraint functions.
        In particular, if you want to control the synchronous phase, you should
        directly optimise phi_s (FLAG_PHI_S_FIT == True) or use PSO algorithm
        (self.wtf['opti method'] == 'PSO').
        """
        solver = minimize
        kwargs = {}
        if self.info['X_0'].shape[0] > 1:
            solver = least_squares
            x_lim = (self.info['X_lim'][:, 0], self.info['X_lim'][:, 1])

            kwargs = {'jac': '2-point',     # Default
                      # 'trf' not ideal as jac is not sparse. 'dogbox' may have
                      # difficulties with rank-defficient jacobian.
                      'method': 'dogbox',
                      'ftol': 1e-8, 'gtol': 1e-8,   # Default
                      # Solver is sometimes 'lazy' and ends with xtol
                      # termination condition, while settings are clearly not
                      #  optimized
                      'xtol': 1e-8,
                      # 'x_scale': 'jac',
                      # 'loss': 'arctan',
                      'diff_step': None, 'tr_solver': None, 'tr_options': {},
                      'jac_sparsity': None,
                      'verbose': debugs['verbose']}

        sol = solver(fun=wrapper_lsq, x0=self.info['X_0'], bounds=x_lim,
                     args=wrapper_args, **kwargs)

        if debugs['fit_progression']:
            debug.output_fit_progress(self.count, sol.fun,
                                      self.info["l_F_str"], final=True)
        if debugs['plot_progression']:
            self.info["hist_F"].append(sol.fun)

        # FIXME may be moved to util/output
        info_string = "Objective functions results:\n"
        for i, fun in enumerate(sol.fun):
            info_string += f"{i}: {' ':>35} | {fun}\n"
        logging.info(info_string)
        info_string = "least_squares algorithm output:"
        info_string += f"\nmessage: {sol.message}\n"
        info_string += f"nfev: {sol.nfev}\tnjev: {sol.njev}\n"
        info_string += f"optimality: {sol.optimality}\nstatus: {sol.status}\n"
        info_string += f"success: {sol.success}\nsolution: {sol.x}\n"
        logging.debug(info_string)
        return sol.success, {'X': sol.x.tolist(), 'F': sol.fun.tolist()}

    def _proper_fix_pso(self, wrapper_args, info_other_sol: dict | None = None
                        ) -> tuple[bool, dict]:
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

    def _set_design_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                         list[str, ...]]:
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
        l_x_str : list of str
            Name of cavities and properties.
        """
        # List of objectives:
        #   - first objective is always a phase
        #   - second objective is always the electric field
        l_x = ['phi_0_rel', 'k_e']
        if con.FLAG_PHI_ABS:
            l_x = ['phi_0_abs', 'k_e']
        if self.wtf['phi_s fit']:
            l_x = ['phi_s', 'k_e']
        # List of constaints:
        l_g = ['phi_s']
        # FIXME should not be initialized if not used

        ref_linac = self.ref_lin
        x_0, x_lim, l_x_str = [], [], []
        for obj in l_x:
            for cav in self.comp['l_cav']:
                ref_cav = ref_linac.equiv_elt(cav)

                args = (con.LINAC, cav, ref_cav, ref_linac)
                x_0.append(initial_value(obj, ref_cav))
                x_lim.append(limits(obj, *args))
                l_x_str.append(' '.join((cav.get('elt_name', to_numpy=False),
                                         d_markdown[obj])))
        g_lim = []
        for const in l_g:
            for cav in self.comp['l_cav']:
                ref_cav = ref_linac.equiv_elt(cav)

                args = (con.LINAC, cav, ref_cav, ref_linac)
                g_lim.append(constraints(const, *args))

        x_0 = np.array(x_0)
        x_lim = np.array(x_lim)
        g_lim = np.array(g_lim)

        logging.info("Design space (handled in "
                     + "optimisation.linacs_design_space, not .ini):\n"
                     + f"Initial guess:\n{x_0}\n"
                     + f"Bounds:\n{x_lim}\n"
                     + f"Constraints (not necessarily used):\n{g_lim}")
        return x_0, x_lim, g_lim, l_x_str

    def get_x_sol_in_real_phase(self) -> None:
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
        if con.FLAG_PHI_ABS:
            key = 'phi_0_abs'

        for i, cav in enumerate(self.comp['l_cav']):
            x_in_real_phase[i] = cav.acc_field.phi_0[key]
            # second half of the array remains untouched
        self.info['X_in_real_phase'] = x_in_real_phase

    def _select_objective(
            self, l_objectives: list[str, ...], l_scales: list[float, ...],
            **d_idx: dict[str, int]) -> tuple[Callable[[dict], np.ndarray],
                                              list[str, ...]]:
        """
        Select the objective to fit.

        Parameters
        ----------
        l_objectives : list of str
            Name of the objectives.
        l_scales : list of floats
            Scaling factor for each objective.
        **d_idx : dict[str, int]
            Holds indices of reference and objective position, where objective
            is evaluated.

        Returns
        -------
        fun_residual : Callable
            Takes a dict of results, returns the residuals between objective
            and results.
        l_f_str : list of str
            To output name of objective as well as position of evaluation.

        """
        # List of indexes for ref_lin object and results dictionary
        idx_ref, idx_brok = d_idx.values()

        # mismatch factor treated differently as it is already calculated from
        # two linacs
        exceptions = ['mismatch factor']

        # List of strings to output the objective names and positions
        l_f_str = [f"{d_markdown[key].replace('[deg]', '[rad]')} @idx{i_r}"
                   for i_r in idx_ref
                   for key in l_objectives]

        # We evaluate all the desired objectives
        l_ref = [self.ref_lin.get(key)[i_r]
                 if key not in exceptions
                 else self.ref_lin.get('twiss_zdelta')[i_r]
                 for i_r in idx_ref
                 for key in l_objectives]

        # TODO move to util/output
        info_str = "Objectives:\n"
        info_str +=\
                f"   {'Objective:':>35} | {'Scale:':>6} | {'Initial value'}\n"
        for i, (f_str, f_scale, ref) in enumerate(zip(l_f_str, l_scales,
                                                      l_ref)):
            info_str += f"{i}: {f_str:>35} | {f_scale:>6} | {ref}\n"
        logging.info(info_str)

        def fun_residual(results: dict) -> np.ndarray:
            """Compute difference between ref value and results dictionary."""
            i = -1
            residues = []
            for i_b in idx_brok:
                for key, scale in zip(l_objectives, l_scales):
                    i += 1

                    # mismatch factor
                    if key == 'mismatch factor':
                        mism = mismatch_factor(l_ref[i],
                                               results['twiss_zdelta'][i_b])[0]
                        residues.append(mism * scale)
                        continue

                    # all other keys
                    residues.append((l_ref[i] - results[key][i_b]) * scale)
            return np.array(residues)
        return fun_residual, l_f_str


def wrapper_lsq(arr_cav_prop: np.ndarray, fault: Fault,
                fun_residual: Callable[[dict], np.ndarray],
                phi_s_fit: bool) -> np.ndarray:
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

    if debugs['fit_progression'] and fault.count % 100 == 0:
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

    if debugs['fit_progression'] and fault.count % 200 == 0:
        debug.output_fit_progress(fault.count, arr_f, fault.info["l_F_str"])
    if debugs['plot_progression']:
        fault.info['hist_F'].append(arr_f)
    fault.count += 1

    return arr_f, results
