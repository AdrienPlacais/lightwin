#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:46:38 2023.

@author: placais
"""
import logging
from typing import Callable
import numpy as np

import config_manager as con
from util.dicts_output import d_markdown
from core.elements import _Element, FieldMap
from core.list_of_elements import ListOfElements
from core.accelerator import Accelerator
from core.emittance import mismatch_factor
from optimisation.variables import initial_value, limits, constraints


class MyFault:
    """To handle and fix a single Fault."""

    def __init__(self, ref_acc: Accelerator, fix_acc: Accelerator, wtf: dict,
                 failed_cav: list[FieldMap], comp_cav: list[FieldMap],
                 elts: list[_Element], idx_eval_objectives: list[int]):
        """Init."""
        self.ref_acc, self.fix_acc = ref_acc, fix_acc
        self.wtf = wtf
        self.failed_cav, self.comp_cav = failed_cav, comp_cav
        self.elts = self._create_list_of_elements(elts)
        self.idx_eval_objectives = idx_eval_objectives

        self.fit_info = {
            'X': [],                # Solution
            'X_0': [],              # Initial guess
            'X_lim': [],            # Bounds
            'X_info': [],           # Name of variables for output
            'X_in_real_phase': [],  # See get_x_sol_in_real_phase
            'F': [],                # Final objective values
            'hist_F': [],           # Objective evaluations
            'F_info': [],           # Name of objectives for output
            'G': [],                # Constraints
            'resume': None,         # For output
        }

    def _create_list_of_elements(self, elts: list[_Element]
                                 ) -> ListOfElements:
        """Create the proper ListOfElements object."""
        idx = elts[0].get('s_in')
        w_kin = self.fix_acc.get('w_kin')[idx]
        phi_abs = self.fix_acc.get('phi_abs')[idx]
        tm_cumul = self.fix_acc.get('tm_cumul')[idx]
        elts = ListOfElements(elts, w_kin, phi_abs, idx_in=idx,
                              tm_cumul=tm_cumul)
        return elts

    def fix(self):
        """Fix the Fault."""
        self._init_status()
        x_0, x_lim, constr, x_info = self._set_design_space()
        compute_residuals, info_objectives = self._select_objective()

        self.fit_info.update({
            'X_0': x_0,
            'X_lim': x_lim,
            'X_info': x_info,
            'F_info': info_objectives,
            'G': constr,
        })

    def _init_status(self):
        """Update status of compensating and failed cavities."""
        for cav, status in zip([self.failed_cav, self.comp_cav],
                               ['failed', 'compensate (in progress)']):
            if cav.get('status') != 'nominal':
                logging.warning(f"Cavity {cav} is already used for another "
                                + "purpose. I will continue anyway...")
            cav.update_status(status)

    # TODO may be simpler
    def _set_design_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                         list[str]]:
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
        x_info : list of str
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

        ref_acc = self.ref_acc
        x_0, x_lim, x_info = [], [], []
        for obj in l_x:
            for cav in self.comp_cav:
                ref_cav = ref_acc.equiv_elt(cav)

                args = (con.LINAC, cav, ref_cav, ref_acc)
                x_0.append(initial_value(obj, ref_cav))
                x_lim.append(limits(obj, *args))
                x_info.append(' '.join((cav.get('elt_name', to_numpy=False),
                                        d_markdown[obj])))
        g_lim = []
        for const in l_g:
            for cav in self.comp_cav:
                ref_cav = ref_acc.equiv_elt(cav)

                args = (con.LINAC, cav, ref_cav, ref_acc)
                g_lim.append(constraints(const, *args))

        x_0 = np.array(x_0)
        x_lim = np.array(x_lim)
        g_lim = np.array(g_lim)

        logging.info("Design space (handled in "
                     + "optimisation.linacs_design_space, not .ini):\n"
                     + f"Initial guess:\n{x_0}\n"
                     + f"Bounds:\n{x_lim}\n"
                     + f"Constraints (not necessarily used):\n{g_lim}")
        return x_0, x_lim, g_lim, x_info

    def _select_objective(self) -> tuple[Callable, list[str]]:
        """Set optimisation objective."""
        objectives = self.wtf['objective']
        scales = self.wtf['scale objective']
        idx_eval_fix = [i - self.elts[0].idx['s_in']
                        for i in self.idx_eval_objectives]

        # List of strings to output the objective names and positions
        info_objectives = [
            f"{d_markdown[key].replace('[deg]', '[rad]')} @idx{i}"
            for i in self.idx_eval_objectives
            for key in objectives]

        # We evaluate all the desired objectives
        exceptions = ['mismatch factor']
        objectives = [self.ref_acc.get(key)[i]
                      if key not in exceptions
                      else self.ref_acc.get('twiss_zdelta')[i]
                      for i in self.idx_eval_objectives
                      for key in objectives]

        # TODO move to util/output
        output = "Objectives:\n"
        output +=\
            f"   {'Objective:':>35} | {'Scale:':>6} | {'Initial value'}\n"
        for i, (info, scale, objective) in enumerate(
                zip(info_objectives, scales, objectives)):
            output += f"{i}: {info:>35} | {scale:>6} | {objective}\n"
        logging.info(output)

        def compute_residuals(results: dict) -> np.ndarray:
            """Compute difference between ref value and results dictionary."""
            i_ref = -1
            residues = []
            for i_fix in idx_eval_fix:
                for key, scale in zip(objectives, scales):
                    i_ref += 1

                    # mismatch factor
                    if key == 'mismatch factor':
                        mism = mismatch_factor(
                            objectives[i_ref],
                            results['twiss_zdelta'][i_fix])[0]
                        residues.append(mism * scale)
                        continue

                    residues.append(
                        (objectives[i_ref] - results[key][i_fix]) * scale)
            return np.array(residues)

        return compute_residuals, info_objectives
