#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:46:38 2023.

@author: placais

This module holds the class `Fault`. Its purpose is to hold information on a
cavity failure and to fix it.
"""
import logging
from collections.abc import Callable
from functools import partial

import numpy as np

import config_manager as con

from core.elements import _Element, FieldMap
from core.list_of_elements import ListOfElements
from core.list_of_elements_factory import (
    subset_of_pre_existing_list_of_elements
)
from core.accelerator import Accelerator
from core.beam_parameters import mismatch_from_arrays

from beam_calculation.output import SimulationOutput

from failures.variables import VariablesAndConstraints
from failures.set_of_cavity_settings import SetOfCavitySettings

from algorithms.least_squares import LeastSquares

from util.dicts_output import markdown


class Fault:
    """To handle and fix a single Fault."""

    def __init__(self,
                 ref_acc: Accelerator,
                 fix_acc: Accelerator,
                 wtf: dict[str, str | int | bool | list[str] | list[float]],
                 failed_cav: list[FieldMap],
                 comp_cav: list[FieldMap],
                 elt_eval_objectives: list[_Element],
                 elts: list[_Element]
                 ) -> None:
        """
        Create the Fault object.

        Parameters
        ----------
        ref_acc : Accelerator
            The reference `Accelerator` (nominal `Accelerator`).
        fix_acc : Accelerator
            The broken `Accelerator` to be fixed.
        wtf : dict[str, str | int | bool | list[str] | list[float]]
            What To Fit dictionary. Holds information on the fixing method.
        failed_cav : list[FieldMap]
            Holds the failed cavities.
        comp_cav : list[FieldMap]
            Holds the compensating cavities.
        elt_eval_objectives : list[_Element]
            `_Element`s at which exit objectives are evaluated.
        elts : list[_Element]
            Holds the portion of the linac that will be computed again and
            again in the optimisation process. It is as short as possible, but
            must contain all `failed_cav`, `comp_cav` and
            `elt_eval_objectives`.

        """
        self.ref_acc, self.fix_acc = ref_acc, fix_acc
        self.wtf = wtf
        self.failed_cav, self.comp_cav = failed_cav, comp_cav
        self.elts = self._create_list_of_elements(elts)
        self.elt_eval_objectives = elt_eval_objectives

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

    def _create_list_of_elements(self, elts: list[_Element]) -> ListOfElements:
        """
        Create a `ListOfElements` object from a list of `_Element` objects.

        We also use the `SimulationOutput` that was calculated with the first
        solver, on the full linac `ListOfElements`.

        """
        first_solver = list(self.ref_acc.simulation_outputs.keys())[0]

        simulation_output = self.ref_acc.simulation_outputs[first_solver]
        files_from_full_list_of_elements = self.ref_acc.elts.files
        elts = subset_of_pre_existing_list_of_elements(
            elts,
            simulation_output,
            files_from_full_list_of_elements)
        return elts

    def fix(self, beam_calculator_run_with_this: Callable[
            [SetOfCavitySettings, ListOfElements, bool], SimulationOutput]
            ) -> tuple[bool, SetOfCavitySettings, dict]:
        """
        Fix the Fault.

        Parameters
        ----------
        beam_calculator_run_with_this : Callable[[
                SetOfCavitySettings, ListOfElements, bool], SimulationOutput]
            The `run_with_this` method from a `BeamCalculator` object.

        Returns
        -------
        success : bool
            Indicates convergence of the optimisation `Algorithm`.
        optimized_cavity_settings : SetOfCavitySettings
            Best cavity settings found by the optimization `Algorithm`.
        self.info : dict
            Useful information, such as the best solution.

        """
        variables_constraints = self._set_design_space()
        compute_residuals, info_objectives = self._select_objective()
        compute_beam_propagation = partial(beam_calculator_run_with_this,
                                           elts=self.elts)

        algorithm = LeastSquares(
            variables_constraints=variables_constraints,
            compute_residuals=compute_residuals,
            compute_beam_propagation=compute_beam_propagation,
            compensating_cavities=self.comp_cav,
            variable_names=self.variable_names,
            phi_s_fit=self.wtf['phi_s fit'],
            elts=self.elts)
        success, optimized_cavity_settings, self.info = algorithm.optimise()
        return success, optimized_cavity_settings, self.info

    def update_cavities_status(self, optimisation: str,
                               success: bool | None = None) -> None:
        """Update status of compensating and failed cavities."""
        if optimisation not in ['not started', 'finished']:
            logging.error("{optimisation =} not understood. Not changing any "
                          + "status...")
            return

        if optimisation == 'not started':
            cavities = self.failed_cav + self.comp_cav
            status = ['failed' for cav in self.failed_cav]
            status += ['compensate (in progress)' for cav in self.comp_cav]

            if {cav.get('status') for cav in cavities} != {'nominal'}:
                logging.error("At least one compensating or failed cavity is "
                              + "already compensating or faulty, probably "
                              + "in another Fault object. Updating its status "
                              + "anyway...")

        elif optimisation == 'finished':
            assert success is not None

            cavities = self.comp_cav
            status = ['compensate (ok)' for cav in cavities]
            if not success:
                status = ['compensate (not ok)' for cav in cavities]

        for cav, stat in zip(cavities, status):
            cav.update_status(stat)

    def _set_design_space(self) -> VariablesAndConstraints:
        """
        Set initial conditions and boundaries for the fit.

        In the returned arrays, first half of components are initial phases
        phi_0, while second half of components are norms.

        Returns
        -------
        variables_constraints : VariablesAndConstraints
            Holds variables, their initial values, their limits, and
            constraints.

        """
        variables = ['phi_0_rel', 'k_e']
        if con.FLAG_PHI_ABS:
            variables = ['phi_0_abs', 'k_e']
        if self.wtf['phi_s fit']:
            variables = ['phi_s', 'k_e']

        constraints = ['phi_s']
        # FIXME should not be initialized if not used

        # FIXME not clean
        self.variable_names = variables

        global_compensation = 'global' in self.wtf['strategy']
        kwargs = {'global_compensation': global_compensation}

        variables_constraints = VariablesAndConstraints(
            con.LINAC, self.ref_acc, self.comp_cav, variables, constraints,
            **kwargs)

        logging.info("Design space (handled in failures.variables, not "
                     f".ini):\n{variables_constraints}")
        return variables_constraints

    def _select_objective(self) -> tuple[Callable, list[str]]:
        """Set optimisation objective."""
        objectives = self.wtf['objective']
        scales = self.wtf['scale objective']

        info_objectives = [
            f"{markdown[key].replace('[deg]', '[rad]')} @{elt = }"
            for elt in self.elt_eval_objectives
            for key in objectives]

        objectives_values = [
            self.ref_acc.get(key, elt=elt, pos='out')
            if 'mismatch_factor' not in key
            else self.ref_acc.get('twiss', elt=elt, pos='out',
                                  phase_space='zdelta')
            for elt in self.elt_eval_objectives
            for key in objectives]

        # TODO move to util/output
        output = "Objectives:\n"
        output +=\
            f"   {'Objective:':>35} | {'Scale:':>6} | {'Initial value'}\n"
        for i, (info, scale, objective) in enumerate(
                zip(info_objectives, scales, objectives_values)):
            output += f"{i}: {info:>35} | {scale:>6} | {objective}\n"
        logging.info(output)

        def compute_residuals(simulation_output: SimulationOutput
                              ) -> np.ndarray:
            """Compute difference between ref value and results dictionary."""
            i_ref = -1
            residues = []
            for elt in self.elt_eval_objectives:
                for key, scale in zip(objectives, scales):
                    i_ref += 1

                    if key == 'mismatch_factor_zdelta':
                        mism = mismatch_from_arrays(
                            objectives_values[i_ref],
                            simulation_output.get('twiss', elt=elt, pos='out',
                                                  phase_space='zdelta'))[0]
                        residues.append(mism * scale)
                        continue

                    residues.append(
                        (objectives_values[i_ref]
                         - simulation_output.get(key, elt=elt, pos='out'))
                        * scale)
            return np.array(residues)

        return compute_residuals, info_objectives

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

        for i, cav in enumerate(self.comp_cav):
            x_in_real_phase[i] = cav.acc_field.phi_0[key]
            # second half of the array remains untouched
        self.info['X_in_real_phase'] = x_in_real_phase
