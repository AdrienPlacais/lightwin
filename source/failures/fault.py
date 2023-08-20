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

from failures.set_of_cavity_settings import SetOfCavitySettings

from optimisation.parameters.variable import Variable
from optimisation.parameters.constraint import Constraint
from optimisation.parameters.objective import Objective
from optimisation.parameters.factories import (variable_factory,
                                               constraint_factory,
                                               objective_factory)

from optimisation.algorithms.least_squares import LeastSquares
from optimisation.algorithms.nsga import NSGA

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
        files_from_full_list_of_elements = self.fix_acc.elts.files
        elts = subset_of_pre_existing_list_of_elements(
            elts,
            simulation_output,
            files_from_full_list_of_elements)
        return elts

    def fix(self, beam_calculator_run_with_this: Callable[
        [SetOfCavitySettings, ListOfElements], SimulationOutput]
            ) -> tuple[bool, SetOfCavitySettings, dict]:
        """
        Fix the Fault.

        Parameters
        ----------
        beam_calculator_run_with_this : Callable[[
                SetOfCavitySettings, ListOfElements], SimulationOutput]
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
        variables, constraints = self._set_design_space()

        solv1 = list(self.ref_acc.simulation_outputs.keys())[0]
        reference_simulation_output = self.ref_acc.simulation_outputs[solv1]

        objectives, compute_residuals = objective_factory(
            names=self.wtf['objective'],
            scales=self.wtf['scale objective'],
            elements=self.elt_eval_objectives,
            reference_simulation_output=reference_simulation_output,
            positions=None)

        compute_beam_propagation = partial(beam_calculator_run_with_this,
                                           elts=self.elts)

        algorithm = LeastSquares(
            variables=variables,
            constraints=constraints,
            compute_beam_propagation=compute_beam_propagation,
            compute_residuals=compute_residuals,
            compensating_cavities=self.comp_cav,
            variable_names=self.variable_names,
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
        self.elts.store_settings_in_dat(self.elts.files['dat_filepath'],
                                        save=True)

    def _set_design_space(self) -> tuple[list[Variable], list[Constraint]]:
        """
        Set initial conditions and boundaries for the fit.

        In the returned arrays, first half of components are initial phases
        phi_0, while second half of components are norms.

        Returns
        -------
        variables : list[Variable]
            Holds variables, their initial values, their limits.
        constraints : list[Constraint]
            Holds constraints and their limits.

        """
        variable_names = ['phi_0_rel', 'k_e']
        if con.FLAG_PHI_ABS:
            variable_names = ['phi_0_abs', 'k_e']
        if self.wtf['phi_s fit']:
            variable_names = ['phi_s', 'k_e']

        global_compensation = 'global' in self.wtf['strategy']
        variables = variable_factory(preset=con.LINAC,
                                     variable_names=variable_names,
                                     compensating_cavities=self.comp_cav,
                                     ref_elts=self.ref_acc.elts,
                                     global_compensation=global_compensation)
        # FIXME should not be initialized if not used
        # FIXME not clean
        self.variable_names = variable_names

        constraint_names = ['phi_s']
        constraints = constraint_factory(preset=con.LINAC,
                                         constraint_names=constraint_names,
                                         compensating_cavities=self.comp_cav,
                                         ref_elts=self.ref_acc.elts)
        return variables, constraints

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
