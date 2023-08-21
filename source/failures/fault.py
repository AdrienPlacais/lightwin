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

import config_manager as con

from core.elements import _Element, FieldMap
from core.list_of_elements import ListOfElements
from core.list_of_elements_factory import (
    subset_of_pre_existing_list_of_elements
)
from beam_calculation.output import SimulationOutput

from failures.set_of_cavity_settings import SetOfCavitySettings

from optimisation.parameters.objective import Objective
from optimisation.parameters.variable import Variable
from optimisation.parameters.constraint import Constraint
from optimisation.parameters.factories import (
    variable_constraint_objective_factory
)

from optimisation.algorithms.least_squares import LeastSquares
from optimisation.algorithms.least_squares_penalty import LeastSquaresPenalty
from optimisation.algorithms.nsga import NSGA


class Fault:
    """
    To handle and fix a single failure.

    Attributes
    ----------
    failed_cavities : list[FieldMap]
        Holds the failed cavities.
    compensating_cavities : list[FieldMap]
        Holds the compensating cavities.
    elts : ListOfElements
        Holds the portion of the linac that will be computed again and again in
        the optimisation process. It is as short as possible, but must contain
        all `failed_cavities`, `compensating_cavities` and
        `elt_eval_objectives`.
    variables : list[Variable]
        Holds information on the optimisation variables.
    constraints : list[Constraint] | None
        Holds infomation on the optimisation constraints.

    Methods
    -------
    compute_constraints : Callable[[SimulationOutput], np.ndarray] | None
        Compute the constraint violation for a given `SimulationOutput`.
    compute_residuals : Callable[[SimulationOutput], np.ndarray]
        A function that takes in a `SimulationOutput` and returns the residues
        of every objective w.r.t the reference one.
    fix : Callable[[Callable], tuple[bool, SetOfCavitySettings, dict]]
        Creates the `OptimisationAlgorithm` object and fix the fault. Needs the
        `run_with_this` method from the proper `BeamCalculator` as argument.
    update_cavities_status : Callable[[None], None]
        Change the `status` of the cavities at the start and the end of the
        optimisation process. Changing the cavities status can modify the
        `FieldMap` objects. In particular, `k_e` is set to 0. when a cavity is
        broken. Also updates the `.dat` file.
    get_x_sol_in_real_phase : Callable[[None], None]
        Set phi_0_abs or phi_0_rel from the given phi_s.

    """

    def __init__(self,
                 reference_elts: ListOfElements,
                 reference_simulation_output: SimulationOutput,
                 files_from_full_list_of_elements: dict[str, str | list[str]],
                 wtf: dict[str, str | int | bool | list[str] | list[float]],
                 failed_cavities: list[FieldMap],
                 compensating_cavities: list[FieldMap],
                 elt_eval_objectives: list[_Element],
                 elts: list[_Element]
                 ) -> None:
        """
        Create the Fault object.

        Parameters
        ----------
        reference_elts : ListOfElements
            `ListOfElements` from reference linac, holding in particular the
            original cavity settings.
        reference_simulation_output : SimulationOutput
            Nominal simulation.
        files_from_full_list_of_elements : dict
            `files` attribute from the linac under fixing. Used to set
            calculation paths.
        wtf : dict[str, str | int | bool | list[str] | list[float]]
            What To Fit dictionary. Holds information on the fixing method.
        failed_cavities : list[FieldMap]
            Holds the failed cavities.
        compensating_cavities : list[FieldMap]
            Holds the compensating cavities.
        elt_eval_objectives : list[_Element]
            `_Element`s at which exit objectives are evaluated.
        elts : list[_Element]
            Holds the portion of the linac that will be computed again and
            again in the optimisation process. It is as short as possible, but
            must contain all `failed_cavities`, `compensating_cavities` and
            `elt_eval_objectives`.

        """
        self.failed_cavities = failed_cavities
        self.compensating_cavities = compensating_cavities

        self.elts: ListOfElements
        self.elts = subset_of_pre_existing_list_of_elements(
            elts,
            reference_simulation_output,
            files_from_full_list_of_elements,
        )

        args = variable_constraint_objective_factory(
                preset=con.LINAC,
                reference_elts=reference_elts,
                reference_simulation_output=reference_simulation_output,
                elements_eval_objective=elt_eval_objectives,
                compensating_cavities=self.compensating_cavities,
                wtf=wtf,
                phi_abs=con.FLAG_PHI_ABS
            )
        self.variables: list[Variable] = args[0]
        self.constraints: list[Constraint] = args[1]
        self.compute_constraints = args[2]
        self.objectives: list[Objective] = args[3]
        self.compute_residuals = args[4]

        algorithms = {
            'least_squares': LeastSquares,
            'least_squares_penalty': LeastSquaresPenalty,
            'nsga': NSGA}
        self._algorithm_class = algorithms[wtf['opti method']]

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
        compute_beam_propagation = partial(beam_calculator_run_with_this,
                                           elts=self.elts)

        algorithm = self._algorithm_class(
            compute_beam_propagation=compute_beam_propagation,
            objectives=self.objectives,
            compute_residuals=self.compute_residuals,
            compensating_cavities=self.compensating_cavities,
            elts=self.elts,
            variables=self.variables,
            constraints=self.constraints,
            compute_constraints=self.compute_constraints,
        )
        self.algorithm_instance = algorithm
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
            cavities = self.failed_cavities + self.compensating_cavities
            status = ['failed' for cav in self.failed_cavities]
            status += ['compensate (in progress)'
                       for cav in self.compensating_cavities]

            if {cav.get('status') for cav in cavities} != {'nominal'}:
                logging.error("At least one compensating or failed cavity is "
                              + "already compensating or faulty, probably "
                              + "in another Fault object. Updating its status "
                              + "anyway...")

        elif optimisation == 'finished':
            assert success is not None

            cavities = self.compensating_cavities
            status = ['compensate (ok)' for cav in cavities]
            if not success:
                status = ['compensate (not ok)' for cav in cavities]

        for cav, stat in zip(cavities, status):
            cav.update_status(stat)
        self.elts.store_settings_in_dat(self.elts.files['dat_filepath'],
                                        save=True)

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

        for i, cav in enumerate(self.compensating_cavities):
            x_in_real_phase[i] = cav.acc_field.phi_0[key]
            # second half of the array remains untouched
        self.info['X_in_real_phase'] = x_in_real_phase
