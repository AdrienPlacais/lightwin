#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:46:38 2023.

@author: placais

This module holds the class :class:`Fault`. Its purpose is to hold information
on a cavity failure and to fix it.

"""
import logging
from typing import Any

import config_manager as con

from core.elements.element import Element
from core.elements.field_map import FieldMap
from core.list_of_elements.list_of_elements import ListOfElements
from core.list_of_elements.helper import equivalent_elt
from core.list_of_elements.factory import (
    subset_of_pre_existing_list_of_elements
)
from beam_calculation.output import SimulationOutput

from failures.set_of_cavity_settings import SetOfCavitySettings

from optimisation.objective.objective import Objective
from optimisation.objective.factory import (
    get_objectives_and_residuals_function
)
from optimisation.design_space.variable import Variable
from optimisation.design_space.constraint import Constraint
from optimisation.design_space.factory import \
    get_design_space_and_constraint_function

from optimisation.algorithms.algorithm import OptimisationAlgorithm


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

    """

    def __init__(self,
                 reference_elts: ListOfElements,
                 reference_simulation_output: SimulationOutput,
                 files_from_full_list_of_elements: dict[str, Any],
                 wtf: dict[str, str | int | bool | list[str] | list[float]],
                 broken_elts: ListOfElements,
                 failed_cavities: list[FieldMap],
                 compensating_cavities: list[FieldMap],
                 ) -> None:
        """
        Create the Fault object.

        Parameters
        ----------
        reference_elts : ListOfElements
            List of elements of the reference linac. In particular, these
            elements hold the original cavity settings.
        reference_simulation_output : SimulationOutput
            Nominal simulation.
        files_from_full_list_of_elements : dict
            ``files`` attribute from the linac under fixing. Used to set
            calculation paths.
        wtf : dict[str, str | int | bool | list[str] | list[float]]
            What To Fit dictionary. Holds information on the fixing method.
        failed_cavities : list[FieldMap]
            Holds the failed cavities.
        compensating_cavities : list[FieldMap]
            Holds the compensating cavities.
        elts : list[Element]
            Holds the portion of the linac that will be computed again and
            again in the optimisation process. It is as short as possible, but
            must contain all altered cavities as well as the elements where
            objectives will be evaluated.

        """
        self.failed_cavities = failed_cavities
        self.compensating_cavities = compensating_cavities

        reference_cavities = [equivalent_elt(reference_elts, cavity)
                              for cavity in self.compensating_cavities]
        design_space = get_design_space_and_constraint_function(
            linac_name=con.LINAC,
            reference_cavities=reference_cavities,
            compensating_cavities=self.compensating_cavities,
            **wtf,
        )
        self.variables, self.constraints, self.compute_constraints = \
            design_space

        objective_preset = wtf['objective_preset']
        assert isinstance(objective_preset, str)
        elts_of_compensation_zone, self.objectives, self.compute_residuals = \
            get_objectives_and_residuals_function(
                linac_name=con.LINAC,
                objective_preset=objective_preset,
                reference_elts=reference_elts,
                reference_simulation_output=reference_simulation_output,
                broken_elts=broken_elts,
                failed_cavities=failed_cavities,
                compensating_cavities=compensating_cavities,
                )
        self.elts: ListOfElements = subset_of_pre_existing_list_of_elements(
            elts_of_compensation_zone,
            reference_simulation_output,
            files_from_full_list_of_elements,
        )

    def fix(self, optimisation_algorithm: OptimisationAlgorithm
            ) -> tuple[bool, SetOfCavitySettings, dict]:
        """
        Fix the Fault.

        Parameters
        ----------
        optimisation_algorithm : OptimisationAlgorithm
            The optimisation algorithm to be used, already initialized.

        Returns
        -------
        success : bool
            Indicates convergence of the :class:`OptimisationAlgorithm`.
        optimized_cavity_settings : SetOfCavitySettings
            Best cavity settings found by the :class:`OptimisationAlgorithm`.
        self.info : dict
            Useful information, such as the best solution.

        """
        outputs = optimisation_algorithm.optimise()
        success, optimized_cavity_settings, self.info = outputs
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
