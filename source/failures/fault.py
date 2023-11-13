#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds the class :class:`Fault`.

It's purpose is to hold information on a failure and to fix it.

"""
import logging
from typing import Any

import config_manager as con

from core.elements.element import Element

from core.beam_parameters.factory import InitialBeamParametersFactory
from core.beam_parameters.initial_beam_parameters import InitialBeamParameters

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
    failed_elements : list[Element]
        Holds the failed elements.
    compensating_elements : list[Element]
        Holds the compensating elements.
    elts : ListOfElements
        Holds the portion of the linac that will be computed again and again in
        the optimisation process. It is as short as possible, but must contain
        all `failed_elements`, `compensating_elements` and
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
                 failed_elements: list[Element],
                 compensating_elements: list[Element],
                 initial_beam_parameters_factory: InitialBeamParametersFactory,
                 ) -> None:
        """
        Create the Fault object.

        Parameters
        ----------
        reference_elts : ListOfElements
            List of elements of the reference linac. In particular, these
            elements hold the original element settings.
        reference_simulation_output : SimulationOutput
            Nominal simulation.
        files_from_full_list_of_elements : dict
            ``files`` attribute from the linac under fixing. Used to set
            calculation paths.
        wtf : dict[str, str | int | bool | list[str] | list[float]]
            What To Fit dictionary. Holds information on the fixing method.
        failed_elements : list[Element]
            Holds the failed elements.
        compensating_elements : list[Element]
            Holds the compensating elements.
        elts : list[Element]
            Holds the portion of the linac that will be computed again and
            again in the optimisation process. It is as short as possible, but
            must contain all altered elements as well as the elements where
            objectives will be evaluated.

        """
        assert all([element.can_be_retuned for element in failed_elements])
        self.failed_elements = failed_elements
        assert all([element.can_be_retuned
                    for element in compensating_elements])
        self.compensating_elements = compensating_elements

        reference_elements = [equivalent_elt(reference_elts, element)
                              for element in self.compensating_elements]
        design_space = get_design_space_and_constraint_function(
            linac_name=con.LINAC,
            reference_elements=reference_elements,
            compensating_elements=self.compensating_elements,
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
                failed_elements=failed_elements,
                compensating_elements=compensating_elements,
                )

        self.elts: ListOfElements = subset_of_pre_existing_list_of_elements(
            elts_of_compensation_zone,
            reference_simulation_output,
            files_from_full_list_of_elements,
            initial_beam_parameters_factory,
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

    def update_elements_status(self, optimisation: str,
                               success: bool | None = None) -> None:
        """Update status of compensating and failed elements."""
        if optimisation not in ['not started', 'finished']:
            logging.error("{optimisation =} not understood. Not changing any "
                          "status...")
            return

        if optimisation == 'not started':
            elements = self.failed_elements + self.compensating_elements
            status = ['failed' for cav in self.failed_elements]
            status += ['compensate (in progress)'
                       for cav in self.compensating_elements]

            if {cav.get('status') for cav in elements} != {'nominal'}:
                logging.error("At least one compensating or failed element is "
                              "already compensating or faulty, probably in"
                              "another Fault object. Updating its status "
                              "anyway...")

        elif optimisation == 'finished':
            assert success is not None

            elements = self.compensating_elements
            status = ['compensate (ok)' for _ in elements]
            if not success:
                status = ['compensate (not ok)' for _ in elements]

        for cav, stat in zip(elements, status):
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
        # First half of X array: phase of elements (relative or synchronous
        # according to the value of wtf['phi_s fit']).
        # Second half is the norms of elements
        x_in_real_phase = self.info["X"].copy()

        key = 'phi_0_rel'
        if con.FLAG_PHI_ABS:
            key = 'phi_0_abs'

        for i, cav in enumerate(self.compensating_elements):
            x_in_real_phase[i] = cav.acc_field.phi_0[key]
            # second half of the array remains untouched
        self.info['X_in_real_phase'] = x_in_real_phase
