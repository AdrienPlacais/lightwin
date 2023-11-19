#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We define factory and presets to handle variables, constraints, limits, etc..

.. note::
    If you add your own DesignSpaceFactory preset, do not forget to add it to
    the list of supported presets in :mod:`config.optimisation.design_space`.

.. todo::
    decorator to auto output the variables and constraints?

"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass


from optimisation.design_space.variable import Variable
from optimisation.design_space.constraint import Constraint
from optimisation.design_space.design_space import DesignSpace
from optimisation.design_space.helper import (same_value_as_nominal,
                                              LIMITS_CALCULATORS,
                                              )

from core.list_of_elements.helper import equivalent_elt
from core.elements.element import Element


# =============================================================================
# Factories / presets
# =============================================================================
@dataclass
class DesignSpaceFactory(ABC):
    """
    A base class to handle :class:`Variable` and :class:`Constraint` creation.

    Attributes
    ----------
    reference_elements : list[Element]
       All the elements with the reference setting.
    compensating_elements : list[Element]
        The elements from the linac under fixing that will be used for
        compensation.
    design_space_kw : dict[str, float | bool
        The entries of ``[design_space]`` in ``.ini`` file.

    """

    design_space_kw: dict[str, float | bool]

    def __post_init__(self):
        """Declare complementary variables."""
        self.filepath_variables: str
        self.filepath_constraints: str

    def _check_can_be_retuned(self, compensating_elements: list[Element]
                              ) -> None:
        """Check that given elements can be retuned."""
        assert all([elt.can_be_retuned for elt in compensating_elements])

    @property
    @abstractmethod
    def variables_names(self) -> tuple[str, ...]:
        """Return the name of the variables."""
        pass

    @property
    def constraints_names(self) -> tuple[str, ...]:
        """Return the name of the constraints."""
        return ()

    def use_files(self,
                  filepath_variables: str,
                  filepath_constraints: str | None = None) -> None:
        """Tell factory to generate design space from the provided files.

        Parameters
        ----------
        filepath_variables : str
            Path to the ``variables.csv`` file.
        filepath_constraints : str | None
            Path to the ``constraints.csv`` file. The default is None.

        """
        self.filepath_variables = filepath_variables
        if filepath_constraints is not None:
            self.filepath_constraints = filepath_constraints
        self.run = self._run_from_file

    def _run_variables(self,
                       compensating_elements: list[Element],
                       reference_elements: list[Element]) -> list[Variable]:
        """Set up all the required variables."""
        assert reference_elements is not None
        variables = []
        for var_name in self.variables_names:
            for element in compensating_elements:
                ref_elt = equivalent_elt(reference_elements, element)
                variable = Variable(
                    name=var_name,
                    element_name=str(element),
                    limits=self._get_limits_from_kw(var_name,
                                                    ref_elt,
                                                    reference_elements),
                    x_0=self._get_initial_value_from_kw(var_name, ref_elt),
                )
                variables.append(variable)
        return variables

    def _run_constraints(self,
                         compensating_elements: list[Element],
                         reference_elements: list[Element]
                         ) -> list[Constraint]:
        """Set up all the required constraints."""
        assert reference_elements is not None
        constraints = []
        for constraint_name in self.constraints_names:
            for element in compensating_elements:
                ref_elt = equivalent_elt(reference_elements, element)
                constraint = Constraint(
                    name=constraint_name,
                    element_name=str(element),
                    limits=self._get_limits_from_kw(constraint_name,
                                                    ref_elt,
                                                    reference_elements),
                )
                constraints.append(constraint)
        return constraints

    def run(self,
            compensating_elements: list[Element],
            reference_elements: list[Element]) -> DesignSpace:
        """Set up variables and constraints."""
        self._check_can_be_retuned(compensating_elements)
        variables = self._run_variables(compensating_elements,
                                        reference_elements)
        constraints = self._run_constraints(compensating_elements,
                                            reference_elements)
        design_space = DesignSpace(variables, constraints)
        logging.info(str(design_space))
        return design_space

    def _get_initial_value_from_kw(self,
                                   variable: str,
                                   reference_element: Element) -> float:
        """Select initial value for given variable.

        The default behavior is to return the value of ``variable`` from
        ``reference_element``, which is a good starting point for optimisation.

        Parameters
        ----------
        variable : {'k_e', 'phi_0_rel', 'phi_0_abs', 'phi_s'}
            The variable from which you want the limits.
        reference_element : Element
            The element in its nominal tuning.

        Returns
        -------
        float
            Initial value.

        """
        return same_value_as_nominal(variable, reference_element)

    def _get_limits_from_kw(self,
                            variable: str,
                            reference_element: Element,
                            reference_elements: list[Element],
                            ) -> tuple[float, float]:
        """
        Select limits for given variable.

        Call this method for classic limits.

        Parameters
        ----------
        variable : {'k_e', 'phi_0_rel', 'phi_0_abs', 'phi_s'}
            The variable from which you want the limits.
        reference_element : Element
            The element in its nominal tuning.
        reference_elements : list[Element]
            List of reference elements.

        Returns
        -------
        tuple[float | None]
            Lower and upper limit for current variable.

        """
        assert reference_elements is not None
        limits_calculator = LIMITS_CALCULATORS[variable]
        return limits_calculator(reference_element=reference_element,
                                 reference_elements=reference_elements,
                                 **self.design_space_kw)

    def _run_from_file(self,
                       compensating_elements: list[Element],
                       variables_names: tuple[str, ...],
                       constraints_names: tuple[str, ...] | None = None,
                       **kwargs: str,
                       ) -> DesignSpace:
        """Use the :meth:`.DesignSpace.from_files` constructor.

        Parameters
        ----------
        variables_names : tuple[str, ...]
            Name of the variables to create.
        constraints_names : tuple[str, ...] | None, optional
            Name of the constraints to create. The default is None.

        Returns
        -------
        DesignSpace

        """
        self._check_can_be_retuned(compensating_elements)
        assert 'filepath_variables' in self.__dir__()
        filepath_constraints = getattr(self, 'filepath_constraints', None)

        elements_names = tuple([str(elt)
                                for elt in compensating_elements])
        design_space = DesignSpace.from_files(elements_names,
                                              self.filepath_variables,
                                              variables_names,
                                              filepath_constraints,
                                              constraints_names,
                                              **kwargs,
                                              )
        return design_space


@dataclass
class Unconstrained(DesignSpaceFactory):
    """Factory to set amplitude and phase of elements, no phi_s."""

    variables_names: tuple[str, str] = ('phi_0_abs', 'k_e')


@dataclass
class ConstrainedSyncPhase(DesignSpaceFactory):
    """Factory to set k_e and phase of elements, with phi_s constraint."""

    variables_names: tuple[str, str] = ('phi_0_abs', 'k_e')
    constraints_names: tuple[str] = ('phi_s', )


@dataclass
class SyncPhaseAsVariable(DesignSpaceFactory):
    """Factory to set k_e and phi_s of elements, no constraint."""

    variables_names: tuple[str, str] = ('phi_s', 'k_e')


# should not exist
@dataclass
class FM4_MYRRHA(DesignSpaceFactory):
    """Factory to set reduce design space around a pre-existing solution."""

    variables_names: tuple[str, str] = ('phi_0_abs', 'k_e')

    def __post_init__(self):
        """Check that we are in the proper case."""
        super().__post_init__()

    def run(self, compensating_elements: list[Element], *args, **kwargs
            ) -> DesignSpace:
        """Classic run but check name of compensating elements first."""
        assert [str(elt) for elt in compensating_elements] == [
            'FM1', 'FM2', 'FM3', 'FM5', 'FM6']
        return super().run(*args,
                           compensating_elements=compensating_elements,
                           **kwargs)

    def _run_variables(self, compensating_elements: list[Element]
                       ) -> list[Variable]:
        """Set up all the required variables."""
        variables = []
        my_initial_values = {'phi_0_abs': {'FM1': 1.2428429564125352,
                                           'FM2': 5.849758478693384,
                                           'FM3': 1.370628110261926,
                                           'FM5': 3.323382937071699,
                                           'FM6': 2.611163043271624
                                           },
                             'k_e': {'FM1': 1.614713,
                                     'FM2': 1.607485,
                                     'FM3': 1.9268,
                                     'FM5': 1.942578,
                                     'FM6': 1.851571,
                                     }
                             }
        tol = 1e-3

        for var_name in self.variables_names:
            for element in compensating_elements:
                my_initial_value = my_initial_values[var_name][str(element)]
                variable = Variable(
                    name=var_name,
                    element_name=str(element),
                    x_0=my_initial_value,
                    limits=(my_initial_value - tol, my_initial_value + tol),
                )
                variables.append(variable)
        return variables


DESIGN_SPACE_FACTORY_PRESETS = {
    'unconstrained': Unconstrained,
    'constrained_sync_phase': ConstrainedSyncPhase,
    'sync_phase_as_variable': SyncPhaseAsVariable,
    'FM4_MYRRHA': FM4_MYRRHA,
}  #:


def get_design_space_factory(design_space_preset: str,
                             **design_space_kw: float | bool
                             ) -> DesignSpaceFactory:
    """Select proper factory, instantiate it and return it.

    Parameters
    ----------
    design_space_preset : str
        design_space_preset
    design_space_kw : float | bool
        design_space_kw

    Returns
    -------
    DesignSpaceFactory

    """
    design_space_factory_class = DESIGN_SPACE_FACTORY_PRESETS[
        design_space_preset]
    design_space_factory = design_space_factory_class(
        design_space_kw=design_space_kw,
    )
    return design_space_factory
