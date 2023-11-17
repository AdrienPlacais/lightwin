#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to hold variables and constraints."""
from dataclasses import dataclass
from optimisation.design_space.constraint import Constraint

from optimisation.design_space.variable import Variable


@dataclass
class DesignSpace:
    """This class hold variables and constraints of an optimisation problem."""

    variables: list[Variable]
    constraints: list[Constraint]

    def to_file(self,
                filepath_variables: str,
                filepath_constraints: str,
                header: str | None = None,
                delimiter: str | None = None) -> None:
        """Save the variables and constraints in a ``.csv`` file.

        This file can be re-used to initialize another
        :class:`DesignSpaceFactory`.

        .. todo::
            Handle elements with different number of constraints (ex: QP and FM
            will not have the same!!!!)

        """
        variables, constraints = self.run()

        n_different_variables = self._check_dimensions(variables)
        # self._check_dimensions(constraints)

        # Sort by element name first, and then by variable name
        variables = sorted(variables,
                           key=lambda var: (var.element_name, var.name))

        # Create a 2d (nested) lists, where one line = one element and
        # one column = min, x0 or xmax of one variable
        lines = [
            # To convert Generator returned by flatten into a list
            list(
                # To have all values of all Variables in same list level
                flatten(
                    # Convert every Variable into a list of strings containing
                    # minimum value, (initial value), maximum value
                    map(lambda var: var.str_for_file(), vars_of_an_elt)
                )
            )
            # To have one line of lines == one element:
            for vars_of_an_elt in chunks(variables)
        ]
        with open(filepath_variables, 'w', encoding='utf-8') as file:
            for line in lines:
                file.write(line)

    def _to_file(self,
                 parameters: list[Variable] | list[Constraint],
                 filepath: str,
                 delimiter: str | None = None) -> None:
        n_different_parameters = self._check_dimensions(parameters)

        # Sort by element name first, and then by variable name
        sorted_parameters = sorted(parameters,
                                   key=lambda param: (param.element_name,
                                                      param.name))

        # Create a 2d (nested) lists, where one line = one element and
        # one column = min, (x0) or xmax of one parameter
        lines = [
            # To convert Generator returned by flatten into a list
            list(
                # To have all values of all Variables in same list level
                flatten(
                    # Convert every Contraint/Variable into a list of strings
                    # containing minimum value, (initial value), maximum value
                    map(lambda param: param.str_for_file(),
                        params_of_an_elt)
                )
            )
            # To have one line of lines == one element:
            for params_of_an_elt in chunks(sorted_parameters)
        ]
        head = [param.str_header1_for_file
                for param in sorted_parameters[:n_different_parameters]
                ]

    def _check_dimensions(self,
                          parameters: list[Variable] | list[Constraint]
                          ) -> int:
        """Ensure that all elements have the same number of var or const."""
        n_parameters = len(parameters)
        n_elements = len(self.compensating_elements)
        if n_parameters % n_elements != 0:
            raise NotImplementedError("As for now, all elements must have the "
                                      "same number of Variables "
                                      "(or Constraints).")
        n_different_parameters = n_parameters // n_elements
        return n_different_parameters
