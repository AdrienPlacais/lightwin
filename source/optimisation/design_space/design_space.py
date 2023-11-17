#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to hold variables and constraints."""
import logging
from typing import Any
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import pandas as pd

from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from optimisation.design_space.constraint import Constraint
from optimisation.design_space.design_space_parameter \
    import DesignSpaceParameter

from optimisation.design_space.variable import Variable


@dataclass
class DesignSpace:
    """This class hold variables and constraints of an optimisation problem."""

    variables: list[Variable]
    constraints: list[Constraint]

    def compute_constraints(self, simulation_output: SimulationOutput
                            ) -> np.ndarray:
        """Compute constraint violation for ``simulation_output``."""
        constraints_with_tuples = [constraint.evaluate(simulation_output)
                                   for constraint in self.constraints]
        constraint_violation = [
            single_constraint
            for constraint_with_tuples in constraints_with_tuples
            for single_constraint in constraint_with_tuples
            if ~np.isnan(single_constraint)
        ]
        return np.array(constraint_violation)

    def __str__(self) -> str:
        """Give nice output of the variables and constraints."""
        return '\n'.join(self._str_variables() + self._str_constraints())

    def _str_variables(self) -> list[str]:
        """Generate information on the variables that were created."""
        info = [str(variable) for variable in self.variables]
        info.insert(0, "Created variables:")
        info.insert(1, "=" * 100)
        info.insert(2, Variable.header_of__str__())
        info.insert(3, "-" * 100)
        info.append("=" * 100)
        return info

    def _str_constraints(self) -> list[str]:
        """Generate information on the constraints that were created."""
        info = [str(constraint) for constraint in self.constraints]
        info.insert(0, "Created constraints:\n")
        info.insert(1, "=" * 100)
        info.insert(2, Constraint.header_of__str__())
        info.insert(3, "-" * 100)
        info.append("=" * 100)
        return info

    def to_pandas_dataframe(self) -> pd.DataFrame:
        """Convert list of variables to a pandas dataframe."""
        to_get = ('element_name', 'x_0', 'limits')
        dicts = [var.to_dict(*to_get) for var in self.variables]
        return pd.DataFrame(dicts, columns=to_get)

    def to_file(self,
                parameters: list[DesignSpaceParameter],
                filepath: str,
                delimiter: str = ',',
                **to_csv_kw: dict[str, Any],
                ) -> None:
        """Save all the design space parameters in a compact file.

        Parameters
        ----------
        parameters : list[DesignSpaceParameter]
            All the defined parameters.
        filepath : str
            Where file will be stored.
        delimiter : str
            Delimiter between two columns. The default is ','.

        """
        elements_and_parameters = _gather_dicts_by_key(parameters,
                                                       'element_name')
        lines = [self._parameters_to_single_file_line(name, param)
                 for name, param in elements_and_parameters.items()]
        as_df = pd.DataFrame(lines, columns=list(lines[0].keys()))
        as_df.to_csv(filepath, sep=delimiter, **to_csv_kw)
        logging.info(f"Variables or Contraints saved in {filepath}")

    def _parameters_to_single_file_line(
            self,
            element_name: str,
            parameters: list[DesignSpaceParameter]
            ) -> dict[str, float | None | tuple[float, float]]:
        """Prepare a dict containing all info of a single element.

        Parameters
        ----------
        element_name : str
            Name of the element, which will be inserted in the output dict.
        parameters : list[DesignSpaceParameter]
            Parameters concerning the element, which ``limits`` (``x_0`` if
            appliable) will be inserted in the dict.

        Returns
        -------
        dict[str, float | None | tuple[float, float]]
            Contains all :class:`.Variable` or :class:`.Constraint` information
            of the element.

        """
        line_as_list_of_dicts = _parameters_to_dict(parameters,
                                                    ('x_0', 'limits'))
        line_as_list_of_dicts.insert(0, {'element_name': element_name})
        line_as_dict = _merge(line_as_list_of_dicts)
        return line_as_dict


# =============================================================================
# Private helpers
# =============================================================================
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


def _gather_dicts_by_key(parameters: list[DesignSpaceParameter],
                         key: str) -> dict[str, list[DesignSpaceParameter]]:
    """Gather parameters with the same ``key`` attribute value in lists.

    Parameters
    ----------
    parameters : list[DesignSpaceParameter]
        Objects to study.
    key : str
        Name of the attribute against which ``parameters`` should be gathered.

    Returns
    -------
    dict[Any, list[DesignSpaceParameter]]
        Keys are all existing values of attribute ``key`` from ``parameters``.
        Values are lists of :class:`.DesignSpaceParameter` with ``key``
        attribute equaling the dict key.

    """
    dict_by_key = defaultdict(list)
    for parameter in parameters:
        dict_by_key[str(getattr(parameter, key))].append(parameter)
    return dict_by_key


def _parameters_to_dict(parameters: list[DesignSpaceParameter],
                        to_get: tuple[str, ...]) -> list[dict]:
    """Convert several design space parameters to dict.

    We use the ``prepend_parameter_name`` argument to prepend the name of each
    ``parameter.name`` to the name of the values ``to_get``. This way, we avoid
    dictionaries sharing the same keys in the output list.

    Parameters
    ----------
    parameters : list[DesignSpaceParameter]
        Where ``to_get`` will be looked for.
    to_get : tuple[str, ...]
        Values to get.

    Returns
    -------
    list[dict]
        Contains ``to_get`` values in dictionaries for every parameter.

    """
    return [parameter.to_dict(*to_get, prepend_parameter_name=True)
            for parameter in parameters]


def _merge(dicts: list[dict]) -> dict:
    """Merge a list of dicts in a single dict."""
    return {key: value for dic in dicts for key, value in dic.items()}
