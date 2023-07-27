#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:20:53 2023.

@author: placais

In this module we define an object that is used to evaluate the quality of a
set of cavity settings - we do not directly evaluate a `SetOfCavitySettings`
though, but rather a `SimulationOutput`.

"""
import logging
from typing import TypeAlias, Callable
from dataclasses import dataclass
from abc import ABC

import numpy as np

from beam_calculation.output import SimulationOutput
from core.elements import _Element


# =============================================================================
# Data post treatments
# =============================================================================
def _do_nothing(*args: np.ndarray | float | None, **kwargs: bool
                ) -> np.ndarray | float:
    """The most advanced of our tests."""
    return args[0]


def _difference(value: np.ndarray | float, reference_value: np.ndarray | float,
                absolute_value: bool = False,
                ) -> np.ndarray | float:
    """Compute the difference."""
    delta = value - reference_value
    if absolute_value:
        return np.abs(delta)
    return delta


def _relative_difference(
    value: np.ndarray | float, reference_value: np.ndarray | float,
    absolute_value: bool = False, replace_zeros_by_nan_in_ref: bool = True
) -> np.ndarray | float:
    """Compute the relative difference."""
    if replace_zeros_by_nan_in_ref:
        if not isinstance(reference_value, np.ndarray):
            logging.warning("You demanded the null values to be removed in "
                            "the `reference_value` array, but it is not an "
                            "array. I will set it to an array of size 1.")
            reference_value = np.array(reference_value)

        reference_value = reference_value.copy()
        reference_value[reference_value == 0.] = np.NaN

    delta_rel = (value - reference_value) / reference_value
    if absolute_value:
        return np.abs(delta_rel)
    return delta_rel


def _rms_error(value: np.ndarray | float, reference_value: np.ndarray | float
               ) -> float:
    """Compute the RMS error."""
    rms = np.sqrt(np.sum((value - reference_value)**2)) / value.shape[0]
    return rms


def _maximum(value: np.ndarray) -> float:
    """Return the maximum of `value`. A bit dumb, but adds consistency."""
    return np.max(value)


def _maximum_of_relative_difference(value: np.ndarray | float,
                                    reference_value: np.ndarray | float,
                                    **kwargs: bool) -> float:
    """Compute the maximum of the relative difference."""
    delta = _relative_difference(value, reference_value, **kwargs)
    return _maximum(delta)


def _relative_difference_of_maxima(value: np.ndarray | float,
                                   reference_value: np.ndarray | float,
                                   **kwargs: bool) -> float:
    """Compute relative difference between maxima of inputs."""
    delta = _relative_difference(_maximum(value), _maximum(reference_value),
                                 **kwargs)
    return delta


# =============================================================================
# Testers
# =============================================================================
def _value_is_within_limits(value: np.ndarray | float,
                            limits: tuple[np.ndarray | float | None,
                                          np.ndarray | float | None]
                            ) -> bool:
    """Test if the given value is within the given limits."""
    return _value_is_above(value, limits[0]) \
        and _value_is_below(value, limits[1])


def _value_is_above(value: np.ndarray | float,
                    lower_limit: np.ndarray | float | None) -> bool:
    """Test if the given value is above a threshold."""
    if lower_limit is None:
        return True
    return np.all(value > lower_limit)


def _value_is_below(value: np.ndarray | float,
                    upper_limit: np.ndarray | float | None) -> bool:
    """Test if the given value is below a threshold."""
    if upper_limit is None:
        return True
    return np.all(value < upper_limit)


def _value_is(value: np.ndarray | float, objective_value: np.ndarray | float,
              tol: float = 1e-10) -> bool:
    """Test if the value equals `objective_value`."""
    return np.all(np.abs(value - objective_value) < tol)


# =============================================================================
# Base class
# =============================================================================
@dataclass
class SimulationOutputEvaluator(ABC):
    """
    A base class for all the possible types of tests.

    Arguments
    ---------
    descriptor : str | None, None
        A sentence or two to describe what the test is about.
    quantity : str
        The physical quantity that is compared. Must be understandable by the
        `SimulationOutput.get` method.
    quantity_kwargs : dict[_Element | str | bool] | None, optional
        Keywords arguments for the `SimulationOutput.get` method. The default
        is None.
    simulation_output_ref : SimulationOutput | None, optional
        The SimulationOutput of a nominal `Accelerator`. It is up to the user
        to verify that the `BeamCalculator` is the same between the reference
        and the fixed `SimulationOutput`. The default value is None.
    quantity_ref_kwargs : dict[_Element | str | bool] | None, optional
        Keywords arguments for the `SimulationOutput.get` method (reference
        data). The default is None.
    post_treat_name : str
        A POST_TREATERS key. Will set the operations performed on the value(s)
        of `quantity`.
    test_name : str | None, optional
        A TESTERS key. Will set the function transforming values into a
        boolean. The default is None.
    test_kwargs : dict, optional
        Keywords arguments for the `test` function. The default is
        None.

    """
    quantity: str

    quantity_kwargs: dict[_Element | str | bool] | None = None

    simulation_output_ref: SimulationOutput | None = None
    quantity_ref_kwargs: dict[_Element | str | bool] | None = None

    post_treat: Callable = _do_nothing
    post_treat_kwargs: dict[str, bool] | None = None

    test: Callable | None = None
    test_kwargs: dict[
        str,
        tuple[np.ndarray | float | None] | np.ndarray | float] | None = None

    descriptor: str | None = None     # or __str__ or __repr__ or even __doc__?

    def __post_init__(self):
        """Raise warnings."""
        if self.descriptor is None:
            logging.warning("No descriptor was given for this evaluator, which"
                            " may be confusing in the long run.")
        self.descriptor = ' '.join(self.descriptor.split())

        if self.quantity_kwargs is None:
            self.quantity_kwargs = {}
        if self.quantity_ref_kwargs is None:
            self.quantity_ref_kwargs = {}
        if self.post_treat_kwargs is None:
            self.post_treat_kwargs = {}
        if self.test_kwargs is None:
            self.test_kwargs = {}

    def __repr__(self) -> str:
        """Output the descriptor string."""
        return self.descriptor

    def run(self, simulation_output: SimulationOutput) -> bool | float:
        """
        Run the test.

        It can return a bool (test passed with success or not), or a float. The
        former is useful for production purposes, when you want to sort the
        settings in valid/invalid categories. The latter is useful for
        development purposes, i.e. to identify the most complex cases in a
        bunch of configurations.

        """
        value = simulation_output.get(self.quantity, **self.quantity_kwargs)

        simulation_output_ref = simulation_output
        if self.simulation_output_ref is not None:
            simulation_output_ref = self.simulation_output_ref
        reference_value = simulation_output_ref.get(self.quantity,
                                                    **self.quantity_ref_kwargs)

        error = self.post_treat(*(value, reference_value),
                                **self.post_treat_kwargs)

        if self.test is None:
            return error

        test_results = self.test(value, **self.test_kwargs)
        return test_results

    def plot(self, simulation_output: SimulationOutput):
        """
        Plot the quantity, the allowed limits.

        """
        pass


# =============================================================================
# Presets
# =============================================================================
PRESETS = {
    "no power loss": {
        'quantity': 'pow_lost',
        'test': _value_is,
        'test_kwargs': {'objective_value': 0.},
        'descriptor': """Lost power shall be null."""
    },
    "longitudinal eps at end": {
        'quantity': 'eps_zdelta',
        'quantity_kwargs': {'elt': 'DR378', 'pos': 'out'},
        'quantity_ref_kwargs': {'elt': 'DR378', 'pos': 'out'},
        'simulation_output_ref': 'yes please',
        'post_treat': _relative_difference,
        'descriptor': """
            Relative difference of emittance in [z-delta] plane between fixed
            and reference linacs.
            """
    }
}


# =============================================================================
# Factory
# =============================================================================
def factory(*args: str,
            reference_simulation_output: SimulationOutput | None = None
            ) -> list[SimulationOutputEvaluator]:
    """Create evaluators using presets."""
    kwarguments = [PRESETS[arg] for arg in args]

    for kwarg in kwarguments:
        if 'simulation_output_ref' in kwarg:
            kwarg['simulation_output_ref'] = reference_simulation_output

    evaluators = [SimulationOutputEvaluator(**kwarg)
                  for kwarg in kwarguments]
    return evaluators
