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
from typing import Callable, Any
from functools import partial
from dataclasses import dataclass
from abc import ABC

import numpy as np

from beam_calculation.output import SimulationOutput
from util.helper import resample
from util.dicts_output import markdown
from visualization import plot


# =============================================================================
# Data post treatments
#  convention: arg[0] is `value` or `treated_value`. arg[1] is
#  `reference_value`.
# =============================================================================
def _do_nothing(*args: np.ndarray | float | None, **kwargs: bool
                ) -> np.ndarray | float:
    """Do nothing."""
    return args[0]


def _difference(value: np.ndarray | float, reference_value: np.ndarray | float,
                to_absolute: bool = False,
                ) -> np.ndarray | float:
    """Compute the difference."""
    delta = value - reference_value
    if to_absolute:
        return np.abs(delta)
    return delta


def _relative_difference(
    value: np.ndarray | float, reference_value: np.ndarray | float,
    to_absolute: bool = False, replace_zeros_by_nan_in_ref: bool = True
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
    if to_absolute:
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
def _value_is_within_limits(treated_value: np.ndarray | float,
                            limits: tuple[np.ndarray | float | None,
                                          np.ndarray | float | None]
                            ) -> bool:
    """Test if the given value is within the given limits."""
    return _value_is_above(treated_value, limits[0]) \
        and _value_is_below(treated_value, limits[1])


def _value_is_above(treated_value: np.ndarray | float,
                    lower_limit: np.ndarray | float | None) -> bool:
    """Test if the given value is above a threshold."""
    if lower_limit is None:
        return True
    return np.all(treated_value > lower_limit)


def _value_is_below(treated_value: np.ndarray | float,
                    upper_limit: np.ndarray | float | None) -> bool:
    """Test if the given value is below a threshold."""
    if upper_limit is None:
        return True
    return np.all(treated_value < upper_limit)


def _value_is(treated_value: np.ndarray | float,
              objective_value: np.ndarray | float, tol: float = 1e-10) -> bool:
    """Test if the value equals `objective_value`."""
    return np.all(np.abs(treated_value - objective_value) < tol)


# =============================================================================
# Other helpers
# =============================================================================
def _need_to_resample(value: np.ndarray | float, ref_value: np.ndarray | float
                      ) -> bool:
    """Determine if we need to resample `value` or `ref_value`."""
    for val in [ref_value, value]:
        if isinstance(val, float):
            return False
        if val.shape == ():
            return False

    if value.shape == ref_value.shape:
        return False

    return True


# =============================================================================
# Base class
# =============================================================================
@dataclass
class SimulationOutputEvaluator(ABC):
    """
    A base class for all the possible types of tests.

    Arguments
    ---------
    value_getter : Callable[SimulationOutput, Any]
        A function that takes the `SimulationOutput` under study as argument,
        and returns the value to be studied.
    ref_value_getter : Callable[[SimulationOutput, SimulationOutput],
                                 Any] | None, optional
        A function that takes the reference `SimulationOutput` and the
        `SimulationOutput` under study as arguments, and returns the reference
        value. In general, only one of the arguments will be used. The default
        is None.
    ref_simulation_output : SimulationOutput | None, optional
        The SimulationOutput of a nominal `Accelerator`. It is up to the user
        to verify that the `BeamCalculator` is the same between the reference
        and the fixed `SimulationOutput`. The default value is None.
    post_treater: Callable[[np.ndarray | float, np.ndarray | float],
                           np.ndarray | float], optional
        A function that takes `value` as first argument, `ref_value` as second
        argument and returns the treated data. The default is `do_nothing`.
    tester : Callable[np.ndarray | float, bool] | None, optional
        A function that takes `value` after post_treatment and returns a
        boolean. The default is None.
    descriptor : str | None, optional
        A sentence or two to describe what the test is about.
    markdown : str | None, optional
        A markdown name for this quantity, used in plots. The default is None.

    """

    value_getter: Callable[SimulationOutput, Any]
    ref_value_getter: Callable[[SimulationOutput, SimulationOutput],
                               Any] | None = None

    ref_simulation_output: SimulationOutput | None = None

    post_treater: Callable[[np.ndarray | float, np.ndarray | float],
                           np.ndarray | float] = _do_nothing

    tester: Callable[np.ndarray | float, bool] | None = None

    descriptor: str | None = None
    markdown: str | None = None

    fignum: int | None = None
    plt_kwargs: dict | None = None

    def __post_init__(self):
        """Raise warnings."""
        if self.descriptor is None:
            logging.warning("No descriptor was given for this evaluator, which"
                            " may be confusing in the long run.")
        self.descriptor = ' '.join(self.descriptor.split())

        if self.markdown is None:
            self.markdown = 'test'
        if self.fignum is not None:
            self._create_plot()

    def __repr__(self) -> str:
        """Output the descriptor string."""
        return self.descriptor

    def run(self, simulation_output: SimulationOutput) -> bool | float | None:
        """
        Run the test.

        It can return a bool (test passed with success or not), or a float. The
        former is useful for production purposes, when you want to sort the
        settings in valid/invalid categories. The latter is useful for
        development purposes, i.e. to identify the most complex cases in a
        bunch of configurations.

        """
        z_abs = None
        value = self.value_getter(simulation_output)
        if value is None:
            logging.error(f"A value misses in {self} test. Skipping test.")
            return None

        ref_value = None
        if self.ref_value_getter is not None:
            ref_value = self.ref_value_getter(self.ref_simulation_output,
                                              simulation_output)

            if _need_to_resample(value, ref_value):
                logging.info("Here I needed to resample!")
                z_abs = simulation_output.get('z_abs')
                ref_z_abs = self.ref_simulation_output.get('z_abs')
                z_abs, value, ref_z_abs, ref_value = resample(
                    z_abs, value, ref_z_abs, ref_value)

        treated_value = self.post_treater(*(value, ref_value))

        if self.tester is None:
            return treated_value
        return self.tester(treated_value)

    def _create_plot(self) -> None:  # returns fig, axx
        """Prepare the plot."""
        fig, axx = plot._create_fig_if_not_exists(axnum=[211, 212],
                                                  sharex=True,
                                                  num=self.fignum,
                                                  clean_fig=False,)
        axx[0].set_ylabel(self.markdown)
        # see what are the kwargs for _create_fig_if_not_exists...


# =============================================================================
# Presets
# =============================================================================
PRESETS = {
    "no power loss": {
        'value_getter': lambda s: s.get('pow_lost'),
        'post_treater': _do_nothing,
        'tester': partial(_value_is, objective_value=0.),
        'markdown': r"$P_{lost}$",
        'descriptor': """Lost power shall be null."""
    },
    "longitudinal eps growth": {
        'value_getter': lambda s: s.get('eps_zdelta'),
        'ref_value_getter': lambda ref_s, s: s.get('eps_zdelta',
                                                   elt='first', pos='in'),
        'post_treater': _maximum_of_relative_difference,
        'tester': partial(_value_is_below, upper_limit=1.2),
        'markdown': r"""$\Delta\epsilon_{z\delta} / \epsilon_{zdelta}$ (ref $z=0$)""",
        'descriptor': """Longitudinal emittance should not grow by more than
                         20% along the linac."""

    },
    "longitudinal eps at end": {
        'value_getter': lambda s: s.get('eps_zdelta', elt='last', pos='out'),
        'ref_value_getter': lambda ref_s, s: ref_s.get('eps_zdelta',
                                                       elt='last', pos='out'),
        'post_treater': _relative_difference,
        'markdown': markdown['eps_zdelta'],
        'descriptor': """Relative difference of emittance in [z-delta] plane
                         between fixed and reference linacs."""
    }
}
