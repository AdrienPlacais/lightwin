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
    """
    Do nothing.

    If you want to plot the data as imported from the `SimulationOutput`, set
    the first of the `post_treaters` keys to:
        partial(_do_nothing, to_plot=True)

    """
    return args[0]


def _difference(value: np.ndarray | float, reference_value: np.ndarray | float,
                **kwargs: bool) -> np.ndarray | float:
    """Compute the difference."""
    delta = value - reference_value
    return delta


def _relative_difference(
    value: np.ndarray | float, reference_value: np.ndarray | float,
    replace_zeros_by_nan_in_ref: bool = True,
    **kwargs: bool,
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
    return delta_rel


def _rms_error(value: np.ndarray | float, reference_value: np.ndarray | float,
               **kwargs: bool) -> float:
    """Compute the RMS error."""
    rms = np.sqrt(np.sum((value - reference_value)**2)) / value.shape[0]
    return rms


def _absolute(*args: np.ndarray | float, **kwargs: bool) -> np.ndarray | float:
    """Return the absolute value `value`. A bit dumb, but adds consistency."""
    return np.abs(args[0])


def _scale_by(*args: np.ndarray | float, scale: np.ndarray | float = 1.,
              **kwargs) -> np.ndarray | float:
    """Return `value` scaled by `scale`."""
    return args[0] * scale


def _maximum(*args: np.ndarray | float, **kwargs: bool) -> float:
    """Return the maximum of `value`. A bit dumb, but adds consistency."""
    return np.max(args[0])


def _minimum(*args: np.ndarray | float, **kwargs: bool) -> float:
    """Return the minimum of `value`. A bit dumb, but adds consistency."""
    return np.min(args[0])


# =============================================================================
# Testers
# =============================================================================
def _value_is_within_limits(treated_value: np.ndarray | float,
                            limits: tuple[np.ndarray | float,
                                          np.ndarray | float],
                            **kwargs: bool) -> bool:
    """Test if the given value is within the given limits."""
    return _value_is_above(treated_value, limits[0]) \
        and _value_is_below(treated_value, limits[1])


def _value_is_above(treated_value: np.ndarray | float,
                    lower_limit: np.ndarray | float, **kwargs: bool
                    ) -> bool:
    """Test if the given value is above a threshold."""
    return np.all(treated_value > lower_limit)


def _value_is_below(treated_value: np.ndarray | float,
                    upper_limit: np.ndarray | float, **kwargs: bool
                    ) -> bool:
    """Test if the given value is below a threshold."""
    return np.all(treated_value < upper_limit)


def _value_is(treated_value: np.ndarray | float,
              objective_value: np.ndarray | float, tol: float = 1e-10,
              **kwargs: bool) -> bool:
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


def _return_value_should_be_plotted(partial_function: Callable) -> bool:
    """
    Determine if keyword 'to_plot' was passed and is True.

    This function only works on functions defined by `functools.partial`. If it
    is not (lambda function, "classic" function), we consider that the plotting
    was not desired.
    Then we check if the 'to_plot' keyword was given in the partial definition,
    and if it is not we also consider that the plot was not wanted.

    """
    if not isinstance(partial_function, partial):
        return False

    keywords = partial_function.keywords
    if 'to_plot' not in keywords:
        return False

    return keywords['to_plot']


def _limits_given_in_functoolspartial_args(partial_function: Callable
                                           ) -> tuple[np.ndarray | float]:
    """Extract the limits given to a test function."""
    if not isinstance(partial_function, partial):
        logging.warning("Given function must be a functools.partial func.")
        return tuple(np.NaN)

    keywords = partial_function.keywords

    if 'limits' in keywords:
        return keywords['limits']

    limits = [keywords[key] for key in keywords.keys()
              if key in ['lower_limit', 'upper_limit', 'objective_value']
              ]
    assert len(limits) in [1, 2]
    return tuple(limits)


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
    post_treaters: tuple[Callable[[np.ndarray | float, np.ndarray | float],
                                  np.ndarray | float]], optional
        A tuple of functions called one after each other. They take `value` as
        first argument, `ref_value` as second argument. It returns an updated
        `value`, which is given to the next function in the tuple. The default
        is (`_do_nothing`,).
    tester : Callable[np.ndarray | float, bool] | None, optional
        A function that takes `value` after post_treatment and returns a
        boolean. The default is None.
    fignum : int | None, optional
        The Figure number. The default is None, in which case no plot is
        produced.
    descriptor : str | None, optional
        A sentence or two to describe what the test is about.
    markdown : str | None, optional
        A markdown name for this quantity, used in plots. The default is None.

    """

    value_getter: Callable[SimulationOutput, Any]
    ref_value_getter: Callable[[SimulationOutput, SimulationOutput],
                               Any] | None = None

    ref_simulation_output: SimulationOutput | None = None

    post_treaters: tuple[Callable[
        [np.ndarray | float, np.ndarray | float],
        np.ndarray | float]
    ] = (_do_nothing,)

    tester: Callable[np.ndarray | float, bool] | None = None

    descriptor: str | None = None
    markdown: str | None = None

    fignum: int | None = None
    main_ax: object | None = None
    plt_kwargs: dict | None = None

    def __post_init__(self):
        """Check inputs, create plot if a `fignum` was provided."""
        if self.descriptor is None:
            logging.warning("No descriptor was given for this evaluator, which"
                            " may be confusing in the long run.")
        self.descriptor = ' '.join(self.descriptor.split())

        if not isinstance(self.post_treaters, tuple):
            logging.warning("You must provide a tuple of post_treaters, even "
                            "if you want to perform only one operation. "
                            "Remember: `(lala)` is not a tuple, but `(lala,)` "
                            "is. Transforming input into a tuple and hoping "
                            "for the best...")
            self.post_treaters = tuple(self.post_treaters)

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
        z_abs = simulation_output.get('z_abs')
        value = self.value_getter(simulation_output)
        if value is None:
            logging.error(f"A value misses in {self} test. Skipping test.")
            return None

        ref_value = None
        if self.ref_value_getter is not None:
            ref_value = self.ref_value_getter(self.ref_simulation_output,
                                              simulation_output)

            if _need_to_resample(value, ref_value):
                ref_z_abs = self.ref_simulation_output.get('z_abs')
                z_abs, value, ref_z_abs, ref_value = resample(
                    z_abs, value, ref_z_abs, ref_value)

        for post_treater in self.post_treaters:
            value = post_treater(*(value, ref_value))
            if _return_value_should_be_plotted(post_treater):
                self._add_a_value_plot(z_abs, value)

        if self.tester is None:
            return value

        test = self.tester(value)
        if _return_value_should_be_plotted(self.tester):
            limits = _limits_given_in_functoolspartial_args(self.tester)
            self._add_a_limit_plot(z_abs, limits)

        return test

    def _create_plot(self) -> None:  # returns fig, axx
        """Prepare the plot."""
        fig, axx = plot._create_fig_if_not_exists(axnum=[211, 212],
                                                  sharex=True,
                                                  num=self.fignum,
                                                  clean_fig=True,)
        fig.suptitle(self.descriptor, fontsize=14)
        axx[0].set_ylabel(self.markdown)
        axx[0].grid(True)
        self.main_ax = axx[0]
        # see what are the kwargs for _create_fig_if_not_exists...

    def _add_a_value_plot(self, z_data: np.ndarray, value: np.ndarray | float
                          ) -> None:
        """Add (treated) data to the plot."""
        assert self.main_ax is not None
        if isinstance(value, float) or value.shape == ():
            self.main_ax.axhline(value, xmin=z_data[0], xmax=z_data[-1])
            return
        self.main_ax.plot(z_data, value)

    def _add_a_limit_plot(self, z_data: np.ndarray,
                          limit: tuple[np.ndarray | float]) -> None:
        """Add limits to the plot."""
        assert self.main_ax is not None

        for lim in limit:
            if isinstance(lim, float) or lim.shape == ():
                self.main_ax.axhline(lim, xmin=z_data[0], xmax=z_data[-1],
                                     c='r', ls='--', lw=5)
                continue
            self.main_ax.plot(z_data, lim)


# =============================================================================
# Presets
# =============================================================================
PRESETS = {
    # Legacy "fit quality"
    # Legacy "Fred tests"
    "no power loss": {
        'value_getter': lambda s: s.get('pow_lost'),
        'post_treaters': (partial(_do_nothing, to_plot=True),),
        'tester': partial(_value_is, objective_value=0., to_plot=True),
        'fignum': 101,
        'markdown': markdown["pow_lost"],
        'descriptor': """Lost power shall be null."""
    },
    "longitudinal eps shall not grow too much": {
        'value_getter': lambda s: s.get('eps_zdelta'),
        'ref_value_getter': lambda ref_s, s: s.get('eps_zdelta',
                                                   elt='first', pos='in'),
        'post_treaters': (_relative_difference,
                          partial(_scale_by, scale=100., to_plot=True),
                          _maximum),
        'tester': partial(_value_is_below, upper_limit=20., to_plot=True),
        'fignum': 102,
        'markdown': r"$\Delta\epsilon_{z\delta} / \epsilon_{z\delta}$ (ref $z=0$) [%]",
        'descriptor': """Longitudinal emittance should not grow by more than
                         20% along the linac."""

    },
    "max of eps shall not be too high": {
        'value_getter': lambda s: s.get('eps_zdelta'),
        'ref_value_getter': lambda ref_s, s: np.max(ref_s.get('eps_zdelta')),
        'post_treaters': (_maximum,
                          partial(_relative_difference,
                                  replace_zeros_by_nan_in_ref=False,
                                  to_plot=True)),
        'tester': partial(_value_is_below, upper_limit=30., to_plot=True),
        'fignum': 103,
        'markdown': r"$\frac{max(\epsilon_{z\delta}) - max(\epsilon_{z\delta}^{ref}))}{max(\epsilon_{z\delta}^{ref})}$",
        'descriptor': """The maximum of longitudinal emittance should not
                         exceed the nominal maximum of longitudinal emittance
                         by more than 30%."""

    },
    # Legacy "Bruce tests"
    "longitudinal eps at end": {
        'value_getter': lambda s: s.get('eps_zdelta', elt='last', pos='out'),
        'ref_value_getter': lambda ref_s, s: ref_s.get('eps_zdelta',
                                                       elt='last', pos='out'),
        'post_treaters': (_relative_difference,),
        'markdown': markdown['eps_zdelta'],
        'descriptor': """Relative difference of emittance in [z-delta] plane
                         between fixed and reference linacs."""
    },
    "mismatch factor at end": {
        'value_getter': lambda s: s.get('mismatch_factor',
                                        elt='last', pos='out'),
        'markdown': markdown['mismatch_factor'],
        'descriptor': """Mismatch factor at the end of the linac."""
    },
}
