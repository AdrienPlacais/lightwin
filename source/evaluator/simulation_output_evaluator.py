#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We define an object to evaluate quality of a set of cavity settings.

.. note::
    We do not directly evaluate a :class:`.SetOfCavitySettings` though, but
    rather a :class:`.SimulationOutput`.

"""
import logging
import os.path
from typing import Callable, Any
from functools import partial
from dataclasses import dataclass
from abc import ABC

import numpy as np

from beam_calculation.output import SimulationOutput
from util.helper import resample
from visualization import plot


# =============================================================================
# Helpers
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
    ] = (post_treaters.do_nothing,)

    tester: Callable[np.ndarray | float, bool] | None = None

    descriptor: str | None = None
    markdown: str | None = None

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

        if self.plt_kwargs is None:
            self.plt_kwargs = {}
        self._fig: object | None = None
        self.main_ax: object | None = None
        self._create_plot(**self.plt_kwargs)

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
        try:
            value = self.value_getter(simulation_output)
        except IndexError:
            logging.error("Mismatch between z_abs and value shapes. Current "
                          "quantity is probably a mismatch_factor, which "
                          "was interpolated. Returning None.")
            value = None
        if value is None:
            logging.error(f"A value misses in test: {self}. Skipping...")
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
            self._save_plot(simulation_output.out_path, **self.plt_kwargs)
            return value

        test = self.tester(value)
        if _return_value_should_be_plotted(self.tester):
            limits = _limits_given_in_functoolspartial_args(self.tester)
            self._add_a_limit_plot(z_abs, limits)

        self._save_plot(simulation_output.out_path, **self.plt_kwargs)
        return test

    def _create_plot(self, fignum: int | None = None, **kwargs) -> None:
        """Prepare the plot."""
        if fignum is None:
            return
        fig, axx = plot._create_fig_if_not_exists(axnum=[211, 212],
                                                  sharex=True,
                                                  num=fignum,
                                                  clean_fig=True,)
        fig.suptitle(self.descriptor, fontsize=14)
        axx[0].set_ylabel(self.markdown)
        axx[0].grid(True)

        self._fig = fig
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

    def _save_plot(self, out_path: str | None, fignum: int | None = None,
                   savefig: bool = False, **kwargs
                   ) -> None:
        """Save the figure if asked, and if `out_path` is defined."""
        if not savefig or self._fig is None:
            return

        if out_path is None:
            logging.error("The attribute `out_path` from `SimuationOutput` is "
                          "not defined, hence I cannot save the Figure. Did "
                          "you call the method "
                          "`Accelerator.keep_simulation_output`?")
            return

        filename = f"simulation_output_evaluator_{fignum}.png"
        filepath = os.path.join(out_path, filename)
        plot._savefig(self._fig, filepath)
