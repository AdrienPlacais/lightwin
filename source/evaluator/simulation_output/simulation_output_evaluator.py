#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to evaluate quality of a set of cavity settings.

.. note::
    We do not directly evaluate a :class:`.SetOfCavitySettings` though, but
    rather a :class:`.SimulationOutput`.

.. todo::
    different factories for evaluation during the fit and evaluation after

"""
import logging
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import util.dicts_output as dic
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from evaluator.post_treaters import do_nothing
from evaluator.types import (
    post_treated_value_t,
    post_treater_t,
    ref_value_t,
    tester_t,
    value_t,
)
from util.helper import resample
from visualization import plot


# =============================================================================
# Helpers
# =============================================================================
def _need_to_resample(value: value_t, ref_value: ref_value_t) -> bool:
    """Determine if we need to resample ``value`` or ``ref_value``."""
    if isinstance(value, float) or isinstance(ref_value, float):
        return False
    assert isinstance(value, np.ndarray) and isinstance(ref_value, np.ndarray)
    if value.shape == () or ref_value.shape == ():
        return False
    if value.shape == ref_value.shape:
        return False
    return True


def _return_value_should_be_plotted(partial_function: Callable) -> bool:
    """Determine if keyword 'to_plot' was passed and is True.

    This function only works on functions defined by ``functools.partial``. If
    it is not (lambda function, "classic" function), we consider that the
    plotting was not desired.
    We check if the 'to_plot' keyword was given in the partial definition, and
    if it is not we also consider that the plot was not wanted.

    """
    if not isinstance(partial_function, partial):
        return False

    keywords = partial_function.keywords
    if "to_plot" not in keywords:
        return False

    return keywords["to_plot"]


def _limits_given_in_functoolspartial_args(
    partial_function: Callable,
) -> Sequence[np.ndarray | float]:
    """Extract the limits given to a test function."""
    if not isinstance(partial_function, partial):
        logging.error("Given function must be a functools.partial func.")
        return (np.NaN, np.NaN)

    keywords = partial_function.keywords
    if "limits" in keywords:
        return keywords["limits"]

    limits = [
        keywords[key]
        for key in keywords.keys()
        if key in ["lower_limit", "upper_limit", "objective_value"]
    ]
    assert len(limits) in (1, 2)
    return tuple(limits)


# =============================================================================
# Base class
# =============================================================================
@dataclass
class SimulationOutputEvaluator(ABC):
    """A base class for all the possible types of tests.

    Arguments
    ---------
    value_getter : Callable[[SimulationOutput], value_t]
        A function that takes the simulation output under study as argument,
        and returns the value to be studied.
    ref_simulation_output : SimulationOutput
        The simulation output of a nominal :class:`.Accelerator`. It is up to
        the user to verify that the :class:`.BeamCalculator` is the same
        between the reference and the fixed :class:`.SimulationOutput`.
    ref_value_getter : Callable[[SimulationOutput, SimulationOutput],\
                                 ref_value_t] | None, optional
        A function that takes the reference simulation ouput and the simulation
        output under study as arguments, and returns the reference value. In
        general, only the first argument will be used. The second argument can
        be used in specific cases, eg for the mismatch factor.  The default is
        None.
    post_treaters: Sequence[post_treater_t], optional
        A tuple of functions that will be called one after each other and
        applied on ``value``, which is returned by ``value_getter``. First
        argument must be ``value``, second argument ``ref_value``. They return
        an update ``value``, which is passed to the next function in
        ``post_treaters``. The default is a tuple containing only
        :func:`._do_nothing`.
    tester : tester_t | None, optional
        A function that takes post-treated ``value`` and test it. It can return
        a boolean or a float. The default is None.
    fignum : int | None, optional
        The Figure number. The default is None, in which case no plot is
        produced.
    descriptor : str, optional
        A sentence or two to describe what the test is about. The default is an
        empty string.
    markdown : str, optional
        A markdown name for this quantity, used in plots y label. The default
        is an empty string.
    plt_kwargs : dict[str, Any] | None = None
        A dictionary with keyword arguments passed to the ``plt.Figure``. The
        default is None.

    """

    value_getter: Callable[[SimulationOutput], value_t]
    ref_simulation_output: SimulationOutput
    ref_value_getter: (
        Callable[[SimulationOutput, SimulationOutput], ref_value_t] | None
    ) = None

    post_treaters: Sequence[post_treater_t] = (do_nothing,)
    tester: tester_t | None = None

    descriptor: str = ""
    markdown: str = ""

    plt_kwargs: dict[str, Any] | None = None
    raise_error_if_value_getter_returns_none: bool = True

    def __post_init__(self):
        """Check inputs, create plot if a ``fignum`` was provided."""
        self.descriptor = _descriptor(self.descriptor)
        self.post_treaters = _post_treaters(self.post_treaters)
        self.plt_kwargs = kwargs(self.plt_kwargs)

        self._fig: Figure | None = None
        self.main_ax: Axes | None = None
        self._create_plot(**self.plt_kwargs)

    def __repr__(self) -> str:
        """Output the descriptor string."""
        return self.descriptor

    def run(
        self, simulation_output: SimulationOutput
    ) -> np.ndarray | bool | float:
        """Run the test.

        It can return a bool (test passed with success or not), or a float. The
        former is useful for production purposes, when you want to sort the
        settings in valid/invalid categories. The latter is useful for
        development purposes, i.e. to identify the most complex cases in a
        bunch of configurations.

        """
        if self.main_ax is not None:
            plot.remove_artists(self.main_ax)
            self._add_structure_plot(simulation_output)

        plt_kw = {}
        x_data, y_data = self._get_data(simulation_output)
        if y_data is None:
            if self.raise_error_if_value_getter_returns_none:
                logging.error(f"A value misses in test: {self}. Skipping...")
            return np.NaN

        y_ref_data = self._get_ref_data(simulation_output)
        if y_ref_data is None:
            # this happens with mismatch
            # return y_data
            # logging.critical(self.descriptor)
            y_ref_data = y_data

        if _need_to_resample(y_data, y_ref_data):
            x_data, y_data, _, y_ref_data = self._resampled(
                x_data, y_data, y_ref_data
            )

        y_data = self._apply_post_treatments(
            x_data, y_data, y_ref_data, **plt_kw
        )

        if self.tester is not None:
            y_data = self._apply_test(x_data, y_data, **plt_kw)

        self._save_plot(simulation_output.out_path, **self.plt_kwargs)
        return y_data

    def _get_data(
        self,
        simulation_output: SimulationOutput,
    ) -> tuple[np.ndarray, np.ndarray | float | None]:
        """Get da data."""
        x_data = simulation_output.get("z_abs")
        try:
            y_data = self.value_getter(simulation_output)
        except IndexError:
            logging.error(
                "Mismatch between x_data and y_data shapes. Current "
                "quantity is probably a mismatch_factor, which "
                "was interpolated. Returning None."
            )
            y_data = None
        return x_data, y_data

    def _get_ref_data(
        self, simulation_output: SimulationOutput
    ) -> np.ndarray | float | None:
        """Get da reference data."""
        if self.ref_value_getter is None:
            return None
        y_ref_data = self.ref_value_getter(
            self.ref_simulation_output, simulation_output
        )
        return y_ref_data

    def _resampled(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray | float,
        y_ref_data: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resample data."""
        x_ref_data = self.ref_simulation_output.get("z_abs")
        x_data, y_data, x_ref_data, y_ref_data = resample(
            x_data,
            np.atleast_1d(y_data),
            x_ref_data,
            np.atleast_1d(y_ref_data),
        )
        return x_data, y_data, x_ref_data, y_ref_data

    def _apply_post_treatments(
        self,
        x_data: np.ndarray | float,
        y_data: np.ndarray | float,
        y_ref_data: np.ndarray | float,
        **plot_kw: str,
    ) -> np.ndarray | float:
        """Apply all the ``post_treaters`` functions.

        Can also plot the post-treated data after all or some of the
        post-treatments have been performed.

        """
        for post_treater in self.post_treaters:
            y_data = post_treater(*(y_data, y_ref_data))

            if _return_value_should_be_plotted(post_treater):
                assert self.main_ax is not None
                assert isinstance(x_data, np.ndarray)
                self._add_a_value_plot(x_data, y_data, **plot_kw)
        return y_data

    def _apply_test(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray | float,
        **plot_kw: str,
    ) -> bool | float | None:
        """Apply da testing functions.

        Can also plot the test results if asked.

        """
        y_data = self.tester(y_data)

        if _return_value_should_be_plotted(self.tester):
            assert self.main_ax is not None
            limits = _limits_given_in_functoolspartial_args(self.tester)
            self._add_a_limit_plot(x_data, limits, **plot_kw)
        return y_data

    def _create_plot(self, fignum: int | None = None, **kwargs) -> None:
        """Prepare the plot."""
        if fignum is None:
            return

        fig, axx = plot.create_fig_if_not_exists(num=fignum, **kwargs)
        fig.suptitle(self.descriptor, fontsize=14)
        axx[0].set_ylabel(self.markdown)
        axx[0].grid(True)

        self._fig = fig
        self.main_ax = axx[0]
        self._struct_ax = axx[1]

    def _add_structure_plot(
        self,
        simulation_output: SimulationOutput,
    ) -> None:
        """Add a plot of the structure in the bottom ax."""
        elts = simulation_output.element_to_index.keywords["_elts"]
        plot.clean_axes((self._struct_ax,))
        self._struct_ax.set_xlabel(dic.markdown["z_abs"])
        plot._plot_structure(elts, self._struct_ax)

    def _add_a_value_plot(
        self,
        z_data: np.ndarray,
        value: np.ndarray | float,
        **plot_kw: str,
    ) -> None:
        """Add (treated) data to the plot."""
        assert self.main_ax is not None
        if isinstance(value, float) or value.shape == ():
            self.main_ax.axhline(
                value, xmin=z_data[0], xmax=z_data[-1], **plot_kw
            )
            self.main_ax.relim()
            self.main_ax.autoscale()
            return
        self.main_ax.plot(z_data, value, **plot_kw)
        self.main_ax.relim()
        self.main_ax.autoscale()

    def _add_a_limit_plot(
        self,
        z_data: np.ndarray,
        limit: tuple[np.ndarray | float, np.ndarray | float],
        **plot_kw: str,
    ) -> None:
        """Add limits to the plot."""
        assert self.main_ax is not None

        for lim in limit:
            if isinstance(lim, float) or lim.shape == ():
                self.main_ax.axhline(
                    lim,
                    xmin=z_data[0],
                    xmax=z_data[-1],
                    c="r",
                    ls="--",
                    lw=5,
                    **plot_kw,
                )
                continue
            self.main_ax.plot(z_data, lim, **plot_kw)
        # self.main_ax.relim()
        # self.main_ax.autoscale()

    def _save_plot(
        self,
        out_path: Path,
        fignum: int | None = None,
        savefig: bool = False,
        **kwargs,
    ) -> None:
        """Save the figure if asked, and if ``out_path`` is defined."""
        if not savefig or self._fig is None:
            return

        if out_path is None:
            logging.error(
                "The attribute `out_path` from `SimulationOutput` is"
                " not defined, hence I cannot save the Figure. Did "
                "you call the method "
                "`Accelerator.keep_simulation_output`?"
            )
            return

        filename = f"simulation_output_evaluator_{fignum}.png"
        filepath = Path(out_path, filename)
        plot._savefig(self._fig, filepath)


def _descriptor(descriptor: str) -> str:
    """Clean the given string, raise warning if it is empty."""
    if not descriptor:
        logging.warning(
            "No descriptor was given for this evaluator, which may be "
            "confusing in the long run."
        )
    descriptor = " ".join(descriptor.split())
    return descriptor


def _post_treaters(
    post_treaters: post_treater_t | Sequence[post_treater_t],
) -> Sequence[post_treater_t]:
    """Check that we have a tuple, convert it to tuple if not."""
    if isinstance(post_treaters, Sequence):
        return post_treaters
    return (post_treaters,)


def kwargs(plt_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Test plot kwargs, add some default values."""
    if plt_kwargs is None:
        plt_kwargs = {}

    default_kwargs = {
        "axnum": 2,
        "clean_fig": True,
        "sharex": True,
    }
    return plt_kwargs | default_kwargs
