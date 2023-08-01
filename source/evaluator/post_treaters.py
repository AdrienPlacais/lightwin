#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:23:24 2023.

@author: placais

In this module we regroup the functions dedicated to treat data, used in
SimulationOutputEvaluators. They all take a value and a reference value as
argumets and return the treated value (ref is unchanged).

"""
import logging
import numpy as np


def do_nothing(*args: np.ndarray | float | None, **kwargs: bool
               ) -> np.ndarray | float:
    """
    Do nothing.

    If you want to plot the data as imported from the `SimulationOutput`, set
    the first of the `post_treaters` keys to:
        partial(_do_nothing, to_plot=True)

    """
    return args[0]


def difference(value: np.ndarray | float, reference_value: np.ndarray | float,
               **kwargs: bool) -> np.ndarray | float:
    """Compute the difference."""
    delta = value - reference_value
    return delta


def relative_difference(
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

    delta_rel = (value - reference_value) / np.abs(reference_value)
    return delta_rel


def rms_error(value: np.ndarray | float, reference_value: np.ndarray | float,
              **kwargs: bool) -> float:
    """Compute the RMS error."""
    rms = np.sqrt(np.sum((value - reference_value)**2)) / value.shape[0]
    return rms


def absolute(*args: np.ndarray | float, **kwargs: bool) -> np.ndarray | float:
    """Return the absolute value `value`. A bit dumb, but adds consistency."""
    return np.abs(args[0])


def scale_by(*args: np.ndarray | float, scale: np.ndarray | float = 1.,
             **kwargs) -> np.ndarray | float:
    """Return `value` scaled by `scale`."""
    return args[0] * scale


def maximum(*args: np.ndarray | float, **kwargs: bool) -> float:
    """Return the maximum of `value`. A bit dumb, but adds consistency."""
    return np.max(args[0])


def minimum(*args: np.ndarray | float, **kwargs: bool) -> float:
    """Return the minimum of `value`. A bit dumb, but adds consistency."""
    return np.min(args[0])
