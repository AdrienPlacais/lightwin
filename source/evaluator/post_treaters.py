#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define the functions used by :class:`.SimulationOutputEvaluator`.

They are dedicated to treat data. They all take a value and a reference value
as arguments and return the treated value (ref is unchanged).

"""
import logging
from typing import overload

import numpy as np


@overload
def do_nothing(*args: float, **kwargs: bool) -> float: ...


@overload
def do_nothing(*args: np.ndarray, **kwargs: bool) -> np.ndarray: ...


def do_nothing(
    *args: np.ndarray | float | None, **kwargs: bool
) -> np.ndarray | float:
    """
    Do nothing.

    If you want to plot the data as imported from the
    :class:`.SimulationOutput`, set the first of the ``post_treaters`` keys to:
    partial(_do_nothing, to_plot=True)

    """
    assert args[0] is not None
    return args[0]


def set_first_value_to(
    *args: np.ndarray, value: float, **kwargs: bool
) -> np.ndarray:
    """Set first element of array to ``value``, sometimes bugs in TW output."""
    args[0][0] = value
    return args[0]


@overload
def difference(
    value: float, reference_value: float, **kwargs: bool
) -> float: ...


@overload
def difference(
    value: np.ndarray, reference_value: np.ndarray, **kwargs: bool
) -> np.ndarray: ...


def difference(
    value: np.ndarray | float,
    reference_value: np.ndarray | float,
    **kwargs: bool,
) -> np.ndarray | float:
    """Compute the difference."""
    delta = value - reference_value
    return delta


@overload
def relative_difference(
    value: float, reference_value: float, **kwargs: bool
) -> float: ...


@overload
def relative_difference(
    value: np.ndarray, reference_value: np.ndarray, **kwargs: bool
) -> np.ndarray: ...


def relative_difference(
    value: np.ndarray | float,
    reference_value: np.ndarray | float,
    replace_zeros_by_nan_in_ref: bool = True,
    **kwargs: bool,
) -> np.ndarray | float:
    """Compute the relative difference."""
    if replace_zeros_by_nan_in_ref:
        if not isinstance(reference_value, np.ndarray):
            logging.warning(
                "You asked the null values to be removed in "
                "the `reference_value` array, but it is not an "
                "array. I will set it to an array of size 1."
            )
            reference_value = np.atleast_1d(reference_value)

        assert isinstance(reference_value, np.ndarray)
        reference_value = reference_value.copy()
        reference_value[reference_value == 0.0] = np.NaN

    delta_rel = (value - reference_value) / np.abs(reference_value)
    return delta_rel


def rms_error(
    value: np.ndarray, reference_value: np.ndarray, **kwargs: bool
) -> float:
    """Compute the RMS error."""
    rms = np.sqrt(np.sum((value - reference_value) ** 2)) / value.shape[0]
    return rms


@overload
def absolute(*args: float, **kwargs: bool) -> float: ...


@overload
def absolute(*args: np.ndarray, **kwargs: bool) -> np.ndarray: ...


def absolute(*args: np.ndarray | float, **kwargs: bool) -> np.ndarray | float:
    """Return the absolute ``value``."""
    if isinstance(args[0], np.ndarray):
        return np.abs(args[0])
    return abs(args[0])


@overload
def scale_by(*args: float, scale: float = 1.0, **kwargs) -> float: ...


@overload
def scale_by(
    *args: np.ndarray, scale: np.ndarray | float = 1.0, **kwargs
) -> np.ndarray: ...


def scale_by(
    *args: np.ndarray | float, scale: np.ndarray | float = 1.0, **kwargs
) -> np.ndarray | float:
    """Return ``value`` scaled by ``scale``."""
    return args[0] * scale


def maximum(*args: np.ndarray, **kwargs: bool) -> float:
    """Return the maximum of ``value``."""
    return np.max(args[0])


def minimum(*args: np.ndarray, **kwargs: bool) -> float:
    """Return the minimum of ``value``."""
    return np.min(args[0])


def take_last(*args: np.ndarray, **kwargs: bool) -> float:
    """Return the last element of ``value``."""
    return args[0][-1]


def sum(*args: np.ndarray, **kwargs: bool) -> float:
    """Sum array over linac."""
    return np.sum(args[0])
