#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We regroup the functions dedicated to testing data.

They are used by :class:`.SimulationOutputEvaluator'. They all take a value and
limits/upper or lower or objective value as arguments, and return a boolean.

"""
import numpy as np


def value_is_within_limits(treated_value: np.ndarray | float,
                           limits: tuple[np.ndarray | float,
                                         np.ndarray | float],
                           **kwargs: bool) -> bool:
    """Test if the given value is within the given limits."""
    return value_is_above(treated_value, limits[0]) \
        and value_is_below(treated_value, limits[1])


def value_is_above(treated_value: np.ndarray | float,
                   lower_limit: np.ndarray | float,
                   **kwargs: bool) -> bool:
    """Test if the given value is above a threshold."""
    return np.all(treated_value > lower_limit)


def value_is_below(treated_value: np.ndarray | float,
                   upper_limit: np.ndarray | float,
                   **kwargs: bool) -> bool:
    """Test if the given value is below a threshold."""
    return np.all(treated_value < upper_limit)


def value_is(treated_value: np.ndarray | float,
             objective_value: np.ndarray | float,
             tol: float = 1e-10,
             **kwargs: bool) -> bool:
    """Test if the value equals ``objective_value`."""
    return np.all(np.abs(treated_value - objective_value) < tol)
