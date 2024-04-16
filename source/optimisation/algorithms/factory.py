#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a factory function to create :class:`.OptimisationAlgorithm`.

.. todo::
    Docstrings

"""
import logging
from abc import ABCMeta
from functools import partial
from typing import Any, Callable

from beam_calculation.beam_calculator import BeamCalculator
from core.elements.field_maps.cavity_settings_factory import (
    CavitySettingsFactory,
)
from failures.fault import Fault
from optimisation.algorithms.algorithm import OptimisationAlgorithm
from optimisation.algorithms.differential_evolution import (
    DifferentialEvolution,
)
from optimisation.algorithms.downhill_simplex import DownhillSimplex
from optimisation.algorithms.downhill_simplex_penalty import (
    DownhillSimplexPenalty,
)
from optimisation.algorithms.explorator import Explorator
from optimisation.algorithms.least_squares import LeastSquares
from optimisation.algorithms.least_squares_penalty import LeastSquaresPenalty
from optimisation.algorithms.nsga import NSGA

ALGORITHM_SELECTOR: dict[str, ABCMeta] = {
    "least_squares": LeastSquares,
    "least_squares_penalty": LeastSquaresPenalty,
    "nsga": NSGA,
    "downhill_simplex": DownhillSimplex,
    "downhill_simplex_penalty": DownhillSimplexPenalty,
    "nelder_mead": DownhillSimplex,
    "nelder_mead_penalty": DownhillSimplexPenalty,
    "differential_evolution": DifferentialEvolution,
    "explorator": Explorator,
    "experimental": Explorator,
}
algorithms = tuple(ALGORITHM_SELECTOR.keys())  #:


def optimisation_algorithm_factory(
    opti_method: str,
    fault: Fault,
    beam_calculator: BeamCalculator,
    **kwargs: Any,
) -> OptimisationAlgorithm:
    """
    Create the proper :class:`.OptimisationAlgorithm` instance.

    Parameters
    ----------
    opti_method : str
        Name of the desired optimisation algorithm.
    fault : Fault
        Fault that will be compensated by the optimisation algorithm.
    compute_beam_propagation : Callable
        Function that takes in a set of cavity settings and a list of elements,
        computes the beam propagation with these, and returns a simulation
        output.

    Returns
    -------
    beam_calculators : OptimisationAlgorithm
        Proper optimisation algorithm.

    """
    run_with_this = beam_calculator.run_with_this
    cavity_settings_factory = beam_calculator.cavity_settings_factory
    new_kwargs = _optimisation_algorithm_kwargs(
        fault, run_with_this, cavity_settings_factory
    )
    _check_common_keys(kwargs, new_kwargs)
    kwargs = new_kwargs | kwargs

    algorithm_base_class = ALGORITHM_SELECTOR[opti_method]
    algorithm: OptimisationAlgorithm = algorithm_base_class(**kwargs)
    return algorithm


def _optimisation_algorithm_kwargs(
    fault: Fault,
    run_with_this: Callable,
    cavity_settings_factory: CavitySettingsFactory,
) -> dict[str, Any]:
    """Set default arguments to instantiate the optimisation algorithm.

    The kwargs for :class:`.OptimisationAlgorithm` that are defined in
    :meth:`.FaultScenario._set_optimisation_algorithms` will override the ones
    defined here.

    Parameters
    ----------
    fault : Fault
        Fault that will be compensated by the optimisation algorithm.
    compute_beam_propagation : Callable
        Function that takes in a set of cavity settings and a list of elements,
        computes the beam propagation with these, and returns a simulation
        output.

    Returns
    -------
    new_kwargs : dict[str, Any]
        A dictionary of keyword arguments for the initialisation of
        :class:`.OptimisationAlgorithm`.

    """
    compute_beam_propagation = partial(run_with_this, elts=fault.elts)
    new_kwargs: dict[str, Any] = {
        "compensating_elements": fault.compensating_elements,
        "elts": fault.elts,
        "objectives": fault.objectives,
        "variables": fault.variables,
        "compute_beam_propagation": compute_beam_propagation,
        "compute_residuals": fault.compute_residuals,
        "constraints": fault.constraints,
        "compute_constraints": fault.compute_constraints,
        "cavity_settings_factory": cavity_settings_factory,
    }
    return new_kwargs


def _check_common_keys(
    kwargs: dict[str, Any], new_kwargs: dict[str, Any]
) -> None:
    """Check keys that are common between the two dictionaries.

    Parameters
    ----------
    kwargs : dict[str, Any]
        kwargs as defined in the
        :meth:`.FaultScenario._set_optimisation_algorithms` (they have
        precedence).
    new_kwargs : [str, Any]
        kwargs as defined in the :func:`_optimisation_algorithm_kwargs` (they
        will be overriden as they are considered as "default" or "fallback"
        values).

    """
    keys = set(kwargs.keys())
    new_keys = set(new_kwargs.keys())
    common_keys = keys.intersection(new_keys)
    if len(common_keys) > 0:
        logging.info(
            "The following OptimisationAlgorithm arguments are "
            "set both in FaultScenario (kwargs) and in "
            "optimisation.algorithms.factory (new_kwargs). We use"
            " the ones from FaultScenario.\n"
            f"{common_keys = })"
        )
