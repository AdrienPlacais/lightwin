#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""We define a factory method to create :class:`.OptimisationAlgorithm`."""
from typing import Any, Callable
from abc import ABCMeta
import logging
from functools import partial

from failures.fault import Fault
from optimisation.algorithms.algorithm import OptimisationAlgorithm
from optimisation.algorithms.least_squares import LeastSquares
from optimisation.algorithms.least_squares_penalty import LeastSquaresPenalty
from optimisation.algorithms.nsga import NSGA
from optimisation.algorithms.downhill_simplex import DownhillSimplex
from optimisation.algorithms.downhill_simplex_penalty import \
    DownhillSimplexPenalty
from optimisation.algorithms.differential_evolution import \
    DifferentialEvolution
from optimisation.algorithms.explorator import Explorator


ALGORITHMS: dict[str, ABCMeta] = {
    'least_squares': LeastSquares,
    'least_squares_penalty': LeastSquaresPenalty,
    'nsga': NSGA,
    'downhill_simplex': DownhillSimplex,
    'nelder_mead': DownhillSimplex,
    'nelder_mead_penalty': DownhillSimplexPenalty,
    'differential_evolution': DifferentialEvolution,
    'explorator': Explorator,
    'experimental': Explorator,
    }


def optimisation_algorithm_factory(
        opti_method: str,
        fault: Fault,
        beam_calculator_run_with_this: Callable,
        **kwargs: Any
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
    new_kwargs = _optimisation_algorithm_kwargs(fault,
                                                beam_calculator_run_with_this)
    _check_common_keys(kwargs, new_kwargs)
    kwargs = new_kwargs | kwargs

    algorithm_base_class = ALGORITHMS[opti_method]
    algorithm: OptimisationAlgorithm = algorithm_base_class(**kwargs)
    return algorithm


def _optimisation_algorithm_kwargs(
        fault: Fault,
        beam_calculator_run_with_this: Callable,
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
    compute_beam_propagation = partial(beam_calculator_run_with_this,
                                       elts=fault.elts)
    new_kwargs: dict[str, Any] = {
        "compute_beam_propagation": compute_beam_propagation,
        "objectives": fault.objectives,
        "compute_residuals": fault.compute_residuals,
        "compensating_cavities": fault.compensating_cavities,
        "elts": fault.elts,
        "variables": fault.variables,
        "constraints": fault.constraints,
        "compute_constraints": fault.compute_constraints,
        }
    return new_kwargs


def _check_common_keys(kwargs: dict[str, Any], new_kwargs: dict[str, Any]
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
        logging.info("The following OptimisationAlgorithm arguments are "
                     "set both in FaultScenario (kwargs) and in "
                     "optimisation.algorithms.factory (new_kwargs). We use"
                     " the ones from FaultScenario.\n"
                     f"{common_keys = })")
