"""Define :class:``.NSGA``, a genetic algorithm for optimisation."""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.result import Result
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from failures.set_of_cavity_settings import SetOfCavitySettings
from optimisation.algorithms.algorithm import OptiInfo, OptimisationAlgorithm


@dataclass
class NSGA(OptimisationAlgorithm):
    """Non-dominated Sorted Genetic Algorithm."""

    supports_constraints = True

    def optimise(
        self,
        keep_history: bool = False,
        save_history: bool = False,
    ) -> tuple[bool, SetOfCavitySettings | None, OptiInfo]:
        """
        Set up the optimisation and solve the problem.

        Returns
        -------
        success : bool
            Tells if the optimisation algorithm managed to converge.
        optimized_cavity_settings : SetOfCavitySettings
            Best solution found by the optimization algorithm.
        info : dict[str, list[float]]] | None
            Gives list of solutions, corresponding objective, convergence
            violation if applicable, etc.

        """
        if keep_history or save_history:
            raise NotImplementedError
        problem: Problem = MyElementwiseProblem(
            _wrapper_residuals=self._wrapper_residuals,
            **self._problem_arguments,
        )

        bias_init = True
        initial_population = None
        n_pop = 200
        if bias_init:
            initial_population = self._set_population(problem, n_pop)

        algorithm = self._set_algorithm(sampling=initial_population)
        termination = self._set_termination()

        result: Result = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=termination,
            selection=TournamentSelection,
            # seed=None,
            verbose=True,
            # display=None,
            # callback=None,
            # return_least_infeasible=False,
            # save_history=False,
        )
        success = True

        # add least squares solution
        # result.X = np.vstack((result.X, self.x_0))
        # f, g = self._wrapper_residuals(self.x_0)
        # result.F = np.vstack((result.F, f))
        # result.G.append(g)

        set_of_cavity_settings, info = self._best_solution(result)
        return success, set_of_cavity_settings, info

    @property
    def _problem_arguments(self) -> dict[str, int | np.ndarray]:
        """Gather arguments required for :class:`ElementwiseProblem`."""
        _xl, _xu = self._format_variables()
        kwargs = {
            "n_var": self.n_var,
            "n_obj": self.n_obj,
            "n_ieq_constr": self.n_constr,
            "xl": _xl,
            "xu": _xu,
        }
        return kwargs

    def _format_variables(self) -> tuple[np.ndarray, np.ndarray]:
        """Format :class:`Variable` for this algorithm."""
        _xl = [var.limits[0] for var in self.variables]
        _xu = [var.limits[1] for var in self.variables]
        return _xl, _xu

    @property
    def x_0(self) -> np.ndarray:
        """Return initial value used in `LeastSquares'."""
        return np.array([var.x_0 for var in self.variables])

    @property
    def x_max_k_e(self) -> np.ndarray:
        """Return a solution with maximum electric fields."""
        x_max = []
        for var in self.variables:
            if var.name == "k_e":
                x_max.append(var.limits[1])
                continue
            x_max.append(var.x_0)
        return x_max

    def _wrapper_residuals(
        self, var: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute residuals from an array of variable values."""
        cav_settings = self._create_set_of_cavity_settings(var)
        simulation_output = self.compute_beam_propagation(cav_settings)

        objective = self.compute_residuals(simulation_output)
        constraints = self.compute_constraints(simulation_output)
        return np.abs(objective), constraints

    def _set_algorithm(self, *args, **kwargs) -> Algorithm:
        """Set `pymoo`s `Algorithm` object."""
        algorithm = NSGA3(*args, **kwargs)
        return algorithm

    def _set_termination(self) -> DefaultMultiObjectiveTermination:
        """Set the termination condition."""
        termination = DefaultMultiObjectiveTermination(
            n_max_gen=1000, n_max_evals=10000, xtol=1e-8, ftol=1e-8
        )
        return termination

    def _set_population(self, problem: Problem, n_pop: int) -> Population:
        """Set population, with some predefined individuals."""
        sampling = LHS()
        initial_variables = sampling(problem, n_pop).get("X")
        initial_variables[0, :] = self.x_0
        initial_variables[1, :] = self.x_max_k_e

        initial_population = Population.new("X", initial_variables)
        Evaluator().eval(problem, initial_population)
        return initial_population

    def _best_solution(
        self, result: Result
    ) -> tuple[SetOfCavitySettings, dict[str, np.ndarray]]:
        """Take the "best" solution."""
        approx = _characteristic_points(result)

        n_f = (result.F - approx["ideal"]) / (
            approx["nadir"] - approx["ideal"]
        )

        n_obj = len(self.objectives)
        weights = np.ones(n_obj) / n_obj
        idx_best = PseudoWeights(weights).do(n_f)

        set_of_cavity_settings = self._create_set_of_cavity_settings(
            result.X[idx_best]
        )
        info = {"X": result.X[idx_best], "F": result.F[idx_best]}
        logging.info(f"I choose {info['F']} (idx {idx_best})")
        return set_of_cavity_settings, info


class MyElementwiseProblem(ElementwiseProblem):
    """A first test implementation, eval single solution at a time."""

    def __init__(
        self,
        _wrapper_residuals: Callable[np.ndarray, np.ndarray],
        **kwargs: int | np.ndarray,
    ) -> None:
        """Create object."""
        self._wrapper_residuals = _wrapper_residuals
        super().__init__(**kwargs)

    def _evaluate(
        self, x: np.ndarray, out: dict[str, np.ndarray], *args, **kwargs
    ) -> dict[str, np.ndarray]:
        """Calculate and return the objectives."""
        out["F"], out["G"] = self._wrapper_residuals(x)
        return out


def _characteristic_points(result: Result) -> dict[str, np.ndarray]:
    """Give the ideal and Nadir points as a dict."""
    ideal_idx = result.F.argmin(axis=0)
    ideal = result.F.min(axis=0)

    # ideal_idx_bis = result.F[:-1].argmin(axis=0)
    # ideal_bis = result.F[:-1].min(axis=0)

    nadir_idx = result.F.argmax(axis=0)
    nadir = result.F.max(axis=0)

    logging.info(
        f"Manually added: idx {result.F.shape[0] - 1}\n"
        f"Ideal points are {ideal} (idx {ideal_idx})\n"
        # f"(without manually added lsq: {ideal_bis} @ {ideal_idx_bis}\n"
        f"Nadir points are {nadir} (idx {nadir_idx})"
    )
    approx = {"ideal": ideal, "nadir": nadir}
    return approx
