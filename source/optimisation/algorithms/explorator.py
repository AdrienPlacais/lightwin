"""Define :class:`Explorator`, a module to explore the design space.

In order to be consistent with the ABC :class:`.OptimisationAlgorithm`,
it also returns the solution with the lowest residue value -- hence it is also
a "brute-force" optimisation algorithm.

.. todo::
    Make this class more robust. In particular: save all objectives (not just
    the norm), handle export when there is more than two variables, also save
    complementary data (e.g.: always save ``phi_s`` even it is not in the
    constraints nor variables).

.. todo::
    Allow for different number of points according to variable.

"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits import mplot3d

from failures.set_of_cavity_settings import SetOfCavitySettings
from optimisation.algorithms.algorithm import (
    ComputeConstraintsT,
    OptiInfo,
    OptimisationAlgorithm,
)
from util.dicts_output import markdown


@dataclass
class Explorator(OptimisationAlgorithm):
    """Method that tries all the possible solutions.

    Notes
    -----
    Very inefficient for optimisation. It is however useful to study a specific
    case.

    All the attributes but ``solution`` are inherited from the Abstract Base
    Class :class:`OptimisationAlgorithm`.

    """

    supports_constraints = True
    compute_constraints: ComputeConstraintsT

    def optimise(
        self,
        keep_history: bool = True,
        save_history: bool = True,
    ) -> tuple[bool, SetOfCavitySettings | None, OptiInfo]:
        """
        Set up the optimisation and solve the problem.

        Returns
        -------
        success : bool
            Tells if the optimisation algorithm managed to converge.
        optimized_cavity_settings : SetOfCavitySettings | None
            Best solution found by the optimization algorithm.
        info : OptiInfo
            Gives list of solutions, corresponding objective, convergence
            violation if applicable, etc.

        """
        is_plottable = self._check_dimensions()
        kwargs = self._algorithm_parameters()
        variables_as_mesh, variables_values = self._generate_combinations(
            **kwargs
        )

        results = [self._wrapper_residuals(var) for var in variables_values]
        objectives_values = np.array([res[0] for res in results])
        constraints_values = np.array([res[1] for res in results])

        objectives_as_mesh = self._array_of_values_to_mesh(
            objectives_values, **kwargs
        )
        constraints_as_mesh = self._array_of_values_to_mesh(
            constraints_values, **kwargs
        )

        best_solution, best_objective = self._take_best_solution(
            variables_values,
            objectives_values,
            criterion="minimize norm of objective",
        )
        info = {
            "X": best_solution,
            "F": best_objective,
            "hist_X": None,
            "hist_F": None,
            "hist_G": None,
        }
        if keep_history:
            info = info | self._generate_optimisation_history(
                variables_values, objectives_values, constraints_values
            )
        if save_history:
            if self.folder is None:
                logging.warning(
                    "You should provide a folder to the optimisation algorithm"
                    " to tell it where it should save history. Using a default"
                    " location..."
                )
                self.folder = Path("/home/placais/LightWin/data")

            my_filepath = self.folder / "optimisation_history.txt"
            self._save_optimization_history(my_filepath, **info)

        if is_plottable:
            axes = self._plot_design_space(
                variables_as_mesh, objectives_as_mesh, constraints_as_mesh
            )
            self._add_the_best_point(axes, info["X"], info["F"])

        optimized_cavity_settings = self._create_set_of_cavity_settings(
            info["X"]
        )
        return True, optimized_cavity_settings, info

    def _check_dimensions(self) -> bool:
        """Check that we have proper number of vars and objectives."""
        if self.n_obj != 1:
            logging.warning(
                "The number of objectives is different from 1. Hence I will "
                "simply plot the norm of objectives."
            )
        if self.n_var != 2:
            logging.warning(
                "Wrong number of variables. Impossible to plot it, but I will "
                "compute all possible solutions anyway."
            )
            return False
        return True

    def _algorithm_parameters(self) -> dict:
        """Create the ``kwargs`` for the optimisation."""
        kwargs = {"n_points": 20}
        return kwargs

    def _generate_combinations(
        self, n_points: int = 10, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate all the possible combinations of the variables."""
        limits = []
        for var in self.variables:
            lim = (var.limits[0], var.limits[1])

            if "phi" in var.name and lim[1] - lim[0] >= 2.0 * np.pi:
                lim = (0.0, 2.0 * np.pi)
            limits.append(lim)

        variables_values = [
            np.linspace(lim[0], lim[1], n_points) for lim in limits
        ]
        variables_mesh = np.array(
            np.meshgrid(*variables_values, indexing="ij")
        )
        variables_combinations = np.concatenate(variables_mesh.T)
        return variables_mesh, variables_combinations

    def _wrapper_residuals(self, var: np.ndarray) -> tuple[float, bool]:
        """Give norm of residual and if phi_s constraint is respected."""
        cav_settings = self._create_set_of_cavity_settings(var)
        simulation_output = self.compute_beam_propagation(cav_settings)
        residuals = self.compute_residuals(simulation_output)
        constraints_evaluations = self.compute_constraints(simulation_output)
        is_ok = (
            constraints_evaluations[0] < 0.0
            and constraints_evaluations[1] < 0.0
        )
        return np.linalg.norm(residuals), is_ok

    def _array_of_values_to_mesh(
        self, objectives_values: np.ndarray, n_points: int = 10, **kwargs
    ) -> np.ndarray:
        """Reformat the results for plotting purposes."""
        return objectives_values.reshape((n_points, n_points)).T

    def _take_best_solution(
        self,
        variable_comb: np.ndarray,
        objectives_values: np.ndarray,
        criterion: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Take the "best" of the calculated solutions.

        Parameters
        ----------
        variable_comb : np.ndarray
            All the set of variables (cavity parameters) that were tried.
        objectives_values : np.ndarray
            The values of the objective corresponding to ``variable_comb``.
        criterion : {'minimize norm of objective', }
            Name of the criterion that will determine which solution is the
            "best". Only one is implemented for now, may add others in the
            future.

        Returns
        -------
        best_solution : np.ndarray | None
            "Best" solution.
        best_objective : np.ndarray | None
            Objective values corresponding to ``best_solution``.

        """
        if criterion == "minimize norm of objective":
            norm_of_objective = objectives_values
            if len(norm_of_objective.shape) > 1:
                norm_of_objective = np.linalg.norm(norm_of_objective, axis=1)
            best_idx = np.nanargmin(norm_of_objective)
            best_solution = variable_comb[best_idx]
            best_objective = objectives_values[best_idx]
            return best_solution, best_objective

        logging.error(
            f"{criterion = } not implemented. Please check your " "input."
        )
        return None, None

    def _plot_design_space(
        self,
        variable_mesh: np.ndarray,
        objective_mesh: np.ndarray,
        constraint_mesh: np.ndarray,
    ) -> mplot3d.Axes3D:
        """Plot the design space."""
        fig = plt.figure(30)
        axes = fig.add_subplot(projection="3d")
        axes.set_xlabel(markdown[self.variable_names[0]].replace("deg", "rad"))
        axes.set_ylabel(markdown[self.variable_names[1]].replace("deg", "rad"))
        axes.set_zlabel("Objective")

        colors = cm.Set1(constraint_mesh)
        rcount, ccount, _ = colors.shape
        axes.plot_wireframe(
            variable_mesh[0],
            variable_mesh[1],
            objective_mesh,
            rcount=rcount,
            ccount=ccount,
        )
        for i in range(len(variable_mesh[0])):
            axes.scatter(
                variable_mesh[0][i],
                variable_mesh[1][i],
                objective_mesh[i],
                s=50,
                c=colors[i],
            )
        plt.show()
        return axes

    def _add_the_best_point(
        self, axes: mplot3d.Axes3D, var: np.ndarray, obj: np.ndarray
    ) -> None:
        """Add the best solution to the plot."""
        axes.scatter3D(var[0], var[1], obj, s=100, c="r")

    def _save_optimization_history(
        self,
        filepath: Path,
        hist_X: np.ndarray | None = None,
        hist_F: np.ndarray | None = None,
        hist_G: np.ndarray | None = None,
        n_phi: int = 10,
        **info: np.ndarray,
    ) -> None:
        if hist_X is None or hist_F is None:
            logging.error(
                "Variable history and/or objective history are None. Maybe you"
                " forgot to set the keep_history flag to True? Or maybe an "
                "error occurred during optimisation?"
            )
            return

        if hist_X.shape[1] != 2:
            logging.error(
                "Number of variables different from 2 not " "implemented."
            )
            return

        with open(filepath, "w", encoding="utf-8") as file:
            for i, (solution, objective) in enumerate(zip(hist_X, hist_F)):
                if i % n_phi == 0 and i != 0:
                    file.write("\n")

                line = f"{solution[0]} {solution[1]} {objective}"
                if hist_G is not None:
                    line += f" {float(hist_G[i])}"
                file.write(line + "\n")
        logging.info(f"Optimisation history saved in {filepath}.")
