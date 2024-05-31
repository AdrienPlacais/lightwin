"""Define the class :class:`Fault`.

Its purpose is to hold information on a failure and to fix it.

.. todo::
    not clear what happens here. separate __init__ in several functions

.. todo::
    store DesignSpace as attribute rather than Variable Constraint
    compute_constraints

"""

import logging
from pathlib import Path
from typing import Any, Self

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.elements.element import Element
from core.list_of_elements.factory import ListOfElementsFactory
from core.list_of_elements.helper import equivalent_elt
from core.list_of_elements.list_of_elements import ListOfElements
from failures.set_of_cavity_settings import SetOfCavitySettings
from optimisation.algorithms.algorithm import OptimisationAlgorithm
from optimisation.design_space.factory import DesignSpaceFactory
from optimisation.objective.factory import (
    get_objectives_and_residuals_function,
)
from util.pickling import MyPickler


class Fault:
    """Handle and fix a single failure.

    Attributes
    ----------
    failed_elements : list[Element]
        Holds the failed elements.
    compensating_elements : list[Element]
        Holds the compensating elements.
    elts : ListOfElements
        Holds the portion of the linac that will be computed again and again in
        the optimisation process. It is as short as possible, but must contain
        all `failed_elements`, `compensating_elements` and
        `elt_eval_objectives`.
    variables : list[Variable]
        Holds information on the optimisation variables.
    constraints : list[Constraint] | None
        Holds infomation on the optimisation constraints.

    Methods
    -------
    compute_constraints : Callable[[SimulationOutput], np.ndarray] | None
        Compute the constraint violation for a given `SimulationOutput`.
    compute_residuals : Callable[[SimulationOutput], np.ndarray]
        A function that takes in a `SimulationOutput` and returns the residues
        of every objective w.r.t the reference one.

    """

    def __init__(
        self,
        reference_elts: ListOfElements,
        reference_simulation_output: SimulationOutput,
        files_from_full_list_of_elements: dict[str, Any],
        wtf: dict[str, Any],
        design_space_factory: DesignSpaceFactory,
        broken_elts: ListOfElements,
        failed_elements: list[Element],
        compensating_elements: list[Element],
        list_of_elements_factory: ListOfElementsFactory,
    ) -> None:
        """Create the Fault object.

        Parameters
        ----------
        reference_elts : ListOfElements
            List of elements of the reference linac. In particular, these
            elements hold the original element settings.
        reference_simulation_output : SimulationOutput
            Nominal simulation.
        files_from_full_list_of_elements : dict
            ``files`` attribute from the linac under fixing. Used to set
            calculation paths.
        wtf : dict[str, str | int | bool | list[str] | list[float]]
            What To Fit dictionary. Holds information on the fixing method.
        design_space_factory : DesignSpaceFactory
            An object to easily create the proper :class:`.DesignSpace`.
        failed_elements : list[Element]
            Holds the failed elements.
        compensating_elements : list[Element]
            Holds the compensating elements.
        elts : list[Element]
            Holds the portion of the linac that will be computed again and
            again in the optimisation process. It is as short as possible, but
            must contain all altered elements as well as the elements where
            objectives will be evaluated.

        """
        assert all([element.can_be_retuned for element in failed_elements])
        self.failed_elements = failed_elements
        assert all(
            [element.can_be_retuned for element in compensating_elements]
        )
        self.compensating_elements = compensating_elements

        reference_elements = [
            equivalent_elt(reference_elts, element)
            for element in self.compensating_elements
        ]
        design_space = design_space_factory.run(
            compensating_elements, reference_elements
        )

        self.variables = design_space.variables
        self.constraints = design_space.constraints
        self.compute_constraints = design_space.compute_constraints

        objective_preset = wtf["objective_preset"]
        assert isinstance(objective_preset, str)
        elts_of_compensation_zone, self.objectives, self.compute_residuals = (
            get_objectives_and_residuals_function(
                objective_preset=objective_preset,
                reference_elts=reference_elts,
                reference_simulation_output=reference_simulation_output,
                broken_elts=broken_elts,
                failed_elements=failed_elements,
                compensating_elements=compensating_elements,
                design_space_kw=design_space_factory.design_space_kw,
            )
        )

        self.elts: ListOfElements = list_of_elements_factory.subset_list_run(
            elts_of_compensation_zone,
            reference_simulation_output,
            files_from_full_list_of_elements,
        )
        self.optimized_cavity_settings: SetOfCavitySettings
        return

    def fix(
        self, optimisation_algorithm: OptimisationAlgorithm
    ) -> tuple[bool, dict]:
        """Fix the :class:`Fault`. Set :attr:`optimized_cavity_settings`.

        Parameters
        ----------
        optimisation_algorithm : OptimisationAlgorithm
            The optimisation algorithm to be used, already initialized.

        Returns
        -------
        success : bool
            Indicates convergence of the :class:`OptimisationAlgorithm`.
        self.info : dict
            Useful information, such as the best solution.

        """
        outputs = optimisation_algorithm.optimise()
        success, optimized_cavity_settings, self.info = outputs
        assert optimized_cavity_settings is not None
        self.optimized_cavity_settings = optimized_cavity_settings
        return success, self.info

    def update_elements_status(
        self, optimisation: str, success: bool | None = None
    ) -> None:
        """Update status of compensating and failed elements."""
        if optimisation not in ("not started", "finished"):
            logging.error(
                "{optimisation = } not understood. Not changing any status..."
            )
            return

        if optimisation == "not started":
            elements = self.failed_elements + self.compensating_elements
            status = ["failed" for _ in self.failed_elements]
            status += [
                "compensate (in progress)" for _ in self.compensating_elements
            ]

            allowed = ("nominal", "rephased (in progress)", "rephased (ok)")
            status_is_invalid = [
                cav.get("status") not in allowed for cav in elements
            ]
            if any(status_is_invalid):
                logging.error(
                    "At least one compensating or failed element is already "
                    "compensating or faulty, probably in another Fault object."
                    " Updating its status anyway..."
                )

        elif optimisation == "finished":
            assert success is not None

            elements = self.compensating_elements
            status = ["compensate (ok)" for _ in elements]
            if not success:
                status = ["compensate (not ok)" for _ in elements]

        for cav, stat in zip(elements, status):
            cav.update_status(stat)

    def pickle(
        self, pickler: MyPickler, path: Path | str | None = None
    ) -> Path:
        """Pickle (save) the object.

        This is useful for debug and temporary saves; do not use it for long
        time saving.

        """
        if path is None:
            path = self.elts.files["accelerator_path"] / "fault.pkl"
        assert isinstance(path, Path)
        pickler.pickle(self, path)

        if isinstance(path, str):
            path = Path(path)
        return path

    @classmethod
    def from_pickle(cls, pickler: MyPickler, path: Path | str) -> Self:
        """Instantiate object from previously pickled file."""
        fault = pickler.unpickle(path)
        return fault  # type: ignore
