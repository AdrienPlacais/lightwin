"""Define a list-based class holding all the :class:`.Fault` to fix.

We also define :func:`fault_scenario_factory`, a factory function creating all
the required :class:`FaultScenario` objects.

"""

import datetime
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from beam_calculation.tracewin.tracewin import TraceWin
from core.accelerator.accelerator import Accelerator
from core.elements.element import Element
from core.list_of_elements.factory import ListOfElementsFactory
from evaluator.list_of_simulation_output_evaluators import (
    FaultScenarioSimulationOutputEvaluators,
)
from failures import strategy
from failures.fault import Fault
from failures.set_of_cavity_settings import FieldMap
from optimisation.algorithms.algorithm import OptimisationAlgorithm
from optimisation.algorithms.factory import optimisation_algorithm_factory
from optimisation.design_space.factory import (
    DesignSpaceFactory,
    get_design_space_factory,
)
from util import debug
from util.pickling import MyPickler

DISPLAY_CAVITIES_INFO = True


class FaultScenario(list):
    """A class to hold all fault related data."""

    def __init__(
        self,
        ref_acc: Accelerator,
        fix_acc: Accelerator,
        beam_calculator: BeamCalculator,
        wtf: dict[str, Any],
        design_space_factory: DesignSpaceFactory,
        fault_idx: list[int] | list[list[int]],
        comp_idx: list[list[int]] | None = None,
        info_other_sol: list[dict] | None = None,
    ) -> None:
        """Create the :class:`FaultScenario` and the :class:`.Fault` objects.

        .. todo::
            Could be cleaner.

        Parameters
        ----------
        ref_acc : Accelerator
            The reference linac (nominal or baseline).
        fix_acc : Accelerator
            The broken linac to be fixed.
        beam_calculator : BeamCalculator
            The solver that will be called during the optimisation process.
        initial_beam_parameters_factory : InitialBeamParametersFactory
            An object to create beam parameters at the entrance of the linac
            portion.
        wtf : dict[str, str | int | bool | list[str] | list[float]]
            What To Fit dictionary. Holds information on the fixing method.
        design_space_factory : DesignSpaceFactory
            An object to easily create the proper :class:`.DesignSpace`.
        fault_idx : list[int | list[int]]
            List containing the position of the errors. If ``strategy`` is
            manual, it is a list of lists (faults already gathered).
        comp_idx : list[list[int]], optional
            List containing the position of the compensating cavities. If
            ``strategy`` is manual, it must be provided. The default is None.
        info_other_sol : list[dict], optional
            Contains information on another fit, for comparison purposes. The
            default is None.

        """
        self.ref_acc, self.fix_acc = ref_acc, fix_acc
        self.beam_calculator = beam_calculator
        self._transfer_phi0_from_ref_to_broken()

        self.wtf = wtf
        self.info_other_sol = info_other_sol
        self.info = {}
        self.optimisation_time: datetime.timedelta

        cavities = strategy.failed_and_compensating(
            fix_acc.elts, failed=fault_idx, compensating_manual=comp_idx, **wtf
        )
        faults = self._set_faults(
            self._get_reference(),
            beam_calculator.list_of_elements_factory,
            design_space_factory,
            *cavities,
        )
        super().__init__(faults)
        self._set_optimisation_algorithms()

        # Change status of cavities after the first one that is down. Idea
        # is to keep relative phi_0 between ref and fix linacs (linac
        # rephasing)
        if not beam_calculator.flag_phi_abs:
            self._set_cavities_to_rephase()

        self._break_cavities()

    def _set_faults(
        self,
        reference_simulation_output: SimulationOutput,
        list_of_elements_factory: ListOfElementsFactory,
        design_space_factory: DesignSpaceFactory,
        *cavities: Sequence[Sequence[FieldMap]],
    ) -> list[Fault]:
        """Create the :class:`.Fault` objects.

        Parameters
        ----------
        reference_simulation_output : SimulationOutput
            The simulation of the nominal linac we'll try to match.
        list_of_elements_factory : ListOfElementsFactory
            An object that can create :class:`.ListOfElements`.
        design_space_factory : DesignSpaceFactory
            An object that can create :class:`.DesignSpace`.
        *cavities : Sequence[Sequence[FieldMap]]
            First if the list of gathered failed cavities. Second is the list
            of corresponding compensating cavities.

        Returns
        -------
        list[Fault]

        """
        faults = []
        files_from_full_list_of_elements = self.fix_acc.elts.files

        for faulty_cavities, compensating_cavities in zip(
            *cavities, strict=True
        ):
            fault = Fault(
                reference_elts=self.ref_acc.elts,
                reference_simulation_output=reference_simulation_output,
                files_from_full_list_of_elements=files_from_full_list_of_elements,
                wtf=self.wtf,
                design_space_factory=design_space_factory,
                broken_elts=self.fix_acc.elts,
                failed_elements=faulty_cavities,
                compensating_elements=compensating_cavities,
                list_of_elements_factory=list_of_elements_factory,
            )
            faults.append(fault)
        return faults

    def _get_reference(self) -> SimulationOutput:
        """Determine wich :class:`.SimulationOutput` is the reference."""
        solvers_already_used = list(self.ref_acc.simulation_outputs.keys())
        assert len(solvers_already_used) > 0, (
            "You must compute propagation of the beam in the reference linac "
            "prior to create a FaultScenario"
        )
        solv1 = solvers_already_used[0]
        reference_simulation_output = self.ref_acc.simulation_outputs[solv1]
        return reference_simulation_output

    def _set_cavities_to_rephase(self) -> None:
        """Change the status of cavities after first failure."""
        logging.warning(
            "The phases in the broken linac are relative. It may be more "
            "relatable to use absolute phases, as it would avoid the rephasing"
            " of the linac at each cavity."
        )
        cavities = self.fix_acc.l_cav
        first_failed_cavity = self[0].failed_elements[0]
        first_failed_index = cavities.index(first_failed_cavity)

        cavities_to_rephase = cavities[first_failed_index:]
        for cav in cavities_to_rephase:
            cav.update_status("rephased (in progress)")

    def _set_optimisation_algorithms(self) -> list[OptimisationAlgorithm]:
        """Set each fault's optimisation algorithm.

        Returns
        -------
        optimisation_algorithms : list[OptimisationAlgorithm]
            The optimisation algorithm for each fault in ``self``.

        """
        opti_method = self.wtf["optimisation_algorithm"]
        assert isinstance(opti_method, str)

        optimisation_algorithms = [
            optimisation_algorithm_factory(
                opti_method, fault, self.beam_calculator, **self.wtf
            )
            for fault in self
        ]
        return optimisation_algorithms

    def _break_cavities(self) -> None:
        """Break the cavities.

        ..todo ::
            At this step I do not save the ``.dat`` anymore. Was it useful for
            something?

        """
        for fault in self:
            fault.update_elements_status(optimisation="not started")
        logging.warning("Not storing elements in dat")
        # self.elts.store_settings_in_dat(
        #     self.elts.files["dat_file"], which_phase="phi_0_rel", save=True
        # )

    def fix_all(self) -> None:
        """
        Fix all the :class:`Fault` objects in self.

        .. todo::
            make this more readable

        """
        start_time = time.monotonic()

        success, info = [], []
        ref_simulation_output = self.ref_acc.simulation_outputs[
            self.beam_calculator.id
        ]
        optimisation_algorithms = self._set_optimisation_algorithms()

        for fault, optimisation_algorithm in zip(
            self, optimisation_algorithms
        ):
            _succ, optimized_cavity_settings, _info = fault.fix(
                optimisation_algorithm
            )

            success.append(_succ)
            info.append(_info)

            simulation_output = (
                self.beam_calculator.post_optimisation_run_with_this(
                    optimized_cavity_settings,
                    self.fix_acc.elts,
                )
            )
            simulation_output.compute_complementary_data(
                self.fix_acc.elts, ref_simulation_output=ref_simulation_output
            )

            self.fix_acc.keep_settings(
                simulation_output, output_phase=self._reference_phase
            )
            self.fix_acc.keep_simulation_output(
                simulation_output, self.beam_calculator.id
            )

            fault.update_elements_status(optimisation="finished", success=True)
            logging.warning("Manually set which_phase")
            fault.elts.store_settings_in_dat(
                fault.elts.files["dat_file"],
                which_phase=self._reference_phase,
                save=True,
            )

            if not self.beam_calculator.flag_phi_abs:
                # Tell LW to keep the new phase of the rephased cavities
                # between the two compensation zones
                self._reupdate_status_of_rephased_cavities(fault)
                logging.critical(
                    "Calculation in relative phase. Check if "
                    "necessary to reperform simulation?"
                )

        self.fix_acc.name = (
            f"Fixed ({str(success.count(True))}" + f" of {str(len(success))})"
        )

        for linac in [self.ref_acc, self.fix_acc]:
            self.info[linac.name + " cav"] = debug.output_cavities(
                linac, DISPLAY_CAVITIES_INFO
            )

        self._evaluate_fit_quality(save=True)

        end_time = time.monotonic()
        delta_t = datetime.timedelta(seconds=end_time - start_time)
        logging.info(f"Elapsed time in optimisation: {delta_t}")

        self.optimisation_time = delta_t
        # Legacy, does not work anymore with the new implementation
        # self.info['fit'] = debug.output_fit(self, FIT_COMPLETE, FIT_COMPACT)

    def _reupdate_status_of_rephased_cavities(self, fault: Fault) -> None:
        """Modify the status of the cavities that were already rephased.

        Change the cavities with status "rephased (in progress)" to
        "rephased (ok)" between the fault in argument and the next one.

        """
        logging.warning("Changed the way of defining idx1 and idx2.")
        elts = self.fix_acc.elts

        idx1 = fault.elts[-1].idx["elt_idx"]
        idx2 = len(elts)
        if fault is not self[-1]:
            next_fault = self[self.index(fault) + 1]
            idx2 = next_fault.elts[0].idx["elt_idx"] + 1

        rephased_cavities_between_two_faults = [
            elt
            for elt in elts[idx1:idx2]
            if elt.get("nature") == "FIELD_MAP"
            and elt.get("status") == "rephased (in progress)"
        ]

        for cav in rephased_cavities_between_two_faults:
            cav.update_status("rephased (ok)")
        return
        # New implementation to test
        idx_start = elts.index(fault.elts[-1])
        for elt in elts[idx_start:]:
            if elt.nature != "FIELD_MAP":
                continue
            if elt.status == "rephased (in progress)":
                elt.update_status("rephased (ok)")
                continue
            if "compensate" in elt.status or "failed" in elt.status:
                return

    def _transfer_phi0_from_ref_to_broken(self) -> None:
        """Transfer the entry phases from reference linac to broken.

        If the absolute initial phases are not kept between reference and
        broken linac, it comes down to rephasing the linac. This is what we
        want to avoid when ``beam_calculator.flag_phi_abs == True``.

        """
        ref_settings = [x.cavity_settings for x in self.ref_acc.l_cav]
        fix_settings = [x.cavity_settings for x in self.fix_acc.l_cav]

        for ref, fix in zip(ref_settings, fix_settings):
            phi_0_ref = getattr(ref, self._reference_phase)
            fix.reference = self._reference_phase
            fix.phi_ref = phi_0_ref

    @property
    def _reference_phase(self) -> str:
        """Give the reference phase: `phi_0_rel` or 'phi_0_abs'."""
        return self.beam_calculator.reference_phase

    def _evaluate_fit_quality(
        self,
        save: bool = True,
        id_solver_ref: str | None = None,
        id_solver_fix: str | None = None,
    ) -> None:
        """
        Compute some quantities on the whole linac to see if fit is good.

        Parameters
        ----------
        save : bool, optional
            To tell if you want to save the evaluation. The default is True.
        id_solver_ref : str | None, optional
            Id of the solver from which you want reference results. The default
            is None. In this case, the first solver is taken
            (``beam_calc_param``).
        id_solver_fix : str | None, optional
            Id of the solver from which you want fixed results. The default is
            None. In this case, the solver is the same as for reference.

        """
        simulations = self._simulations_that_should_be_compared(
            id_solver_ref, id_solver_fix
        )

        quantities_to_evaluate = (
            "w_kin",
            "phi_abs",
            "envelope_pos_phiw",
            "envelope_energy_phiw",
            "eps_phiw",
            "mismatch_factor_zdelta",
        )
        my_evaluator = FaultScenarioSimulationOutputEvaluators(
            quantities_to_evaluate, [fault for fault in self], simulations
        )
        my_evaluator.run(output=True)

        # if save:
        #     fname = 'evaluations_differences_between_simulation_output.csv'
        #     out = os.path.join(self.fix_acc.get('beam_calc_path'), fname)
        #     df_eval.to_csv(out)

    def _set_evaluation_elements(
        self,
        additional_elt: list[Element] | None = None,
    ) -> dict[str, Element]:
        """Set a the proper list of where to check the fit quality."""
        evaluation_elements = [fault.elts[-1] for fault in self]
        if additional_elt is not None:
            evaluation_elements += additional_elt
        evaluation_elements.append(self.fix_acc.elts[-1])
        return evaluation_elements

    def _simulations_that_should_be_compared(
        self, id_solver_ref: str | None, id_solver_fix: str | None
    ) -> tuple[SimulationOutput, SimulationOutput]:
        """Get proper :class:`SimulationOutput` for comparison."""
        if id_solver_ref is None:
            id_solver_ref = list(self.ref_acc.simulation_outputs.keys())[0]

        if id_solver_fix is None:
            id_solver_fix = id_solver_ref

        if id_solver_ref != id_solver_fix:
            logging.warning(
                "You are trying to compare two SimulationOutputs "
                "created by two different solvers. This may lead "
                "to errors, as interpolations in this case are not"
                " implemented yet."
            )

        ref_simu = self.ref_acc.simulation_outputs[id_solver_ref]
        fix_simu = self.fix_acc.simulation_outputs[id_solver_fix]
        return ref_simu, fix_simu

    def pickle(
        self, pickler: MyPickler, path: Path | str | None = None
    ) -> Path:
        """Pickle (save) the object.

        This is useful for debug and temporary saves; do not use it for long
        time saving.

        """
        if path is None:
            path = self.fix_acc.accelerator_path / "fault_scenario.pkl"
        assert isinstance(path, Path)
        pickler.pickle(self, path)

        if isinstance(path, str):
            path = Path(path)
        return path

    @classmethod
    def from_pickle(cls, pickler: MyPickler, path: Path | str) -> Self:
        """Instantiate object from previously pickled file."""
        fault_scenario = pickler.unpickle(path)
        return fault_scenario  # type: ignore


def fault_scenario_factory(
    accelerators: list[Accelerator],
    beam_calculator: BeamCalculator,
    wtf: dict[str, Any],
    design_space_kw: dict[str, str | bool | float],
) -> list[FaultScenario]:
    """Create the :class:`FaultScenario` objects (factory template).

    Parameters
    ----------
    accelerators : list[Accelerator]
        Holds all the linacs. The first one must be the reference linac,
        while all the others will be to be fixed.
    beam_calculator : BeamCalculator
        The solver that will be called during the optimisation process.
    wtf : dict[str, str | int | bool | list[str] | list[float]]
        What To Fit dictionary. Holds information on the fixing method.
    design_space_kw : dict[str, str | bool | float]
        The ``[design_space]`` entries from the ``.ini`` file.

    Returns
    -------
    fault_scenarios : list[FaultScenario]
        Holds all the initialized :class:`FaultScenario` objects, holding their
        already initialied :class:`Fault` objects.

    """
    # TODO may be better to move this to beam_calculator.init_solver_parameters
    need_to_force_element_to_index_creation = (TraceWin,)
    if isinstance(beam_calculator, *need_to_force_element_to_index_creation):
        _force_element_to_index_method_creation(
            accelerators[1], beam_calculator
        )
    scenarios_fault_idx = wtf.pop("failed")

    scenarios_comp_idx = [None for _ in accelerators[1:]]
    if "compensating_manual" in wtf:
        scenarios_comp_idx = wtf.pop("compensating_manual")

    _ = [
        beam_calculator.init_solver_parameters(accelerator)
        for accelerator in accelerators
    ]

    design_space_factory: DesignSpaceFactory
    design_space_factory = get_design_space_factory(**design_space_kw)

    fault_scenarios = [
        FaultScenario(
            ref_acc=accelerators[0],
            fix_acc=accelerator,
            beam_calculator=beam_calculator,
            wtf=wtf,
            design_space_factory=design_space_factory,
            fault_idx=fault_idx,
            comp_idx=comp_idx,
        )
        for accelerator, fault_idx, comp_idx in zip(
            accelerators[1:], scenarios_fault_idx, scenarios_comp_idx
        )
    ]

    return fault_scenarios


def _force_element_to_index_method_creation(
    accelerator: Accelerator,
    beam_calculator: BeamCalculator,
) -> None:
    """Run a first simulation to link :class:`.Element` with their index.

    .. note::
        To initalize a :class:`.Fault`, you need a sub:class:`.ListOfElements`.
        To create the latter, you need a ``_element_to_index`` method. It can
        only be created if you know the number of steps in every
        :class:`.Element`. So, for :class:`.TraceWin`, we run a first
        simulation.

    """
    beam_calculator.compute(accelerator)
