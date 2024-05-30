"""Define a factory to easily create :class:`.Accelerator`."""

import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from beam_calculation.beam_calculator import BeamCalculator
from core.accelerator.accelerator import Accelerator
from core.elements.field_maps.field_map import FieldMap


@dataclass
class AcceleratorFactory(ABC):
    """A class to create accelerators."""

    def __init__(
        self,
        dat_file: Path,
        project_folder: Path,
        beam_calculators: BeamCalculator | Sequence[BeamCalculator | None],
        **files_kw: str | Path,
    ) -> None:
        """Create the object from the ``project_folder``.

        Parameters
        ----------
        dat_file : Path
            The original ``.dat`` file, as understood by TraceWin.
        project_folder : Path
            Base folder where results will be saved. If the ``project_folder``
            key is not given in the configuration file, it is a folder in the
            same base folder as ``dat_file``.
        files_kw :
            Other arguments from the ``file`` entry of the ``.ini``.

        """
        self.dat_file = dat_file
        self.project_folder = project_folder

        if isinstance(beam_calculators, BeamCalculator):
            beam_calculators = (beam_calculators,)
        assert (
            beam_calculators[0] is not None
        ), "Need at least one working BeamCalculator."
        self.beam_calculators = beam_calculators

    def run(self, *args, **kwargs) -> Accelerator:
        """Create the object."""
        accelerator = Accelerator(*args, **kwargs)
        self._check_consistency_absolute_phases(accelerator.l_cav)
        return accelerator

    def _generate_folders_tree_structure(
        self,
        out_folders: Sequence[Path],
        n_simulations: int,
    ) -> list[Path]:
        """Create the proper folders for every :class:`.Accelerator`.

        The default structure is:

        where_original_dat_is/
            YYYY.MM.DD_HHhMM_SSs_MILLIms/         <- project_folder (abs)
                000000_ref/                       <- accelerator_path (abs)
                    0_FirstBeamCalculatorName/    <- out_folder (rel)
                    (1_SecondBeamCalculatorName/) <- out_folder (rel)
                000001/
                    0_FirstBeamCalculatorName/
                    (1_SecondBeamCalculatorName/)
                000002/
                    0_FirstBeamCalculatorName/
                    (1_SecondBeamCalculatorName/)
                etc

        Parameters
        ----------
        out_folders : Sequence[Path]
            Name of the folders that will store outputs. By default, it is the
            name of the solver, preceeded by its position in the list of
            :class:`.BeamCalculator`.

        """
        accelerator_paths = [
            self.project_folder / f"{i:06d}" for i in range(n_simulations)
        ]
        accelerator_paths[0] = accelerator_paths[0].with_name(
            f"{accelerator_paths[0].name}_ref"
        )
        for accel_path in accelerator_paths:
            for out_folder in out_folders:
                path = accel_path / out_folder
                path.mkdir(parents=True, exist_ok=True)
        return accelerator_paths

    def _check_consistency_absolute_phases(
        self, cavities: Sequence[FieldMap]
    ) -> None:
        """Check that solvers phases are consistent with ``.dat`` file."""
        beam_calculators = [x for x in self.beam_calculators if x is not None]
        beam_calculators_flags = set(
            [
                beam_calculator.flag_phi_abs
                for beam_calculator in beam_calculators
            ]
        )

        if len(beam_calculators_flags) > 1:
            logging.debug(
                "BeamCalculator objects have different flag_phi_abs values."
                " Warning already raised in BeamCalculatorsFactory"
            )
            return

        cavities_references = [x.cavity_settings.reference for x in cavities]
        if len(set(cavities_references)) > 1:
            logging.warning(
                "The cavities do not all have the same reference phase. This "
                "may lead to inconsistencies."
            )
            return

        if (
            cavities_references[0] == "phi_0_abs"
            and not beam_calculators[0].flag_phi_abs
        ):
            logging.warning(
                "You asked LW a simulation in relative phase, while there "
                "is at least one cavity in absolute phase in the .dat file "
                "used by TW. You can expect a phase difference between "
                "LightWin results and TraceWin results for the same input "
                ".dat, after the first failed cavity. No difference should "
                "appear with the output .dat, or when using TraceWin solver "
                "within LightWin."
            )
            return

        if (
            cavities_references[0] == "phi_0_rel"
            and beam_calculators[0].flag_phi_abs
        ):
            logging.warning(
                "You asked LW a simulation in absolute phase, while there "
                "is at least one cavity in relative phase in the .dat file "
                "used by TW. You can expect a phase difference between "
                "LightWin results and TraceWin results for the same input "
                ".dat, after the first failed cavity. No difference should "
                "appear with the output .dat, or when using TraceWin solver "
                "within LightWin."
            )
            return


class NoFault(AcceleratorFactory):
    """Factory used to generate a single accelerator, no faults."""

    def __init__(
        self,
        dat_file: Path,
        project_folder: Path,
        beam_calculator: BeamCalculator,
        **files_kw,
    ) -> None:
        """Initialize."""
        super().__init__(
            dat_file,
            project_folder,
            beam_calculators=beam_calculator,
            **files_kw,
        )

    @property
    def beam_calculator(self) -> BeamCalculator:
        """Shortcut to get the only existing :class:`.BeamCalculator`."""
        return self.beam_calculators[0]

    def run(self, *args, **kwargs) -> Accelerator:
        """Create a single accelerator."""
        out_folders = (self.beam_calculator.out_folder,)
        accelerator_path = self._generate_folders_tree_structure(
            out_folders,
            n_simulations=1,
        )[0]
        list_of_elements_factory = (
            self.beam_calculator.list_of_elements_factory
        )
        name = "Working"

        accelerator = super().run(
            name=name,
            dat_file=self.dat_file,
            accelerator_path=accelerator_path,
            list_of_elements_factory=list_of_elements_factory,
        )
        return accelerator


class StudyWithoutFaultsAcceleratorFactory(NoFault):
    """Alias for :class:`.NoFault`."""


class WithFaults(AcceleratorFactory):
    """Factory used to generate several accelerators for a fault study."""

    def __init__(
        self,
        dat_file: Path,
        project_folder: Path,
        beam_calculators: BeamCalculator | Sequence[BeamCalculator | None],
        failed: list[list[int]] | None = None,
        **kwargs: Path | str | float | list[int],
    ) -> None:
        """Initialize."""
        super().__init__(dat_file, project_folder, beam_calculators, **kwargs)
        self.failed = failed

        self._n_simulations = 0

    @property
    def n_simulations(self) -> int:
        """Determine how much simulations will be made."""
        if self._n_simulations > 0:
            return self._n_simulations

        self._n_simulations = 1

        if self.failed is not None:
            self._n_simulations += len(self.failed)

        return self._n_simulations

    def run(self, *args, **kwargs) -> Accelerator:
        """Return a single accelerator."""
        return super().run(*args, **kwargs)

    def run_all(self, **kwargs) -> list[Accelerator]:
        """Create the required Accelerators as well as their output folders."""
        out_folders = [
            beam_calculator.out_folder
            for beam_calculator in self.beam_calculators
            if beam_calculator is not None
        ]

        accelerator_paths = self._generate_folders_tree_structure(
            out_folders, n_simulations=self.n_simulations
        )

        names = [
            "Working" if i == 0 else "Broken"
            for i in range(self.n_simulations)
        ]

        list_of_elements_factory = self.beam_calculators[
            0
        ].list_of_elements_factory

        accelerators = [
            self.run(
                name=name,
                dat_file=self.dat_file,
                accelerator_path=accelerator_path,
                list_of_elements_factory=list_of_elements_factory,
            )
            for name, accelerator_path in zip(names, accelerator_paths)
        ]
        return accelerators


class FullStudyAcceleratorFactory(WithFaults):
    """Alias for :class:`WithFaults`."""
