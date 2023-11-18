#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a factory to easily create :class:`.Accelerator`."""
from abc import ABC, abstractmethod
import os.path
from dataclasses import dataclass

from core.accelerator.accelerator import Accelerator
from beam_calculation.beam_calculator import BeamCalculator


@dataclass
class AcceleratorFactory(ABC):
    """A class to create accelerators."""

    def __init__(self,
                 dat_file: str,
                 project_folder: str,
                 **files_kw: str) -> None:
        """Create the object from the ``project_folder``.

        Parameters
        ----------
        project_folder : str
            Base folder where ``.dat`` should be found.
        files_kw :
            Other arguments from the ``file`` entry of the ``.ini``.

        """
        self.dat_file = dat_file
        self.project_folder = project_folder

    @abstractmethod
    def run(self, *args, **kwargs) -> Accelerator:
        """Create the object."""
        return Accelerator(*args, **kwargs)


class FullStudyAcceleratorFactory(AcceleratorFactory):
    """Factory used to generate several accelerators for a fault study."""

    def __init__(self,
                 dat_file: str,
                 project_folder: str,
                 beam_calculators: tuple[BeamCalculator | None, ...],
                 failed: list[list[int]] | None = None,
                 **kwargs: str | float | list[int],
                 ) -> None:
        """Initialize."""
        super().__init__(dat_file,
                         project_folder,
                         **kwargs)
        assert beam_calculators[0] is not None, "Need at least one working "\
            "BeamCalculator."
        self.beam_calculators = beam_calculators
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
        return Accelerator(*args, **kwargs)

    def run_all(self,
                **kwargs
                ) -> list[Accelerator]:
        """Create the required Accelerators as well as their output folders."""
        out_folders = tuple([beam_calculator.out_folder
                            for beam_calculator in self.beam_calculators
                            if beam_calculator is not None
                             ])

        accelerator_paths = self._generate_folders_tree_structure(
            out_folders=out_folders
        )

        names = ['Working' if i == 0 else 'Broken'
                 for i in range(self.n_simulations)]

        list_of_elements_factory = \
            self.beam_calculators[0].list_of_elements_factory

        accelerators = [self.run(
            name=name,
            dat_file=self.dat_file,
            project_folder=self.project_folder,
            accelerator_path=accelerator_path,
            list_of_elements_factory=list_of_elements_factory,
        ) for name, accelerator_path in zip(names, accelerator_paths)]
        return accelerators

    def _generate_folders_tree_structure(self,
                                         out_folders: tuple[str, ...]
                                         ) -> list[str]:
        """
        Create the proper folders for every :class:`.Accelerator`.

        The default structure is:

        where_original_dat_is/
            YYYY.MM.DD_HHhMM_SSs_MILLIms/              <- project_folder
                000000_ref/                            <- accelerator_path
                    beam_calculation_toolname/         <- beam_calc
                    (beam_calculation_post_toolname)/  <- beam_calc_post
                000001/
                    beam_calculation_toolname/
                    (beam_calculation_post_toolname)/
                000002/
                    beam_calculation_toolname/
                    (beam_calculation_post_toolname)/
                etc

        """
        accelerator_paths = [os.path.join(self.project_folder, f"{i:06d}")
                             for i in range(self.n_simulations)]
        accelerator_paths[0] += '_ref'

        _ = [os.makedirs(os.path.join(fault_scenar, out_folder))
             for fault_scenar in accelerator_paths
             for out_folder in out_folders]
        return accelerator_paths
