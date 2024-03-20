#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds a factory to create the :class:`.BeamCalculator`."""
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.envelope_1d.envelope_1d import Envelope1D
from beam_calculation.envelope_3d.envelope_3d import Envelope3D
from beam_calculation.tracewin.tracewin import TraceWin

IMPLEMENTED_BEAM_CALCULATORS = {
    'Envelope1D': Envelope1D,
    'TraceWin': TraceWin,
    'Envelope3D': Envelope3D,
}  #:


class BeamCalculatorsFactory:
    """A class to create :class:`.BeamCalculator` objects."""

    def __init__(
            self,
            beam_calculator: dict[str, Any],
            files: dict[str, Any],
            beam_calculator_post: dict[str, Any] | None = None,
            out_folder: Path | Sequence[Path] | None = None,
            **other_kw: dict) -> None:
        """Create the object."""
        self.all_beam_calculator_kw = beam_calculator,
        if beam_calculator_post is not None:
            self.all_beam_calculator_kw = (beam_calculator,
                                           beam_calculator_post)

        self.out_folders = self._set_out_folders(self.all_beam_calculator_kw,
                                                 out_folder)

        self.beam_calculators_id: list[str] = []
        self._patch_to_remove_misunderstood_key()
        self._original_dat_dir: Path = files['dat_file'].parent

    def _set_out_folders(self,
                         all_beam_calculator_kw: Sequence[dict[str, Any]],
                         out_folder: Path | Sequence[Path] | None = None,
                         ) -> list[Path]:
        """Set in which subfolder the results will be saved."""
        if isinstance(out_folder, Path):
            out_folders = [out_folder for _ in all_beam_calculator_kw]
            if len(out_folders) > 1:
                logging.warning(f"You asked for several BeamCalculator but "
                                f"provided only one {out_folder = }. Results "
                                "from all the BeamCalculators will be saved in"
                                " the same folder, which may lead to "
                                "conflicts.")
            return out_folders

        if isinstance(out_folder, Sequence):
            out_folders = out_folder
            assert (n_out_folders := len(out_folders)) \
                == (n_beam_calculators := len(all_beam_calculator_kw)), (
                f"Mismatch between {n_out_folders = } and "
                f"{n_beam_calculators = }")
            return list(out_folders)

        out_folders = [Path(f"{i}_{kw['tool']}")
                       for i, kw in enumerate(all_beam_calculator_kw)]
        return out_folders

    def _patch_to_remove_misunderstood_key(self) -> None:
        """Patch to remove a key not understood by TraceWin. Declare id list.

        .. todo::
            fixme

        """
        for beam_calculator_kw in self.all_beam_calculator_kw:
            if 'simulation type' in beam_calculator_kw:
                del beam_calculator_kw['simulation type']

    def run(self, tool: str, **beam_calculator_kw) -> BeamCalculator:
        """Create a single :class:`.BeamCalculator`.

        Parameters
        ----------
        beam_calculator_class : ABCMeta
            The specific beam calculator.

        Returns
        -------
        BeamCalculator

        """
        beam_calculator_class = IMPLEMENTED_BEAM_CALCULATORS[tool]
        beam_calculator = beam_calculator_class(
            out_folder=self.out_folders.pop(0),
            default_field_map_folder=self._original_dat_dir,
            **beam_calculator_kw)
        self.beam_calculators_id.append(beam_calculator.id)
        return beam_calculator

    def run_all(self) -> tuple[BeamCalculator, ...]:
        """Create all the beam calculators."""
        beam_calculators = [
            self.run(**beam_calculator_kw)
            for beam_calculator_kw in self.all_beam_calculator_kw]
        return tuple(beam_calculators)
