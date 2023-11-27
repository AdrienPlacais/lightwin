#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds a factory to create the :class:`.BeamCalculator`."""
from typing import Any
from pathlib import Path

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
            **other_kw: dict) -> None:
        """Create the object."""
        self.all_beam_calculator_kw = (beam_calculator, )

        if beam_calculator_post is not None:
            self.all_beam_calculator_kw = (beam_calculator,
                                           beam_calculator_post)

        self.beam_calculators_id: list[str] = []
        self._patch_to_remove_misunderstood_key()
        self._original_dat_dir: Path = files['dat_file'].parent

    def _patch_to_remove_misunderstood_key(self) -> None:
        """Patch to remove a key not understood by TraceWin. Declare id list.

        .. todo::
            fixme

        """
        for beam_calculator_kw in self.all_beam_calculator_kw:
            if 'simulation type' in beam_calculator_kw:
                del beam_calculator_kw['simulation type']

    def _run(self,
             tool: str,
             out_folder: Path,
             **beam_calculator_kw) -> BeamCalculator:
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
            out_folder=out_folder,
            default_field_map_folder=self._original_dat_dir,
            **beam_calculator_kw)
        self.beam_calculators_id.append(beam_calculator.id)
        return beam_calculator

    def run_all(self) -> tuple[BeamCalculator, ...]:
        """Create all the beam calculators."""
        out_folders = (Path(f"beam_calculation_{i}")
                       for i, _ in enumerate(self.all_beam_calculator_kw))
        beam_calculators = [
            self._run(out_folder=out_folder,
                      **beam_calculator_kw)
            for beam_calculator_kw, out_folder
            in zip(self.all_beam_calculator_kw, out_folders)]
        return tuple(beam_calculators)
