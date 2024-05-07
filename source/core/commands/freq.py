#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a useless command to serve as place holder."""
import logging

from core.commands.command import Command
from core.elements.field_maps.field_map import FieldMap
from core.instruction import Instruction


class Freq(Command):
    """Used to get the frequency of every Section."""

    is_implemented = True
    n_attributes = 1

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Save frequency as attribute."""
        super().__init__(line, dat_idx)
        self.f_rf_mhz = float(line[1])

    def set_influenced_elements(
        self, instructions: list[Instruction], **kwargs: float
    ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx["dat_idx"] + 1
        stop = start
        for instruction in instructions[start:]:
            if isinstance(instruction, Freq):
                self.influenced = slice(start, stop)
                return
            stop += 1
        self.influenced = slice(start, stop)

    def apply(
        self,
        instructions: list[Instruction],
        freq_bunch: float | None = None,
        **kwargs: float,
    ) -> list[Instruction]:
        """Set :class:`.FieldMap` frequency.

        If another :class:`Freq` is found, we stop and the new :class:`Freq`
        will be dealt with later.

        """
        if freq_bunch is None:
            logging.warning(
                "The bunch frequency was not provided. Setting it to RF "
                "frequency..."
            )
            freq_bunch = self.f_rf_mhz

        for instruction in instructions[self.influenced]:
            if isinstance(instruction, FieldMap):
                instruction.cavity_settings.set_bunch_to_rf_freq_func(
                    self.f_rf_mhz
                )
        return instructions
