#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a useless command to serve as place holder."""
from core.instruction import Instruction
from core.commands.command import Command
from core.elements.field_maps.field_map import FieldMap


class FieldMapPath(Command):
    """Used to get the base path of field maps."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Save the given path as attribute."""
        super().__init__(line, dat_idx, is_implemented=True)
        self.path = line[1]

    def set_influenced_elements(self,
                                instructions: list[Instruction],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx['dat_idx'] + 1
        stop = start
        for instruction in instructions[start:]:
            if isinstance(instruction, FieldMapPath):
                self.idx['influenced'] = slice(start, stop)
                return
            stop += 1
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              instructions: list[Instruction],
              **kwargs: float
              ) -> list[Instruction]:
        """
        Set :class:`FieldMap` field folder up.

        If another :class:`FieldMapPath` is found, we stop and this command
        will be applied later.

        """
        for instruction in instructions[self.idx['influenced']]:
            if isinstance(instruction, FieldMap):
                instruction.field_map_folder = self.path
        return instructions