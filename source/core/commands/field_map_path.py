#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a useless command to serve as place holder."""
import logging
from pathlib import Path

from core.commands.command import Command
from core.elements.field_maps.field_map import FieldMap
from core.instruction import Instruction


class FieldMapPath(Command):
    """Used to get the base path of field maps."""

    is_implemented = True
    n_attributes = 1

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        default_field_map_folder: Path,
        **kwargs: str,
    ) -> None:
        """Save the given path as attribute."""
        super().__init__(line, dat_idx)
        path = Path(line[1])

        self.path = path
        if self.path.is_dir():
            return

        self.path = default_field_map_folder / path
        if self.path.is_dir():
            return

        self.path = default_field_map_folder
        if self.path.is_dir():
            return

        logging.critical(
            f"The {path = } given in FIELD_MAP_PATH was not found."
        )

    def set_influenced_elements(
        self, instructions: list[Instruction], **kwargs: float
    ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx["dat_idx"] + 1
        stop = start
        for instruction in instructions[start:]:
            if isinstance(instruction, FieldMapPath):
                self.influenced = slice(start, stop)
                return
            stop += 1
        self.influenced = slice(start, stop)

    def apply(
        self, instructions: list[Instruction], **kwargs: float
    ) -> list[Instruction]:
        """Set :class:`.FieldMap` field folder up.

        If another :class:`FieldMapPath` is found, we stop and this command
        will be applied later.

        """
        for instruction in instructions[self.influenced]:
            if isinstance(instruction, FieldMap):
                instruction.field_map_folder = self.path
        return instructions
