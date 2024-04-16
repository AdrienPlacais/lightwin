#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a command to indicate end of the linac."""
from core.commands.command import Command
from core.instruction import Instruction


class End(Command):
    """The end of the linac."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Call mother ``__init__``."""
        super().__init__(line, dat_idx, is_implemented=True)

    def set_influenced_elements(self, *args, **kwargs: float) -> None:
        """Determine the index of the element concerned by :func:`apply`."""
        start = 0
        stop = self.idx["dat_idx"] + 1
        self.influenced = slice(start, stop)

    def apply(
        self, instructions: list[Instruction], **kwargs: float
    ) -> list[Instruction]:
        """Remove everything in ``instructions`` after this object."""
        return instructions[self.influenced]
