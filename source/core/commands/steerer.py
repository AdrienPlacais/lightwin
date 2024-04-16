#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a useless command to serve as place holder."""
import logging

from core.commands.command import Command
from core.elements.element import Element
from core.instruction import Instruction


class Steerer(Command):
    """Dummy class."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Call the mother ``__init__`` method."""
        super().__init__(line, dat_idx, is_implemented=False)

    def set_influenced_elements(
        self, instructions: list[Instruction], **kwargs: float
    ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        next_element = list(
            filter(
                lambda elt: isinstance(elt, Element),
                instructions[self.idx["dat_idx"] :],
            )
        )[0]
        start = next_element.idx["dat_idx"]
        stop = start + 1
        self.influenced = slice(start, stop)

    def apply(
        self, instructions: list[Instruction], **kwargs: float
    ) -> list[Instruction]:
        """Do nothing."""
        logging.error("Steerer not implemented.")
        return instructions
