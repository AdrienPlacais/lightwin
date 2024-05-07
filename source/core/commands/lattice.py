#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define ``LATTICE`` and ``LATTICE_END`` instructions."""
import logging

from core.commands.command import Command
from core.commands.superpose_map import SuperposeMap
from core.elements.element import Element
from core.instruction import Comment, Instruction


class Lattice(Command):
    """Used to get the number of elements per lattice."""

    is_implemented = True
    n_attributes = (1, 2)

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Save lattice structure."""
        super().__init__(line, dat_idx)
        self.n_lattice = int(line[1])

        self.n_macro_lattice = 1
        if len(line) > 2:
            self.n_macro_lattice = int(line[2])

            if self.n_macro_lattice > 1:
                logging.warning(
                    "Macro-lattice not implemented. LightWin will consider "
                    "that number of macro-lattice per lattice is 1 or 0."
                )

    def set_influenced_elements(
        self, instructions: list[Instruction], **kwargs: float
    ) -> None:
        """Determine the index of the elements concerned by :func:`apply`.

        Here, this is all the elements between this command and the next
        ``LATTICE`` or ``LATTICE_END`` instruction.

        """
        start = self.idx["dat_idx"] + 1
        self.influenced = self._indexes_between_this_command_and(
            instructions[start:], Lattice, LatticeEnd
        )

    def apply(
        self, instructions: list[Instruction], **kwargs: float
    ) -> list[Instruction]:
        """Set lattice section number of elements in current lattice."""
        index = self.idx["dat_idx"]

        current_lattice_number = self._current_lattice_number(
            instructions, index
        )
        current_section_number = self._current_section_number(instructions)

        index_in_current_lattice = 0
        for instruction in instructions[self.influenced]:
            if isinstance(instruction, SuperposeMap):
                logging.error(
                    "SuperposeMap not implemented. Will mess with indexes..."
                )

            if isinstance(instruction, (Command, Comment)):
                continue
            assert isinstance(element := instruction, Element)
            if not element.increment_lattice_idx:
                continue

            element.idx["lattice"] = current_lattice_number
            element.idx["section"] = current_section_number
            element.idx["idx_in_lattice"] = index_in_current_lattice

            index_in_current_lattice += 1
            if index_in_current_lattice == self.n_lattice:
                current_lattice_number += 1
                index_in_current_lattice = 0

        return instructions

    def _current_section_number(self, instructions: list[Instruction]) -> int:
        """Get section number of ``self``."""
        all_lattice_commands = list(
            filter(
                lambda instruction: isinstance(instruction, Lattice),
                instructions,
            )
        )
        return all_lattice_commands.index(self)

    def _current_lattice_number(
        self, instructions: list[Instruction], index: int
    ) -> int:
        """Get lattice number of current object.

        We look for :class:`Element` in ``instructions`` in reversed order,
        starting from ``self``. We take the first non negative lattice index
        that we find, and return it + 1.
        If we do not find anything, this is because no :class:`Element` had a
        defined lattice number before.

        This approach allows for :class:`Element` without a lattice number, as
        for examples drifts between a :class:`LatticeEnd` and a
        :class:`Lattice`.

        """
        instructions_before_self = instructions[:index]
        reversed_instructions_before_self = instructions_before_self[::-1]

        for instruction in reversed_instructions_before_self:
            if isinstance(element := instruction, Element):
                previous_lattice_number = instruction.idx["lattice"]

                if previous_lattice_number >= 0:
                    return previous_lattice_number + 1
        return 0


class LatticeEnd(Command):
    """Define the end of lattice."""

    is_implemented = True
    n_attributes = 0

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Call mother ``__init__`` method."""
        super().__init__(line, dat_idx)

    def set_influenced_elements(
        self, instructions: list[Instruction], **kwargs: float
    ) -> None:
        """Determine the index of the elements concerned by :func:`apply`.

        Here, this is all the elements that are between this command and the
        next ``LATTICE`` instruction.

        """
        start = self.idx["dat_idx"] + 1
        self.influenced = self._indexes_between_this_command_and(
            instructions[start:], Lattice
        )

    def apply(
        self, instructions: list[Instruction], **kwargs: float
    ) -> list[Instruction]:
        """Reset the lattice index of every influenced element.

        .. todo::
            As for now, the effect of this command will be overriden by the
            _force_a_lattice_for_every_element. See how I should handle this...

        """
        for instruction in instructions[self.influenced]:
            if not isinstance(element := instruction, Element):
                continue
            element.idx["lattice"] = -1
            element.idx["idx_in_lattice"] = -1
        return instructions
