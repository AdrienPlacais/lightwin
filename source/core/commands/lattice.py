#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define ``LATTICE`` and ``LATTICE_END`` instructions."""
import logging

from core.instruction import Instruction
from core.commands.command import Command
from core.commands.superpose_map import SuperposeMap
from core.instruction import Comment
from core.elements.element import Element


class Lattice(Command):
    """Used to get the number of elements per lattice."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Save lattice structure."""
        super().__init__(line, dat_idx, is_implemented=True)
        self.n_lattice = int(line[1])

        self.n_macro_lattice = 1
        if len(line) > 2:
            self.n_macro_lattice = int(line[2])

            if self.n_macro_lattice > 1:
                logging.warning("Macro-lattice not implemented. LightWin will "
                                "consider that number of macro-lattice per "
                                "lattice is 1 or 0.")

    def set_influenced_elements(self,
                                instructions: list[Instruction],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx['dat_idx'] + 1
        stop = start
        for instruction in instructions[start:]:
            if isinstance(instruction, Lattice | LatticeEnd):
                self.idx['influenced'] = slice(start, stop)
                return

            stop += 1
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              instructions: list[Instruction],
              **kwargs: float
              ) -> list[Instruction]:
        """Set lattice section number of elements in current lattice."""
        index = self.idx['dat_idx']

        current_lattice_number = self._current_lattice_number(instructions,
                                                              index)
        current_section_number = self._current_section_number(instructions)

        index_in_current_lattice = 0
        for instruction in instructions[self.idx['influenced']]:
            if isinstance(instruction, SuperposeMap):
                logging.error("SuperposeMap not implemented. Will mess with "
                              "indexes...")

            if isinstance(instruction, (Command, Comment)):
                continue

            instruction.idx['lattice'] = current_lattice_number
            instruction.idx['section'] = current_section_number
            instruction.idx['idx_in_lattice'] = index_in_current_lattice

            index_in_current_lattice += 1
            if index_in_current_lattice == self.n_lattice:
                current_lattice_number += 1
                index_in_current_lattice = 0

        return instructions

    def _current_section_number(self,
                                instructions: list[Instruction],
                                ) -> int:
        """Get section number of ``self``."""
        all_lattice_commands = list(filter(
            lambda instruction: isinstance(instruction, Lattice),
            instructions))
        return all_lattice_commands.index(self)

    def _current_lattice_number(self,
                                instructions: list[Instruction],
                                index: int
                                ) -> int:
        """
        Get lattice number of current object.

        We look for :class:`Element` in ``instructions``in reversed order,
        starting from ``self``. We take the first non None lattice index that
        we find, and return it + 1.
        If we do not find anything, this is because no :class:`Element` had a
        defined lattice number before.

        This approach allows for :class:`Element` without a lattice number, as
        for examples drifts between a :class:`LatticeEnd` and a
        :class:`Lattice`.

        """
        instructions_before_self = instructions[:index]
        reversed_instructions_before_self = instructions_before_self[::-1]

        for instruction in reversed_instructions_before_self:
            if isinstance(instruction, Element):
                previous_lattice_number = instruction.get('lattice',
                                                          to_numpy=False)

                if previous_lattice_number is not None:
                    return previous_lattice_number + 1
        return 0


class LatticeEnd(Command):
    """Define the end of lattice."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Call mother ``__init__`` method."""
        super().__init__(line, dat_idx, is_implemented=True)

    def set_influenced_elements(self,
                                instructions: list[Instruction],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx['dat_idx']
        stop = start
        for instruction in instructions[start:]:
            if isinstance(instruction, Lattice):
                self.idx['influenced'] = slice(start, stop)
                return
            stop += 1
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              instructions: list[Instruction],
              **kwargs: float
              ) -> list[Instruction]:
        """Do nothing, everything is handled by :class:`Lattice`."""
        return instructions
