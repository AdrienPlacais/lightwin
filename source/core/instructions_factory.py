#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define methods to easily create :class:`.Command` or :class:`.Element`."""
from typing import Any

from core.instruction import Instruction, Dummy
from core.elements.factory import ElementFactory, IMPLEMENTED_ELEMENTS
from core.commands.command import Command
from core.commands.factory import CommandFactory, IMPLEMENTED_COMMANDS


class InstructionsFactory:
    """Define a factory class to easily create commands and elements."""

    def __init__(self,
                 freq_bunch: float,
                 dat_filepath: str,
                 **factory_kw: Any) -> None:
        """Instantiate the command and element factories."""
        self._freq_bunch = freq_bunch
        self._command_factory = CommandFactory(**factory_kw)
        self._element_factory = ElementFactory(
            default_field_map_folder=dat_filepath,
            **factory_kw)

    def run(self,
            dat_content: list[list[str]],
            ) -> list[Instruction]:
        """Create all the elements and commands."""
        instructions = [self._call_proper_factory(line, dat_idx)
                        for dat_idx, line in enumerate(dat_content)]
        instructions = self._apply_commands(instructions)
        return instructions

    def _call_proper_factory(self,
                             line: list[str],
                             dat_idx: int,
                             **instruction_kw: str) -> Instruction:
        """Create proper :class:`.Instruction`, or :class:`.Dummy`.

        We go across every word of ``line``, and create the first instruction
        that we find. If we do not recognize it, we return a dummy instruction
        instead.

        Parameters
        ----------
        line : list[str]
            A single line of the ``.dat`` file.
        dat_idx : int
            Line number of the line (starts at 0).
        command_fac : CommandFactory
            A factory to create :class:`.Command`.
        element_fac : ElementFactory
            A factory to create :class:`.Element`.
        instruction_kw : dict
            Keywords given to the ``run`` method of the proper factory.

        Returns
        -------
        Instruction
            Proper :class:`.Command` or :class:`.Element`, or :class:`.Dummy`.

        """
        for word in line:
            word = word.upper()
            if word in IMPLEMENTED_COMMANDS:
                return self._command_factory.run(line,
                                                 dat_idx,
                                                 **instruction_kw)
            if word in IMPLEMENTED_ELEMENTS:
                return self._element_factory.run(line,
                                                 dat_idx,
                                                 **instruction_kw)
        return Dummy(line, dat_idx, warning=True)

    def _apply_commands(self,
                        instructions: list[Instruction]) -> list[Instruction]:
        """Apply all the implemented commands."""
        index = 0
        while index < len(instructions):
            instruction = instructions[index]

            if isinstance(instruction, Command):
                instruction.set_influenced_elements(instructions)
                if instruction.is_implemented:
                    instructions = instruction.apply(
                        instructions,
                        freq_bunch=self._freq_bunch)
            index += 1
        return instructions
