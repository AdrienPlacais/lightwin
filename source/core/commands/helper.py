#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define helper function applying on commands."""
from core.instruction import Instruction
from core.commands.command import Command


def apply_commands(instructions: list[Instruction],
                   freq_bunch: float) -> list[Instruction]:
    """Apply all the implemented commands."""
    index = 0
    while index < len(instructions):
        instruction = instructions[index]

        if isinstance(instruction, Command):
            instruction.set_influenced_elements(instructions)
            if instruction.is_implemented:
                instructions = instruction.apply(instructions,
                                                 freq_bunch=freq_bunch)
        index += 1
    return instructions
