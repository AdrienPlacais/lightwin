#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to easily create :class:`.Command` objects."""
from typing import Any
from core.commands.command import Command
from core.commands.dummy_command import DummyCommand
from core.commands.end import End
from core.commands.field_map_path import FieldMapPath
from core.commands.freq import Freq
from core.commands.lattice import Lattice, LatticeEnd
from core.commands.shift import Shift
from core.commands.steerer import Steerer
from core.commands.superpose_map import SuperposeMap

IMPLEMENTED_COMMANDS = {
    'ADJUST': DummyCommand,
    'ADJUST_STEERER': DummyCommand,
    'DIAG_ACHROMAT': DummyCommand,
    'DIAG_POSITION': DummyCommand,
    'DIAG_SIZE': DummyCommand,
    'DIAG_DSIZE': DummyCommand,
    'DUMMY_COMMAND': DummyCommand,
    'END': End,
    'ERROR_BEAM_STAT': DummyCommand,
    'ERROR_CAV_NCPL_STAT': DummyCommand,
    'ERROR_QUAD_NCPL_STAT': DummyCommand,
    'ERROR_GAUSSIAN_CUT_OFF': DummyCommand,
    'FIELD_MAP_PATH': FieldMapPath,
    'FREQ': Freq,
    'LATTICE': Lattice,
    'LATTICE_END': LatticeEnd,
    'SHIFT': Shift,
    'STEERER': Steerer,
    'SUPERPOSE_MAP': SuperposeMap,
}  #:


class CommandFactory:
    """An object to create :class:`.Command` objects."""

    def __init__(self, **factory_kw: Any) -> None:
        """Do nothing for now.

        .. todo::
            Check if it would be relatable to hold some arguments? As for now,
            I would be better off with a run function instead of a class.

        """
        return

    def run(self,
            line: list[str],
            dat_idx: int,
            **command_kw) -> Command:
        """Call proper constructor."""
        command_creator = IMPLEMENTED_COMMANDS[line[0]]
        command = command_creator(line, dat_idx, **command_kw)
        return command
