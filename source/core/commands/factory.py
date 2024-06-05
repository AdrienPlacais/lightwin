"""Define a class to easily create :class:`.Command` objects.

.. todo::
    handle personalized name of commands (marker)

"""

from pathlib import Path
from typing import Any

from core.commands.adjust import Adjust
from core.commands.command import Command
from core.commands.dummy_command import DummyCommand
from core.commands.end import End
from core.commands.error import (
    ErrorBeamDyn,
    ErrorBeamStat,
    ErrorBendCPLDyn,
    ErrorBendCPLStat,
    ErrorBendNCPLDyn,
    ErrorBendNCPLStat,
    ErrorCavCPLDyn,
    ErrorCavCPLStat,
    ErrorCavNCPLDyn,
    ErrorCavNCPLStat,
    ErrorCavNCPLStatFile,
    ErrorGaussianCutOff,
    ErrorQuadNCPLDyn,
    ErrorQuadNCPLStat,
    ErrorRFQCelNCPLDyn,
    ErrorRFQCelNCPLStat,
    ErrorSetRatio,
    ErrorStatFile,
)
from core.commands.field_map_path import FieldMapPath
from core.commands.freq import Freq
from core.commands.lattice import Lattice, LatticeEnd
from core.commands.marker import Marker
from core.commands.set_adv import SetAdv
from core.commands.shift import Shift
from core.commands.steerer import Steerer
from core.commands.superpose_map import SuperposeMap

IMPLEMENTED_COMMANDS = {
    "ADJUST": Adjust,
    "ADJUST_STEERER": DummyCommand,
    "DUMMY_COMMAND": DummyCommand,
    "END": End,
    "ERROR_BEAM_DYN": ErrorBeamDyn,
    "ERROR_BEAM_STAT": ErrorBeamStat,
    "ERROR_BEND_CPL_DYN": ErrorBendCPLDyn,
    "ERROR_BEND_CPL_STAT": ErrorBendCPLStat,
    "ERROR_BEND_NCPL_DYN": ErrorBendNCPLDyn,
    "ERROR_BEND_NCPL_STAT": ErrorBendNCPLStat,
    "ERROR_CAV_CPL_DYN": ErrorCavCPLDyn,
    "ERROR_CAV_CPL_STAT": ErrorCavCPLStat,
    "ERROR_CAV_NCPL_DYN": ErrorCavNCPLDyn,
    "ERROR_CAV_NCPL_STAT": ErrorCavNCPLStat,
    "ERROR_CAV_NCPL_STAT_FILE": ErrorCavNCPLStatFile,
    "ERROR_GAUSSIAN_CUT_OFF": ErrorGaussianCutOff,
    "ERROR_QUAD_NCPL_DYN": ErrorQuadNCPLDyn,
    "ERROR_QUAD_NCPL_STAT": ErrorQuadNCPLStat,
    "ERROR_RFQ_CEL_NCPL_DYN": ErrorRFQCelNCPLDyn,
    "ERROR_RFQ_CEL_NCPL_STAT": ErrorRFQCelNCPLStat,
    "ERROR_STAT_FILE": ErrorStatFile,
    "ERROR_SET_RATIO": ErrorSetRatio,
    "FIELD_MAP_PATH": FieldMapPath,
    "FREQ": Freq,
    "LATTICE": Lattice,
    "LATTICE_END": LatticeEnd,
    "MARKER": Marker,
    "PLOT_DST": DummyCommand,
    "SET_ADV": SetAdv,
    "SHIFT": Shift,
    "STEERER": Steerer,
    "SUPERPOSE_MAP": SuperposeMap,
}  #:


class CommandFactory:
    """An object to create :class:`.Command` objects."""

    def __init__(
        self, default_field_map_folder: Path, **factory_kw: Any
    ) -> None:
        """Do nothing for now.

        .. todo::
            Check if it would be relatable to hold some arguments? As for now,
            I would be better off with a run function instead of a class.

        """
        self.default_field_map_folder = default_field_map_folder
        return

    def run(self, line: list[str], dat_idx: int, **command_kw) -> Command:
        """Call proper constructor."""
        name, line = self._personalized_name(line)
        command_creator = IMPLEMENTED_COMMANDS[line[0].upper()]
        command = command_creator(
            line,
            dat_idx,
            default_field_map_folder=self.default_field_map_folder,
            name=name,
            **command_kw,
        )
        return command

    def _personalized_name(
        self, line: list[str]
    ) -> tuple[str | None, list[str]]:
        """
        Extract the user-defined name of the Element if there is one.

        .. todo::
            Make this robust.

        """
        original_line = " ".join(line)
        line_delimited_with_name = original_line.split(":", maxsplit=1)

        if len(line_delimited_with_name) == 2:
            name = line_delimited_with_name[0].strip()
            cleaned_line = line_delimited_with_name[1].split()
            return name, cleaned_line

        return None, line
