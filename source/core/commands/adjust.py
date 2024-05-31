"""Define the ADJUST command.

As for now, ADJUST commands are not used by LightWin.

Functionnality under implementation: LightWin will be able to add ADJUST and
DIAGNOSTIC commands to perform a beauty pass.

.. todo::
    How should I save the min/max variables?? For now, use None.

.. note::
    This is TraceWin's equivalent of :class:`.Variable`.

"""

from core.commands.command import Command
from core.elements.element import Element
from core.instruction import Instruction


class Adjust(Command):
    """A dummy command."""

    is_implemented = False
    n_attributes = range(2, 8)

    def __init__(self, line: list[str], dat_idx: int, **kwargs) -> None:
        """Instantiate the object."""
        super().__init__(line, dat_idx, **kwargs)
        self.number = int(line[1])
        self.vth_variable = int(line[2])
        self.n_link = int(line[3]) if len(line) > 3 else 0
        self.min = float(line[4]) if len(line) > 4 else None
        self.max = float(line[5]) if len(line) > 5 else None
        self.start_step = float(line[6]) if len(line) > 6 else None
        self.k_n = float(line[7]) if len(line) > 7 else None

    def set_influenced_elements(
        self, instructions: list[Instruction], **kwargs: float
    ) -> None:
        """Apply command to first :class:`.Element` that is found."""
        start = self.idx["dat_idx"] + 1
        indexes_between_this_cmd_and_element = (
            self._indexes_between_this_command_and(
                instructions[start:], Element
            )
        )
        self.influenced = indexes_between_this_cmd_and_element.stop
        return

    def apply(self, *args, **kwargs) -> list[Instruction]:
        """Do not apply anything."""
        raise NotImplementedError
