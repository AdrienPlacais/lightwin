#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we define a dummy :class:`.Element`/:class:`.Command`.

We use it to keep track of non-implemented elements/commands.


.. todo::
    Clarify nature of ``dat_idx``.

"""
import logging
from abc import ABC


class Instruction(ABC):
    """An object corresponding to a line in a ``.dat`` file."""

    line: list[str]
    idx: dict[str, str | slice]

    def __init__(self,
                 line: list[str],
                 dat_idx: dict[str, str | slice],
                 is_implemented: bool) -> None:
        """Instantiate corresponding line and line number in ``.dat`` file."""
        self.line = line
        self.idx = {'dat_idx': dat_idx}
        self.is_implemented = is_implemented


class Dummy(Instruction):
    """An object corresponding to a non-implemented element or command."""

    def __init__(self,
                 line: list[str],
                 dat_idx: dict[str, str | slice],
                 warning: bool = False,
                 ) -> None:
        """Create the dummy object, raise a warning if necessary.

        Parameters
        ----------
        line : list[str]
            Arguments of the line in the ``.dat`` file.
        dat_idx : dict[str, str | slice]
            dat_idx
        warning : bool, optional
            To raise a warning when the element is not implemented. The default
            is False.

        """
        super().__init__(line, dat_idx, is_implemented=False)
        if warning:
            logging.warning("A dummy element was added as the corresponding "
                            "element or command is not implemented. If the "
                            "BeamCalculator is not TraceWni, this may be a "
                            "problem. In particular if the missing element "
                            "has a length that is non-zero. You can disable "
                            "this warning in tracewin_utils.dat_files._create"
                            "_element_n_command_objects. Line with a problem:"
                            f"\n{line}")
