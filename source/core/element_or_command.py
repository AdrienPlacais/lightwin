#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:07:06 2023.

@author: placais

Here we define a dummy class, which can be a :class:`Element` or a
:class:`Command`. We use it to keep track of an non-implemented
element/command.

.. note ::
    Maybe :class:`Element` and :class:`Command` should inherit from this? It
    would be much more consistent and pythonic.

"""
import logging
from abc import ABC


class ElementOrCommand(ABC):
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


class Dummy(ElementOrCommand):
    """An object corresponding to a non-implemented element or command."""

    def __init__(self, line: list[str],
                 dat_idx: dict[str, str | slice],
                 warning: bool = False,
                 ) -> None:
        super().__init__(line, dat_idx, is_implemented=False)
        if warning:
            logging.warning("A dummy element was added as the corresponding "
                            "element or command is not implemented. If the "
                            "BeamCalculator is Envelope1D, this may be a "
                            "problem. In particular if the missing element "
                            "has a length that is non-zero. You can disable "
                            "this warning in tracewin_utils.dat_files._create"
                            "_element_n_command_objects. Line with a problem:"
                            f"\n{line}")
