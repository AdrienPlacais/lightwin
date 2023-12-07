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
    idx: dict[str, int | bool | None]

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 is_implemented: bool,
                 name: str | None = None) -> None:
        """Instantiate corresponding line and line number in ``.dat`` file."""
        self.line = line
        self.idx = {'dat_idx': dat_idx}
        self.is_implemented = is_implemented

        self._personalized_name = name
        self._default_name: str

    def __str__(self) -> str:
        """Give information on current command."""
        return f"{self.__class__.__name__:15s} {self.line}"

    def __repr__(self) -> str:
        """Say same thing as ``__str__``."""
        return self.__str__()

    @property
    def name(self) -> str:
        """Give personal. name of instruction if exists, default otherwise."""
        if self._personalized_name is None:
            if hasattr(self, '_default_name'):
                return self._default_name
            return str(self.line)
        return self._personalized_name



class Dummy(Instruction):
    """An object corresponding to a non-implemented element or command."""

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 warning: bool = False,
                 ) -> None:
        """Create the dummy object, raise a warning if necessary.

        Parameters
        ----------
        line : list[str]
            Arguments of the line in the ``.dat`` file.
        dat_idx : int
            Line number in the ``.dat`` file.
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


class Comment(Dummy):
    """An object corresponding to a comment."""

    def __init__(self, line: list[str], dat_idx: int) -> None:
        """Create the object, but never raise a warning.

        Parameters
        ----------
        line : list[str]
            Arguments of the line in the ``.dat`` file.
        dat_idx : int
            Line number in the ``.dat`` file.

        """
        super().__init__(line, dat_idx, warning=False)
