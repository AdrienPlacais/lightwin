#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a useless command to keep track of ``MARKER``."""
import logging

from core.commands.command import Command
from core.instruction import Instruction


class Marker(Command):
    """Dummy class."""

    is_implemented = False

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Instantiate the dummy command."""
        super().__init__(line, dat_idx, name=name)

    def set_influenced_elements(
        self, instructions: list[Instruction], **kwargs: float
    ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        return

    def apply(
        self, instructions: list[Instruction], **kwargs: float
    ) -> list[Instruction]:
        """Do nothing."""
        logging.error("DummyElement not implemented.")
        return instructions

    def concerns_one_of(self, dat_indexes: list[int]) -> bool:
        """Tell if ``self`` concerns an element, which ``dat_idx`` is given.

        Internally, we convert the ``self.influenced`` from a
        :class:`set` to a :class:`list` object and check intersections with
        ``dat_indexes``.

        Parameters
        ----------
        dat_indexes : list[int]
            Indexes in the ``.dat`` file of the sub-list of elements under
            creation.

        """
        return False
