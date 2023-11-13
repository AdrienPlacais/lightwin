#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds all the commands.

.. warning::
    As for now, if ``is_implemented`` is ``False``, the command will still
    appear in the ``.dat`` subset! Is this what I want?

"""
from abc import abstractmethod

from core.instruction import Instruction


class Command(Instruction):
    """
    A generic Command class.

    Parameters
    ----------
    idx : dict[str, int]
        Dictionary holding useful indexes. Keys are ``'dat_idx'`` (position in
        the ``.dat`` file) and ``'influenced_elements'`` (position in the
        ``.dat`` file of the elements concerned by current command).
    is_implemented : bool
        Determine if current command is implemented. If not, it will be skipped
        and its :func:`apply` method will not be used.
    line : list[str]
        Line in the ``.dat`` file corresponding to current command.

    See Also
    --------
    :func:`.core.list_of_elements.factory.subset_of_pre_existing_list_of_elements`
    :func:`.tracewin_utils.dat_files.dat_filecontent_from_smaller_list_of_elements`

    """

    is_implemented: bool

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 is_implemented: bool) -> None:
        """Instantiate mandatory attributes."""
        super().__init__(line, dat_idx, is_implemented)
        self.idx['influenced'] = slice(0, 1)
        self.is_implemented = is_implemented
        # self.idx = {'dat_idx': dat_idx,
        #             'influenced': slice}
        # self.is_implemented = is_implemented
        # self.line = line

    @abstractmethod
    def set_influenced_elements(self,
                                instructions: list[Instruction],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        self.idx['influenced'] = slice(0, 1)

    @abstractmethod
    def apply(self,
              instructions: list[Instruction],
              **kwargs: float
              ) -> list[Instruction]:
        """Apply the command."""
        return instructions

    def concerns_one_of(self, dat_indexes: list[int]) -> bool:
        """
        Tell if ``self`` concerns an element, which ``dat_idx`` is given.

        Internally, we convert the ``self.idx['influenced']`` from a
        :class:`set` to a :class:`list` object and check intersections with
        ``dat_indexes``.

        Parameters
        ----------
        dat_indexes : list[int]
            Indexes in the ``.dat`` file of the sub-list of elements under
            creation.

        """
        idx_influenced = range(self.idx['influenced'].start,
                               self.idx['influenced'].stop)
        idx_influenced = [i for i in idx_influenced]

        intersect = list(set(idx_influenced).intersection(dat_indexes))
        return len(intersect) > 0
