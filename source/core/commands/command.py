#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

This module holds all the commands.

.. warning::
    As for now, if ``is_implemented`` is ``False``, the command will still
    appear in the ``.dat`` subset! Is this what I want?

"""
import logging
from typing import Self
from abc import ABC, abstractmethod

from core.elements.element import Element
from core.elements.field_map import FieldMap


class Command(ABC):
    """
    A generic Command class.

    Parameters
    ----------
    idx : dict[str, int]
        Dictionary holding useful indexes. Keys are ``'dat_idx'`` (position in
        the ``.dat`` file) and ``'influenced_elements'`` (position in the
        ``.dat`` file of the elements concerned by current command).
    is_implemented : bool
        Determine if implementation of current command is over. If not, it will
        be skipped and its :func:`apply` method will not be used. Its
        ``influenced_elements`` will not be set either.
    line : list[str]
        Line in the ``.dat`` file corresponding to current command.

    See also
    --------
    :func:`core.list_of_elements_factory.subset_of_pre_existing_list_of_elements`
    :func:`tracewin_utils.dat_files.`dat_filecontent_from_smaller_list_of_elements`

    """

    idx: dict[str, int | slice]
    is_implemented: bool
    line: list[str]

    def __init__(self, line: list[str],
                 dat_idx: int,
                 is_implemented: bool) -> None:
        """Instantiate mandatory attributes."""
        self.idx = {'dat_idx': dat_idx,
                    'influenced': slice}
        self.is_implemented = is_implemented
        self.line = line

    @abstractmethod
    def set_influenced_elements(self,
                                elts_n_cmds: list[Element | Self],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        self.idx['influenced'] = slice(0, 1)

    @abstractmethod
    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """Apply the command."""
        return elts_n_cmds

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


class End(Command):
    """The end of the linac."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        super().__init__(line, dat_idx, is_implemented=True)

    def set_influenced_elements(self, *args, **kwargs: float) -> None:
        """Determine the index of the element concerned by :func:`apply`."""
        start = 0
        stop = self.idx['dat_idx'] + 1
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """Remove everything in ``elts_n_cmds`` after this object."""
        return elts_n_cmds[self.idx['influenced']]


class FieldMapPath(Command):
    """Used to get the base path of field maps."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        super().__init__(line, dat_idx, is_implemented=True)
        self.path = line[1]

    def set_influenced_elements(self,
                                elts_n_cmds: list[Element | Self],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx['dat_idx'] + 1
        stop = start
        for elt_or_cmd in elts_n_cmds[start:]:
            if isinstance(elt_or_cmd, FieldMapPath):
                self.idx['influenced'] = slice(start, stop)
                return
            stop += 1
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """Set :class:`FieldMap` field folder up to next :class:`FieldMapPath`

        """
        for elt_or_cmd in elts_n_cmds[self.idx['influenced']]:
            if isinstance(elt_or_cmd, FieldMap):
                elt_or_cmd.field_map_folder = self.path
        return elts_n_cmds


class Freq(Command):
    """Used to get the frequency of every Section."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        super().__init__(line, dat_idx, is_implemented=True)
        self.f_rf_mhz = float(line[1])

    def set_influenced_elements(self,
                                elts_n_cmds: list[Element | Self],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx['dat_idx'] + 1
        stop = start
        for elt_or_cmd in elts_n_cmds[start:]:
            if isinstance(elt_or_cmd, Freq):
                self.idx['influenced'] = slice(start, stop)
                return
            stop += 1
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              elts_n_cmds: list[Element | Self],
              freq_bunch: float | None = None,
              **kwargs: float
              ) -> list[Element | Self]:
        """Set :class:`FieldMap` freq, number of cells up to next :class:`Freq`

        """
        if freq_bunch is None:
            logging.warning("The bunch frequency was not provided. Setting it "
                            "to RF frequency...")
            freq_bunch = self.f_rf_mhz

        for elt_or_cmd in elts_n_cmds[self.idx['influenced']]:
            if isinstance(elt_or_cmd, FieldMap):
                n_cell = int(self.f_rf_mhz / freq_bunch)
                elt_or_cmd.acc_field.set_pulsation_ncell(self.f_rf_mhz, n_cell)
        return elts_n_cmds


class Lattice(Command):
    """Used to get the number of elements per lattice."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        super().__init__(line, dat_idx, is_implemented=True)
        self.n_lattice = int(line[1])

        self.n_macro_lattice = 1
        if len(line) > 2:
            self.n_macro_lattice = int(line[2])

            if self.n_macro_lattice > 1:
                logging.warning("Macro-lattice not implemented. LightWin will "
                                "consider that number of macro-lattice per "
                                "lattice is 1 or 0.")

    def set_influenced_elements(self,
                                elts_n_cmds: list[Element | Self],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx['dat_idx'] + 1
        stop = start
        for elt_or_cmd in elts_n_cmds[start:]:
            if isinstance(elt_or_cmd, Lattice | LatticeEnd):
                self.idx['influenced'] = slice(start, stop)
                return

            stop += 1
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """Set lattice section number of elements in current lattice."""
        index = self.idx['dat_idx']

        current_lattice_number = self._current_lattice_number(elts_n_cmds,
                                                              index)
        current_section_number = self._current_section_number(elts_n_cmds)

        index_in_current_lattice = 0
        for elt_or_cmd in elts_n_cmds[self.idx['influenced']]:
            if isinstance(elt_or_cmd, SuperposeMap):
                logging.error("SuperposeMap not implemented. Will mess with "
                              "indexes...")

            if isinstance(elt_or_cmd, Command):
                continue

            if isinstance(elt_or_cmd, Element):
                elt_or_cmd.idx['lattice'] = current_lattice_number
                elt_or_cmd.idx['section'] = current_section_number

            index_in_current_lattice += 1
            if index_in_current_lattice == self.n_lattice:
                current_lattice_number += 1
                index_in_current_lattice = 0

        return elts_n_cmds

    def _current_section_number(self,
                                elts_n_cmds: list[Element | Self],
                                ) -> int:
        """Get section number of ``self``."""
        all_lattice_commands = list(filter(
            lambda elt_or_cmd: isinstance(elt_or_cmd, Lattice), elts_n_cmds))
        return all_lattice_commands.index(self)

    def _current_lattice_number(self,
                                elts_n_cmds: list[Element | Self],
                                index: int
                                ) -> int:
        """
        Get lattice number of current object.

        We look for :class:`Element` in ``elts_n_cmds``in reversed order,
        starting from ``self``. We take the first non None lattice index that
        we find, and return it + 1.
        If we do not find anything, this is because no :class:`Element` had a
        defined lattice number before.

        This approach allows for :class:`Element` without a lattice number, as
        for examples drifts between a :class:`LatticeEnd` and a
        :class:`Lattice`.

        """
        elts_n_cmds_before_self = elts_n_cmds[:index]
        reversed_elts_n_cmds_before_self = elts_n_cmds_before_self[::-1]

        for elt_or_cmd in reversed_elts_n_cmds_before_self:
            if isinstance(elt_or_cmd, Element):
                previous_lattice_number = elt_or_cmd.get('lattice',
                                                         to_numpy=False)

                if previous_lattice_number is not None:
                    return previous_lattice_number + 1
        return 0


class LatticeEnd(Command):
    """Dummy class."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        super().__init__(line, dat_idx, is_implemented=True)

    def set_influenced_elements(self,
                                elts_n_cmds: list[Element | Self],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx['dat_idx']
        stop = start
        for elt_or_cmd in elts_n_cmds[start:]:
            if isinstance(elt_or_cmd, Lattice):
                self.idx['influenced'] = slice(start, stop)
                return
            stop += 1
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """Do nothing, everything is handled by :class:`Lattice`."""
        return elts_n_cmds


class Shift(Command):
    """Dummy class."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        super().__init__(line, dat_idx, is_implemented=False)

    def set_influenced_elements(self,
                                elts_n_cmds: list[Element | Self],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx['dat_idx']
        stop = start + 1
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        logging.error("Shift not implemented.")
        return elts_n_cmds


class Steerer(Command):
    """Dummy class."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        super().__init__(line, dat_idx, is_implemented=False)

    def set_influenced_elements(self,
                                elts_n_cmds: list[Element | Self],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        next_element = list(filter(lambda elt: isinstance(elt, Element),
                                   elts_n_cmds[self.idx['dat_idx']:]))[0]
        start = next_element.idx['dat_idx']
        stop = start + 1
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        logging.error("Steerer not implemented.")
        return elts_n_cmds


class SuperposeMap(Command):
    """Dummy class."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        super().__init__(line, dat_idx, is_implemented=False)

    def set_influenced_elements(self,
                                elts_n_cmds: list[Element | Self],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        start = self.idx['dat_idx']
        stop = start + 2
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        logging.error("SuperposeMap not implemented.")
        return elts_n_cmds
