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
from core.elements.dummy import DummyElement
from core.elements.field_map import FieldMap
from core.elements.superposed_field_map import SuperposedFieldMap

from core.electric_field import RfField


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
        Determine if current command is implemented. If not, it will be skipped
        and its :func:`apply` method will not be used.
    line : list[str]
        Line in the ``.dat`` file corresponding to current command.

    See Also
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

    def __str__(self) -> str:
        """Give information on current command."""
        return str(self.line)

    def __repr__(self) -> str:
        """Give information on current command."""
        return str(self)

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


class DummyCommand(Command):
    """Dummy class."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        super().__init__(line, dat_idx, is_implemented=False)

    def set_influenced_elements(self,
                                elts_n_cmds: list[Element | Self],
                                **kwargs: float
                                ) -> None:
        """Determine the index of the elements concerned by :func:`apply`."""
        return range(0, 1)

    def apply(self, elts_n_cmds: list[Element | Self], **kwargs: float
              ) -> list[Element | Self]:
        """Do nothing."""
        logging.error("DummyElement not implemented.")
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
        return False


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
        """
        Set :class:`FieldMap` field folder up.

        If another :class:`FieldMapPath` is found, we stop and this command
        will be applied later.

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
        """
        Set :class:`FieldMap` frequency, number of cells.

        If another :class:`Freq` is found, we stop and the new :class:`Freq`
        will be dealt with later.

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
                elt_or_cmd.idx['idx_in_lattice'] = index_in_current_lattice

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
        """Do nothing."""
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
        """Do nothing."""
        logging.error("Steerer not implemented.")
        return elts_n_cmds


class SuperposeMap(Command):
    """
    Command to merge several field maps.

    Attributes
    ----------
    z_0 : float
        Position at which the next field map should be inserted.

    """

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        super().__init__(line, dat_idx, is_implemented=True)
        self.z_0 = float(line[1]) * 1e-3

    def set_influenced_elements(self,
                                elts_n_cmds: list[Element | Self],
                                **kwargs: float
                                ) -> None:
        """
        Determine the index of the elements concerned by :func:`apply`.

        It spans from the current ``SUPERPOSE_MAP`` command, up to the next
        element that is not a field map. It allows to consider situations where
        we field_map is not directly after the ``SUPERPOSE_MAP`` command.

        Example
        -------
        ```
        SUPERPOSE_MAP
        STEERER
        FIELD_MAP
        ```

        .. warning::
        Only the first of the ``SUPERPOSE_MAP`` command will have the entire
        valid range of elements.

        """
        start = self.idx['dat_idx']
        next_element_but_not_field_map = list(filter(
            lambda elt: (isinstance(elt, Element)
                         and not isinstance(elt, FieldMap)),
            elts_n_cmds[self.idx['dat_idx']:]))[0]
        stop = next_element_but_not_field_map.idx['dat_idx']
        self.idx['influenced'] = slice(start, stop)

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """
        Apply the command.

        Only the first :class:`SuperposeMap` of a bunch of field maps should be
        applied. In order to avoid messing with indexes in the ``.dat`` file,
        all Commands are replaced by dummy commands. All field maps are
        replaced by dummy elements of length 0, except the first field_map that
        is replaced by a SuperposedFieldMap.

        """
        elts_n_cmds_to_merge = elts_n_cmds[self.idx['influenced']]
        total_length = self._total_length(elts_n_cmds_to_merge)

        elts_n_cmds[self.idx['influenced']], number_of_superposed = \
            self._update_class(elts_n_cmds_to_merge, total_length)

        elts_after_self = list(filter(lambda elt: isinstance(elt, Element),
                                      elts_n_cmds[self.idx['dat_idx'] + 1:]))
        self._decrement_lattice_indexes(elts_after_self, number_of_superposed)

        elts_n_cmds[self.idx['influenced']] = elts_n_cmds_to_merge
        return elts_n_cmds

    def _total_length(self, elts_n_cmds_to_merge: list[Element | Command]
                      ) -> float:
        """Compute length of the superposed field maps."""
        z_max = 0.
        z_0 = None
        for elt_or_cmd in elts_n_cmds_to_merge:
            if isinstance(elt_or_cmd, SuperposeMap):
                z_0 = elt_or_cmd.z_0
                continue

            if isinstance(elt_or_cmd, FieldMap):
                if z_0 is None:
                    logging.error("There is no SUPERPOSE_MAP for current "
                                  "FIELD_MAP.")
                    z_0 = 0.

                z_1 = z_0 + elt_or_cmd.length_m
                if z_1 > z_max:
                    z_max = z_1
                z_0 = None
        return z_max

    def _update_class(self,
                      elts_n_cmds_to_merge: list[Element | Command],
                      total_length: float
                      ) -> tuple[list[Element | Command], int]:
        """Replace elements and commands by dummies, except first field map."""
        number_of_superposed = 0
        superposed_field_map_is_already_inserted = False
        for i, elt_or_cmd in enumerate(elts_n_cmds_to_merge):
            args = (elt_or_cmd.line, elt_or_cmd.idx['dat_idx'])

            if isinstance(elt_or_cmd, Command):
                elts_n_cmds_to_merge[i] = DummyCommand(*args)
                continue

            if superposed_field_map_is_already_inserted:
                elts_n_cmds_to_merge[i] = DummyElement(*args)
                elts_n_cmds_to_merge[i].nature = 'SUPERPOSED_FIELD_MAP'
                elts_n_cmds_to_merge[i].acc_field = RfField()
                number_of_superposed += 1
                continue

            elts_n_cmds_to_merge[i] = SuperposedFieldMap(
                *args,
                total_length=total_length)
            elts_n_cmds_to_merge[i].nature = 'SUPERPOSED_FIELD_MAP'
            elts_n_cmds_to_merge[i].acc_field = RfField()
            superposed_field_map_is_already_inserted = True
        return elts_n_cmds_to_merge, number_of_superposed

    def _decrement_lattice_indexes(self, elts_after_self: list[Element],
                                   number_of_superposed: int) -> None:
        """Decrement some lattice numbers to take removed elts into account."""
        for i, elt in enumerate(elts_after_self):
            if elt.idx['lattice'] is None:
                continue

            if not elt.idx['increment_idx']:
                continue

            elt.idx['idx_in_lattice'] -= number_of_superposed - 1

            if elt.idx['idx_in_lattice'] < 0:
                previous_elt = elts_after_self[i - 1]
                elt.idx['idx_in_lattice'] = \
                    previous_elt.idx['idx_in_lattice'] + 1
