#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

This module holds all the commands.

.. todo::
    Implement a dummy command, that will not be considered but will be kept.

"""
import logging
from typing import Self
from abc import ABC, abstractmethod

from core.elements.element import Element
from core.elements.field_map import FieldMap


COMMANDS = [
    # 'END',
    # 'FIELD_MAP_PATH',
    'FREQ',
    'LATTICE',
    'LATTICE_END',
    'SUPERPOSE_MAP',
]


class Command(ABC):
    """A generic Command class."""

    implemented: bool

    @abstractmethod
    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """Apply the command."""
        return elts_n_cmds


class End(Command):
    """The end of the linac."""

    def __init__(self, line: list[str]) -> None:
        self.implemented = True

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """Remove everything in ``elts_n_cmds`` after this object."""
        end_index = elts_n_cmds.index(self)
        return elts_n_cmds[end_index + 1]


class FieldMapPath(Command):
    """Used to get the base path of field maps."""

    def __init__(self, line: list[str]) -> None:
        self.implemented = True
        self.path = line[1]

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """Set :class:`FieldMap` field folder up to next :class:`FieldMapPath`

        """
        index = elts_n_cmds.index(self)
        for elt_or_cmd in elts_n_cmds[index:]:
            if isinstance(elt_or_cmd, FieldMap):
                elt_or_cmd.field_map_folder = self.path

            if isinstance(elt_or_cmd, FieldMapPath):
                return elts_n_cmds
        return elts_n_cmds


class Freq(Command):
    """Used to get the frequency of every Section."""

    def __init__(self, line: list[str]) -> None:
        self.implemented = True
        self.f_rf_mhz = float(line[1])

    def apply(self,
              elts_n_cmds: list[Element | Self],
              f_bunch: float | None = None,
              **kwargs: float
              ) -> list[Element | Self]:
        """Set :class:`FieldMap` freq, number of cells up to next :class:`Freq`

        """
        if f_bunch is None:
            logging.warning("The bunch frequency was not provided. Setting it "
                            "to RF frequency...")
            f_bunch = self.f_rf_mhz

        index = elts_n_cmds.index(self)
        for elt_or_cmd in elts_n_cmds[index:]:
            if isinstance(elt_or_cmd, FieldMap):
                n_cells = int(self.f_rf_mhz / f_bunch)
                elt_or_cmd.acc_field.set_pulsation_ncell(self.f_rf_mhz,
                                                         n_cells)

            if isinstance(elt_or_cmd, Freq):
                return elts_n_cmds
        return elts_n_cmds


class Lattice(Command):
    """Used to get the number of elements per lattice."""

    def __init__(self, line: list[str]) -> None:
        self.implemented = True
        self.n_lattice = int(line[1])

        self.n_macro_lattice = 1
        if len(line) > 2:
            self.n_macro_lattice = int(line[2])

            if self.n_macro_lattice != 1:
                logging.warning("Macro-lattice not implemented. LightWin will "
                                "consider that number of macro-lattice per "
                                "lattice is 1.")

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """Set lattice section number of elements in current lattice."""
        index = elts_n_cmds.index(self)

        current_lattice_number = self._current_lattice_number(elts_n_cmds,
                                                              index)
        current_section_number = self._current_section_number(elts_n_cmds)

        index_in_current_lattice = 0
        for elt_or_cmd in elts_n_cmds[index:]:
            if isinstance(elt_or_cmd, Lattice | LatticeEnd):
                return elts_n_cmds

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

    def __init__(self, line: list[str]) -> None:
        self.implemented = True

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        """Do nothing, everything is handled by :class:`Lattice`."""
        return elts_n_cmds


class SuperposeMap(Command):
    """Dummy class."""

    def __init__(self, line: list[str]) -> None:
        self.implemented = False

    def apply(self,
              elts_n_cmds: list[Element | Self],
              **kwargs: float
              ) -> list[Element | Self]:
        logging.error("SuperposeMap not implemented.")
        return elts_n_cmds
