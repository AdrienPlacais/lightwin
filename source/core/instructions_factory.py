#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define methods to easily create :class:`.Command` or :class:`.Element`.

.. todo::
    Instantiate this in :class:`.BeamCalculator`. It could be initialized with
    the ``load_electromagnetic_files`` flag (False for TraceWin), the list of
    implemented elements/commands (ex Envelope3D, not everything is set).

.. todo::
    maybe ElementFactory and CommandFactory should be instantiated from this?
    Or from another class, but they do have a lot in common

.. todo::
    for now, forcing loading of cython field maps

"""
import logging
from collections.abc import Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

import config_manager as con
from core.commands.factory import IMPLEMENTED_COMMANDS, CommandFactory
from core.commands.helper import apply_commands
from core.elements.dummy import DummyElement
from core.elements.element import Element
from core.elements.factory import ElementFactory, implemented_elements
from core.elements.field_maps.field_map import FieldMap
from core.elements.helper import (
    force_a_lattice_for_every_element,
    force_a_section_for_every_element,
    give_name_to_elements,
)
from core.instruction import Comment, Dummy, Instruction
from core.list_of_elements.helper import (
    group_elements_by_section,
    group_elements_by_section_and_lattice,
)
from tracewin_utils.electromagnetic_fields import load_electromagnetic_fields


class InstructionsFactory:
    """Define a factory class to easily create commands and elements."""

    def __init__(
        self,
        freq_bunch_mhz: float,
        default_field_map_folder: Path,
        load_field_maps: bool,
        field_maps_in_3d: bool,
        load_cython_field_maps: bool,
        **factory_kw: Any,
    ) -> None:
        """Instantiate the command and element factories.

        Parameters
        ----------
        freq_bunch_mhz : float
            Beam bunch frequency in MHz.
        default_field_map_folder : Path
            Where to look for field maps when no ``FIELD_MAP_PATH`` is
            precised. This is also the folder where the ``.dat`` is.
        load_field_maps : bool
            To load or not the field maps (useless to do it with
            :class:`.TraceWin`).
        field_maps_in_3d : bool
            To load or not the field maps in 3D (useful only with
            :class:`.Envelope3D`... Except that this is not supported yet, so
            it is never useful.
        load_cython_field_maps : bool
            To load or not the field maps for Cython (useful only with
            :class:`.Envelope1D` and :class:`.Envelope3D` used with Cython).
        phi_s_definition : str, optional
            Definition for the synchronous phases that will be used. Allowed
            values are in
            :var:`util.synchronous_phases.SYNCHRONOUS_PHASE_FUNCTIONS`. The
            default is ``'historical'``.
        factory_kw : Any
            Other parameters passed to the :class:`.CommandFactory` and
            :class:`.ElementFactory`.

        """
        # arguments for commands
        self._freq_bunch_mhz = freq_bunch_mhz

        if load_field_maps:
            assert default_field_map_folder.is_dir()

        # factories
        self._command_factory = CommandFactory(
            default_field_map_folder=default_field_map_folder, **factory_kw
        )
        self.element_factory = ElementFactory(
            default_field_map_folder=default_field_map_folder,
            freq_bunch_mhz=freq_bunch_mhz,
            **factory_kw,
        )

        self._load_field_maps = load_field_maps
        if field_maps_in_3d:
            raise NotImplementedError(
                "No solver can handle 3D field maps yet. Except TraceWin, but "
                "you do not need to load the field maps with this solver, it "
                "does it itself."
            )
        self._field_maps_in_3d = field_maps_in_3d
        self._load_cython_field_maps = load_cython_field_maps

        self._cython: bool = con.FLAG_CYTHON
        # would be better without config dependency

    def run(self, dat_content: list[list[str]]) -> list[Instruction]:
        """Create all the elements and commands.

        .. todo::
            Check if the return value from ``apply_commands`` is necessary.

        Parameters
        ----------
        dat_content : list[list[str]]
            List containing all the lines of ``dat_filepath``.

        """
        instructions = [
            self._call_proper_factory(line, dat_idx)
            for dat_idx, line in enumerate(dat_content)
        ]

        new = apply_commands(instructions, self._freq_bunch_mhz)
        # Remove lines after 'end'
        n_instructions = len(new)
        instructions = instructions[:n_instructions]
        assert new == instructions

        elts = [elt for elt in instructions if isinstance(elt, Element)]
        give_name_to_elements(elts)
        self._handle_lattice_and_section(elts)
        self._check_every_elt_has_lattice_and_section(elts)
        self._check_last_lattice_of_every_lattice_is_complete(elts)

        if self._load_field_maps:
            field_maps = [elt for elt in elts if isinstance(elt, FieldMap)]
            load_electromagnetic_fields(field_maps, True)

        return instructions

    def _call_proper_factory(
        self, line: list[str], dat_idx: int, **instruction_kw: str
    ) -> Instruction:
        """Create proper :class:`.Instruction`, or :class:`.Dummy`.

        We go across every word of ``line``, and create the first instruction
        that we find. If we do not recognize it, we return a dummy instruction
        instead.

        Parameters
        ----------
        line : list[str]
            A single line of the ``.dat`` file.
        dat_idx : int
            Line number of the line (starts at 0).
        command_fac : CommandFactory
            A factory to create :class:`.Command`.
        element_fac : ElementFactory
            A factory to create :class:`.Element`.
        instruction_kw : dict
            Keywords given to the ``run`` method of the proper factory.

        Returns
        -------
        Instruction
            Proper :class:`.Command` or :class:`.Element`, or :class:`.Dummy`,
            or :class:`.Comment`.

        """
        if line == [""]:
            return Comment(line, dat_idx)

        for word in line:
            if ";" in word:
                return Comment(line, dat_idx)

            word = word.upper()
            if word in IMPLEMENTED_COMMANDS:
                return self._command_factory.run(
                    line, dat_idx, **instruction_kw
                )
            if word in implemented_elements:
                return self.element_factory.run(
                    line, dat_idx, **instruction_kw
                )
            if "DIAG" in word:
                logging.critical(
                    "A DIAG was detected but not in implemented_elements."
                    "Maybe it has a weight and the parenthesis were not"
                    " correctly detected?"
                )
                return self.element_factory.run(
                    line, dat_idx, **instruction_kw
                )

        return Dummy(line, dat_idx, warning=True)

    def _handle_lattice_and_section(self, elts: list[Element]) -> None:
        """Ensure that every element has proper lattice, section indexes."""
        elts_without_dummies = [
            elt for elt in elts if not isinstance(elt, DummyElement)
        ]
        force_a_section_for_every_element(elts_without_dummies)
        force_a_lattice_for_every_element(elts_without_dummies)

    def _check_every_elt_has_lattice_and_section(
        self, elts: list[Element]
    ) -> None:
        """Check that every element has a lattice and section index."""
        for elt in elts:
            if elt.idx["lattice"] == -1:
                logging.error(
                    "At least one Element is outside of any lattice. This may "
                    "cause problems..."
                )
                break

        for elt in elts:
            if elt.idx["section"] == -1:
                logging.error(
                    "At least one Element is outside of any section. This may "
                    "cause problems..."
                )
                break

    def _check_last_lattice_of_every_lattice_is_complete(
        self, elts: Sequence[Element]
    ) -> None:
        """Check that last lattice of every section is complete."""
        by_section_and_lattice = group_elements_by_section_and_lattice(
            group_elements_by_section(elts)
        )
        for sec, lattices in enumerate(by_section_and_lattice):
            if len(lattices) <= 1:
                continue
            if (ultim := len(lattices[-1])) == (penult := len(lattices[-2])):
                continue
            joined = "\n".join(
                (
                    f"{str(x):>20}\t{str(y):<20}"
                    for x, y in zip_longest(
                        lattices[-2], lattices[-1], fillvalue="-"
                    )
                )
            )
            joined = f"{'Penultimate:':>20}\t{'Ultimate:':<20}\n" + joined
            logging.warning(
                f"Lattice length mismatch in the {sec}th section. The last "
                f"lattice of this section has {ultim} elements, while "
                f"penultimate has {penult} elements. This may create problems "
                "if you rely on lattices identification to compensate faults. "
                f"\n{joined}"
            )
