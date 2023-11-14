#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define helper functions applying on elements."""
from core.instruction import Instruction
from core.elements.element import Element
from core.elements.drift import Drift
from core.elements.field_maps.field_map import FieldMap
from core.elements.quad import Quad
from core.elements.solenoid import Solenoid

CIVIL_REGISTER = {
    Quad: 'QP',
    Drift: 'DR',
    FieldMap: 'FM',
    Solenoid: 'SOL',
}  #:


def give_name_to_elements(elts: list[Element]) -> None:
    """Give a name (the same as TW) to every element."""
    for key, value in CIVIL_REGISTER.items():
        sub_list = list(filter(lambda elt: isinstance(elt, key), elts))
        for i, elt in enumerate(sub_list, start=1):
            if elt.elt_info['elt_name'] is None:
                elt.elt_info['elt_name'] = value + str(i)

    other_elements = list(filter(lambda elt: type(elt) not in CIVIL_REGISTER,
                          elts))
    for i, elt in enumerate(other_elements, start=1):
        if elt.elt_info['elt_name'] is None:
            elt.elt_info['elt_name'] = 'ELT' + str(i)


def force_a_section_for_every_element(elts_without_dummies: list[Element]
                                       ) -> None:
    """Give a section index to every element."""
    idx_section = 0
    for elt in elts_without_dummies:
        idx = elt.idx['section']
        if idx is None:
            elt.idx['section'] = idx_section
            continue
        idx_section = idx


def force_a_lattice_for_every_element(elts_without_dummies: list[Element]
                                       ) -> None:
    """
    Give a lattice index to every element.

    Elements before the first LATTICE command will be in the same lattice as
    the elements after the first LATTICE command.

    Elements after the first LATTICE command will be in the previous lattice.

    Example
    -------
    .. list-table ::
        :widths: 10 10 10
        :header-rows: 1

        * - Element/Command
          - Lattice before
          - Lattice after
        * - ``QP1``
          - None
          - 0
        * - ``DR1``
          - None
          - 0
        * - ``LATTICE``
          -
          -
        * - ``QP2``
          - 0
          - 0
        * - ``DR2``
          - 0
          - 0
        * - ``END LATTICE``
          -
          -
        * - ``QP3``
          - None
          - 0
        * - ``LATTICE``
          -
          -
        * - ``DR3``
          - 1
          - 1
        * - ``END LATTICE``
          -
          -
        * - ``QP4``
          - None
          - 1
    """
    idx_lattice = 0
    for elt in elts_without_dummies:
        idx = elt.idx['lattice']
        if idx is None:
            elt.idx['lattice'] = idx_lattice
            continue
        idx_lattice = idx
