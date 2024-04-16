#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define helper functions applying on elements."""
import logging

from core.elements.element import Element


def give_name_to_elements(
    elts: list[Element], warn_default_element_names: bool = True
) -> None:
    """Give to every :class:`.Element the name TraceWin would give it."""
    civil_register: dict[str, int] = {}
    for elt in elts:
        if (name := elt._personalized_name) is not None:
            assert name not in civil_register, (
                f"You are trying to give to {elt = } the personalized name "
                f"{name}, which is already taken."
            )
            civil_register[name] = 1
            continue

        nth = civil_register.get(name := elt._id, 0) + 1
        elt._default_name = f"{name}{nth}"
        civil_register[name] = nth

    if not warn_default_element_names:
        return

    if (fallback_name := Element._id) not in civil_register:
        return
    logging.warning(
        f"Used a fallback name for {civil_register[fallback_name]} elements. "
        "Check that every subclass of Element that you use overrides the "
        "default Element._id = {fallback_name}."
    )


def force_a_section_for_every_element(
    elts_without_dummies: list[Element],
) -> None:
    """Give a section index to every element."""
    idx_section = 0
    for elt in elts_without_dummies:
        idx = elt.idx["section"]
        if idx is None:
            elt.idx["section"] = idx_section
            continue
        idx_section = idx
    return


def force_a_lattice_for_every_element(
    elts_without_dummies: list[Element],
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
        idx = elt.idx["lattice"]
        if idx is None:
            elt.idx["lattice"] = idx_lattice
            continue
        idx_lattice = idx
