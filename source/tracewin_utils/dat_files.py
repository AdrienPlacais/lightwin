#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds function to load, modify and create .dat structure files.

.. todo::
    Insert line skip at each section change in the output.dat

Non-exhaustive list of non implemented commands:
    'SPACE_CHARGE_COMP',
    'SET_SYNC_PHASE',
    'STEERER',
    'ADJUST',
    'ADJUST_STEERER',
    'ADJUST_STEERER_BX',
    'ADJUST_STEERER_BY',
    'DIAG_SIZE',
    'DIAG_DSIZE',
    'DIAG_DSIZE2',
    'DIAG_DSIZE3',
    'DIAG_DSIZE4',
    'DIAG_DENERGY',
    'DIAG_ENERGY',
    'DIAG_TWISS',
    'DIAG_WAIST',
    'DIAG_POSITION',
    'DIAG_DPHASE',
    'ERROR_CAV_NCPL_STAT',
    'ERROR_CAV_NCPL_DYN',
    'SET_ADV',
    'SHIFT',
    'THIN_STEERING',
    'APERTURE',

"""
import logging
from typing import TypeVar

import numpy as np
import config_manager as con

from core.instruction import Instruction, Dummy

from core.elements.element import Element
from core.elements.field_maps.field_map import FieldMap

from core.commands.command import Command
from core.instructions_factory import InstructionsFactory

from tracewin_utils.electromagnetic_fields import load_electromagnetic_fields

try:
    import beam_calculation.envelope_1d.transfer_matrices_c as tm_c
except ModuleNotFoundError:
    MESSAGE = 'Cython module not compilated. Check elements.py and setup.py'\
        + ' for more information.'
    if con.FLAG_CYTHON:
        raise ModuleNotFoundError(MESSAGE)
    logging.warning(MESSAGE)
    # Load Python version as Cython to allow the execution of the code.
    import beam_calculation.envelope_1d.transfer_matrices_p as tm_c

ListOfElements = TypeVar('ListOfElements')


def create_structure(dat_content: list[list[str]],
                     dat_filepath: str,
                     **kwargs: str | float) -> list[Instruction]:
    """
    Create structure using the loaded ``.dat`` file.

    Parameters
    ----------
    dat_content : list[list[str]]
        List containing all the lines of ``dat_filepath``.
    dat_path : str
        Absolute path to the ``.dat``.
    force_a_section_to_each_element : bool
        To force each element to have a section.
    force_a_lattice_to_each_element : bool
        To force each element to have a lattice.
    load_electromagnetic_files : bool
        Load the files for the FIELD_MAPs.
    check_consistency : bool
        Check the structure of the accelerator, in particular lattices and
        sections.
    **kwargs : float
        Dict transmitted to commands. Must contain the bunch frequency in MHz.

    Returns
    -------
    elts : list[Element]
        List containing all the :class:`Element` objects.

    """
    instructions_factory = InstructionsFactory(dat_filepath=dat_filepath,
                                               **kwargs)
    instructions = instructions_factory.run(dat_content,
                                            cython=con.FLAG_CYTHON,
                                            )
    _check_every_elt_has_lattice_and_section(instructions)

    return instructions


def _check_every_elt_has_lattice_and_section(
        instructions: list[Instruction]) -> None:
    """Check that every element has a lattice and section index."""
    elts = list(filter(lambda elt: isinstance(elt, Element),
                       instructions))
    for elt in elts:
        if elt.get('lattice', to_numpy=False) is None:
            logging.error("At least one Element is outside of any lattice. "
                          "This may cause problems...")
            break

    for elt in elts:
        if elt.get('section', to_numpy=False) is None:
            logging.error("At least one Element is outside of any section. "
                          "This may cause problems...")
            break


def update_field_maps_in_dat(
    elts: ListOfElements,
    new_phases: dict[Element, float],
    new_k_e: dict[Element, float],
    new_abs_phase_flag: dict[Element, float]
) -> None:
    """
    Create a new dat with given elements and settings.

    In constrary to `dat_filecontent_from_smaller_list_of_elements`, does not
    modify the number of `Element`s in the .dat.

    """
    dat_content: list[list[str]] = []
    for instruction in elts.files['elts_n_cmds']:
        line = instruction.line

        if instruction in new_phases:
            line[3] = str(np.rad2deg(new_phases[instruction]))
        if instruction in new_k_e:
            line[6] = str(new_k_e[instruction])
        if instruction in new_abs_phase_flag:
            line[10] = str(new_abs_phase_flag[instruction])

        dat_content.append(line)


def dat_filecontent_from_smaller_list_of_elements(
    original_instructions: list[Instruction],
    elts: list[Element],
) -> tuple[list[list[str]], list[Instruction]]:
    """
    Create a ``.dat`` with only elements of ``elts`` (and concerned commands).

    Properties of the FIELD_MAP, i.e. amplitude and phase, remain untouched, as
    it is the job of :func:`update_field_maps_in_dat`.

    """
    indexes_to_keep = [elt.get('dat_idx', to_numpy=False) for elt in elts]
    last_index = indexes_to_keep[-1] + 1

    new_dat_filecontent: list[list[str]] = []
    new_instructions: list[Instruction] = []
    for instruction in original_instructions[:last_index]:
        element_to_keep = (isinstance(instruction, Element | Dummy)
                           and instruction.idx['dat_idx'] in indexes_to_keep)

        useful_command = (isinstance(instruction, Command)
                          and instruction.concerns_one_of(indexes_to_keep))

        if not (element_to_keep or useful_command):
            continue

        new_dat_filecontent.append(instruction.line)
        new_instructions.append(instruction)

    end = original_instructions[-1]
    new_dat_filecontent.append(end.line)
    new_instructions.append(end)
    return new_dat_filecontent, new_instructions


def save_dat_filecontent_to_dat(dat_content: list[list[str]],
                                dat_path: str) -> None:
    """Save the content of the updated dat to a `.dat`."""
    with open(dat_path, 'w', encoding='utf-8') as file:
        for line in dat_content:
            file.write(' '.join(line) + '\n')
    logging.info(f"New dat saved in {dat_path}.")
