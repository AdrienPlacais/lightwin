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
from pathlib import Path

import math

from core.commands.command import Command
from core.elements.element import Element
from core.instruction import Dummy, Instruction


def update_field_maps_in_dat(elts: object,
                             new_phases: dict[Element, float],
                             new_k_e: dict[Element, float],
                             new_abs_phase_flag: dict[Element, float]
                             ) -> None:
    """
    Create a new dat with given elements and settings.

    In constrary to ``dat_filecontent_from_smaller_list_of_elements``, does not
    modify the number of :class:`.Element` in the ``.dat``.

    .. todo::
        handle personalized name of elements better

    """
    dat_content: list[list[str]] = []
    for instruction in elts.files['elts_n_cmds']:
        line = instruction.line

        # remove personalized name to always have same arg position
        if len(line) > 1 and ':' in line[0]:
            del line[0]

        if instruction in new_phases:
            line[3] = str(math.degrees(new_phases[instruction]))
        if instruction in new_k_e:
            line[6] = str(new_k_e[instruction])
        if instruction in new_abs_phase_flag:
            line[10] = str(new_abs_phase_flag[instruction])

        # read personalized name
        if hasattr(instruction, '_personalized_name') \
                and instruction._personalized_name is not None:
            line.insert(0, f"{instruction._personalized_name} :")

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
                                dat_path: Path) -> None:
    """Save the content of the updated dat to a `.dat`."""
    with open(dat_path, 'w', encoding='utf-8') as file:
        for line in dat_content:
            file.write(' '.join(line) + '\n')
    logging.info(f"New dat saved in {dat_path}.")
