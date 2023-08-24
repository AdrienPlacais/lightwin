#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:24:56 2023.

@author: placais

This module holds function to load, modify and create .dat structure files.

TODO insert line skip at each section change in the output.dat

"""
import logging
from typing import TypeVar

import numpy as np
import config_manager as con

from core.elements.element import Element
from core.elements.quad import Quad
from core.elements.drift import Drift
from core.elements.field_map import FieldMap
from core.elements.solenoid import Solenoid

from core.commands.command import (
    COMMANDS,
    Command,
    End,
    FieldMapPath,
    Freq,
    Lattice,
    LatticeEnd,
    SuperposeMap,
)

# from core.list_of_elements import ListOfElements
ListOfElements = TypeVar('ListOfElements')


TO_BE_IMPLEMENTED = [
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
]


def create_structure(dat_filecontent: list[list[str]]) -> list[Element]:
    """
    Create structure using the loaded dat file.

    Parameters
    ----------
    dat_filecontent : list[list[str]]
        List containing all the lines of dat_filepath.

    Returns
    -------
    elts : list[Element]
        List containing all the `Element` objects.

    """
    subclasses_dispatcher = {
        # Elements
        'DRIFT': Drift,
        'FIELD_MAP': FieldMap,
        'QUAD': Quad,
        'SOLENOID': Solenoid,
        # Commands
        'END': End,
        'FIELD_MAP_PATH': FieldMapPath,
        'FREQ': Freq,
        'LATTICE': Lattice,
        'LATTICE_END': LatticeEnd,
        'SUPERPOSE_MAP': SuperposeMap,
    }

    # elements_iterable = itertools.takewhile(
    #     lambda elt: not isinstance(elt, End),
    #     [subclasses_dispatcher[elem[0]](elem) for elem in dat_filecontent
    #      if elem[0] not in TO_BE_IMPLEMENTED]
    # )
    # elts = list(elements_iterable)

    elts_n_cmds = [subclasses_dispatcher[line[0]](line)
                   for line in dat_filecontent
                   if line[0] not in TO_BE_IMPLEMENTED]
    elts_n_cmds = _apply_commands(elts_n_cmds)
    _check_consistency(elts_n_cmds)
    return elts_n_cmds


def _apply_commands(elts_n_cmds: list[Element | Command]
                    ) -> list[Element | Command]:
    """Apply all the commands that are implemented."""
    kwargs = {'freq_bunch': con.F_BUNCH_MHZ,
              }

    index = 0
    while index < len(elts_n_cmds):
        elt_or_cmd = elts_n_cmds[index]

        if isinstance(elt_or_cmd, Command):
            if elt_or_cmd.is_implemented:
                elts_n_cmds = elt_or_cmd.apply(elts_n_cmds, **kwargs)
        index += 1
    return elts_n_cmds


# Handle when no lattice
def _check_consistency(elts_n_cmds: list[Element | Command]) -> None:
    """Check that every element has a lattice index."""
    elts = list(filter(lambda elt: isinstance(elt, Element),
                       elts_n_cmds))
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


def give_name(elts: list[Element]) -> None:
    """Give a name (the same as TW) to every element."""
    civil_register = {
        'QUAD': 'QP',
        'DRIFT': 'DR',
        'FIELD_MAP': 'FM',
        'SOLENOID': 'SOL',
    }
    for key, value in civil_register.items():
        sub_list = list(filter(lambda elt: elt.get('nature') == key, elts))
        for i, elt in enumerate(sub_list, start=1):
            elt.elt_info['elt_name'] = value + str(i)


# TO UPDATE ?
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
    idx_elt = 0
    dat_filecontent = elts.files['dat_content']
    field_map_folder = elts.files['field_map_folder']

    for line in dat_filecontent:
        if line[0] in TO_BE_IMPLEMENTED or line[0] in COMMANDS:
            continue

        if line[0] == 'FIELD_MAP_PATH':
            line[1] = field_map_folder
            continue

        if line[0] == 'FIELD_MAP':
            elt = elts[idx_elt]
            if elt in new_phases:
                line[3] = str(np.rad2deg(new_phases[elt]))
            if elt in new_k_e:
                line[6] = str(new_k_e[elt])
            if elt in new_abs_phase_flag:
                line[10] = str(new_abs_phase_flag[elt])

        idx_elt += 1


# TO UPDATE
def dat_filecontent_from_smaller_list_of_elements(
        dat_filecontent: list[list[str]],
        elts: list[Element],
) -> list[list[str]]:
    """
    Create a new `.dat` containing only the `Element`s of `elts`.

    Properties of the FIELD_MAP, i.e. amplitude and phase, remain untouched, as
    it is the job of `update_field_maps_in_dat`.

    """
    idx_elt = 0
    indexes_to_keep = [elt.get('elt_idx', to_numpy=False)
                       for elt in elts]
    smaller_dat_filecontent = []
    for line in dat_filecontent:
        if line[0] in TO_BE_IMPLEMENTED + COMMANDS + ['FIELD_MAP_PATH']:
            smaller_dat_filecontent.append(line)
            continue

        if idx_elt in indexes_to_keep:
            smaller_dat_filecontent.append(line.copy())

        idx_elt += 1

    smaller_dat_filecontent = _remove_empty_lattices(smaller_dat_filecontent)
    smaller_dat_filecontent.append(["END"])
    return smaller_dat_filecontent


# FIXME not implemented. Low priority
def _remove_empty_lattices(dat_filecontent: list[list[str]]
                           ) -> list[list[str]]:
    """Remove useless LATTICE and FREQ commands."""
    logging.debug("_remove_empty_lattices not implemented.")
    return dat_filecontent


def save_dat_filecontent_to_dat(dat_content: list[list[str]],
                                dat_path: str) -> None:
    """Save the content of the updated dat to a `.dat`."""
    with open(dat_path, 'w', encoding='utf-8') as file:
        for line in dat_content:
            file.write(' '.join(line) + '\n')
    logging.info(f"New dat saved in {dat_path}.")
