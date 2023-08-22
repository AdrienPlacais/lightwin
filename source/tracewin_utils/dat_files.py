#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:24:56 2023.

@author: placais

This module holds function to load, modify and create .dat structure files.

TODO insert line skip at each section change in the output.dat

"""
import os.path
import logging
from typing import TypeVar
import itertools

import numpy as np

import config_manager as con
from core.elements import (_Element, Quad, Drift, FieldMap, Solenoid, Lattice,
                           Freq, FieldMapPath, End)

# from core.list_of_elements import ListOfElements
ListOfElements = TypeVar('ListOfElements')


TO_BE_IMPLEMENTED = [
    'SPACE_CHARGE_COMP', 'SET_SYNC_PHASE', 'STEERER',
    'ADJUST', 'ADJUST_STEERER', 'ADJUST_STEERER_BX', 'ADJUST_STEERER_BY',
    'DIAG_SIZE', 'DIAG_DSIZE', 'DIAG_DSIZE2', 'DIAG_DSIZE3', 'DIAG_DSIZE4',
    'DIAG_DENERGY', 'DIAG_ENERGY', 'DIAG_TWISS', 'DIAG_WAIST',
    'DIAG_POSITION', 'DIAG_DPHASE',
    'ERROR_CAV_NCPL_STAT', 'ERROR_CAV_NCPL_DYN',
    'SET_ADV', 'LATTICE_END', 'SHIFT', 'THIN_STEERING', 'APERTURE']
NOT_AN_ELEMENT = ['LATTICE', 'FREQ']


def create_structure(dat_filecontent: list[list[str]]) -> list[_Element]:
    """
    Create structure using the loaded dat file.

    Parameters
    ----------
    dat_filecontent : list[list[str]]
        List containing all the lines of dat_filepath.

    Returns
    -------
    elts : list[_Element]
        List containing all the `_Element` objects.

    """
    subclasses_dispatcher = {
        'QUAD': Quad,
        'DRIFT': Drift,
        'FIELD_MAP': FieldMap,
        'SOLENOID': Solenoid,
        'LATTICE': Lattice,
        'FREQ': Freq,
        'FIELD_MAP_PATH': FieldMapPath,
        'END': End,
    }

    elements_iterable = itertools.takewhile(
        lambda elt: not isinstance(elt, End),
        [subclasses_dispatcher[elem[0]](elem) for elem in dat_filecontent
         if elem[0] not in TO_BE_IMPLEMENTED]
    )
    elts = list(elements_iterable)
    return elts


def set_field_map_files_paths(elts: list[_Element],
                              default_field_map_folder: str
                              ) -> tuple[list[_Element], str]:
    """Load FIELD_MAP_PATH, remove it from the list of elements."""
    field_map_paths = list(filter(lambda elt: isinstance(elt, FieldMapPath),
                                  elts))

    # FIELD_MAP_PATH are not physical elements, so we remove them
    for field_map_path in field_map_paths:
        elts.remove(field_map_path)

    if len(field_map_paths) == 0:
        field_map_paths = list(
            FieldMapPath(['FIELD_MAP_PATH', default_field_map_folder])
        )

    if len(field_map_paths) != 1:
        logging.error("Change of field maps base folder not supported.")
        field_map_paths = [field_map_paths[0]]

    field_map_paths = [os.path.abspath(field_map_path.path)
                       for field_map_path in field_map_paths]
    field_map_folder = field_map_paths[0]
    return elts, field_map_folder


def give_name(elts: list[_Element]) -> None:
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


def update_field_maps_in_dat(
    elts: ListOfElements,
    new_phases: dict[_Element, float],
    new_k_e: dict[_Element, float],
    new_abs_phase_flag: dict[_Element, float]
) -> None:
    """
    Create a new dat with given elements and settings.

    In constrary to `dat_filecontent_from_smaller_list_of_elements`, does not
    modify the number of `_Element`s in the .dat.

    """
    idx_elt = 0
    dat_filecontent, field_map_folder = elts.get('dat_content',
                                                 'field_map_folder',
                                                 to_numpy=False)
    for line in dat_filecontent:
        if line[0] in TO_BE_IMPLEMENTED or line[0] in NOT_AN_ELEMENT:
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


def dat_filecontent_from_smaller_list_of_elements(
        dat_filecontent: list[list[str]],
        elts: list[_Element],
) -> list[list[str]]:
    """
    Create a new `.dat` containing only the `_Element`s of `elts`.

    Properties of the FIELD_MAP, i.e. amplitude and phase, remain untouched, as
    it is the job of `update_field_maps_in_dat`.

    """
    idx_elt = 0
    indexes_to_keep = [elt.get('elt_idx', to_numpy=False)
                       for elt in elts]
    smaller_dat_filecontent = []
    for line in dat_filecontent:
        if line[0] in TO_BE_IMPLEMENTED + NOT_AN_ELEMENT + ['FIELD_MAP_PATH']:
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
