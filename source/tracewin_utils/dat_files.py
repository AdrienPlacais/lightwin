#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:24:56 2023.

@author: placais

This module holds function to load, modify and create .dat structure files.

TODO insert line skip at each section change in the output.dat

"""
import logging
import itertools

import config_manager as con
from core.elements import (_Element, Quad, Drift, FieldMap, Solenoid, Lattice,
                           Freq, FieldMapPath, End)
from core.list_of_elements import ListOfElements

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
    elements_list : list[_Element]
        List containing all the Element objects.

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
    elements_list = list(elements_iterable)
    return elements_list


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


def update_dat_with_fixed_cavities(dat_filecontent: list[list[str]],
                                   elts: ListOfElements, fm_folder: str
                                   ) -> None:
    """Create a new dat with updated cavity phase and amplitude."""
    idx_elt = 0

    phi = {
        True: lambda elt: str(elt.get('phi_0_abs', to_deg=True)),
        False: lambda elt: str(elt.get('phi_0_rel', to_deg=True)),
    }

    for line in dat_filecontent:
        if line[0] in TO_BE_IMPLEMENTED or line[0] in NOT_AN_ELEMENT:
            continue

        if line[0] == 'FIELD_MAP':
            elt = elts[idx_elt]
            line[3] = phi[con.FLAG_PHI_ABS](elt)
            line[6] = str(elt.get('k_e'))
            # '1' if True, '0' if False
            line[10] = str(int(con.FLAG_PHI_ABS))

        elif line[0] == 'FIELD_MAP_PATH':
            line[1] = fm_folder
            continue

        idx_elt += 1


def create_dat_from_smaller_list_of_elements(
        dat_filecontent: list[list[str]], elts: ListOfElements) -> None:
    """Create a new `.dat` containing only the `_Element`s of `elts`."""
    idx_elt = 0
    indexes_to_keep = elts.get('elt_idx', to_numpy=False)
    smaller_dat_filecontent = []

    for line in dat_filecontent:
        if line[0] in TO_BE_IMPLEMENTED + NOT_AN_ELEMENT + ['FIELD_MAP_PATH']:
            smaller_dat_filecontent.append(line)
            continue

        if idx_elt in indexes_to_keep:
            smaller_dat_filecontent.append(line)
        idx_elt += 1

    smaller_dat_filecontent = _remove_empty_lattices(smaller_dat_filecontent)
    return smaller_dat_filecontent


def _remove_empty_lattices(dat_filecontent: list[list[str]]
                           ) -> list[list[str]]:
    """Remove useless LATTICE and FREQ commands."""
    logging.warning("_remove_empty_lattices not implemented.")
    return dat_filecontent


def save_dat_filecontent_to_dat(dat_filecontent: list[list[str]],
                                dat_filepath: str) -> None:
    """Save the content of the updated dat to a `.dat`."""
    with open(dat_filepath, 'w', encoding='utf-8') as file:
        for line in dat_filecontent:
            file.write(' '.join(line) + '\n')
    logging.info(f"New dat saved in {dat_filepath}.")
