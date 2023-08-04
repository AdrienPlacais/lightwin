#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:24:56 2023.

@author: placais

This module holds function to load, modify and create .dat structure files.

"""
import itertools

import config_manager as con
from core.elements import (_Element, Quad, Drift, FieldMap, Solenoid, Lattice,
                           Freq, FieldMapPath, End)

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
                                   elts: list[_Element], fm_folder: str
                                   ) -> None:
    """Create a new dat containing the new linac settings."""
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
