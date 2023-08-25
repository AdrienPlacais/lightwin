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

from core.element_or_command import Dummy

from core.elements.element import Element
from core.elements.quad import Quad
from core.elements.drift import Drift
from core.elements.field_map import FieldMap
from core.elements.solenoid import Solenoid

from core.commands.command import (COMMANDS,
                                   Command,
                                   End,
                                   FieldMapPath,
                                   Freq,
                                   Lattice,
                                   LatticeEnd,
                                   SuperposeMap,
                                   )

from tracewin_utils.electromagnetic_fields import (
    geom_to_field_map_type,
    file_map_extensions,
    load_field_map_file,
)

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


def create_structure(dat_content: list[list[str]],
                     dat_filepath: str,
                     **kwargs: str) -> list[Element | Command]:
    """
    Create structure using the loaded ``.dat`` file.

    Parameters
    ----------
    dat_content : list[list[str]]
        List containing all the lines of ``dat_filepath``.
    dat_path : str
        Absolute path to the ``.dat``.

    Returns
    -------
    elts : list[Element]
    List containing all the :class:`Element` objects.

    """
    elts_n_cmds = _create_element_n_command_objects(dat_content, dat_filepath)
    elts_n_cmds = _apply_commands(elts_n_cmds)

    field_maps = list(filter(lambda field_map: isinstance(field_map, FieldMap),
                             elts_n_cmds))
    _load_electromagnetic_fields(field_maps)

    _check_consistency(elts_n_cmds)

    return elts_n_cmds


def _create_element_n_command_objects(dat_content: list[list[str]],
                                      dat_filepath: str
                                      ) -> list[Element | Command]:
    """Initialize the :class:`Element` and :class:`Command`."""
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
    kwargs = {'default_field_map_folder': dat_filepath}

    elts_n_cmds = [subclasses_dispatcher[line[0]](line, dat_idx, **kwargs)
                   if line[0] in subclasses_dispatcher
                   else Dummy(line, dat_idx, warning=True)
                   for dat_idx, line in enumerate(dat_content)
                   ]
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
            elt_or_cmd.set_influenced_elements(elts_n_cmds)
            if elt_or_cmd.is_implemented:
                elts_n_cmds = elt_or_cmd.apply(elts_n_cmds, **kwargs)
        index += 1
    return elts_n_cmds


def _load_electromagnetic_fields(field_maps: list[FieldMap]) -> None:
    """
    Load field map files.

    As for now, only 1D RF electric field are handled by :class:`Envelope1D`.
    With :class:`TraceWin`, every field is supported.

    """
    for field_map in field_maps:
        field_map_types = geom_to_field_map_type(field_map.geometry,
                                                 remove_no_field=True)
        extensions = file_map_extensions(field_map_types)

        field_map.set_full_path(extensions)

        e_spat, n_z = load_field_map_file(field_map)
        field_map.e_spat = e_spat
        field_map.n_z = n_z

    if con.FLAG_CYTHON:
        _load_electromagnetic_fields_for_cython(field_maps)


def _load_electromagnetic_fields_for_cython(field_maps: list[FieldMap]
                                            ) -> None:
    """Load one electric field per section."""
    valid_files = [field_map.field_map_file_name
                   for field_map in field_maps
                   if field_map.e_spat is not None
                   and field_map.n_z is not None]
    # Trick to remouve duplicates and keep order
    valid_files = list(dict.fromkeys(valid_files))

    for valid_file in valid_files:
        if isinstance(valid_file, list):
            logging.error("A least one FieldMap still has several file maps, "
                          "which Cython will not support. Skipping...")
            valid_files.remove(valid_file)
    tm_c.init_arrays(valid_files)


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


# Check if could use dat_content instead of re-creating it
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
    # idx_elt = 0
    # dat_filecontent = elts.files['dat_content']
    # field_map_folder = elts.files['field_map_folder']
    dat_content = [elt_or_cmd.line
                   for elt_or_cmd in elts.files['elts_n_cmds']]

    dat_content: list[list[str]] = []
    for elt_or_cmd in elts.files['elts_n_cmds']:
        line = elt_or_cmd.line

        if elt_or_cmd in new_phases:
            line[3] = str(np.rad2deg(new_phases[elt_or_cmd]))
        if elt_or_cmd in new_k_e:
            line[6] = str(new_k_e[elt_or_cmd])
        if elt_or_cmd in new_abs_phase_flag:
            line[10] = str(new_abs_phase_flag[elt_or_cmd])

        dat_content.append(line)


def dat_filecontent_from_smaller_list_of_elements(
    original_elts_n_cmds: list[Element | Command],
    elts: list[Element],
) -> list[list[str], list[Element | Command]]:
    """
    Create a ``.dat`` with only elements of ``elts`` (and concerned commands).

    Properties of the FIELD_MAP, i.e. amplitude and phase, remain untouched, as
    it is the job of :func:`update_field_maps_in_dat`.

    """
    indexes_to_keep = [elt.get('dat_idx', to_numpy=False) for elt in elts]
    last_index = indexes_to_keep[-1] + 1

    new_dat_filecontent: list[list[str]] = []
    new_elts_n_cmds: list[Element | Command] = []
    for i, elt_or_cmd in enumerate(original_elts_n_cmds[:last_index]):
        element_to_keep = (isinstance(elt_or_cmd, Element | Dummy)
                           and elt_or_cmd.idx['dat_idx'] in indexes_to_keep)

        useful_command = (isinstance(elt_or_cmd, Command)
                          and elt_or_cmd.concerns_one_of(indexes_to_keep))

        if not (element_to_keep or useful_command):
            continue

        new_dat_filecontent.append(elt_or_cmd.line)
        new_elts_n_cmds.append(elt_or_cmd)

    end = original_elts_n_cmds[-1]
    new_dat_filecontent.append(end.line)
    new_elts_n_cmds.append(end)
    return new_dat_filecontent, new_elts_n_cmds


def save_dat_filecontent_to_dat(dat_content: list[list[str]],
                                dat_path: str) -> None:
    """Save the content of the updated dat to a `.dat`."""
    with open(dat_path, 'w', encoding='utf-8') as file:
        for line in dat_content:
            file.write(' '.join(line) + '\n')
    logging.info(f"New dat saved in {dat_path}.")
