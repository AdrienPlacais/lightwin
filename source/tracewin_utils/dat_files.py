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

from core.element_or_command import Dummy

from core.elements.element import Element
from core.elements.aperture import Aperture
from core.elements.drift import Drift
from core.elements.dummy import DummyElement
from core.elements.field_maps.field_map import FieldMap
from core.elements.quad import Quad
from core.elements.solenoid import Solenoid
from core.elements.thin_steering import ThinSteering

from core.commands.command import (Command,
                                   End,
                                   FieldMapPath,
                                   Freq,
                                   Lattice,
                                   LatticeEnd,
                                   Shift,
                                   Steerer,
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

ListOfElements = TypeVar('ListOfElements')


def create_structure(dat_content: list[list[str]],
                     dat_filepath: str,
                     force_a_lattice_to_each_element: bool = True,
                     force_a_section_to_each_element: bool = True,
                     load_electromagnetic_files: bool = True,
                     check_consistency: bool = True,
                     **kwargs: str | float) -> list[Element | Command]:
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
    elts_n_cmds = _create_element_n_command_objects(dat_content, dat_filepath)
    elts_n_cmds = _apply_commands(elts_n_cmds, kwargs['freq_bunch'])

    elts = list(filter(lambda elt: isinstance(elt, Element), elts_n_cmds))
    give_name(elts)

    elts_no_dummies = list(filter(
        lambda elt: not isinstance(elt, DummyElement),
        elts))
    if force_a_section_to_each_element:
        _force_a_section_for_every_element(elts_no_dummies)
    if force_a_lattice_to_each_element:
        _force_a_lattice_for_every_element(elts_no_dummies)

    field_maps = list(filter(lambda field_map: isinstance(field_map, FieldMap),
                             elts))
    if load_electromagnetic_files:
        _load_electromagnetic_fields(field_maps)

    if check_consistency:
        _check_consistency(elts_n_cmds)

    return elts_n_cmds


def _create_element_n_command_objects(dat_content: list[list[str]],
                                      dat_filepath: str
                                      ) -> list[Element | Command]:
    """
    Initialize the :class:`.Element` and :class:`.Command`.

    .. note::
        Elements and Command names in the ``subclasses_dispatcher`` dictionary
        are in uppercase. In the ``.dat`` file, you can use lower or uppercase,
        but they will be converted to uppercase in the routine anyway.

    """
    subclasses_dispatcher = {
        # Elements
        'APERTURE': Aperture,
        'DRIFT': Drift,
        'FIELD_MAP': FieldMap,
        'QUAD': Quad,
        'SOLENOID': Solenoid,
        'THIN_STEERING': ThinSteering,
        # Commands
        'END': End,
        'FIELD_MAP_PATH': FieldMapPath,
        'FREQ': Freq,
        'LATTICE': Lattice,
        'LATTICE_END': LatticeEnd,
        'SHIFT': Shift,
        'STEERER': Steerer,
        'SUPERPOSE_MAP': SuperposeMap,
    }
    kwargs = {'default_field_map_folder': dat_filepath}

    classes = []
    for line in dat_content:
        name = line[0].upper()
        if name in subclasses_dispatcher:
            classes.append(subclasses_dispatcher[name])
            continue
        name = line[2].upper()
        if name in subclasses_dispatcher:
            classes.append(subclasses_dispatcher[name])
            continue
        classes.append(None)

    elts_n_cmds = [classes[dat_idx](line, dat_idx, **kwargs)
                   if classes[dat_idx] is not None
                   else Dummy(line, dat_idx, warning=True)
                   for dat_idx, line in enumerate(dat_content)
                   ]
    return elts_n_cmds


def _apply_commands(elts_n_cmds: list[Element | Command],
                    freq_bunch: float,
                    ) -> list[Element | Command]:
    """Apply all the commands that are implemented."""
    index = 0
    while index < len(elts_n_cmds):
        elt_or_cmd = elts_n_cmds[index]

        if isinstance(elt_or_cmd, Command):
            elt_or_cmd.set_influenced_elements(elts_n_cmds)
            if elt_or_cmd.is_implemented:
                elts_n_cmds = elt_or_cmd.apply(elts_n_cmds,
                                               freq_bunch=freq_bunch)
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

        args = load_field_map_file(field_map)
        if args is not None:
            field_map.acc_field.e_spat = args[0]
            field_map.acc_field.n_z = args[1]

    if con.FLAG_CYTHON:
        _load_electromagnetic_fields_for_cython(field_maps)


def _load_electromagnetic_fields_for_cython(field_maps: list[FieldMap]
                                            ) -> None:
    """Load one electric field per section."""
    valid_files = [field_map.field_map_file_name
                   for field_map in field_maps
                   if field_map.acc_field.e_spat is not None
                   and field_map.acc_field.n_z is not None]
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


def _force_a_section_for_every_element(elts_without_dummies: list[Element]
                                       ) -> None:
    """Give a section index to every element."""
    idx_section = 0
    for elt in elts_without_dummies:
        idx = elt.idx['section']
        if idx is None:
            elt.idx['section'] = idx_section
            continue
        idx_section = idx


def _force_a_lattice_for_every_element(elts_without_dummies: list[Element]
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


def give_name(elts: list[Element]) -> None:
    """Give a name (the same as TW) to every element."""
    civil_register = {
        Quad: 'QP',
        Drift: 'DR',
        FieldMap: 'FM',
        Solenoid: 'SOL',
    }
    for key, value in civil_register.items():
        sub_list = list(filter(lambda elt: isinstance(elt, key), elts))
        for i, elt in enumerate(sub_list, start=1):
            if elt.elt_info['elt_name'] is None:
                elt.elt_info['elt_name'] = value + str(i)
    other_elements = list(filter(lambda elt: type(elt) not in civil_register,
                          elts))
    for i, elt in enumerate(other_elements, start=1):
        if elt.elt_info['elt_name'] is None:
            elt.elt_info['elt_name'] = 'ELT' + str(i)


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
) -> tuple[list[list[str]], list[Element | Command]]:
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
