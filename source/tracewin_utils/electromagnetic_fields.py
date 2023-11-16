#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds functions to handle TraceWin electromagnetic fields.

.. note::
    Last compatibility check: TraceWin v2.22.1.0

"""
import logging
import os.path
from typing import Callable
import pandas as pd
import numpy as np

from core.elements.field_maps.field_map import FieldMap
import tracewin_utils.load


def load_electromagnetic_fields(field_maps: list[FieldMap],
                                cython: bool) -> None:
    """
    Load field map files.

    As for now, only 1D RF electric field are handled by :class:`.Envelope1D`.
    With :class:`.TraceWin`, every field is supported.

    """
    for field_map in field_maps:
        field_map_types = _geom_to_field_map_type(field_map.geometry,
                                                  remove_no_field=True)
        extensions = _file_map_extensions(field_map_types)

        field_map.set_full_path(extensions)

        args = _load_field_map_file(field_map)
        if args is not None:
            field_map.acc_field.e_spat = args[0]
            field_map.acc_field.n_z = args[1]

    if cython:
        _load_electromagnetic_fields_for_cython(field_maps)


def _load_electromagnetic_fields_for_cython(field_maps: list[FieldMap]
                                            ) -> None:
    """Load one electric field per section."""
    valid_files = [field_map.field_map_file_name
                   for field_map in field_maps
                   if field_map.acc_field.e_spat is not None
                   and field_map.acc_field.n_z is not None]
    # Trick to remove duplicates and keep order
    valid_files = list(dict.fromkeys(valid_files))

    for valid_file in valid_files:
        if isinstance(valid_file, list):
            logging.error("A least one FieldMap still has several file maps, "
                          "which Cython will not support. Skipping...")
            valid_files.remove(valid_file)
    tm_c.init_arrays(valid_files)


def _geom_to_field_map_type(geom: int,
                            remove_no_field: bool = True
                            ) -> dict[str, str]:
    """
    Determine the field map type from TraceWin's ``geom`` parameter.

    Notes
    -----
    Last compatibility check: TraceWin v2.22.1.0

    """
    figures = (int(i) for i in f"{abs(geom):0>5}")
    field_types = ('static electric field',
                   'static magnetic field',
                   'RF electric field',
                   'RF magnetic field',
                   '3D aperture map')
    field_geometries = {
        0: 'no field',
        1: '1D: F(z)',
        2: 'not available',
        3: 'not available',
        4: '2D cylindrical static or RF electric field',
        5: '2D cylindrical static or RF magnetic field',
        6: '2D cartesian field',
        7: '3D cartesian field',
        8: '3D cylindrical field',
        9: '1D: G(z)',
    }
    out = {field_type: field_geometries[figure]
           for figure, field_type in zip(figures, field_types)}
    if 'not available' in out.values():
        logging.warning("At least one invalid field geometry was given in the "
                        ".dat.")
    if not remove_no_field:
        return out

    for key in list(out):
        if out[key] == 'no field':
            out.pop(key)
    return out


def _file_map_extensions(field_map_type: dict[str, str]
                         ) -> dict[str, list[str]]:
    """
    Get the proper field map extensions.

    Parameters
    ----------
    field_map_type : dict[str, str]
        Dictionary which keys are the type of electromagnetic field, and values
        are the geometry.

    Returns
    -------
    extensions : dict[str, list[str]]
        Dictionary with the same keys as input. The values are lists containing
        all the extensions of the files to load (no "." in front of extension).

    """
    extensions = {field_type: None
                  for field_type in field_map_type}

    char_1 = {'electric': 'e',
              'magnetic': 'b'}
    char_2 = {'static': 's',
              'RF': 'd'}
    char_3 = {'1D:': ['z'],
              '2D cylindrical': ['r', 'z', 'q'],
              '2D cartesian': ['x', 'y'],
              '3D cartesian': ['x', 'y', 'z'],
              '3D cylindrical': ['r', 'q', 'z']
              }

    for field_type, geometry in field_map_type.items():
        if geometry == 'not available':
            continue

        if field_type == '3D aperture map':
            extensions[field_type] = ['ouv']
            continue

        splitted = field_type.split(' ')
        base_extension = [char_1.get(splitted[1], None),
                          char_2.get(splitted[0], None)]

        geometry_as_a_key = geometry.split(' ')
        if geometry_as_a_key[0] == '1D:':
            geometry_as_a_key = geometry_as_a_key[0]
        else:
            geometry_as_a_key = ' '.join(geometry_as_a_key[:2])

        extension = [base_extension + [last_char]
                     for last_char in char_3[geometry_as_a_key]]
        extensions[field_type] = [''.join(ext) for ext in extension]
    return extensions


def _load_field_map_file(
    field_map: FieldMap) -> tuple[Callable[[float | np.ndarray],
                                           float | np.ndarray] | None,
                                  int | None]:
    """
    Go across the field map file names and load the first recognized.

    For now, only ``.edz`` files (1D electric RF) are implemented. This will be a
    problem with :class:`Envelope1D`, but :class:`TraceWin` does not care.

    """
    if len(field_map.field_map_file_name) > 1:
        logging.debug("Loading of several field_maps not handled")
        return None, None

    for file_name in field_map.field_map_file_name:
        _, extension = os.path.splitext(file_name)

        if extension not in tracewin_utils.load.FIELD_MAP_LOADERS:
            logging.debug("Field map extension not handled.")
            continue

        import_function = tracewin_utils.load.FIELD_MAP_LOADERS[extension]

        # this will require an update if I want to implement new field map
        # extensions
        n_z, zmax, norm, f_z = import_function(file_name)
        if n_z is None:
            return None, None

        assert _is_a_valid_electric_field(n_z,
                                          zmax,
                                          norm,
                                          f_z,
                                          field_map.length_m), \
            f"Error loading {field_map}'s field map."

        z_cavity_array = np.linspace(0., zmax, n_z + 1) / norm

        def e_spat(pos: float | np.ndarray) -> float | np.ndarray:
            return np.interp(x=pos, xp=z_cavity_array, fp=f_z,
                             left=0., right=0.)

        # Patch to keep one filepath per FieldMap. Will require an update in
        # the future...
        field_map.field_map_file_name = file_name

        return e_spat, n_z


def _is_a_valid_electric_field(n_z: int, zmax: float, norm: float,
                               f_z: np.ndarray, cavity_length: float) -> bool:
    """Assert that the electric field that we loaded is valid."""
    if f_z.shape[0] != n_z + 1:
        logging.error(f"The electric field file should have {n_z + 1} "
                      + f"lines, but it is {f_z.shape[0]} lines long. ")
        return False

    tolerance = 1e-6
    if abs(zmax - cavity_length) > tolerance:
        logging.error(f"Mismatch between the length of the field map {zmax = }"
                      + f" and {cavity_length = }.")
        return False

    if abs(norm - 1.) > tolerance:
        logging.warning("Field map scaling factor (second line of the file) "
                        " is different from unity. It may enter in conflict "
                        + "with k_e (6th argument of FIELD_MAP in the .dat).")
    return True

# FIXME Cannot import Accelerator type (circular import)
# Maybe this routine would be better in Accelerator?
# |-> more SimulationOutput


def output_data_in_tw_fashion(linac) -> pd.DataFrame:
    """Mimick TW's Data tab."""
    larousse = {
        '#': lambda lin, elt: elt.get('elt_idx', to_numpy=False),
        'Name': lambda lin, elt: elt.get('elt_name', to_numpy=False),
        'Type': lambda lin, elt: elt.get('nature', to_numpy=False),
        'Length (mm)': lambda lin, elt: elt.length_m * 1e3,
        'Grad/Field/Amp': lambda lin, elt:
            elt.grad if (elt.get('nature', to_numpy=False) == 'QUAD')
            else np.NaN,
        'EoT (MV/m)': lambda lin, elt: None,
        'EoTLc (MV)': lambda lin, elt: elt.get('v_cav_mv'),
        'Input_Phase (deg)': lambda lin, elt: elt.get('phi_0_rel',
                                                      to_deg=True),
        'Sync_Phase (deg)': lambda lin, elt: elt.get('phi_s', to_deg=True),
        'Energy (MeV)': lambda lin, elt: lin.get('w_kin', elt=elt, pos='out'),
        'Beta Synch.': lambda lin, elt: lin.get('beta', elt=elt, pos='out'),
        'Full length (mm)': lambda lin, elt: lin.get('z_abs', elt=elt,
                                                     pos='out') * 1e3,
        'Abs. phase (deg)': lambda lin, elt: lin.get('phi_abs', to_deg=True,
                                                     elt=elt, pos='out'),
    }

    data = []
    n_latt = 1
    i = 0
    for lattice in linac.elts.by_lattice:
        lattice_n = '--------M' + str(n_latt)
        data.append([np.NaN, lattice_n, '', np.NaN, np.NaN, np.NaN, np.NaN,
                     np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
        n_latt += 1
        for elt in lattice:
            row = []
            for value in larousse.values():
                row.append(value(linac, elt))
                data.append(row)
                i += 1

    data = pd.DataFrame(data, columns=larousse.keys())
    return data
