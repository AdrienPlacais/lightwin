#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:52:37 2022.

@author: placais

This module holds all the functions to transfer and convert data between
LightWin and TraceWin.

TODO insert line skip at each section change in the output.dat
"""
import logging
import itertools
import os.path
from typing import Callable
import pandas as pd
import numpy as np

import config_manager as con
from core.elements import (_Element, Quad, Drift, FieldMap, Solenoid, Lattice,
                           Freq, FieldMapPath, End)
import tracewin_utils.load


try:
    import beam_calculation.transfer_matrices_c as tm_c
except ModuleNotFoundError:
    MESSAGE = 'Cython module not compilated. Check elements.py and setup.py'\
        + ' for more information.'
    if con.FLAG_CYTHON:
        raise ModuleNotFoundError(MESSAGE)
    logging.warning(MESSAGE)
    # Load Python version as Cython to allow the execution of the code.
    import beam_calculation.transfer_matrices_p as tm_c


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
    # Dictionnary linking element nature with correct sub-class
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


# TODO is it necessary to load all the electric fields when _p?
def set_all_electric_field_maps(files: dict, sections: list[list[_Element]]
                                ) -> None:
    """
    Load all the filemaps.

    Parameters
    ----------
    files : dict
        `Accelerator.files` dictionary.
    sections : list[list[_Element]]
        List of sections containing lattices containing `_Element` objects.

    """
    filepaths = []
    for i, section in enumerate(sections):
        for lattice in section:
            for elt in lattice:
                if elt.get('nature') == 'FIELD_MAP':
                    elt.field_map_file_name = os.path.join(
                        files['field_map_folder'], elt.field_map_file_name)
                    a_f = elt.acc_field
                    a_f.e_spat, a_f.n_z = get_single_electric_field_map(elt)

                    # For Cython, we need one filepath per section
                    if con.FLAG_CYTHON and len(filepaths) == i:
                        filepaths.append(elt.field_map_file_name)
    # Init arrays
    if con.FLAG_CYTHON:
        tm_c.init_arrays(filepaths)


def get_single_electric_field_map(
    cav: FieldMap) -> tuple[Callable[[float | np.ndarray], float | np.ndarray],
                            int]:
    """
    Select the field map file and call the proper loading function.

    Warning, `filename` is directly extracted from the `.dat` file used by
    TraceWin. Thus, the relative filepath may be misunderstood by this script.

    Also check that the extension of the file is `.edz`, or manually change
    this function.

    Only 1D electric field map are implemented.

    """
    # FIXME
    cav.field_map_file_name += ".edz"
    assert tracewin_utils.load.is_loadable(
        cav.field_map_file_name, cav.geometry, cav.aperture_flag), \
        f"Error preparing {cav}'s field map."

    _, extension = os.path.splitext(cav.field_map_file_name)
    import_function = tracewin_utils.load.FIELD_MAP_LOADERS[extension]

    n_z, zmax, norm, f_z = import_function(cav.field_map_file_name)
    assert is_a_valid_electric_field(n_z, zmax, norm, f_z, cav.length_m), \
        f"Error loading {cav}'s field map."

    z_cavity_array = np.linspace(0., zmax, n_z + 1) / norm

    def e_spat(pos: float | np.ndarray) -> float | np.ndarray:
        return np.interp(x=pos, xp=z_cavity_array, fp=f_z, left=0., right=0.)

    return e_spat, n_z


def is_a_valid_electric_field(n_z: int, zmax: float, norm: float,
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


# FIXME Cannot import Acclerator type (cricular import)
# Maybe this routine would be better in Accelerator?
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


def resample_tracewin_results(ref: object,
                              fix: object) -> None:
    """Interpolate the `fix` results @ `ref` positions."""
    logging.critical("resample_tracewin_results should not exist anymore.")
    logging.critical("resample_tracewin_results should be moved away")
    ref_results = ref.results_multipart
    fix_results = fix.results_multipart

    if ref_results is None or fix_results is None:
        logging.error("At least one multiparticle simulation was not "
                      + "performed (or not loaded).")
        return

    z_ref = ref_results['z(m)']
    z_fix = fix_results['z(m)'].copy()

    for key, val in fix_results.items():
        if isinstance(val, float):
            continue

        if val.ndim == 2:
            for axis in range(val.shape[1]):
                fix_results[key][:, axis] = np.interp(z_ref, z_fix,
                                                      val[:, axis])
            continue

        fix_results[key] = np.interp(z_ref, z_fix, val)

    fix.results_multipart = fix_results
