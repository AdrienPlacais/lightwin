#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:52:37 2022.

@author: placais

TODO insert line skip at each section change in the output.dat
"""
import logging
import re
import os.path
import subprocess
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np

import config_manager as con
from core.elements import (_Element, Quad, Drift, FieldMap, Solenoid, Lattice,
                           Freq, FieldMapPath)
from core.electric_field import load_field_map_file


try:
    import core.transfer_matrices_c as tm_c
except ModuleNotFoundError:
    MESSAGE = 'Cython module not compilated. Check elements.py and setup.py'\
        + ' for more information.'
    if con.FLAG_CYTHON:
        raise ModuleNotFoundError(MESSAGE)
    logging.warning(MESSAGE)
    # Load Python version as Cython to allow the execution of the code.
    import core.transfer_matrices_p as tm_c


to_be_implemented = [
    'SPACE_CHARGE_COMP', 'SET_SYNC_PHASE', 'STEERER',
    'ADJUST', 'ADJUST_STEERER', 'ADJUST_STEERER_BX', 'ADJUST_STEERER_BY',
    'DIAG_SIZE', 'DIAG_DSIZE', 'DIAG_DSIZE2', 'DIAG_DSIZE3', 'DIAG_DSIZE4',
    'DIAG_DENERGY', 'DIAG_ENERGY', 'DIAG_TWISS', 'DIAG_WAIST',
    'DIAG_POSITION', 'DIAG_DPHASE',
    'ERROR_CAV_NCPL_STAT', 'ERROR_CAV_NCPL_DYN',
    'END',
    'SET_ADV', 'LATTICE_END', 'SHIFT', 'THIN_STEERING', 'APERTURE']
not_an_element = ['LATTICE', 'FREQ']

# Dict of data that can be imported from TW's "Data" table.
# More info in load_tw_results
d_tw_data_table = {
    'v_cav_mv': 6,
    'phi_0_rel': 7,
    'phi_s': 8,
    'w_kin': 9,
    'beta': 10,
    'z_abs': 11,
    'phi_abs_array': 12,
}


def load_dat_file(dat_filepath: str) -> tuple[list[list[str]], list[_Element]]:
    """
    Load the dat file and convert it into a list of lines.

    Parameters
    ----------
    dat_filepath : string
        Filepath to the .dat file, as understood by TraceWin.

    Return
    ------
    dat_filecontent : list[list[str]]
        List containing all the lines of dat_filepath.
    elts : list[_Element]
        The _Element objects.
    """
    dat_filecontent = []
    logging.warning("Personalized name of elements not handled for now.")

    # Load and read data file
    with open(dat_filepath) as file:
        for line in file:
            # Remove trailing whitespaces
            line = line.strip()

            # We check that the current line is not empty or that it is not
            # reduced to a comment only
            if len(line) == 0 or line[0] == ';':
                continue

            # Remove any trailing comment
            line = line.split(';')[0]
            # Remove element name
            line = line.split(':')[-1]
            # Remove everything between parenthesis
            # https://stackoverflow.com/questions/14596884/remove-text-between-and
            line = re.sub("([\(\[]).*?([\)\]])", "", line)

            dat_filecontent.append(line.split())

    elts = _create_structure(dat_filecontent)

    return dat_filecontent, elts


def _create_structure(dat_filecontent: list[list[str]]) -> list[_Element]:
    """
    Create structure using the loaded dat file.

    Parameters
    ----------
    dat_filecontent : list[list[str]]
        List containing all the lines of dat_filepath.

    Return
    ------
    elements_list: list[_Element]
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
    }

    # We look at each element in dat_filecontent, and according to the
    # value of the 1st column string we create the appropriate Element
    # subclass and store this instance in l_elts
    elements_list = [subclasses_dispatcher[elem[0]](elem)
                     for elem in dat_filecontent
                     if elem[0] not in to_be_implemented]

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
        sub_list = [elt
                    for elt in elts
                    if elt.get('nature') == key
                    ]
        for i, elt in enumerate(sub_list, start=1):
            elt.elt_info['elt_name'] = value + str(i)


# TODO is it necessary to load all the electric fields when _p?
def load_filemaps(files: dict, sections: list[list[_Element]],
                  freqs: list[float], freq_bunch: float) -> None:
    """
    Load all the filemaps.

    Parameters
    ----------
    files : dict
        Accelerator.files dictionary.
    sections: list of lists of Element
        List of sections containing lattices containing Element objects.
    freqs : list
        List of the RF frequencies in MHz in every section.
    freq_bunch : float
        Bunch frequency in MHz.
    """
    assert len(sections) == len(freqs)

    filepaths = []
    for i, section in enumerate(sections):
        f_mhz = freqs[i].f_rf_mhz
        n_cell = int(f_mhz / freq_bunch)   # FIXME
        for lattice in section:
            for elt in lattice:
                if elt.get('nature') == 'FIELD_MAP':
                    elt.field_map_file_name = os.path.join(
                        files['field_map_folder'], elt.field_map_file_name)
                    a_f = elt.acc_field
                    a_f.e_spat, a_f.n_z = load_field_map_file(elt)
                    a_f.init_freq_ncell(f_mhz, n_cell)

                    # For Cython, we need one filepath per section
                    if con.FLAG_CYTHON and len(filepaths) == i:
                        filepaths.append(elt.field_map_file_name)
    # Init arrays
    if con.FLAG_CYTHON:
        tm_c.init_arrays(filepaths)


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
        if line[0] in to_be_implemented or line[0] in not_an_element:
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


def load_tw_results(filepath: str, prop: str) -> np.ndarray:
    """
    Load a property from TraceWin's "Data" table.

    Parameters
    ----------
    filepath : string
        Path to results file. It must be saved from TraceWin:
            Data > Save table to file.
    prop : string
        Name of the desired property. Must be in d_property.

    Return
    ------
    data_ref: numpy array
        Array containing the desired property.
    """
    if not os.path.isfile(filepath):
        logging.warning(
            "Filepath to results is incorrect. Provide another one.")
        Tk().withdraw()
        filepath = askopenfilename(
            filetypes=[("TraceWin energies file", ".txt")])

    idx = d_tw_data_table[prop]

    data_ref = []
    with open(filepath) as file:
        for line in file:
            try:
                int(line.split('\t')[0])
            except ValueError:
                continue
            splitted_line = line.split('\t')
            new_data = splitted_line[idx]
            if new_data == '-':
                new_data = np.NaN
            data_ref.append(new_data)
    data_ref = np.array(data_ref).astype(float)
    return data_ref


def load_transfer_matrices(filepath_list: list[str]) -> np.ndarray:
    """Load transfer matrices saved in 4 files by components."""
    i = 0
    for path in filepath_list:
        assert os.path.isfile(path), \
            'Incorrect filepath in plot_transfer_matrices.'

        if i == 0:
            r_zz_tot_ref = np.loadtxt(filepath_list[i])

        else:
            tmp = np.loadtxt(filepath_list[i])[:, 1]
            tmp = np.expand_dims(tmp, axis=1)
            r_zz_tot_ref = np.hstack((r_zz_tot_ref, tmp))
        i += 1

    return r_zz_tot_ref


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
        'Energy (MeV)': lambda lin, elt: lin.get('w_kin')[elt.idx['s_out']],
        'Beta Synch.': lambda lin, elt: lin.get('beta')[elt.idx['s_out']],
        'Full length (mm)': lambda lin, elt: lin.get('z_abs')[
            elt.idx['s_out']] * 1e3,
        'Abs. phase (deg)': lambda lin, elt: lin.get(
            'phi_abs_array', to_deg=True)[elt.idx['s_out']],
    }

    data = []
    n_latt = 1
    i = 0
    for section in linac.elements['l_sections']:
        for lattice in section:
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


def run_tw(ini_path: str, path_cal: str, dat_file: str,
           tw_path: str = "/usr/local/bin/./TraceWin", **kwargs) -> None:
    """
    Run TraceWin.

    Parameters
    ----------
    ini_path : str
        Path to the .ini TraceWin file.
    path_cal : str
        Path to the output folder, where TW results will be stored. Overrides
        the path_cal defined in .ini.
    dat_file : str
        Path to the TraceWin .dat file, with accelerator structure. Overrides
        the dat_file defined in .ini.
    tw_path : str, optional
        Path to the TraceWin command. The default is
        "/usr/local/bin/./TraceWin".
    **kwargs : dict
        TraceWin optional arguments. Override what is defined in .ini.

    """
    cmd = _tw_cmd(tw_path, ini_path, path_cal, dat_file, **kwargs)
    logging.info(f"Running TW with command {cmd}...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    process.wait()
    for line in process.stdout:
        print(line)
    logging.info("TW finished!")


def _tw_cmd(tw_path: str, ini_path: str, path_cal: str, dat_file: str,
            **kwargs) -> str:
    """Make the command line to launch TraceWin."""
    cmd = [tw_path, ini_path, f"path_cal={path_cal}", f"dat_file={dat_file}"]
    for key, value in kwargs.items():
        if value is None:
            cmd.append(key)
            continue
        cmd.append(key + "=" + str(value))
    return cmd


def get_multipart_tw_results(folder: str, filename: str = 'partran1.out'
                             ) -> dict:
    """Get the results."""
    f_p = os.path.join(folder, filename)
    n_lines_header = 9
    d_out = {}

    with open(f_p) as file:
        for i, line in enumerate(file):
            if i == n_lines_header:
                headers = line.strip().split()
                break

    out = np.loadtxt(f_p, skiprows=n_lines_header)
    for i, key in enumerate(headers):
        d_out[key] = out[:, i]
    logging.info(f"successfully loaded {f_p}")
    return d_out


def get_transfer_matrices(folder: str, filename: str = 'Transfer_matrix1.dat',
                          high_def: bool = False) -> tuple[np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray]:
    """Get the full transfer matrices calculated by TraceWin."""
    if high_def:
        raise IOError("High definition not implemented. Can only import"
                      + "transfer matrices @ element positions.")
    f_p = os.path.join(folder, filename)

    data = None
    num = []
    z_m = []
    t_m = []

    with open(f_p) as file:
        for i, line in enumerate(file):
            if i % 7 == 0:
                # Get element # and position
                data = line.split()
                num.append(int(data[1]))
                z_m.append(float(data[3]))

                # Re-initialize data
                data = []
                continue

            data.append([float(dat) for dat in line.split()])

            # Save transfer matrix
            if (i + 1) % 7 == 0:
                t_m.append(data)
    logging.info(f"successfully loaded {f_p}")
    return np.array(num), np.array(z_m), np.array(t_m)


def get_tw_cav_param(folder: str, filename: str = 'Cav_set_point_res.dat'
                     ) -> dict:
    """Get the cavity parameters."""
    f_p = os.path.join(folder, filename)
    n_lines_header = 1
    d_out = {}

    with open(f_p) as file:
        for i, line in enumerate(file):
            if i == n_lines_header - 1:
                headers = line.strip().split()
                break

    out = np.loadtxt(f_p, skiprows=n_lines_header)
    for i, key in enumerate(headers):
        d_out[key] = out[:, i]
    logging.info(f"successfully loaded {f_p}")
    return d_out
