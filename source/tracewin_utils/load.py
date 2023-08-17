#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:49:23 2023.

@author: placais

This module holds the function to load and pre-process the TraceWin files.

TODO : handle personalized name of elements

"""
import logging
import os.path
import re
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np


# Dict of data that can be imported from TW's "Data" table.
# More info in results
TRACEWIN_IMPORT_DATA_TABLE = {
    'v_cav_mv': 6,
    'phi_0_rel': 7,
    'phi_s': 8,
    'w_kin': 9,
    'beta': 10,
    'z_abs': 11,
    'phi_abs_array': 12,
}


def dat_file(dat_filepath: str) -> list[list[str]]:
    """
    Load the dat file, convert it into a list of lines and a list of _Element.

    Parameters
    ----------
    dat_filepath : string
        Filepath to the .dat file, as understood by TraceWin.

    Returns
    -------
    dat_filecontent : list[list[str]]
        List containing all the lines of dat_filepath.

    """
    dat_filecontent = []
    logging.info("Personalized name of elements not handled for now.")

    with open(dat_filepath) as file:
        for line in file:
            line = line.strip()

            if len(line) == 0 or line[0] == ';':
                continue

            line = line.split(';')[0]
            line = line.split(':')[-1]
            # Remove everything between parenthesis
            # https://stackoverflow.com/questions/14596884/remove-text-between-and
            line = re.sub("([\(\[]).*?([\)\]])", "", line)

            dat_filecontent.append(line.split())
    return dat_filecontent


def results(filepath: str, prop: str) -> np.ndarray:
    """
    Load a property from TraceWin's "Data" table.

    Parameters
    ----------
    filepath : string
        Path to results file. It must be saved from TraceWin:
            Data > Save table to file.
    prop : string
        Name of the desired property. Must be in d_property.

    Returns
    -------
    data_ref: numpy array
        Array containing the desired property.

    """
    if not os.path.isfile(filepath):
        logging.warning(
            "Filepath to results is incorrect. Provide another one.")
        Tk().withdraw()
        filepath = askopenfilename(
            filetypes=[("TraceWin energies file", ".txt")])

    idx = TRACEWIN_IMPORT_DATA_TABLE[prop]

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


def electric_field_1d(path: str) -> tuple[int, float, float, np.ndarray]:
    """
    Load a 1D electric field (.edz extension).

    Parameters
    ----------
    path : string
        The path to the .edz file to load.

    Returns
    -------
    n_z : int
        Number of steps in the array.
    zmax : float
        z position of the filemap end.
    norm : float
        Electric field normalisation factor. It is different from k_e (6th
        argument of the FIELD_MAP command). Electric fields are normalised by
        k_e/norm, hence norm should be unity by default.
    f_z : np.ndarray
        Array of electric field in MV/m.

    """
    f_z = []

    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == 0:
                line_splitted = line.split(' ')

                # Sometimes the separator is a tab and not a space:
                if len(line_splitted) < 2:
                    line_splitted = line.split('\t')

                n_z = int(line_splitted[0])
                # Sometimes there are several spaces or tabs between numbers
                zmax = float(line_splitted[-1])
                continue

            if i == 1:
                norm = float(line)
                continue

            f_z.append(float(line))

    return n_z, zmax, norm, np.array(f_z)


def is_loadable(field_map_file_name: str, geometry: int, aperture_flag: int
                ) -> bool:
    """Assert that the options for the FIELD_MAP in the .dat are ok."""
    _, extension = os.path.splitext(field_map_file_name)
    if extension not in FIELD_MAP_LOADERS:
        logging.error(f"Field map file extension is {extension}, "
                      + f"while only {FIELD_MAP_LOADERS.keys()} are "
                      + "implemented.")
        return False

    if geometry < 0:
        logging.error("Second order off-axis development not implemented.")
        return False

    field_nature = int(np.log10(geometry))
    if field_nature != 2:
        logging.error("Only RF electric fields implemented.")
        return False

    field_geometry = int(str(geometry)[0])
    if field_geometry != 1:
        logging.error("Only 1D field implemented.")
        return False

    if aperture_flag > 0:
        logging.warning("Space charge compensation maps not implemented.")

    return True


FIELD_MAP_LOADERS = {
    ".edz": electric_field_1d
}
