#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:49:23 2023.

@author: placais
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

    Return
    ------
    dat_filecontent : list[list[str]]
        List containing all the lines of dat_filepath.
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


