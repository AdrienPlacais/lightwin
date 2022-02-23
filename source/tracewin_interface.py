#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:52:37 2022

@author: placais
"""
import os.path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
from electric_field import load_field_map_file
import elements

to_be_implemented = ['SPACE_CHARGE_COMP', 'FREQ', 'FIELD_MAP_PATH',
                     'LATTICE', 'END']
# Dict of data that can be imported from TW's "Data" table.
# More info in load_tw_results
dict_tw_data_table = {
    'v_cav_mv': 6,
    'input_phase': 7,
    'synch_phase': 8,
    'phi_s_deg': 8,
    'energy': 9,
    'beta_synch': 10,
    'full_length': 11,
    'abs_phase': 12,
    }


def load_dat_file(dat_filepath):
    """
    Load the dat file and convert it into a list of lines.

    Parameters
    ----------
    dat_filepath: string
        Filepath to the .dat file, as understood by TraceWin.

    Return
    ------
    dat_filecontent: list, opt
        List containing all the lines of dat_filepath.
    """
    dat_filecontent = []

    # Load and read data file
    with open(dat_filepath) as file:
        for line in file:
            # Remove trailing whitespaces
            line = line.strip()

            # We check that the current line is not empty or that it is not
            # reduced to a comment only
            if(len(line) == 0 or line[0] == ';'):
                continue

            # Remove any trailing comment
            line = line.split(';')[0]
            line = line.split()

            dat_filecontent.append(line)

    list_of_elements = _create_structure(dat_filepath, dat_filecontent)

    return dat_filecontent, list_of_elements


def _create_structure(dat_filepath, dat_filecontent):
    """
    Create structure using the loaded dat file.

    Parameters
    ----------
    dat_filecontent: list, opt
        List containing all the lines of dat_filepath.

    Return
    ------
    list_of_elements: list of Element
        List containing all the Element objects.
    """
    # @TODO Implement lattice
    # Dictionnary linking element name with correct sub-class
    subclasses_dispatcher = {
        'QUAD': elements.Quad,
        'DRIFT': elements.Drift,
        'FIELD_MAP': elements.FieldMap,
        'CAVSIN': elements.CavSin,
        'SOLENOID': elements.Solenoid,
    }

    # We look at each element in dat_filecontent, and according to the
    # value of the 1st column string we create the appropriate Element
    # subclass and store this instance in list_of_elements
    try:
        list_of_elements = [subclasses_dispatcher[elem[0]](elem)
                            for elem in dat_filecontent if elem[0]
                            not in to_be_implemented]
    except KeyError:
        print('Warning, an element from filepath was not recognized.')

    _load_filemaps(dat_filepath, dat_filecontent, list_of_elements)

    return list_of_elements


def _load_filemaps(dat_filepath, dat_filecontent, list_of_elements):
    """
    Load all the filemaps.

    Parameters
    ----------
    dat_filepath: string
        Filepath to the .dat file, as understood by TraceWin.
    dat_filecontent: list, opt
        List containing all the lines of dat_filepath.
    list_of_elements: list of Element
        List containing all the Element objects.
    """
    # Get folder of all field maps
    for line in dat_filecontent:
        if line[0] == 'FIELD_MAP_PATH':
            field_map_folder = line[1]

    field_map_folder = os.path.dirname(dat_filepath) + field_map_folder[1:]

    for elt in list_of_elements:
        if 'field_map_file_name' in vars(elt):
            elt.field_map_file_name = field_map_folder + '/' \
                + elt.field_map_file_name
            load_field_map_file(elt, elt.acc_field)


def save_new_dat(fixed_linac, filepath_old):
    """Save a new dat with the new linac settings."""
    _update_dat_with_fixed_cavities(fixed_linac.files['dat_filecontent'],
                                    fixed_linac.list_of_elements)

    filepath_new = filepath_old[:-4] + '_fixed.dat'
    with open(filepath_new, 'w') as file:
        for line in fixed_linac.files['dat_filecontent']:
            file.write(' '.join(line) + '\n')


def _update_dat_with_fixed_cavities(dat_filecontent, list_of_elements):
    """Create a new dat containing the new linac settings."""
    idx_elt = 0
    for line in dat_filecontent:
        if line[0] in to_be_implemented:
            continue

        else:
            if line[0] == 'FIELD_MAP':
                elt = list_of_elements[idx_elt]

                if elt.status['failed'] or elt.status['compensate']:
                    line[3] = str(np.rad2deg(elt.acc_field.phi_0))
                    line[6] = str(elt.acc_field.norm)
            idx_elt += 1


def load_tw_results(filepath, prop):
    """
    Load a property from TraceWin's "Data" table.

    Parameters
    ----------
    filepath: string
        Path to results file. It must be saved from TraceWin:
            Data > Save table to file.
    prop: string
        Name of the desired property. Must be in dict_property.

    Return
    ------
    data_ref: numpy array
        Array containing the desired property.
    """
    if not os.path.isfile(filepath):
        print('debug/compare_energies error:')
        print('The filepath to the energy file is invalid. Please check the')
        print('source code for more info. Enter a valid filepath:')
        Tk().withdraw()
        filepath = askopenfilename(
            filetypes=[("TraceWin energies file", ".txt")])

    idx = dict_tw_data_table[prop]

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
