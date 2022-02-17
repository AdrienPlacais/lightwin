#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:52:37 2022

@author: placais
"""
import os.path
from electric_field import load_field_map_file
import elements


def load_dat_file(dat_filepath):
    """
    Load the dat file and convert it into a list of lines.

    Parameters
    ----------
    dat_filepath: string
        Filepath to the .dat file, as understood by TraceWin.

    Return
    ------
    dat_file_content: list, opt
        List containing all the lines of dat_filepath.
    """
    dat_file_content = []

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

            dat_file_content.append(line)
    return dat_file_content


def load_filemaps(dat_filepath, dat_file_content, list_of_elements):
    """Assign filemaps paths and load them."""
    # Get folder of all field maps
    for line in dat_file_content:
        if line[0] == 'FIELD_MAP_PATH':
            field_map_folder = line[1]

    field_map_folder = os.path.dirname(dat_filepath) + field_map_folder[1:]

    for elt in list_of_elements:
        if 'field_map_file_name' in vars(elt):
            elt.field_map_file_name = field_map_folder + '/' \
                + elt.field_map_file_name
            load_field_map_file(elt, elt.acc_field)


def create_structure(dat_file_content):
    """Create structure using the loaded dat file."""
    # @TODO Implement lattice
    # Dictionnary linking element name with correct sub-class
    subclasses_dispatcher = {
        'QUAD': elements.Quad,
        'DRIFT': elements.Drift,
        'FIELD_MAP': elements.FieldMap,
        'CAVSIN': elements.CavSin,
        'SOLENOID': elements.Solenoid,
    }
    to_be_implemented = ['SPACE_CHARGE_COMP', 'FREQ', 'FIELD_MAP_PATH',
                         'LATTICE', 'END']
    # TODO Maybe some non-elements such as FREQ or LATTICE would be better
    # off another file/module

    # We look at each element in dat_file_content, and according to the
    # value of the 1st column string we create the appropriate Element
    # subclass and store this instance in list_of_elements
    try:
        list_of_elements = [subclasses_dispatcher[elem[0]](elem)
                            for elem in dat_file_content if elem[0]
                            not in to_be_implemented]
    except KeyError:
        print('Warning, an element from filepath was not recognized.')

    return list_of_elements
