#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:52:37 2022

@author: placais
"""


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
