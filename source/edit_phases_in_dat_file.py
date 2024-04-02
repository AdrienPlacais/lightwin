#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:49:36 2023.

@author: placais

In this module we define utility functions to format ``.dat`` files. In
particular, set new definition of phase in FIELD_MAP and/or remove the first
elements from the dat (injector...).

"""
from tkinter import Tk
from tkinter.simpledialog import askfloat
from tkinter.filedialog import askopenfilename

from tracewin_utils.load import dat_file, table_structure_file
from tracewin_utils.dat_files import create_structure


if __name__ == '__main__':
    freq_bunch = askfloat("First question",
                          "What is the bunch frequency in MHz?",
                          initialvalue=88.0525)

    Tk().withdraw()
    dat_path = askopenfilename(
        filetypes=[("Linac structure file", ".dat")],
        title="Gimme your .dat file",
        initialfile='/home/placais/LightWin/data/SPIRAL2_post_scraper/'
                    'SP2linac_BD_AsurQ_2.01355_Eout_20.0.dat',
    )
    dat_content = dat_file(dat_path)
    elements = create_structure(dat_content,
                                dat_path,
                                force_a_lattice_to_each_element=False,
                                force_a_section_to_each_element=False,
                                load_electromagnetic_files=False,
                                check_consistency=False,
                                freq_bunch=freq_bunch)

    txt_path = askopenfilename(
        filetypes=[("Table structure file", ".txt")],
        title="Run a TW simulation, save the 'Data' tab, give it to me",
        initialfile='/home/placais/LightWin/data/SPIRAL2_post_scraper/results/'
                    'table_structure_file.txt',
    )
    txt_content = table_structure_file(txt_path)
    # Get entry phase, synchronous phase, relative OR absolute phase of each
    # cavity

    print("Now tell me how the phases should be defined.")
    print("\t1 - Synchronous phases (SET_SYNC_PHASE)")
    print("\t2 - Absolute phases (last FIELD_MAP argument = 1)")
    print("\t3 - Relative phases (last FIELD_MAP argument = 0)")

    print("And now I need to know the name of the element at which the new "
          ".dat should start. It will modify the phases of the FIELD_MAPs if "
          "you asked for absolute phases.")
    # Get phase shift

    # Compute phase that should be saved

    # Save it in dat

    # Save new dat

    print("Now you may want to restart your Python kernel to avoid confusions "
          "between the variables declared here and the ones from the main "
          "program.")
