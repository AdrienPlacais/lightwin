#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:52:37 2022.

@author: placais

TODO insert line skip at each section change in the output.dat
"""
import os.path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
from constants import FLAG_PHI_ABS, FLAG_CYTHON, F_BUNCH_MHZ
from electric_field import load_field_map_file
import elements
from helper import printc

try:
    import transfer_matrices_c as tm_c
except ModuleNotFoundError:
    MESSAGE = ', Cython module not compilated. Check elements.py and setup.py'\
        + ' for more information.'
    if FLAG_CYTHON:
        raise ModuleNotFoundError('Error' + MESSAGE)
    print('Warning' + MESSAGE)
    # Load Python version as Cython to allow the execution of the code.
    import transfer_matrices_p as tm_c


to_be_implemented = [
    'SPACE_CHARGE_COMP', 'FIELD_MAP_PATH', 'SET_SYNC_PHASE', 'STEERER',
    'ADJUST', 'ADJUST_STEERER',
    'DIAG_DSIZE3', 'DIAG_ENERGY', 'DIAG_TWISS', 'DIAG_POSITION', 'DIAG_DPHASE',
    'DIAG_DENERGY',
    'ERROR_CAV_NCPL_STAT',
    'END']
not_an_element = ['LATTICE', 'FREQ']

# Dict of data that can be imported from TW's "Data" table.
# More info in load_tw_results
d_tw_data_table = {
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

    l_elts = _create_structure(dat_filecontent)

    return dat_filecontent, l_elts


def _create_structure(dat_filecontent):
    """
    Create structure using the loaded dat file.

    Parameters
    ----------
    dat_filecontent : list of str
        List containing all the lines of dat_filepath.

    Return
    ------
    l_elts: list of Element
        List containing all the Element objects.
    """
    # Dictionnary linking element nature with correct sub-class
    subclasses_dispatcher = {
        'QUAD': elements.Quad,
        'DRIFT': elements.Drift,
        'FIELD_MAP': elements.FieldMap,
        'SOLENOID': elements.Solenoid,
        'LATTICE': elements.Lattice,
        'FREQ': elements.Freq,
    }

    # We look at each element in dat_filecontent, and according to the
    # value of the 1st column string we create the appropriate Element
    # subclass and store this instance in l_elts
    try:
        l_elts = [subclasses_dispatcher[elem[0]](elem)
                  for elem in dat_filecontent
                  if elem[0] not in to_be_implemented]
    except KeyError:
        print('Warning, an element from filepath was not recognized.')

    return l_elts


def give_name(l_elts):
    """Give a name (the same as TW) to every element."""
    civil_register = {
        'QUAD': 'QP',
        'DRIFT': 'DR',
        'FIELD_MAP': 'FM',
        'SOLENOID': 'SOL',
    }
    for key, value in civil_register.items():
        sub_list = [elt
                    for elt in l_elts
                    if elt.get('nature') == key
                    ]
        for i, elt in enumerate(sub_list, start=1):
            elt.elt_info['name'] = value + str(i)


# TODO is it necessary to load all the electric fields when _p?
def load_filemaps(dat_filepath, dat_filecontent, sections, freqs):
    """
    Load all the filemaps.

    Parameters
    ----------
    dat_filepath: string
        Filepath to the .dat file, as understood by TraceWin.
    dat_filecontent: list, opt
        List containing all the lines of dat_filepath.
    sections: list of lists of Element
        List of sections containing lattices containing Element objects.
    freqs:
        List of the RF frequencies in MHz in every section
    """
    assert len(sections) == len(freqs)

    field_map_folder = [line[1]
                        for line in dat_filecontent
                        if line[0] == 'FIELD_MAP_PATH']

    if len(field_map_folder) == 0:
        printc("tracewin_interface warning: ", opt_message="No field map" +
               " folder specified. Assuming that field maps are in the same" +
               " folder as the .dat")
        field_map_folder = os.path.dirname(dat_filepath)

    elif len(field_map_folder) > 1:
        raise IOError("Several field map folders are specified, which is ",
                      "currently not supported.")

    else:
        field_map_folder = os.path.dirname(dat_filepath) \
            + field_map_folder[0][1:]

    l_filepaths = []
    for i, section in enumerate(sections):
        f_mhz = freqs[i].f_rf_mhz
        n_cell = int(f_mhz / F_BUNCH_MHZ)   # FIXME
        for lattice in section:
            for elt in lattice:
                if elt.get('nature') == 'FIELD_MAP':
                    elt.field_map_file_name = field_map_folder + '/' \
                        + elt.field_map_file_name  # TODO with join
                    a_f = elt.acc_field
                    a_f.e_spat, a_f.n_z = load_field_map_file(elt)
                    a_f.init_freq_ncell(f_mhz, n_cell)

                    # For Cython, we need one filepath per section
                    if FLAG_CYTHON and len(l_filepaths) == i:
                        l_filepaths.append(elt.field_map_file_name)
    # Init arrays
    if FLAG_CYTHON:
        tm_c.init_arrays(l_filepaths)


def save_new_dat(fixed_linac, filepath_old):
    """Save a new dat with the new linac settings."""
    print('saving new dat\n\n')
    _update_dat_with_fixed_cavities(fixed_linac.files['dat_filecontent'],
                                    fixed_linac.elements['list'])

    filepath_new = filepath_old[:-4] + '_fixed.dat'
    with open(filepath_new, 'w') as file:
        for line in fixed_linac.files['dat_filecontent']:
            file.write(' '.join(line) + '\n')


def _update_dat_with_fixed_cavities(dat_filecontent, l_elts):
    """Create a new dat containing the new linac settings."""
    idx_elt = 0

    d_phi = {
        True: lambda elt: [str(np.rad2deg(elt.acc_field.phi_0['phi_0_abs'])), '1'],
        False: lambda elt: [str(np.rad2deg(elt.acc_field.phi_0['phi_0_rel'])), '0']
    }

    for line in dat_filecontent:
        if line[0] in to_be_implemented or line[0] in not_an_element:
            continue

        if line[0] == 'FIELD_MAP':
            elt = l_elts[idx_elt]
            line[3] = d_phi[FLAG_PHI_ABS](elt)[0]
            line[6] = str(elt.acc_field.k_e)
            line[10] = d_phi[FLAG_PHI_ABS](elt)[1]

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
        Name of the desired property. Must be in d_property.

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


def load_transfer_matrices(filepath_list):
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


def output_data_in_tw_fashion(linac):
    """Mimick TW's Data tab."""
    larousse = {
        '#': lambda i, elt, synch: i,
        'Name': lambda i, elt, synch: elt.get('name'),
        'Type': lambda i, elt, synch: elt.get('nature'),
        'Length (mm)': lambda i, elt, synch: elt.length_m * 1e3,
        'Grad/Field/Amp': lambda i, elt, synch:
        elt.grad
        if(elt.get('nature') == 'QUAD')
        else np.NaN,
        'EoT (MV/m)': lambda i, elt, synch: None,
        'EoTLc (MV)': lambda i, elt, synch:
        elt.acc_field.cav_params['v_cav_mv']
        if(elt.get('nature') == 'FIELD_MAP')
        else np.NaN,
        'Input_Phase (deg)': lambda i, elt, synch:
        np.rad2deg(elt.acc_field.phi_0['phi_0_rel'])
        if(elt.get('nature') == 'FIELD_MAP')
        else np.NaN,
        'Sync_Phase (deg)': lambda i, elt, synch:
        elt.acc_field.cav_params['phi_s_deg']
        if(elt.get('nature') == 'FIELD_MAP')
        else np.NaN,
        'Energy (MeV)': lambda i, elt, synch:
        synch.energy['kin_array_mev'][elt.idx['s_out']],
        'Beta Synch.': lambda i, elt, synch:
        synch.energy['beta_array'][elt.idx['s_out']],
        'Full length (mm)': lambda i, elt, synch:
        synch.z['abs_array'][elt.idx['s_out']] * 1e3,
        'Abs. phase (deg)': lambda i, elt, synch:
        np.rad2deg(synch.phi['abs_array'][elt.idx['s_out']]),
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
                    row.append(value(i, elt, linac.synch))
                data.append(row)
                i += 1

    data = pd.DataFrame(data, columns=larousse.keys())
    return data
