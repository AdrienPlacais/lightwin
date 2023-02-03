#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:52:37 2022.

@author: placais

TODO insert line skip at each section change in the output.dat
"""
import os.path
import subprocess
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
    'SPACE_CHARGE_COMP', 'SET_SYNC_PHASE', 'STEERER',
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
    'phi_0_rel': 7,
    'phi_s': 8,
    'w_kin': 9,
    'beta': 10,
    'z_abs': 11,
    'phi_abs_array': 12,
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
        'FIELD_MAP_PATH': elements.FieldMapPath,
    }

    # We look at each element in dat_filecontent, and according to the
    # value of the 1st column string we create the appropriate Element
    # subclass and store this instance in l_elts
    l_elts = [subclasses_dispatcher[elem[0]](elem)
              for elem in dat_filecontent
              if elem[0] not in to_be_implemented]

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
            elt.elt_info['elt_name'] = value + str(i)


# TODO is it necessary to load all the electric fields when _p?
def load_filemaps(files, sections, freqs):
    """
    Load all the filemaps.

    Parameters
    ----------
    files: dict
        Accelerator.files dict
    dat_filecontent: list, opt
        List containing all the lines of dat_filepath.
    sections: list of lists of Element
        List of sections containing lattices containing Element objects.
    freqs:
        List of the RF frequencies in MHz in every section
    """
    assert len(sections) == len(freqs)

    l_filepaths = []
    for i, section in enumerate(sections):
        f_mhz = freqs[i].f_rf_mhz
        n_cell = int(f_mhz / F_BUNCH_MHZ)   # FIXME
        for lattice in section:
            for elt in lattice:
                if elt.get('nature') == 'FIELD_MAP':
                    elt.field_map_file_name = os.path.join(
                        files['field_map_folder'], elt.field_map_file_name)
                    a_f = elt.acc_field
                    a_f.e_spat, a_f.n_z = load_field_map_file(elt)
                    a_f.init_freq_ncell(f_mhz, n_cell)

                    # For Cython, we need one filepath per section
                    if FLAG_CYTHON and len(l_filepaths) == i:
                        l_filepaths.append(elt.field_map_file_name)
    # Init arrays
    if FLAG_CYTHON:
        tm_c.init_arrays(l_filepaths)


def save_new_dat(fixed_linac, filepath, *args):
    """Save a new dat with the new linac settings."""
    printc("tracewin_interface.save_new_dat info: ",
           opt_message=f"new dat saved in {filepath}\n\n")

    _update_dat_with_fixed_cavities(fixed_linac.files['dat_filecontent'],
                                    fixed_linac.elts,
                                    fixed_linac.files['field_map_folder'])

    for i, arg in enumerate(args):
        arg.to_csv(filepath + str(i) + '.csv')

    with open(filepath, 'w') as file:
        for line in fixed_linac.files['dat_filecontent']:
            file.write(' '.join(line) + '\n')

    fixed_linac.files['dat_filepath'] = filepath


def _update_dat_with_fixed_cavities(dat_filecontent, l_elts, fm_folder):
    """Create a new dat containing the new linac settings."""
    idx_elt = 0

    d_phi = {
        True: lambda elt: str(elt.get('phi_0_abs', to_deg=True)),
        False: lambda elt: str(elt.get('phi_0_rel', to_deg=True)),
    }

    for line in dat_filecontent:
        if line[0] in to_be_implemented or line[0] in not_an_element:
            continue

        if line[0] == 'FIELD_MAP':
            elt = l_elts[idx_elt]
            line[3] = d_phi[FLAG_PHI_ABS](elt)
            line[6] = str(elt.get('k_e'))
            # '1' if True, '0' if False
            line[10] = str(int(FLAG_PHI_ABS))

        elif line[0] == 'FIELD_MAP_PATH':
            line[1] = fm_folder
            continue

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
        __s = "Filepath to results is incorrect. Provide another one."
        printc("tracewin_interface.load_tw_results warning: ", opt_message=__s)
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
        '#': lambda lin, elt: elt.get('elt_idx', to_numpy=False),
        'Name': lambda lin, elt: elt.get('elt_name', to_numpy=False),
        'Type': lambda lin, elt: elt.get('nature', to_numpy=False),
        'Length (mm)': lambda lin, elt: elt.length_m * 1e3,
        'Grad/Field/Amp': lambda lin, elt:
            elt.grad if(elt.get('nature', to_numpy=False) == 'QUAD')
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


def run_tw(linac, ini_path, tw_path="/usr/local/bin/./TraceWin",
           **kwargs):
    """Prepare arguments and run TraceWin."""
    l_keys = ["dat_file", "path_cel"]
    l_values = [os.path.abspath(linac.get("dat_filepath")),
                os.path.abspath(linac.get("out_tw"))]

    for key, val in zip(l_keys, l_values):
        if key not in kwargs.keys():
            __s = f"The key {key} was not found in kwargs. Used the default"
            __s += f" value {val} instead."
            printc("tracewin_interface.run_tw info: ", opt_message=__s)

    cmd = _tw_cmd(tw_path, ini_path, **kwargs)
    printc("tracewin_interface.run_tw info: ",
           opt_message=f"Running TW with command {cmd}.")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    process.wait()
    for line in process.stdout:
        print(line)
    printc("tracewin_interface.run_tw info: ", opt_message="TW finished!")


def _tw_cmd(tw_path, ini_path, **kwargs):
    """Make the command line to launch TraceWin."""
    cmd = tw_path + " " + ini_path + " hide"
    for key, value in kwargs.items():
        cmd += " " + key + "=" + str(value)
    return cmd


def get_multipart_tw_results(linac, filename='partran1.out'):
    """Get the results."""
    f_p = os.path.join(linac.get('out_tw'), filename)
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

    return d_out


def check_results_after_tw(linac):
    """Check that some criterions are matched."""
    # We check Power Loss
    f_p = os.path.join(linac.get('out_tw'), 'partran1.out')
    out = np.loadtxt(f_p, skiprows=10)

    pow_lost = out[:, 35]
    if pow_lost[-1] > 1e-10:
        print("Loss of power!")

    # Normalized RMS emittances [mm.mrad, mm.mrad, pi.deg.MeV]
    eps_rms = out[:, 15:18]
    var_eps_rms = np.abs((eps_rms[0] - eps_rms) / eps_rms[0])
    if np.any(np.where(var_eps_rms > 2e-2)):
        print("The RMS emittance is too damn high!")

    # Emittance at 99%
    eps_99 = out[:, 25:28]
    ref_eps_99 = []
    maxs = []
    for i in range(3):
        maxs.append(np.max(eps_99[:, i]) / np.max(ref_eps_99[:, i]))
