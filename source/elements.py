#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021

@author: placais
"""
import os
import numpy as np
import helper
import transfer_matrices


def add_drift(accelerator, line, i):
    """Add a linear drift to the Accelerator object."""
    n_attributes = len(line) - 1

    # First, check validity of input
    if((n_attributes != 2) and
       (n_attributes != 3) and
       (n_attributes != 5)):
        raise IOError(
            'Wrong number of arguments for DRIFT element at position '
            + str(i))

    accelerator.elements_resume[i] = str(i) + ' \t' + '\t'.join(line)
    accelerator.elements_nature[i] = 'DRIFT'
    accelerator.L_mm[i] = float(line[1])
    accelerator.L_m[i] = accelerator.L_mm[i] * 1e-3
    accelerator.R[i] = float(line[2])

    try:
        accelerator.R_y[i] = float(line[3])
        accelerator.R_x_shift[i] = float(line[4])
        accelerator.R_y_shift[i] = float(line[5])
    except IndexError:
        pass

    accelerator.z_transfer_func[i] = transfer_matrices.z_drift


def add_quad(accelerator, line, i):
    """Add a quadrupole to the Accelerator object."""
    n_attributes = len(line) - 1

    # First, check validity of input
    if((n_attributes < 3) or
       (n_attributes > 9)):
        raise IOError(
            'Wrong number of arguments for QUAD element at position '
            + str(i))

    accelerator.elements_resume[i] = str(i) + ' \t' + '\t'.join(line)
    accelerator.elements_nature[i] = 'QUAD'

    accelerator.L_mm[i] = float(line[1])
    accelerator.L_m[i] = accelerator.L_mm[i] * 1e-3
    accelerator.G[i] = float(line[2])
    accelerator.R[i] = float(line[3])

    try:
        accelerator.Theta[i] = float(line[4])
        accelerator.G3_over_u3[i] = float(line[5])
        accelerator.G4_over_u4[i] = float(line[6])
        accelerator.G5_over_u5[i] = float(line[7])
        accelerator.G6_over_u6[i] = float(line[8])
        accelerator.GFR[i] = float(line[9])
    except IndexError:
        pass

    accelerator.z_transfer_func[i] = transfer_matrices.z_drift


def add_solenoid(accelerator, line, i):
    """Add a solenoid to the Accelerator object."""
    n_attributes = len(line) - 1

    # First, check validity of input
    if(n_attributes != 3):
        raise IOError(
            'Wrong number of arguments for SOLENOID element at position '
            + str(i))

    accelerator.elements_resume[i] = str(i) + ' \t' + '\t'.join(line)
    accelerator.elements_nature[i] = 'SOLENOID'

    accelerator.L_mm[i] = float(line[1])
    accelerator.L_m[i] = accelerator.L_mm[i] * 1e-3
    accelerator.B[i] = float(line[2])
    accelerator.R[i] = float(line[3])

    accelerator.z_transfer_func[i] = transfer_matrices.z_drift


def add_field_map(accelerator, line, i, TraceWin_dat_filename, f_MHz):
    """
    Add a field map to the Accelerator object.

    Attributes
    ----------
    geom: int
        Field map type.
    L_mm: float
        Field map length (mm).
    theta_i: float
        RF input field phase (deg).
    R: float
        Aperture (mm).
    k_b: float
        Magnetic field intensity factor.
    k_e: float
        Electric field intensity factor.
    K_i: float
        Space charge compensation factor.
    K_a: int
        Aperture flag.
    FileName: string
        File name without extension (abs. or relative path).
    P: int, opt
        0: theta_i is relative phase.
        1: theta_i is absolute phase.
    """
    n_attributes = len(line) - 1

    # FIXME
    accelerator.TraceWin_dat_filename = TraceWin_dat_filename

    if((n_attributes < 9) or (n_attributes > 10)):
        raise IOError(
            'Wrong number of arguments for FIELD_MAP element at position '
            + str(i))

    accelerator.elements_resume[i] = str(i) + ' \t' + '\t'.join(line)
    accelerator.elements_nature[i] = 'FIELD_MAP'

    accelerator.geom[i] = int(line[1])
    accelerator.L_mm[i] = float(line[2])
    accelerator.L_m[i] = accelerator.L_mm[i] * 1e-3
    accelerator.theta_i[i] = float(line[3])
    accelerator.R[i] = float(line[4])
    accelerator.k_b[i] = float(line[5])
    accelerator.k_e[i] = float(line[6])
    accelerator.K_i[i] = float(line[7])
    accelerator.K_a[i] = int(line[8])     # FIXME according to doc, may also be float
    accelerator.FileName[i] = str(line[9])

    try:
        accelerator.P[i] = int(line[10])
    except IndexError:
        pass

    accelerator.z_transfer_func[i] = transfer_matrices.dummy

    nz, zmax, Norm, Fz_array = select_and_load_field_map_file(
        accelerator.TraceWin_dat_filename,
        accelerator.geom[i],
        accelerator.K_a[i],
        accelerator.FileName[i])

    accelerator.nz[i] = nz
    accelerator.zmax[i] = zmax
    accelerator.Norm[i] = Norm
    accelerator.Fz_array[i] = Fz_array

    return


def select_and_load_field_map_file(TraceWin_dat_filename, geom, K_a, FileName):
    """
    Select the field map file and call the proper loading function.

    Warning, FileName is directly extracted from the .dat file used by
    TraceWin. Thus, the relative filepath may be misunderstood by this
    script.
    Also check that the extension of the file is .edz, or manually change
    this function.
    Finally, only 1D electric field map are implemented.
    """
    # Flag to show or not the loading file info:
    debug_verbose = False
    # Check nature and geometry of the field map, and select proper file
    # extension and import function
    extension, import_function = check_geom(geom, K_a)

    # Now we select the proper filepath
    # Warning, the "/"s may have to be changed to "\"s on Windows.
    absolute_path = TraceWin_dat_filename.split('/')[:-1]

    # Hypothesis on the structure of the TraceWin project
    absolute_path = "/".join(absolute_path) + "/field_maps_1D/"
    absolute_path = absolute_path + FileName + extension

    if(os.path.exists(FileName)):
        path = FileName
        if(debug_verbose):
            #  accelerator.show_element_info()  # FIXME
            print("Loading field map with relative filepath...")

    elif(os.path.exists(absolute_path)):
        path = absolute_path
        if(debug_verbose):
            #  accelerator.show_element_info()  # FIXME
            print("Loading field map with absolute filepath...")

    else:
        msg = "Field Map file not found.\n"
        msg = msg + "Please check select_and_load_field_map_file function."
        raise IOError(msg)
    # TODO check doc, this part may be simpler

    # Load the field map
    nz, zmax, Norm, Fz = import_function(path)
    return nz, zmax, Norm, Fz


def check_geom(geom, K_a):
    """
    Verify that the file can be correctly imported.

    Returns
    -------
    extension: str
        Extension of the file to load. See TraceWin documentation.
    import_function: fun
        Function adapted to the nature and geometry of the field.
    """
    # TODO: autodetect extensions
    # TODO: implement import of magnetic fields
    # TODO: implement 2D and 3D maps
    # First, we check the nature of the given file
    if(geom < 0):
        raise IOError("Second order off-axis development not implemented.")

    field_nature = int(np.log10(geom))
    field_geometry = int(str(geom)[0])

    if(field_nature != 2):
        raise IOError("Only RF electric fields implemented.")

    if(field_geometry != 1):
        raise IOError("Only 1D field implemented.")

    if(K_a > 0):
        print("Warning! Space charge compensation maps not implemented.")

    extension = ".edz"
    import_function = helper.load_electric_field_1D
    return extension, import_function
