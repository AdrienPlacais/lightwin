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


class Element():
    """Super class holding methods and properties common to all elements."""

    def __init__(self, line, i):
        """
        Initialize what is common to all ELEMENTs.

        Attributes
        ----------
        n_attributes: integer
            The number of attributes in the .dat file.
        element_pos: integer
            Position of the element. Should match TraceWin's.
        resume: string
            Resume of the element properties. Should match the corresponding
            line in the .dat file, at the exception of the first character
            that is the elemet position.
        """
        self.n_attributes = len(line) - 1
        self.element_pos = i
        self.resume = [str(self.element_pos)] + line
        self.resume = ' '.join(self.resume)

    def show_element_info(self):
        """
        Output information on the element.

        Should match the corresponding line in the .dat file.
        """
        print(self.resume)

    def transfer_matrix_z(self, gamma):
        """Return dummy transfer matrix."""
        msg = "Transfer matrix not yet implemented: "
        helper.printc(msg, color="info", opt_message=self.resume)
        R_zz = np.full((2, 2), np.NaN)
        return R_zz


class Drift(Element):
    """Linear drift."""

    def __init__(self, line, i):
        """Add a drift to structure."""
        super().__init__(line, i)

        # First, check validity of input
        if((self.n_attributes != 2) and
           (self.n_attributes != 3) and
           (self.n_attributes != 5)):
            raise IOError(
                'Wrong number of arguments for DRIFT element at position '
                + str(self.element_pos))

        self.L_mm = float(line[1])
        self.L_m = self.L_mm * 1e-3
        self.R = float(line[2])

        try:
            self.R_y = float(line[3])
            self.R_x_shift = float(line[4])
            self.R_y_shift = float(line[5])
        except IndexError:
            pass

    def transfer_matrix_z(self, gamma):
        """
        Compute the longitudinal transfer matrix of the drift.

        Parameters
        ----------
        gamma: float
            Lorentz factor of the particle.

        Returns
        -------
        R_zz: np.array
            Transfer longitudinal sub-matrix.
        """
        R_zz = transfer_matrices.z_drift(self.L_m, gamma)
        return R_zz


class Quad(Element):
    """Quadrupole."""

    def __init__(self, line, i):
        """Add a quadrupole to structure."""
        super().__init__(line, i)

        # First, check validity of input
        if((self.n_attributes < 3) or
           (self.n_attributes > 9)):
            raise IOError(
                'Wrong number of arguments for QUAD element at position '
                + str(self.element_pos))

        self.L_mm = float(line[1])
        self.L_m = self.L_mm * 1e-3
        self.G = float(line[2])
        self.R = float(line[3])

        try:
            self.Theta = float(line[4])
            self.G3_over_u3 = float(line[5])
            self.G4_over_u4 = float(line[6])
            self.G5_over_u5 = float(line[7])
            self.G6_over_u6 = float(line[8])
            self.GFR = float(line[9])
        except IndexError:
            pass

    def transfer_matrix_z(self, gamma):
        """
        Compute the longitudinal transfer matrix of the quadrupole.

        Parameters
        ----------
        gamma: float
            Lorentz factor of the particle.

        Returns
        -------
        R_zz: np.array
            Transfer longitudinal sub-matrix.
        """
        R_zz = transfer_matrices.z_drift(self.L_m, gamma)
        return R_zz


class Solenoid(Element):
    """Solenoid."""

    def __init__(self, line, i):
        """Add a solenoid to structure."""
        super().__init__(line, i)

        # First, check validity of input
        if(self.n_attributes != 3):
            raise IOError(
                'Wrong number of arguments for SOLENOID element at position '
                + str(self.element_pos))

        self.L_mm = float(line[1])
        self.L_m = self.L_mm * 1e-3
        self.B = float(line[2])
        self.R = float(line[3])

    def transfer_matrix_z(self, gamma):
        """
        Compute the longitudinal transfer matrix of the solenoid.

        Parameters
        ----------
        gamma: float
            Lorentz factor of the particle.

        Returns
        -------
        R_zz: np.array
            Transfer longitudinal sub-matrix.
        """
        R_zz = transfer_matrices.z_drift(self.L_m, gamma)
        return R_zz


class FieldMap(Element):
    """Field map."""

    def __init__(self, line, i, TraceWin_dat_filename):
        """Add a field map to the structure."""
        super().__init__(line, i)

        self.TraceWin_dat_filename = TraceWin_dat_filename

        if((self.n_attributes < 9) or (self.n_attributes > 10)):
            raise IOError(
                'Wrong number of arguments for FIELD_MAP element at position '
                + str(self.element_pos))

        self.geom = int(line[1])
        self.L = float(line[2])
        self.theta_i = float(line[3])
        self.R = float(line[4])
        self.k_b = float(line[5])
        self.k_e = float(line[6])
        self.K_i = float(line[7])
        self.K_a = int(line[8])     # FIXME according to doc, may also be float
        self.FileName = str(line[9])

        try:
            self.P = int(line[10])
        except IndexError:
            pass

        self.select_and_load_field_map_file()

    def select_and_load_field_map_file(self):
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
        extension, import_function = self.check_geom()

        # Now we select the proper filepath
        # Warning, the "/"s may have to be changed to "\"s on Windows.
        absolute_path = self.TraceWin_dat_filename.split('/')[:-1]

        # Hypothesis on the structure of the TraceWin project
        absolute_path = "/".join(absolute_path) + "/field_maps_1D/"
        absolute_path = absolute_path + self.FileName + extension

        if(os.path.exists(self.FileName)):
            path = self.FileName
            if(debug_verbose):
                self.show_element_info()
                print("Loading field map with relative filepath...")

        elif(os.path.exists(absolute_path)):
            path = absolute_path
            if(debug_verbose):
                self.show_element_info()
                print("Loading field map with absolute filepath...")

        else:
            msg = "Field Map file not found.\n"
            msg = msg + "Please check FieldMap.load_field_map_file function."
            raise IOError(msg)
        # TODO check doc, this part may be simpler

        # Finally, load the field map
        self.map = import_function(path)
        if(debug_verbose):
            print("Field map loaded.")

    def check_geom(self):
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
        if(self.geom < 0):
            raise IOError("Second order off-axis development not implemented.")

        self.field_nature = int(np.log10(self.geom))
        self.field_geometry = int(str(self.geom)[0])

        if(self.field_nature != 2):
            raise IOError("Only RF electric fields implemented.")

        if(self.field_geometry != 1):
            raise IOError("Only 1D field implemented.")

        if(self.K_a > 0):
            print("Warning! Space charge compensation maps not implemented.")

        extension = ".edz"
        import_function = helper.load_electric_field_1D
        return extension, import_function
