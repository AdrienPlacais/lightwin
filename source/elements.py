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
from constants import m_MeV


# =============================================================================
# Element class
# =============================================================================
# TODO implement a way to output each instance attributes, such as old
# elements_resume.
class _Element():
    """Generic element. _ ensures that it is not called from another file."""

    def __init__(self, elem):
        """
        Init parameters common to all elements.

        Some attributes, such as length_m for FIELD_MAP, are stored differently
        and will be changed later.

        Parameters
        ----------
        elem: list of string
            A valid line of the .dat file.
        """
        # FIXME too many instance attributes
        self.name = elem[0]
        self.length_mm = float(elem[1])
        self.length_m = 1e-3 * float(elem[1])

        self.entrance_pos_m = np.NaN
        self.exit_pos_m = np.NaN

        # TODO bidul_in and bidul_out in the same array.
        # bidul_in = bidul[0] and bidul_out=bidul[-1].
        # Other values in between if needed.
        self.gamma_in = np.NaN      # TODO remove
        self.gamma_out = np.NaN     # TODO remove
        self.gamma_array = np.full((2), np.NaN)

        self.energy_array_mev = np.full((2), np.NaN)

        self.frequency_mhz = 352.2  # FIXME import of frequency

        self.transfer_matrix = np.full((2, 2), np.NaN)

    def compute_acceleration(self):
        """Compute acceleration in current element."""
        print('Warning, compute_acceleration not yet implemented.')
        return 1.


# =============================================================================
# More specific classes
# =============================================================================
class Drift(_Element):
    """Sub-class of Element, with parameters specific to DRIFTs."""

    def __init__(self, elem):
        n_attributes = len(elem) - 1
        assert n_attributes in [2, 3, 5]

        super().__init__(elem)
        self.aperture_mm = float(elem[2])   # R
        self._try_to_load_optional_parameters(elem)

    def _try_to_load_optional_parameters(self, elem):
        """Try to load optional parameters."""
        self.aperture_y_mm = np.NaN
        self.horizontal_aperture_shift_mm = np.NaN
        self.vertical_aperture_shift_mm = np.NaN

        try:
            self.aperture_y_mm = float(elem[3])                 # R_y
            self.horizontal_aperture_shift_mm = float(elem[4])  # R_x_shift
            self.vertical_aperture_shift_mm = float(elem[5])    # R_y_shift
        except IndexError:
            pass

    def compute_transfer_matrix(self):
        """Compute longitudinal matrix."""
        transfer_matrix = transfer_matrices.z_drift(self.length_m,
                                                    self.gamma_array[0])
        self.transfer_matrix = np.expand_dims(transfer_matrix, 0)


class Quad(_Element):
    """Sub-class of Element, with parameters specific to QUADs."""

    def __init__(self, elem):
        # FIXME too many instance attributes
        n_attributes = len(elem) - 1
        assert n_attributes in range(3, 10)

        super().__init__(elem)
        self.magnetic_field_gradient = float(elem[2])  # G
        self.aperture_mm = float(elem[3])              # R
        self._try_to_load_optional_parameters(elem)

    def _try_to_load_optional_parameters(self, elem):
        """Load the optional parameters if they are given."""
        # Following assignations may have to go in the __init__.
        self.skew_angle_deg = np.NaN        # Theta
        self.sextupole_gradient = np.NaN    # G3_over_u3
        self.octupole_gradient = np.NaN     # G4_over_u4
        self.decapole_gradient = np.NaN     # G5_over_u5
        self.dodecapole_gradient = np.NaN   # G6_over_u6j
        self.good_field_radius_mm = np.NaN  # GFR

        try:
            self.skew_angle_deg = float(elem[4])
            self.sextupole_gradient = float(elem[5])
            self.octupole_gradient = float(elem[6])
            self.decapole_gradient = float(elem[7])
            self.dodecapole_gradient = float(elem[8])
            self.good_field_radius_mm = float(elem[9])
        except IndexError:
            pass

    def compute_transfer_matrix(self):
        """Compute longitudinal matrix."""
        transfer_matrix = transfer_matrices.z_drift(self.length_m,
                                                    self.gamma_array[0])
        self.transfer_matrix = np.expand_dims(transfer_matrix, 0)


class Solenoid(_Element):
    """Sub-class of Element, with parameters specific to SOLENOIDs."""

    def __init__(self, elem):
        n_attributes = len(elem) - 1
        assert n_attributes == 3

        super().__init__(elem)
        self.magnetic_field = float(elem[2])  # B
        self.aperture_mm = float(elem[3])     # R

    def compute_transfer_matrix(self):
        """Compute longitudinal matrix."""
        transfer_matrix = transfer_matrices.z_drift(self.length_m,
                                                    self.gamma_array[0])
        self.transfer_matrix = np.expand_dims(transfer_matrix, 0)


class FieldMap(_Element):
    """Sub-class of Element, with parameters specific to FIELD_MAPs."""

    def __init__(self, elem):
        n_attributes = len(elem) - 1
        assert n_attributes in [9, 10]

        super().__init__(elem)
        self.geometry = int(elem[1])
        self.length_mm = float(elem[2])
        self.length_m = 1e-3 * float(elem[2])
        self.theta_i_deg = float(elem[3])
        self.aperture_mm = float(elem[4])               # R
        self.magnetic_field_factor = float(elem[5])     # k_b
        self.electric_field_factor = float(elem[6])     # k_e
        self.space_charge_comp_factor = float(elem[7])  # K_i
        self.aperture_flag = int(elem[8])               # K_a
        # FIXME according to doc, may also be float
        self.field_map_file_name = str(elem[9])         # FileName

        self.transfer_matrix = np.full((2, 2), np.NaN)

        try:
            self.relative_phase_flag = int(elem[10])    # P
        except IndexError:
            pass

        print('Warning, field map loading not fully implemented.')
        print('Loading default field map...')
        # FIXME
        self.field_map_file_name = '../data/Ez.edz'
        TraceWin_dat_filename = '~/TraceWin/work_field_map/work_field_map.dat'
        self.select_and_load_field_map_file(TraceWin_dat_filename)

    def select_and_load_field_map_file(self, TraceWin_dat_filename):
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

        # Warning, the "/"s may have to be changed to "\"s on Windows.
        absolute_path = TraceWin_dat_filename.split('/')[:-1]

        # Hypothesis on the structure of the TraceWin project
        absolute_path = "/".join(absolute_path) + "/field_maps_1D/"
        absolute_path = absolute_path + self.field_map_file_name + extension

        # TODO check filename with assert
        if os.path.exists(self.field_map_file_name):
            path = self.field_map_file_name
            if debug_verbose:
                print("Loading field map with relative filepath...")

        elif os.path.exists(absolute_path):
            path = absolute_path
            if debug_verbose:
                print("Loading field map with absolute filepath...")

        else:
            msg = "Field Map file not found.\n"
            msg = msg + "Please check select_and_load_field_map_file function."
            raise IOError(msg)
        # TODO check doc, this part may be simpler

        # Load the field map
        self.nz, zmax, self.norm, self.fz = import_function(path)

        assert abs(zmax - self.length_m) < 1e-6

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
        if(self.geometry < 0):
            raise IOError("Second order off-axis development not implemented.")

        field_nature = int(np.log10(self.geometry))
        field_geometry = int(str(self.geometry)[0])

        if(field_nature != 2):
            raise IOError("Only RF electric fields implemented.")

        if(field_geometry != 1):
            raise IOError("Only 1D field implemented.")

        if(self.aperture_flag > 0):
            print("Warning! Space charge compensation maps not implemented.")

        extension = ".edz"
        import_function = helper.load_electric_field_1D
        return extension, import_function

    def compute_transfer_matrix(self):
        """Compute longitudinal matrix."""
        # TODO Check this Ncell truc.
        R_zz_single, MT_and_energy_evolution, V_cav_MV, phi_s_deg = \
            transfer_matrices.z_field_map_electric_field(
                        self.energy_array_mev[0],
                        self.frequency_mhz,
                        self.fz,
                        self.electric_field_factor,
                        self.theta_i_deg,
                        2,
                        self.nz,
                        self.length_m)

        #  self.E_MeV[i+1] = MT_and_energy_evolution[-1, 0, 1]
        #  self.gamma[i+1] = 1. + self.E_MeV[i+1] / m_MeV
        #  self.V_cav_MV[i] = V_cav_MV
        #  self.phi_s_deg[i] = phi_s_deg
        self.absolute_position_m = MT_and_energy_evolution[:, 0, 0]   \
            + self.entrance_pos_m
        self.energy_array_mev = MT_and_energy_evolution[:, 0, 1]
        self.gamma_array = helper.mev_to_gamma(self.energy_array_mev, m_MeV)
        self.transfer_matrix = MT_and_energy_evolution[:, 1:, :]


class CavSin(_Element):
    """Sub-class of Element, with parameters specific to CAVSINs."""

    def __init__(self, elem):
        super().__init__(elem)
        print('Warning, SinCav not yet implemented.')


class NotAnElement():
    """Dummy."""

    def __init__(self, elem):
        self.length_m = 0.
        print('Warning, NotAnElement not yet implemented:', elem)


# =============================================================================
# Old
# =============================================================================
def add_sinus_cavity(accelerator, line, i, TraceWin_dat_filename, f_MHz):
    """
    Add a sinus cavity to the Accelerator object.

    Attributes
    ----------
    L_mm: float
        Field map length (mm).
    N: int
        Cell number.
    EoT: float
        Average accelerating field (V/m).
    theta_s: float
        Phase of the synchronous particle at the entrance (deg). Can be
        absolute or relative.
    R: float
        Aperture (mm).
    P: int
        0: theta_s is relative phase.
        1: theta_s is absolute phase.
    """
    n_attributes = len(line) - 1

    # First, check validity of input
    if(n_attributes != 6):
        raise IOError(
            'Wrong number of arguments for CAVSIN element at position '
            + str(i))

    accelerator.elements_resume[i] = str(i) + ' \t' + '\t'.join(line)
    accelerator.elements_nature[i] = 'CAVSIN'

    accelerator.L_mm[i] = float(line[1])
    accelerator.L_m[i] = accelerator.L_mm[i] * 1e-3
    accelerator.N[i] = int(line[2])
    accelerator.EoT[i] = float(line[3])
    accelerator.theta_s[i] = float(line[4])
    accelerator.R[i] = float(line[5])
    accelerator.P[i] = int(line[6])
