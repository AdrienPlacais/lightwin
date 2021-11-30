#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021

@author: placais
"""
import numpy as np
import cmath
from scipy.interpolate import interp1d
from collections import namedtuple
import helper
import transfer_matrices
from electric_field import rf_field
from constants import m_MeV, c


solver_param = namedtuple('solver_param', 'method n_steps d_z')


# =============================================================================
# Element class
# =============================================================================
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
        self.name = elem[0]
        self.length_m = 1e-3 * float(elem[1])

        # Absolute pos, gamma and energy of in and out.
        # In accelerating elements, the size of these arrays will be > 2.
        self.pos_m = np.full((2), np.NaN)
        self.gamma_array = np.full((2), np.NaN)
        self.energy_array_mev = np.full((2), np.NaN)
        self.transfer_matrix = np.full((1, 2, 2), np.NaN)
        # Dummy rf field
        self.acc_field = rf_field(352.2)    # FIXME frequency import


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
        self._load_optional_parameters(elem)

    def _load_optional_parameters(self, elem):
        """Try to load optional parameters."""
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
        self.gamma_array[-1] = self.gamma_array[0]
        self.energy_array_mev[-1] = self.energy_array_mev[0]


class Quad(_Element):
    """Sub-class of Element, with parameters specific to QUADs."""

    def __init__(self, elem):
        n_attributes = len(elem) - 1
        assert n_attributes in range(3, 10)

        super().__init__(elem)
        self.magnetic_field_gradient = float(elem[2])  # G
        self.aperture_mm = float(elem[3])              # R
        self._load_optional_parameters(elem)

    def _load_optional_parameters(self, elem):
        """Load the optional parameters if they are given."""
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
        self.gamma_array[-1] = self.gamma_array[0]
        self.energy_array_mev[-1] = self.energy_array_mev[0]


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
        self.gamma_array[-1] = self.gamma_array[0]
        self.energy_array_mev[-1] = self.energy_array_mev[0]


class FieldMap(_Element):
    """Sub-class of Element, with parameters specific to FIELD_MAPs."""

    def __init__(self, elem):
        n_attributes = len(elem) - 1
        assert n_attributes in [9, 10]

        super().__init__(elem)
        self.geometry = int(elem[1])
        self.length_m = 1e-3 * float(elem[2])
        self.theta_i_deg = float(elem[3])
        self.aperture_mm = float(elem[4])               # R
        self.magnetic_field_factor = float(elem[5])     # k_b
        self.electric_field_factor = float(elem[6])     # k_e
        self.space_charge_comp_factor = float(elem[7])  # K_i
        self.aperture_flag = int(elem[8])               # K_a
        # FIXME according to doc, may also be float
        self.field_map_file_name = str(elem[9])         # FileName

        try:
            self.relative_phase_flag = int(elem[10])    # P
        except IndexError:
            pass

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
        # Check nature and geometry of the field map, and select proper file
        # extension and import function
        extension, import_function = self.check_geom()
        self.field_map_file_name = self.field_map_file_name + extension

        # Load the field map
        n_z, zmax, norm, f_z = import_function(self.field_map_file_name)
        assert abs(zmax - self.length_m) < 1e-6
        assert abs(norm - 1.) < 1e-6, 'Warning, imported electric field ' \
            + 'different from 1. Conflict with electric_field_factor?'

        # Interpolation
        z_cavity_array = np.linspace(0., zmax, n_z + 1)
        f_z_scaled = self.electric_field_factor * f_z * norm
        ez_spat = interp1d(z_cavity_array, f_z_scaled, bounds_error=False,
                           kind='linear', fill_value=0.,
                           assume_sorted=True)

        self.acc_field = rf_field(352.2, np.deg2rad(self.theta_i_deg), 2)
        self.acc_field.e_spat = ez_spat

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
        # First, we check the nature of the given file
        assert self.geometry >= 0, \
            "Second order off-axis development not implemented."

        field_nature = int(np.log10(self.geometry))
        field_geometry = int(str(self.geometry)[0])

        assert field_nature == 2, "Only RF electric fields implemented."
        assert field_geometry == 1, "Only 1D field implemented."
        assert self.aperture_flag <= 0, \
            "Warning! Space charge compensation maps not implemented."

        extension = ".edz"
        import_function = helper.load_electric_field_1d
        return extension, import_function

    def compute_transfer_matrix(self):
        """Compute longitudinal matrix."""
        entry = self.pos_m[0]

        n_steps = 100 * self.acc_field.n_cell
        d_z = self.length_m / n_steps
        self.transfer_matrix_solver = solver_param('RK', n_steps, d_z)

        f_e = transfer_matrices.z_field_map_electric_field(
            self, self.acc_field, self.transfer_matrix_solver)

        self.pos_m += entry
        self.gamma_array = helper.mev_to_gamma(self.energy_array_mev, m_MeV)
        # Remove first slice of transfer matrix (indentity matrix)
        self.transfer_matrix = self.transfer_matrix[1:, :, :]
        self._compute_synch_phase_and_acc_pot(f_e)

    def _compute_synch_phase_and_acc_pot(self, f_e):
        """Compute the sychronous phase and accelerating potential."""
        phi_s = cmath.phase(f_e)
        self.phi_s_deg = np.rad2deg(phi_s)
        self.v_cav_mv = np.abs(
            (self.energy_array_mev[0] - self.energy_array_mev[-1])
            / np.cos(phi_s))


class CavSin(_Element):
    """Sub-class of Element, with parameters specific to CAVSINs."""

    def __init__(self, elem):
        n_attributes = len(elem) - 1
        assert n_attributes == 6

        super().__init__(elem)
        self.cell_number = int(elem[1])                 # N
        self.eff_gap_voltage = float(elem[2])           # EoT
        self.sync_phase = float(elem[3])                # theta_s
        self.aperture_mm = float(elem[4])               # R
        self.transfer_matrix = np.full((2, 2), np.NaN)

        try:
            self.relative_phase_flag = int(elem[5])     # P
        except IndexError:
            pass

    def compute_transfer_matrix(self):
        """Compute longitudinal matrix."""
        print('Warning, MT of sin cav not implemented.')
        transfer_matrix = transfer_matrices.z_drift(self.length_m,
                                                    self.gamma_array[0])
        self.transfer_matrix = np.expand_dims(transfer_matrix, 0)
        self.gamma_array[-1] = self.gamma_array[0]
        self.energy_array_mev[-1] = self.energy_array_mev[0]


class NotAnElement():
    """Dummy."""

    def __init__(self, elem):
        self.length_m = 0.
        print('Warning, NotAnElement not yet implemented:', elem)


# =============================================================================
# Old
# =============================================================================
def add_sinus_cavity(accelerator, line, i, dat_filename, f_mhz):
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
    assert n_attributes == 6, \
        'Wrong number of arguments for CAVSIN element at position ' + str(i)

    accelerator.elements_resume[i] = str(i) + ' \t' + '\t'.join(line)
    accelerator.elements_nature[i] = 'CAVSIN'

    accelerator.L_mm[i] = float(line[1])
    accelerator.L_m[i] = accelerator.L_mm[i] * 1e-3
    accelerator.N[i] = int(line[2])
    accelerator.EoT[i] = float(line[3])
    accelerator.theta_s[i] = float(line[4])
    accelerator.R[i] = float(line[5])
    accelerator.P[i] = int(line[6])
