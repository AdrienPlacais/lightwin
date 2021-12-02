#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021

@author: placais
"""
import cmath
from collections import namedtuple
import numpy as np
import helper
import transfer_matrices
import transport
from electric_field import RfField
from constants import m_MeV

# TODO separate functions for RK / leapfrog?
SolverParam = namedtuple('SolverParam', 'method n_steps d_z')


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
        self._init_solver()

    def _init_solver(self):
        """Solver properties."""
        self.acc_field = RfField(352.2)    # FIXME frequency import

        # By default, 1 step for non-accelerating elements
        method = 'RK'
        n_steps = 1
        d_z = self.length_m / n_steps
        self.solver_transf_mat = SolverParam(method, n_steps, d_z)

        # By default, most elements are z drifts
        self.dict_transf_mat = {
            'RK': transfer_matrices.z_drift,
            'leapfrog': transfer_matrices.z_drift,
            'transport': transport.transport_beam,
            }


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
        transfer_matrix = self.dict_transf_mat[
            self.solver_transf_mat.method](self)
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
        transfer_matrix = self.dict_transf_mat[
            self.solver_transf_mat.method](self)
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
        transfer_matrix = self.dict_transf_mat[
            self.solver_transf_mat.method](self)
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

        # As FieldMap is an accelerating element, some solver parameters should
        # be changed.
        self._update_solver()

    def _update_solver(self):
        """Replace dummy solving parameters by real ones."""
        # Replace dummy accelerating field by a true one
        freq = self.acc_field.frequency_mhz
        self.acc_field = RfField(freq, np.deg2rad(self.theta_i_deg), 2)

        # By default, 1 step for most elements
        n_steps = 100 * self.acc_field.n_cell
        d_z = self.length_m / n_steps
        meth = self.solver_transf_mat.method
        self.solver_transf_mat = SolverParam(meth, n_steps, d_z)

        self.dict_transf_mat = {
            'RK': transfer_matrices.z_field_map_electric_field,
            'leapfrog': transfer_matrices.z_field_map_electric_field,
            'transport': transport.transport_beam,
            }

    def compute_transfer_matrix(self):
        """Compute longitudinal matrix."""
        # Save this as pos_m will be replaced in z_field_map_electric_field.
        entry = self.pos_m[0]

        f_e = self.dict_transf_mat[self.solver_transf_mat.method](self)

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
