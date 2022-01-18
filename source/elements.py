#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021

@author: placais
"""
import cmath
from collections import namedtuple
import numpy as np
import transfer_matrices
import transport
from electric_field import RfField

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

        # By default, an element is non accelerating and has a dummy
        # accelerating field.
        self.accelerating = False
        self.acc_field = RfField(0.)

        self.pos_m = {
            'abs': None,
            'rel': None,
            }
        self.idx = {
            'in': None,
            'out': None,
            }

        self.transfer_matrix = None

        self.dict_transf_mat = None
        self.solver_transf_mat = None

    def init_solver_settings(self, method):
        """Initialize solver properties."""
        if self.accelerating:
            if self.name == 'FIELD_MAP':
                n_steps = 100 * self.acc_field.n_cell
                self.dict_transf_mat = {
                    'RK': transfer_matrices.z_field_map_electric_field,
                    'leapfrog': transfer_matrices.z_field_map_electric_field,
                    'transport': transport.transport_beam,
                    }
            else:
                raise IOError('Accelerating element not implemented.')

        else:
            # By default, 1 step for non-accelerating elements
            n_steps = 1
            self.dict_transf_mat = {
                'RK': transfer_matrices.z_drift,
                'leapfrog': transfer_matrices.z_drift,
                'transport': transport.transport_beam,
             }

        self.pos_m['rel'] = np.linspace(0., self.length_m, n_steps + 1)
        self.transfer_matrix = np.full((n_steps, 2, 2), np.NaN)

        d_z = self.length_m / n_steps
        self.solver_transf_mat = SolverParam(method, n_steps, d_z)

    def compute_transfer_matrix(self, synch):
        """
        Compute longitudinal matrix.

        This default function is used for non accelerating elements.
        """
        assert ~self.accelerating
        self.transfer_matrix = self.dict_transf_mat[
            self.solver_transf_mat.method](self, gamma=np.NaN, synch=synch)


# =============================================================================
# More specific classes
# =============================================================================
class Drift(_Element):
    """Sub-class of Element, with parameters specific to DRIFTs."""

    def __init__(self, elem):
        n_attributes = len(elem) - 1
        assert n_attributes in [2, 3, 5]
        super().__init__(elem)


class Quad(_Element):
    """Sub-class of Element, with parameters specific to QUADs."""

    def __init__(self, elem):
        n_attributes = len(elem) - 1
        assert n_attributes in range(3, 10)
        super().__init__(elem)


class Solenoid(_Element):
    """Sub-class of Element, with parameters specific to SOLENOIDs."""

    def __init__(self, elem):
        n_attributes = len(elem) - 1
        assert n_attributes == 3
        super().__init__(elem)


class FieldMap(_Element):
    """Sub-class of Element, with parameters specific to FIELD_MAPs."""

    def __init__(self, elem):
        n_attributes = len(elem) - 1
        assert n_attributes in [9, 10]

        super().__init__(elem)
        self.accelerating = True
        self.geometry = int(elem[1])
        self.length_m = 1e-3 * float(elem[2])
        self.theta_i_rad = np.deg2rad(float(elem[3]))
        self.electric_field_factor = float(elem[6])     # k_e
        self.aperture_flag = int(elem[8])               # K_a
        # FIXME according to doc, may also be float
        self.field_map_file_name = str(elem[9])         # FileName

        try:
            self.relative_phase_flag = int(elem[10])    # P
        except IndexError:
            pass

        # Replace dummy accelerating field by a true one
        self.acc_field = RfField(352.2,     # FIXME frequency import
                                 self.theta_i_rad, 2)

        self.f_e = None
        self.phi_s_deg = None
        self.v_cav_mv = None

    def compute_transfer_matrix(self, synch):
        """Compute longitudinal matrix."""
        # Init f_e
        self.f_e = 0.

        # Compute transfer matrix
        self.dict_transf_mat[self.solver_transf_mat.method](self, synch)

        # Remove first slice of transfer matrix (indentity matrix)
        self.transfer_matrix = self.transfer_matrix[1:, :, :]
        self._compute_synch_phase_and_acc_pot(synch)

    def _compute_synch_phase_and_acc_pot(self, synch):
        """Compute the sychronous phase and accelerating potential."""
        phi_s = cmath.phase(self.f_e)
        self.phi_s_deg = np.rad2deg(phi_s)
        energy_now = synch.energy['kin_mev']
        energy_before = synch.energy['kin_array_mev'][
            -self.solver_transf_mat.n_steps]
        self.v_cav_mv = np.abs(energy_now - energy_before) / np.cos(phi_s)


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

    def compute_transfer_matrix(self, synch):
        """Compute longitudinal matrix."""
        print('Warning, MT of sin cav not implemented.')
