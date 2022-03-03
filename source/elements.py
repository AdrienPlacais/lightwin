#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021

@author: placais
"""
import cmath
import numpy as np
import transfer_matrices
import transport
from electric_field import RfField


# =============================================================================
# Element class
# =============================================================================
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
            'in': None,         # @TODO: replace by synch_in
            'out': None,
            }

        self.transfer_matrix = None

        self.solver_param_transf_mat = {'method': None, 'n_steps': None,
                                        'd_z': None}
        # Function to compute transfer_matrix, given by
        # solver_param_transf_mat['method']
        self.func_transf_mat = {'RK': None, 'leapfrog': None,
                                'transport': None}

    def init_solvers(self):
        """Initialize solvers as well as general properties."""
        if self.accelerating:
            assert self.name == 'FIELD_MAP'
            n_steps = 10 * self.acc_field.n_cell

            if self.status['failed']:
                func_transf_mat = {
                    'RK': transfer_matrices.z_drift_element,
                    'leapfrog': transfer_matrices.z_drift_element,
                    'transport': transport.transport_beam,
                    }
            else:
                func_transf_mat = {
                    'RK': transfer_matrices.z_field_map_electric_field,
                    'leapfrog': transfer_matrices.
                    z_field_map_electric_field,
                    'transport': transport.transport_beam,
                    }

        else:
            # By default, 1 step for non-accelerating elements
            n_steps = 1
            func_transf_mat = {
                'RK': transfer_matrices.z_drift_element,
                'leapfrog': transfer_matrices.z_drift_element,
                'transport': transport.transport_beam,
              }

        self.pos_m['rel'] = np.linspace(0., self.length_m, n_steps + 1)
        self.transfer_matrix = np.full((n_steps, 2, 2), np.NaN)

        self.func_transf_mat = func_transf_mat
        self.solver_param_transf_mat = {
            'method': None,
            'n_steps': n_steps,
            'd_z': self.length_m / n_steps,
            }

    def compute_transfer_matrix(self, synch):
        """Compute longitudinal matrix."""
        self.transfer_matrix = self.func_transf_mat[
            self.solver_param_transf_mat['method']](self, synch=synch)

        if self.name == 'FIELD_MAP':
            self._compute_synch_phase_and_acc_pot(synch)


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
        self.aperture_flag = int(elem[8])               # K_a
        # FIXME according to doc, may also be float
        self.field_map_file_name = str(elem[9])         # FileName

        try:
            relative_phase_flag = int(elem[10])    # P
        except IndexError:
            # Relative by default
            relative_phase_flag = 0
            # pass

        self.acc_field = RfField(352.2, norm=float(elem[6]),
                                 relative_phase_flag=relative_phase_flag,
                                 phi_0=np.deg2rad(float(elem[3])))
        # FIXME frequency import
        self.status = {
            'failed': False,
            'compensate': False
            }

    def _compute_synch_phase_and_acc_pot(self, synch):
        """Compute the sychronous phase and accelerating potential."""
        if self.status['failed']:
            self.acc_field.phi_s_deg = np.NaN
            self.acc_field.v_cav_mv = np.NaN

        else:
            phi_s = cmath.phase(self.acc_field.f_e)
            self.acc_field.phi_s_deg = np.rad2deg(phi_s)

            energy_now = synch.energy['kin_array_mev'][self.idx['out']]
            energy_before = synch.energy['kin_array_mev'][self.idx['in']]
            self.acc_field.v_cav_mv = np.abs(energy_now - energy_before) \
                / np.cos(phi_s)

    def fail(self):
        """Break this nice cavity."""
        self.status['failed'] = True
        self.acc_field.norm = 0.


class Lattice():
    """Used to get the number of elements per lattice."""

    def __init__(self, elem):
        self.n_lattice = int(elem[1])
