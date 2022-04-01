#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais
"""
import numpy as np
from scipy.optimize import minimize_scalar
import transfer_matrices_p
import transport
from electric_field import RfField, compute_param_cav
from constants import N_STEPS_PER_CELL, STR_PHI_ABS, E_rest_MeV, METHOD
import helper


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
        self.info = {
            'name': None,
            'nature': elem[0],
            'status': None,    # Only make sense for cavities
            'zone': None,
            }
        self.length_m = 1e-3 * float(elem[1])

        # By default, an element is non accelerating and has a dummy
        # accelerating field.
        self.acc_field = RfField()

        self.pos_m = {
            'abs': None,
            'rel': None,
            }
        self.idx = {
            's_in': None,
            's_out': None,
            'element': None,
            'lattice': None,
            'section': None,
            }
        # tmat stands for 'transfer matrix'
        self.tmat = {
            'matrix': None,
            'solver_param': {'n_steps': None, 'd_z': None},
            'func': {'RK': None, 'leapfrog': None, 'transport': None},
            }

    def init_solvers(self):
        """Initialize solvers as well as general properties."""
        functions_transf_mat = {
            'non_acc': {'RK': transfer_matrices_p.z_drift_p,
                        'leapfrog': transfer_matrices_p.z_drift_p,
                        'transport': transport.transport_beam,
                        },
            'accelerating': {
                'RK': transfer_matrices_p.z_field_map_p,
                'leapfrog': transfer_matrices_p.z_field_map_p,
                'transport': transport.transport_beam,
                }}
        key = 'non_acc'
        n_steps = 1
        if self.info['nature'] == 'FIELD_MAP':
            n_steps = N_STEPS_PER_CELL * self.acc_field.n_cell
            if self.info['status'] != 'failed':
                key = 'accelerating'

        self.pos_m['rel'] = np.linspace(0., self.length_m, n_steps + 1)
        self.tmat['matrix'] = np.full((n_steps, 2, 2), np.NaN)

        self.tmat['func'] = functions_transf_mat[key]
        self.tmat['solver_param'] = {
            'n_steps': n_steps,
            'd_z': self.length_m / n_steps,
            }

    def compute_transfer_matrix(self, W_kin_in, omega0, **kwargs):
        """Compute longitudinal matrix."""
        n_steps, d_z = self.tmat['solver_param'].values()
        tmat_fun = self.tmat['func'][METHOD]

        if self.info['nature'] == 'FIELD_MAP':
            acc_f = self.acc_field
            r_zz, l_gamma, l_beta, l_phi_rel, itg_field = \
                tmat_fun(d_z, W_kin_in, n_steps, **kwargs)

            acc_f.cav_params = compute_param_cav(itg_field,
                                                 self.info['status'])
            l_delta_phi = [
                phi_rf * omega0 / kwargs['omega0_rf']
                for phi_rf in l_phi_rel
                ]

        else:
            r_zz, l_gamma, l_beta, l_delta_phi = tmat_fun(d_z, W_kin_in,
                                                          n_steps, omega0)

        self.tmat['matrix'] = r_zz
        return l_gamma, l_beta, l_delta_phi

    def update_status(self, new_status):
        """
        Change the status of a cavity.

        We also ensure that the value new_status is correct. If the new value
        is 'failed', we also set the norm of the electric field to 0.
        """
        assert self.info['nature'] == 'FIELD_MAP', 'The status of an ' + \
            'element only makes sense for cavities.'

        authorized_values = [
            'nominal',
            'failed',
            'compensate',
            'rephased',
            ]
        assert new_status in authorized_values

        self.info['status'] = new_status
        if new_status == 'failed':
            self.acc_field.norm = 0.


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
        self.grad = float(elem[2])


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
        self.geometry = int(elem[1])
        self.length_m = 1e-3 * float(elem[2])
        self.aperture_flag = int(elem[8])               # K_a
        # FIXME according to doc, may also be float
        self.field_map_file_name = str(elem[9])         # FileName

        try:
            absolute_phase_flag = int(elem[10])    # P
        except IndexError:
            # Relative by default
            elem.append('0')
            absolute_phase_flag = int(elem[10])

        self.acc_field = RfField(norm=float(elem[6]),
                                 absolute_phase_flag=absolute_phase_flag,
                                 phi_0=np.deg2rad(float(elem[3])))
        self.update_status('nominal')

    def match_synch_phase(self, W_kin_in, omega0, phi_s_rad):
        """Sweeps phi_0 until the cavity synch phase matches phi_s_rad."""
        bounds = (0, 2.*np.pi)
        acc_f = self.acc_field
        kwargs = {
            'omega0_rf': acc_f.omega0_rf,
            'norm': acc_f.norm,
            'phi_0_rel': acc_f.phi_0['rel'],
            'e_spat': acc_f.e_spat,
            }

        def _wrapper_synch(phi_0_rad):
            kwargs['phi_0_rel'] = phi_0_rad
            l_gamma, l_beta, l_delta_phi =\
                self.compute_transfer_matrix(W_kin_in, omega0, **kwargs)
            diff = helper.diff_angle(
                phi_s_rad,
                np.deg2rad(self.acc_field.cav_params['phi_s_deg']))
            return diff**2

        res = minimize_scalar(_wrapper_synch, bounds=bounds)
        if not res.success:
            print('match synch phase not found')

        return res.x


class Lattice():
    """Used to get the number of elements per lattice."""

    def __init__(self, elem):
        self.n_lattice = int(elem[1])


class Freq():
    """Used to get the frequency of every Section."""

    def __init__(self, elem):
        self.f_rf_mhz = float(elem[1])
