#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais
"""
import numpy as np
from scipy.optimize import minimize_scalar
import transport
from electric_field import RfField, compute_param_cav, convert_phi_0
from constants import N_STEPS_PER_CELL, FLAG_PHI_ABS, METHOD, STR_PHI_0_ABS, \
    OMEGA_0_BUNCH, FLAG_PHI_S_FIT, FLAG_CYTHON
import transfer_matrices_c as tm_c
import transfer_matrices_p as tm_p

import helper


d_fun_tm = {
    'non_acc': {'RK_p': tm_p.z_drift,
                'RK_c': tm_c.z_drift,
                'leapfrog_p': tm_p.z_drift,
                'leapfrog_c': tm_c.z_drift,
                'transport': transport.transport_beam,
                },
    'accelerating': {'RK_p': tm_p.z_field_map_rk4,
                     'RK_c': tm_c.z_field_map_rk4,
                     'leapfrog_p': tm_p.z_field_map_leapfrog,
                     'leapfrog_c': tm_c.z_field_map_leapfrog,
                     'transport': transport.transport_beam,
                     }
}


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

        self.pos_m = {'abs': None, 'rel': None}
        self.idx = {'s_in': None,
                    's_out': None,
                    'element': None,
                    'lattice': [],
                    'section': [],
                    }
        # tmat stands for 'transfer matrix'
        self.tmat = {
            'func': {'RK': lambda: None,
                     'leapfrog': lambda: None,
                     'transport': lambda: None},
            'matrix': None,
            'solver_param': {'n_steps': None, 'd_z': None},
        }

    def init_solvers(self):
        """Initialize solvers as well as general properties."""
        if self.info['nature'] == 'FIELD_MAP':
            if self.info['status'] == 'failed':
                key = 'non_acc'
            else:
                key = 'accelerating'
            n_steps = N_STEPS_PER_CELL * self.acc_field.n_cell

        else:
            key = 'non_acc'
            n_steps = 1

        self.pos_m['rel'] = np.linspace(0., self.length_m, n_steps + 1)

        self.tmat['matrix'] = np.full((n_steps, 2, 2), np.NaN)
        self.tmat['func'] = d_fun_tm[key][METHOD]
        self.tmat['solver_param'] = {'n_steps': n_steps,
                                     'd_z': self.length_m / n_steps,
                                     }

    def calc_transf_mat(self, w_kin_in, **kwargs):
        """Compute longitudinal matrix."""
        n_steps, d_z = self.tmat['solver_param'].values()

        # Initialisation of electric field arrays
        # FIXME
        if self.idx['element'] == 0 and FLAG_CYTHON:
            tm_c.init_arrays()

        if self.info['nature'] == 'FIELD_MAP' and \
                self.info['status'] != 'failed':

            # The field map functions from transfer_matrices_c and
            # transfer_matrices_p do not take exactly the same argumemts
            if FLAG_CYTHON:
                last_arg = self.idx['section'][0][0]
            else:
                last_arg = kwargs['e_spat']

            r_zz, w_phi, itg_field = \
                self.tmat['func'](d_z, w_kin_in, n_steps, kwargs['omega0_rf'],
                                  kwargs['norm'], kwargs['phi_0_rel'],
                                  last_arg)
            w_phi[:, 1] *= OMEGA_0_BUNCH / kwargs['omega0_rf']
            cav_params = compute_param_cav(itg_field, self.info['status'])

        else:
            r_zz, w_phi, _ = self.tmat['func'](d_z, w_kin_in, n_steps)
            cav_params = None

        results = {'r_zz': r_zz, 'cav_params': cav_params,
                   'w_kin': w_phi[:, 0], 'phi_rel': w_phi[:, 1]}

        return results

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
            self.init_solvers()


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

    def _import_from_acc_f(self, kwargs):
        """Import norm and phi_0 from the accelerating field."""
        kwargs['norm'] = self.acc_field.norm
        kwargs['phi_0_rel'] = self.acc_field.phi_0['rel']
        kwargs['phi_0_abs'] = self.acc_field.phi_0['abs']
        return kwargs

    def set_cavity_parameters(self, synch, phi_abs_in, w_kin_in, d_fit=None):
        """
        Set the properties of the electric field.

        If the cavity is in nominal workking condition, we use the properties
        (norm, phi_0) defined in the RfField object. If we are compensating
        a fault, we use the properties given by the optimisation algorithm.
        """
        if d_fit is None:
            d_fit = {'flag': False}
        acc_f = self.acc_field

        # FIXME Equiv of synch._set_omega_rf:
        new_omega = 2. * OMEGA_0_BUNCH
        synch.omega0['rf'] = new_omega
        synch.omega0['ref'] = new_omega
        synch.frac_omega['rf_to_bunch'] = OMEGA_0_BUNCH / new_omega
        synch.frac_omega['bunch_to_rf'] = new_omega / OMEGA_0_BUNCH
        phi_rf_abs = phi_abs_in * acc_f.omega0_rf / OMEGA_0_BUNCH

        kwargs = {
            'omega0_rf': acc_f.omega0_rf,
            'norm': np.NaN,
            'phi_0_rel': np.NaN,
            'phi_0_abs': np.NaN,
            'phi_s_objective': None,
            'e_spat': acc_f.e_spat,
        }

        assert synch.info['synchronous'], 'Not sure what should happen here.'
        # Ref linac: we compute every missing phi_0
        if synch.info['reference']:
            acc_f.phi_0['rel'], acc_f.phi_0['abs'] = convert_phi_0(
                phi_rf_abs, acc_f.phi_0['abs_phase_flag'], acc_f.phi_0['rel'],
                acc_f.phi_0['abs'])
            # acc_f.convert_phi_0(phi_rf_abs, acc_f.phi_0['abs_phase_flag'])
            kwargs = self._import_from_acc_f(kwargs)

        else:
            # Phases should have been imported from reference linac
            if self.info['status'] == 'nominal':
                # We already have the phi0's from the reference linac. We
                # recompute the relative or absolute one according to
                # FLAG_PHI_ABS
                acc_f.phi_0['rel'], acc_f.phi_0['abs'] = convert_phi_0(
                    phi_rf_abs, FLAG_PHI_ABS, acc_f.phi_0['rel'],
                    acc_f.phi_0['abs'])
                kwargs = self._import_from_acc_f(kwargs)

            elif self.info['status'] == 'rephased':
                # We must keep the relative phase equal to reference linac
                acc_f.rephase_cavity(phi_rf_abs)
                print('set_proper_cavity_parameters: cav rephasing not',
                      'reimplemented yet.')

            elif self.info['status'] == 'fault':
                # Useless, as we used drift functions when there is a fault
                raise IOError('Faulty cavity should not have parameters.')

            elif self.info['status'] == 'compensate':
                # The phi0's are set by the fitting algorithm. We compute
                # the missing (abs or rel) value of phi0 for the sake of
                # completeness, but it won't be used to calculate the
                # matrix
                if d_fit['flag']:
                    kwargs['norm'] = d_fit['norm']

                    if FLAG_PHI_S_FIT:
                        kwargs['phi_s_objective'] = d_fit['phi']
                        kwargs['phi_0_rel'] = \
                            self.match_synch_phase(w_kin_in, **kwargs)

                        kwargs['phi_0_rel'], kwargs['phi_0_abs'] = \
                            convert_phi_0(phi_rf_abs, False,
                                          kwargs['phi_0_rel'],
                                          kwargs['phi_0_abs'])
                    else:
                        # fit['phi'] is phi_0_rel or phi_0_abs according to
                        # FLAG_PHI_ABS.
                        # We set it and calculate the abs/rel phi_0 that is
                        # missing.
                        kwargs[STR_PHI_0_ABS] = d_fit['phi']
                        kwargs['phi_0_rel'], kwargs['phi_0_abs'] = \
                            convert_phi_0(
                            phi_rf_abs, FLAG_PHI_ABS, kwargs['phi_0_rel'],
                            kwargs['phi_0_abs'])

                else:
                    acc_f.phi_0['rel'], acc_f.phi_0['abs'] = convert_phi_0(
                        phi_rf_abs, FLAG_PHI_ABS, acc_f.phi_0['rel'],
                        acc_f.phi_0['abs'])
                    kwargs = self._import_from_acc_f(kwargs)

        return kwargs

    def match_synch_phase(self, w_kin_in, **kwargs):
        """Sweeps phi_0_rel until the cavity synch phase matches phi_s_rad."""
        bounds = (0, 2. * np.pi)

        def _wrapper_synch(phi_0_rad):
            kwargs['phi_0_rel'] = phi_0_rad
            results = self.calc_transf_mat(w_kin_in, **kwargs)
            diff = helper.diff_angle(
                kwargs['phi_s_objective'],
                results['cav_params']['phi_s_rad'])
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
