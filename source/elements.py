#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

TODO : check FLAG_PHI_S_FIT
TODO : set_cavity_parameters should also return phi_rf_rel. Will be necessary
for non-synch particles.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from electric_field import RfField, compute_param_cav, convert_phi_0
import constants

try:
    import transfer_matrices_c as tm_c

except ModuleNotFoundError:
    MESSAGE = ', Cython module not compilated. Check elements.py and setup.py'\
        + ' for more information.'

    # If Cython was asked, raise Error.
    if constants.FLAG_CYTHON:
        raise ModuleNotFoundError('Error' + MESSAGE)
    # Else, only issue a Warning.
    print('Warning' + MESSAGE)
    # Load Python version as Cython to allow the execution of the code.
    # It will not be used.
    import transfer_matrices_p as tm_c

import transfer_matrices_p as tm_p
import helper

# Force reload of the module constants, as a modification of constants.METHOD
# between two executions is not taken into account
# (alternative is to restart kernel each time)
# import importlib
# importlib.reload(constants)
# print(f"METHOD: {constants.METHOD}")

# =============================================================================
# Module dictionaries
# =============================================================================
# Dict to select the proper transfer matrix function
d_mod = {'p': tm_p,     # Pure Python
         'c': tm_c}     # Cython
d_func_tm = {'RK': lambda mod: mod.z_field_map_rk4,
             'leapfrog': lambda mod: mod.z_field_map_leapfrog,
             'jm': lambda mod: mod.z_field_map_jm,
             'non_acc': lambda mod: mod.z_drift}

# Dict to select the proper number of steps for the transfer matrix, the
# energy, the phase, etc
d_n_steps = {
    'RK': lambda elt: constants.N_STEPS_PER_CELL * elt.acc_field.n_cell,
    'leapfrog': lambda elt: constants.N_STEPS_PER_CELL * elt.acc_field.n_cell,
    'jm': lambda elt: elt.acc_field.n_z,
    'drift': lambda elt: 1,
}


# =============================================================================
# Element class
# =============================================================================
class _Element():
    """Generic element. _ ensures that it is not called from another module."""

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
        self.idx = {'s_in': None, 's_out': None,
                    'element': None,
                    'lattice': [],
                    'section': []}

        # tmat stands for 'transfer matrix'
        self.tmat = {
            'func': {'RK': lambda: None,
                     'leapfrog': lambda: None,
                     'transport': lambda: None},
            'matrix': None,
            'solver_param': {
                'n_steps': None,
                'd_z': None,
                # 'delta_phi_norm': None,
                # 'delta_gamma_norm': None,
            },
        }

    def init_solvers(self):
        """Initialize how transfer matrices will be calculated."""
        l_method = constants.METHOD.split('_')

        # Select proper module (Python or Cython)
        mod = d_mod[l_method[1]]

        # Select proper number of steps
        key_n_steps = l_method[0]
        if self.info['nature'] != 'FIELD_MAP':
            key_n_steps = 'drift'
        n_steps = d_n_steps[key_n_steps](self)

        self.pos_m['rel'] = np.linspace(0., self.length_m, n_steps + 1)
        self.tmat['matrix'] = np.full((n_steps, 2, 2), np.NaN)
        self.tmat['solver_param']['n_steps'] = n_steps
        d_z = self.length_m / n_steps
        self.tmat['solver_param']['d_z'] = d_z

        # Select proper function to compute transfer matrix
        key_fun = l_method[0]
        if self.info['nature'] != 'FIELD_MAP' \
                or self.info['status'] == 'failed':
            key_fun = 'non_acc'
        # else:
            # # Precopute some constants to speed up calculation
            # delta_phi_norm = self.acc_field.omega0_rf * d_z / constants.c
            # delta_gamma_norm = constants.q_adim * d_z / constants.E_rest_MeV
            # self.tmat['solver_param']['delta_phi_norm'] = delta_phi_norm
            # self.tmat['solver_param']['delta_gamma_norm'] = delta_gamma_norm
        self.tmat['func'] = d_func_tm[key_fun](mod)

    def calc_transf_mat(self, w_kin_in, **rf_field_kwargs):
        """
        Compute longitudinal matrix.

        Parameters
        ----------
        w_kin_in : float
            Kinetic energy at the entrance of the element in MeV.
        rf_field_kwargs : dict
            Holds all the rf field parameters. The mandatory keys are:
                omega0_rf, k_e, phi_0_rel
            For Python implementation, also need e_spat.
            For Cython implementation, also need section_idx.
        """
        n_steps, d_z = self.tmat['solver_param'].values()
        gamma = 1. + w_kin_in * constants.inv_E_rest_MeV

        # Initialisation of electric field arrays
        # FIXME
        if self.idx['element'] == 0 and constants.FLAG_CYTHON:
            tm_c.init_arrays()

        if self.info['nature'] == 'FIELD_MAP' and \
                self.info['status'] != 'failed':

            r_zz, gamma_phi, itg_field = \
                self.tmat['func'](d_z, gamma, n_steps, rf_field_kwargs)

            gamma_phi[:, 1] *= constants.OMEGA_0_BUNCH \
                / rf_field_kwargs['omega0_rf']
            cav_params = compute_param_cav(itg_field, self.info['status'])

        else:
            r_zz, gamma_phi, _ = self.tmat['func'](d_z, gamma, n_steps)
            cav_params = None

        w_kin = (gamma_phi[:, 0] - 1.) * constants.E_rest_MeV

        results = {'r_zz': r_zz, 'cav_params': cav_params,
                   'w_kin': w_kin, 'phi_rel': gamma_phi[:, 1]}

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
            # Cavity settings not changed from .dat
            "nominal",
            # Cavity ABSOLUTE phase changed; relative phase unchanged
            "rephased (in progress)",
            "rephased (ok)",
            # Cavity norm is 0
            "failed",
            # Trying to fit
            "compensate (in progress)",
            # Compensating, proper setting found
            "compensate (ok)",
            # Compensating, proper setting not found
            "compensate (not ok)",
        ]
        assert new_status in authorized_values

        self.info['status'] = new_status
        if new_status == 'failed':
            self.acc_field.k_e = 0.
            self.init_solvers()

    def keep_mt_and_rf_field(self, elt_results, rf_field):
        """
        Save some data calculated by Accelerator.compute_transfer_matrices.
        """
        self.tmat["matrix"] = elt_results["r_zz"]

        if elt_results['cav_params'] is not None:
            self.acc_field.cav_params = elt_results['cav_params']
            self.acc_field.phi_0['abs'] = rf_field['phi_0_abs']
            self.acc_field.phi_0['rel'] = rf_field['phi_0_rel']


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

        self.acc_field = RfField(k_e=float(elem[6]),
                                 absolute_phase_flag=absolute_phase_flag,
                                 phi_0=np.deg2rad(float(elem[3])))
        self.update_status('nominal')

    def set_cavity_parameters(self, synch, phi_bunch_abs, w_kin_in,
                              d_fit=None):
        """
        Set the properties of the electric field.

        In this routine, all phases are rf phases, i.e. defined as:
            phi = omega0_rf * t
        while phases are omega0_bunch * t in most of the program.

        Return
        ------
        phi_rf_rel: float
            Relative phase of the particle at the entry of the cavity.
        """
        a_f = self.acc_field

        # Set pulsation inside cavity, convert bunch phase into rf phase
        new_omega = 2. * constants.OMEGA_0_BUNCH
        phi_rf_abs = synch.set_omega_rf(new_omega, phi_bunch_abs)
        # FIXME new_omega not necessarily 2*omega_bunch

        err_msg = 'Should not look for cavity parameters of a broken cavity.'
        assert self.info['status'] != 'fault', err_msg

        err_msg = 'Out of synch particle to be implemented.'
        assert synch.info['synchronous'], err_msg

        # By definition, the synchronous particle has a relative input phase of
        # 0.
        # FIXME :
        # phi_rf_rel = 0.

        # Add the parameters that are independent from the cavity status
        rf_field_kwargs = {'omega0_rf': a_f.omega0_rf,
                           'e_spat': a_f.e_spat,
                           'section_idx': self.idx['section'][0][0]
                          }

        # Set norm and phi_0 of the cavity
        d_cav_param_setter = {
            "nominal": _take_parameters_from_rf_field_object,
            "rephased (in progress)": _find_new_absolute_entry_phase,
            "rephased (ok)": _take_parameters_from_rf_field_object,
            "compensate (in progress)": _try_parameters_from_d_fit,
            "compensate (ok)": _take_parameters_from_rf_field_object,
            "compensate (not ok)": _take_parameters_from_rf_field_object,
        }
        # Argument for the functions in d_cav_param_setter
        arg = (a_f,)
        if self.info['status'] == "compensate (in progress)":
            arg = (d_fit, w_kin_in, self)

        # Apply
        rf_field_kwargs, flag_abs_to_rel = \
            d_cav_param_setter[self.info['status']](*arg, **rf_field_kwargs)

        # Compute phi_0_rel in the general case. Compute instead phi_0_abs if
        # the cavity is rephased
        rf_field_kwargs['phi_0_rel'], rf_field_kwargs['phi_0_abs'] = \
            convert_phi_0(phi_rf_abs, flag_abs_to_rel, rf_field_kwargs)

        return rf_field_kwargs

    def match_synch_phase(self, w_kin_in, phi_s_objective, **rf_field_kwargs):
        """
        Sweeps phi_0_rel until the cavity synch phase matches phi_s_rad.

        Parameters
        ----------
        w_kin_in : float
            Kinetic energy at the cavity entrance in MeV.
        rf_field_kwargs : dict
            Holds all rf electric field parameters.

        Return
        ------
        phi_0_rel : float
            The relative cavity entrance phase that leads to a synchronous
            phase of phi_s_objective.
        """
        bounds = (0, 2. * np.pi)

        def _wrapper_synch(phi_0_rel):
            rf_field_kwargs['phi_0_rel'] = phi_0_rel
            rf_field_kwargs['phi_0_abs'] = None
            results = self.calc_transf_mat(w_kin_in, **rf_field_kwargs)
            diff = helper.diff_angle(
                phi_s_objective,
                results['cav_params']['phi_s_rad'])
            return diff**2

        res = minimize_scalar(_wrapper_synch, bounds=bounds)
        if not res.success:
            print('match synch phase not found')
        phi_0_rel = res.x

        return phi_0_rel


class Lattice():
    """Used to get the number of elements per lattice."""

    def __init__(self, elem):
        self.n_lattice = int(elem[1])


class Freq():
    """Used to get the frequency of every Section."""

    def __init__(self, elem):
        self.f_rf_mhz = float(elem[1])


def _take_parameters_from_rf_field_object(a_f, **rf_field_kwargs):
    """Extract RfField object parameters."""
    rf_field_kwargs['k_e'] = a_f.k_e
    rf_field_kwargs['phi_0_rel'] = None
    rf_field_kwargs['phi_0_abs'] = a_f.phi_0['abs']
    flag_abs_to_rel = True

    # If we are calculating the transfer matrices of the nominal linac and the
    # initial phases are defined in the .dat as relative phases, phi_0_abs is
    # not defined
    if rf_field_kwargs['phi_0_abs'] is None:
        rf_field_kwargs['phi_0_rel'] = a_f.phi_0['rel']
        flag_abs_to_rel = False

    return rf_field_kwargs, flag_abs_to_rel


def _find_new_absolute_entry_phase(a_f, **rf_field_kwargs):
    """Extract RfField parameters, except phi_0_abs that is recalculated."""
    rf_field_kwargs['k_e'] = a_f.k_e
    rf_field_kwargs['phi_0_rel'] = a_f.phi_0['rel']
    rf_field_kwargs['phi_0_abs'] = None
    flag_abs_to_rel = False
    return rf_field_kwargs, flag_abs_to_rel


def _try_parameters_from_d_fit(d_fit, w_kin, obj_cavity, **rf_field_kwargs):
    """Extract parameters from d_fit."""
    assert d_fit['flag'], "Inconsistency between cavity status and d_fit flag."
    rf_field_kwargs['k_e'] = d_fit['k_e']
    rf_field_kwargs['phi_0_rel'] = d_fit['phi']
    rf_field_kwargs['phi_0_abs'] = d_fit['phi']

    flag_abs_to_rel = constants.FLAG_PHI_ABS

    if constants.FLAG_PHI_S_FIT:
        phi_0 = obj_cavity.match_synch_phase(
            w_kin, phi_s_objective=d_fit['phi'], **rf_field_kwargs)
        rf_field_kwargs['phi_0_rel'] = phi_0
        rf_field_kwargs['phi_0_abs'] = None
        flag_abs_to_rel = False

    # TODO modify the fit process in order to always fit on the
    # relative phase. Absolute phase can easily be calculated
    # afterwards.
    return rf_field_kwargs, flag_abs_to_rel
