#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

TODO : check FLAG_PHI_S_FIT
TODO : rf_param should also return phi_rf_rel. Will be necessary
for non-synch particles.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from electric_field import RfField, compute_param_cav, convert_phi_0
from constants import OMEGA_0_BUNCH, E_REST_MEV, INV_E_REST_MEV,\
    N_STEPS_PER_CELL, METHOD, FLAG_CYTHON, FLAG_PHI_ABS
from helper import recursive_items, recursive_getter

try:
    import transfer_matrices_c as tm_c

except ModuleNotFoundError:
    MESSAGE = ', Cython module not compilated. Check elements.py and setup.py'\
        + ' for more information.'

    # If Cython was asked, raise Error.
    if FLAG_CYTHON:
        raise ModuleNotFoundError('Error' + MESSAGE)
    # Else, only issue a Warning.
    print('Warning' + MESSAGE)
    # Load Python version as Cython to allow the execution of the code.
    # It will not be used.
    import transfer_matrices_p as tm_c

import transfer_matrices_p as tm_p
import helper

# Force reload of the module constants, as a modification of METHOD between
# two executions is not taken into account (alternative is to restart kernel
# each time):
# import importlib
# importlib.reload(constants)
# print(f"METHOD: {METHOD}")

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
d_n_steps = {'RK': lambda elt: N_STEPS_PER_CELL * elt.get('n_cell'),
             'leapfrog': lambda elt: N_STEPS_PER_CELL * elt.get('n_cell'),
             'jm': lambda elt: elt.get('n_z'),
             'drift': lambda elt: 1}


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
        self.elt_info = {
            'elt_name': None,
            'nature': elem[0],
            'status': 'none',    # Only make sense for cavities
        }
        self.length_m = 1e-3 * float(elem[1])

        # By default, an element is non accelerating and has a dummy
        # accelerating field.
        self.acc_field = RfField()

        self.idx = {'s_in': None, 's_out': None,
                    'elt_idx': None, 'lattice': None, 'section': None}

        self._tm_func = lambda d_z, gamma, n_steps, rf_field=None: \
            (np.empty([10, 2, 2]), np.empty([10, 2]), None)
        self.solver_param = {'n_steps': None, 'd_z': None,
                             'abs_mesh': None, 'rel_mesh': None}

    def has(self, key):
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys, to_numpy=True, **kwargs):
        """Shorthand to get attributes."""
        val = {}
        for key in keys:
            val[key] = []

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)
            # Easier to concatenate lists that stack numpy arrays, so convert
            # to list
            if not to_numpy and isinstance(val[key], np.ndarray):
                val[key] = val[key].tolist()

        # Convert to list; elements of the list are numpy is required, except
        # strings that are never converted
        out = [np.array(val[key]) if to_numpy and not isinstance(val[key], str)
               else val[key]
               for key in keys]

        # Return as tuple or single value
        if len(out) == 1:
            return out[0]
        # implicit else:
        return tuple(out)

    def init_solvers(self):
        """Initialize how transfer matrices will be calculated."""
        l_method = METHOD.split('_')

        # Select proper module (Python or Cython)
        mod = d_mod[l_method[1]]

        # Select proper number of steps
        key_n_steps = l_method[0]
        if self.elt_info['nature'] != 'FIELD_MAP':
            key_n_steps = 'drift'
        n_steps = d_n_steps[key_n_steps](self)

        self.solver_param['n_steps'] = n_steps
        self.solver_param['d_z'] = self.length_m / n_steps
        self.solver_param['rel_mesh'] = np.linspace(0., self.length_m,
                                                    n_steps + 1)

        # Select proper function to compute transfer matrix
        key_fun = l_method[0]
        if (self.get('nature') != 'FIELD_MAP'
                or self.get('status') == 'failed'):
            key_fun = 'non_acc'

        self._tm_func = d_func_tm[key_fun](mod)

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
        n_steps, d_z = self.get('n_steps', 'd_z')
        gamma = 1. + w_kin_in * INV_E_REST_MEV

        if self.get('nature') == 'FIELD_MAP' and \
                self.get('status') != 'failed':

            r_zz, gamma_phi, itg_field = \
                self._tm_func(d_z, gamma, n_steps, rf_field_kwargs)

            gamma_phi[:, 1] *= OMEGA_0_BUNCH / rf_field_kwargs['omega0_rf']
            cav_params = compute_param_cav(itg_field, self.get('status'))

        else:
            r_zz, gamma_phi, _ = self._tm_func(d_z, gamma, n_steps)
            cav_params = None

        w_kin = (gamma_phi[:, 0] - 1.) * E_REST_MEV

        results = {'r_zz': r_zz, 'cav_params': cav_params,
                   'w_kin': w_kin, 'phi_rel': gamma_phi[:, 1]}

        return results

    def update_status(self, new_status):
        """
        Change the status of a cavity.

        We also ensure that the value new_status is correct. If the new value
        is 'failed', we also set the norm of the electric field to 0.
        """
        assert self.elt_info['nature'] == 'FIELD_MAP', 'The status of an ' + \
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

        self.elt_info['status'] = new_status
        if new_status == 'failed':
            self.acc_field.k_e = 0.
            self.init_solvers()

    def keep_rf_field(self, rf_field, cav_params):
        """Save data calculated by Accelerator.compute_transfer_matrices."""
        if rf_field != {}:
            self.acc_field.cav_params = cav_params
            self.acc_field.phi_0['phi_0_abs'] = rf_field['phi_0_abs']
            self.acc_field.phi_0['phi_0_rel'] = rf_field['phi_0_rel']
            self.acc_field.k_e = rf_field['k_e']

    def rf_param(self, phi_bunch_abs, w_kin_in, d_fit=None):
        """Set the properties of the rf field (returns {} by default)."""
        # Remove unused arguments warning:
        del phi_bunch_abs, w_kin_in, d_fit

        rf_field_kwargs = {}
        return rf_field_kwargs


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

    def rf_param(self, phi_bunch_abs, w_kin_in, d_fit=None):
        """Set the properties of the electric field."""
        if self.get('status') == 'failed':
            rf_field_kwargs = {}
            return rf_field_kwargs

        # assert synch.get('synchronous'), "Out of synch particle to be " \
            # + "implemented."
        # FIXME By definition, the synchronous particle has a relative input
        # phase of 0. phi_rf_rel = 0.

        # Add the parameters that are independent from the cavity status
        rf_field_kwargs = {'omega0_rf': self.get('omega0_rf'),
                           'e_spat': self.get('e_spat'),
                           'section_idx': self.idx['section'],
                           'k_e': None, 'phi_0_rel': None, 'phi_0_abs': None}

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
        arg = (self.acc_field,)
        if self.elt_info['status'] == "compensate (in progress)":
            arg = (d_fit, w_kin_in, self)

        # Apply
        rf_field_kwargs, abs_to_rel = \
            d_cav_param_setter[self.elt_info['status']](
                *arg, **rf_field_kwargs)

        # Compute phi_0_rel in the general case. Compute instead phi_0_abs if
        # the cavity is rephased
        phi_rf_abs = phi_bunch_abs * rf_field_kwargs['omega0_rf'] \
            / OMEGA_0_BUNCH

        rf_field_kwargs['phi_0_rel'], rf_field_kwargs['phi_0_abs'] = \
            convert_phi_0(phi_rf_abs, abs_to_rel, rf_field_kwargs)

        return rf_field_kwargs

    def match_synch_phase(self, w_kin_in, phi_s_objective, **rf_field_kwargs):
        """
        Sweeps phi_0_rel until the cavity synch phase matches phi_s.

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
            diff = helper.diff_angle(phi_s_objective,
                                     results['cav_params']['phi_s'])
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
    rf_field_kwargs['k_e'] = a_f.get('k_e')
    rf_field_kwargs['phi_0_rel'] = None
    rf_field_kwargs['phi_0_abs'] = a_f.get('phi_0_abs')
    abs_to_rel = True

    # If we are calculating the transfer matrices of the nominal linac and the
    # initial phases are defined in the .dat as relative phases, phi_0_abs is
    # not defined
    if a_f.get('phi_0_abs') is None:
        rf_field_kwargs['phi_0_rel'] = a_f.get('phi_0_rel')
        abs_to_rel = False

    return rf_field_kwargs, abs_to_rel


def _find_new_absolute_entry_phase(a_f, **rf_field_kwargs):
    """Extract RfField parameters, except phi_0_abs that is recalculated."""
    rf_field_kwargs['k_e'] = a_f.get('k_e')
    rf_field_kwargs['phi_0_rel'] = a_f.get('phi_0_rel')
    rf_field_kwargs['phi_0_abs'] = None
    abs_to_rel = False
    return rf_field_kwargs, abs_to_rel


def _try_parameters_from_d_fit(d_fit, w_kin, obj_cavity, **rf_field_kwargs):
    """Extract parameters from d_fit."""
    assert d_fit['flag'], "Inconsistency between cavity status and d_fit flag."
    rf_field_kwargs['k_e'] = d_fit['k_e']
    rf_field_kwargs['phi_0_rel'] = d_fit['phi']
    rf_field_kwargs['phi_0_abs'] = d_fit['phi']

    abs_to_rel = FLAG_PHI_ABS

    if d_fit['phi_s fit']:
        phi_0 = obj_cavity.match_synch_phase(
            w_kin, phi_s_objective=d_fit['phi'], **rf_field_kwargs)
        rf_field_kwargs['phi_0_rel'] = phi_0
        rf_field_kwargs['phi_0_abs'] = None
        abs_to_rel = False

    # TODO modify the fit process in order to always fit on the relative phase.
    # Absolute phase can easily be calculated afterwards.
    return rf_field_kwargs, abs_to_rel
