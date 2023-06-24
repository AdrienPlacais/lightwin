#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

TODO : check FLAG_PHI_S_FIT
TODO : rf_param should also return phi_rf_rel. Will be necessary
for non-synch particles.
FIXME : __repr__ won't work with retuned elements
"""
import logging
from typing import Any
import numpy as np
from scipy.optimize import minimize_scalar

import config_manager as con
from core.electric_field import (RfField, compute_param_cav,
                                 phi_0_rel_corresponding_to,
                                 phi_0_abs_corresponding_to)
from util.helper import recursive_items, recursive_getter, diff_angle
import util.converters as convert
from optimisation.set_of_cavity_settings import SingleCavitySettings

try:
    import core.transfer_matrices_c as tm_c

except ModuleNotFoundError:
    MESSAGE = 'Cython module not compilated. Check setup.py and elements.py'\
        + ' for more information.'

    # If Cython was asked, raise Error.
    if con.FLAG_CYTHON:
        logging.error(MESSAGE)
        raise ModuleNotFoundError(MESSAGE)
    # Else, only issue a Warning.
    logging.warning(MESSAGE)
    # Load Python version as Cython to allow the execution of the code.
    # It will not be used.
    import core.transfer_matrices_p as tm_c

import core.transfer_matrices_p as tm_p

# =============================================================================
# Module dictionaries
# =============================================================================
# Dict to select the proper transfer matrix function
d_mod = {'p': tm_p,     # Pure Python
         'c': tm_c}     # Cython
d_func_tm = {'RK': lambda mod: mod.z_field_map_rk4,
             'leapfrog': lambda mod: mod.z_field_map_leapfrog,
             'non_acc': lambda mod: mod.z_drift}

# Dict to select the proper number of steps for the transfer matrix, the
# energy, the phase, etc
d_n_steps = {'RK': lambda elt: con.N_STEPS_PER_CELL * elt.get('n_cell'),
             'leapfrog': lambda elt: con.N_STEPS_PER_CELL * elt.get('n_cell'),
             'drift': lambda elt: 1}


# =============================================================================
# Element class
# =============================================================================
class _Element():
    """Generic element. _ ensures that it is not called from another module."""

    def __init__(self, elem: list[str]) -> None:
        """
        Init parameters common to all elements.

        Some attributes, such as length_m for FIELD_MAP, are stored differently
        and will be changed later.

        Parameters
        ----------
        elem : list of string
            A valid line of the .dat file.
        """
        self.__elem = elem
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

    def __str__(self) -> str:
        return self.elt_info['elt_name']

    def __repr__(self) -> str:
        # if self.elt_info['status'] not in ['none', 'nominal']:
        #     logging.warning("Element properties where changed.")
        # return f"{self.__class__}(elem={self.__elem})"
        return self.__str__()

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True,
            **kwargs: dict) -> Any:
        """Shorthand to get attributes."""
        val: dict[str, Any] = {}
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
        if len(out) == 1:
            return out[0]

        return tuple(out)

    def init_solvers(self) -> None:
        """Initialize how transfer matrices will be calculated."""
        l_method = con.METHOD.split('_')

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

    def init_mesh(self, n_steps: int) -> None:
        """Initialize meshing in the _Element."""
        self.solver_param['n_steps'] = n_steps
        self.solver_param['d_z'] = self.length_m / n_steps
        self.solver_param['rel_mesh'] = np.linspace(0., self.length_m,
                                                    n_steps + 1)

    def calc_transf_mat(self, w_kin_in: float, **rf_field_kwargs: dict
                        ) -> dict:
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

        Returns
        -------
        results : dict
            Holds the results. Keys are 'r_zz', 'cav_params', 'w_kin' and
            'phi_rel'.
        """
        n_steps, d_z = self.get('n_steps', 'd_z')
        gamma = convert.energy(w_kin_in, "kin to gamma")

        if self.get('nature') == 'FIELD_MAP' and \
                self.get('status') != 'failed':

            r_zz, gamma_phi, itg_field = \
                self._tm_func(d_z, gamma, n_steps, rf_field_kwargs)

            gamma_phi[:, 1] /= rf_field_kwargs['n_cell']
            cav_params = compute_param_cav(itg_field, self.get('status'))

        else:
            r_zz, gamma_phi, _ = self._tm_func(d_z, gamma, n_steps)
            cav_params = None

        w_kin = convert.energy(gamma_phi[:, 0], "gamma to kin")

        results = {'r_zz': r_zz, 'cav_params': cav_params,
                   'w_kin': w_kin, 'phi_rel': gamma_phi[:, 1]}

        return results

    def update_status(self, new_status: str) -> None:
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

    def keep_rf_field(self, rf_field: dict, cav_params: dict) -> None:
        """Save data calculated by Accelerator.compute_transfer_matrices."""
        if rf_field != {}:
            self.acc_field.cav_params = cav_params
            self.acc_field.phi_0['phi_0_abs'] = rf_field['phi_0_abs']
            self.acc_field.phi_0['phi_0_rel'] = rf_field['phi_0_rel']
            self.acc_field.k_e = rf_field['k_e']

    def rf_param(self, phi_bunch_abs: float, w_kin_in: float,
                 cavity_settings: SingleCavitySettings | None = None
                 ) -> dict:
        """
        Set the properties of the rf field; in the default case, returns None.

        Parameters
        ----------
        phi_bunch_abs : float
            Absolute phase of the particle (bunch frequency).
        w_kin_in : float
            Kinetic energy at the Element entrance in MeV.
        cavity_settings : SingleCavitySettings | None, optional
            Cavity settings. Should be None in a non-accelerating element such
            as a Drift or a broken FieldMap, and in accelerating elements
            outside the fit process. The default is None.

        Returns
        -------
        rf_parameters : dict
            Always {} by default.

        """
        return {}

    def is_accelerating(self) -> bool:
        """Say if this _Element is accelerating or not."""
        return self.get('nature') == 'FIELD_MAP' \
            and self.get('status') != 'failed'


# =============================================================================
# More specific classes
# =============================================================================
class Drift(_Element):
    """Sub-class of Element, with parameters specific to DRIFTs."""

    def __init__(self, elem: list[str]) -> None:
        n_attributes = len(elem) - 1
        assert n_attributes in [2, 3, 5]
        super().__init__(elem)


class Quad(_Element):
    """Sub-class of Element, with parameters specific to QUADs."""

    def __init__(self, elem: list[str]) -> None:
        n_attributes = len(elem) - 1
        assert n_attributes in range(3, 10)
        super().__init__(elem)
        self.grad = float(elem[2])


class Solenoid(_Element):
    """Sub-class of Element, with parameters specific to SOLENOIDs."""

    def __init__(self, elem: list[str]) -> None:
        n_attributes = len(elem) - 1
        assert n_attributes == 3
        super().__init__(elem)


class FieldMap(_Element):
    """Sub-class of Element, with parameters specific to FIELD_MAPs."""

    def __init__(self, elem: list[str]) -> None:
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
                                 absolute_phase_flag=bool(absolute_phase_flag),
                                 phi_0=np.deg2rad(float(elem[3])))
        self.update_status('nominal')

    def rf_param(self, phi_bunch_abs: float, w_kin_in: float,
                 cavity_settings: SingleCavitySettings | None = None
                 ) -> dict:
        """
        Set the properties of the rf field; specific to FieldMap.

        Parameters
        ----------
        phi_bunch_abs : float
            Absolute phase of the particle (bunch frequency).
        w_kin_in : float
            Kinetic energy at the Element entrance in MeV.
        cavity_settings : SingleCavitySettings | None, optional
            Cavity settings. Should be None in a non-accelerating element such
            as a Drift or a broken FieldMap, and in accelerating elements
            outside the fit process. The default is None.

        Returns
        -------
        rf_parameters : dict
            Holds parameters that Envelope1d needs to solve the motion in the
            FieldMap. If this is a non-accelerating Element, return {}.

        """
        status = self.elt_info['status']
        if status in ['none', 'failed']:
            return {}

        generic_rf_param = {
            'omega0_rf': self.get('omega0_rf'),
            'e_spat': self.acc_field.e_spat,
            'section_idx': self.idx['section'],
            'n_cell': self.get('n_cell')
        }

        if status in ['nominal', 'rephased (ok)', 'compensate (ok)',
                      'compensate (not ok)']:
            norm_and_phases, abs_to_rel = _get_from(self.acc_field)

        elif status in ['rephased (in progress)']:
            norm_and_phases, abs_to_rel = _get_from(self.acc_field,
                                                    force_rephasing=True)

        elif status in ['compensate (in progress)']:
            assert isinstance(cavity_settings, SingleCavitySettings)
            norm_and_phases, abs_to_rel = \
                _try_this(cavity_settings, w_kin_in, self, generic_rf_param)
        else:
            logging.error(f'{self} {status = } is not allowed.')
            return {}

        phi_rf_abs = phi_bunch_abs * generic_rf_param['n_cell']

        if abs_to_rel:
            norm_and_phases['phi_0_rel'] = phi_0_rel_corresponding_to(
                norm_and_phases['phi_0_abs'], phi_rf_abs)
        else:
            norm_and_phases['phi_0_abs'] = phi_0_abs_corresponding_to(
                norm_and_phases['phi_0_rel'], phi_rf_abs)

        # '|' merges the two dictionaries
        rf_parameters = generic_rf_param | norm_and_phases
        return rf_parameters

    def match_synch_phase(self, w_kin_in: float, phi_s_objective: float,
                          **rf_field_kwargs: dict) -> float:
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
            diff = diff_angle(phi_s_objective, results['cav_params']['phi_s'])
            return diff**2

        res = minimize_scalar(_wrapper_synch, bounds=bounds)
        if not res.success:
            logging.error('Synch phase not found')
        phi_0_rel = res.x

        return phi_0_rel

    def phi_0_rel_matching_this(self, phi_s: float, w_kin_in: float,
                                **rf_parameters: dict) -> float:
        """
        Sweeps phi_0_rel until the cavity synch phase matches phi_s.

        Parameters
        ----------
        phi_s : float
            Synchronous phase to be matched.
        w_kin_in : float
            Kinetic energy at the cavity entrance in MeV.
        rf_parameters : dict
            Holds all rf electric field parameters.

        Return
        ------
        phi_0_rel : float
            The relative cavity entrance phase that leads to a synchronous
            phase of phi_s_objective.
        """
        bounds = (0, 2. * np.pi)

        def _wrapper_synch(phi_0_rel):
            rf_parameters['phi_0_rel'] = phi_0_rel
            rf_parameters['phi_0_abs'] = None
            results = self.calc_transf_mat(w_kin_in, **rf_parameters)
            diff = diff_angle(phi_s, results['cav_params']['phi_s'])
            return diff**2

        res = minimize_scalar(_wrapper_synch, bounds=bounds)
        if not res.success:
            logging.error('Synch phase not found')
        phi_0_rel = res.x
        return phi_0_rel


class Lattice:
    """Used to get the number of elements per lattice."""

    def __init__(self, elem: list[str]) -> None:
        self.n_lattice = int(elem[1])


class Freq:
    """Used to get the frequency of every Section."""

    def __init__(self, elem: list[str]) -> None:
        self.f_rf_mhz = float(elem[1])


class FieldMapPath:
    """Used to get the base path of field maps."""

    def __init__(self, elem: list[str]) -> None:
        self.path = elem[1]


class End:
    """The end of the linac."""

    def __init__(self, elem: list[str]) -> None:
        pass


def _get_from(rf_field: RfField, force_rephasing: bool = False
              ) -> tuple[dict, bool]:
    """Get norms and phases from RfField object."""
    norm_and_phases = {
        'k_e': rf_field.get('k_e'),
        'phi_0_rel': None,
        'phi_0_abs': rf_field.get('phi_0_abs')
    }
    abs_to_rel = True

    # If we are calculating the transfer matrices of the nominal linac and the
    # initial phases are defined in the .dat as relative phases, phi_0_abs is
    # not defined yet
    if norm_and_phases['phi_0_abs'] is None or force_rephasing:
        norm_and_phases['phi_0_rel'] = rf_field.get('phi_0_rel')
        abs_to_rel = False
    return norm_and_phases, abs_to_rel


def _try_this(cavity_settings: SingleCavitySettings, w_kin: float,
              cav: FieldMap, generic_rf_param: dict[str, Any]
              ) -> tuple[dict, bool]:
    """Extract parameters from cavity_parameters."""
    if cavity_settings.phi_s is not None:
        generic_rf_param['k_e'] = cavity_settings.k_e
        phi_0_rel = cav.phi_0_rel_matching_this(cavity_settings.phi_s,
                                                w_kin_in=w_kin,
                                                **generic_rf_param)

        norm_and_phases = {
            'k_e': cavity_settings.k_e,
            'phi_0_rel': phi_0_rel,
        }
        abs_to_rel = False
        return norm_and_phases, abs_to_rel

    norm_and_phases = {key: cavity_settings.get(key)
                       for key in ['k_e', 'phi_0_abs', 'phi_0_rel']}

    norm_and_phases = {key: cavity_settings.get(key)
                       for key in ['k_e', 'phi_0_abs', 'phi_0_rel']}
    abs_to_rel = con.FLAG_PHI_ABS
    return norm_and_phases, abs_to_rel
