#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:43:39 2021.

@author: placais

TODO : phi_s_rad_objective should not be used too
"""
import os.path
import logging
from typing import Callable
import cmath
import numpy as np

from util.helper import recursive_items, recursive_getter


def compute_param_cav(integrated_field, status):
    """Compute synchronous phase and accelerating field."""
    if status == 'failed':
        polar_itg = np.array([np.NaN, np.NaN])
    else:
        polar_itg = cmath.polar(integrated_field)
    cav_params = {'v_cav_mv': polar_itg[0],
                  'phi_s': polar_itg[1]}
    return cav_params


class RfField():
    """
    Cos-like RF field.

    Warning, all phases are defined as:
        phi = omega_0_rf * t
    While in the rest of the code and in particular in Particle, it is defined
    as:
        phi = omega_0_bunch * t
    """

    def __init__(self, k_e=np.NaN, absolute_phase_flag=0, phi_0=None):
        # By default, electric field spatial function is null.
        self.e_spat = lambda x: 0.
        self.k_e = k_e

        self.phi_0 = {'phi_0_rel': None,
                      'phi_0_abs': None,
                      'nominal_rel': None,
                      'abs_phase_flag': bool(absolute_phase_flag)}
        if self.phi_0['abs_phase_flag']:
            self.phi_0['phi_0_abs'] = phi_0
        else:
            self.phi_0['phi_0_rel'] = phi_0
            self.phi_0['nominal_rel'] = phi_0

        self.cav_params = {'v_cav_mv': np.NaN,
                           'phi_s': np.NaN}
        self.phi_s_rad_objective = None

        # Initialized later as it depends on the Section the cavity is in
        self.omega0_rf, self.n_cell, self.n_z = None, None, None

    def has(self, key):
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys, to_deg=False, **kwargs):
        """Shorthand to get attributes."""
        val = {}
        for key in keys:
            val[key] = []

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)

            if val[key] is not None and to_deg and 'phi' in key:
                val[key] = np.rad2deg(val[key])

        # Convert to list
        out = [val[key] for key in keys]

        if len(out) == 1:
            return out[0]
        # implicit else:
        return tuple(out)

    def init_freq_ncell(self, f_mhz, n_cell):
        """Initialize the frequency and the number of cells."""
        self.omega0_rf = 2e6 * np.pi * f_mhz
        self.n_cell = n_cell

    def rephase_cavity(self, phi_rf_abs):
        """
        Rephase the cavity.

        In other words, we want the particle to enter with the same relative
        phase as in the nominal linac.
        """
        assert self.phi_0['nominal_rel'] is not None
        self.phi_0['phi_0_rel'] = self.phi_0['nominal_rel']
        self.phi_0['phi_0_abs'] = np.mod(
            self.phi_0['nominal_rel'] - phi_rf_abs, 2. * np.pi)


# =============================================================================
# Helper functions dedicated to electric fields
# =============================================================================
def load_field_map_file(cav) -> tuple[Callable[[float | np.ndarray],
                                               float | np.ndarray],
                                                int]:
    """
    Select the field map file and call the proper loading function.

    Warning, filename is directly extracted from the .dat file used by
    TraceWin. Thus, the relative filepath may be misunderstood by this
    script.
    Also check that the extension of the file is .edz, or manually change
    this function.
    Finally, only 1D electric field map are implemented.
    """
    # FIXME
    cav.field_map_file_name += ".edz"
    assert is_loadable(cav.field_map_file_name, cav.geometry,
                       cav.aperture_flag), \
            f"Error preparing {cav}'s field map."

    _, extension = os.path.splitext(cav.field_map_file_name)
    import_function = FIELD_MAP_LOADERS[extension]

    n_z, zmax, norm, f_z = import_function(cav.field_map_file_name)
    assert is_a_valid_electric_field(n_z, zmax, norm, f_z, cav.length_m), \
            f"Error loading {cav}'s field map."

    z_cavity_array = np.linspace(0., zmax, n_z + 1) / norm

    def e_spat(pos: float | np.ndarray) -> float | np.ndarray:
        return np.interp(x=pos, xp=z_cavity_array, fp=f_z, left=0., right=0.)

    return e_spat, n_z


def is_loadable(field_map_file_name: str, geometry: int, aperture_flag: int
               ) -> bool:
    """Assert that the options for the FIELD_MAP in the .dat are ok."""
    _, extension = os.path.splitext(field_map_file_name)
    if extension not in FIELD_MAP_LOADERS:
        logging.error(f"Field map file extension is {extension}, "
                      + f"while only {FIELD_MAP_LOADERS.keys()} are "
                      + "implemented.")
        return False

    if geometry < 0:
        logging.error("Second order off-axis development not implemented.")
        return False

    field_nature = int(np.log10(geometry))
    if field_nature != 2:
        logging.error("Only RF electric fields implemented.")
        return False

    field_geometry = int(str(geometry)[0])
    if field_geometry != 1:
        logging.error("Only 1D field implemented.")
        return False

    if aperture_flag > 0:
        logging.warning("Space charge compensation maps not implemented.")

    return True


def _load_electric_field_1d(path: str) -> tuple[int, float, float, np.ndarray]:
    """
    Load a 1D electric field (.edz extension).

    Parameters
    ----------
    path : string
        The path to the .edz file to load.

    Returns
    -------
    n_z : int
        Number of steps in the array.
    zmax : float
        z position of the filemap end.
    norm : float
        Electric field normalisation factor. It is different from k_e (6th
        argument of the FIELD_MAP command). Electric fields are normalised by
        k_e/norm, hence norm should be unity by default.
    f_z : np.ndarray
        Array of electric field in MV/m.

    """
    f_z = []

    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == 0:
                line_splitted = line.split(' ')

                # Sometimes the separator is a tab and not a space:
                if len(line_splitted) < 2:
                    line_splitted = line.split('\t')

                n_z = int(line_splitted[0])
                # Sometimes there are several spaces or tabs between numbers
                zmax = float(line_splitted[-1])
                continue

            if i == 1:
                norm = float(line)
                continue

            f_z.append(float(line))

    return n_z, zmax, norm, np.array(f_z)

def is_a_valid_electric_field(n_z: int, zmax: float, norm: float,
                              f_z: np.ndarray, cavity_length: float) -> bool:
    """Assert that the electric field that we loaded is valid."""
    if f_z.shape[0] != n_z + 1:
        logging.error(f"The electric field file should have {n_z + 1} "
                      + f"lines, but it is {f_z.shape[0]} lines long. ")
        return False

    tolerance = 1e-6
    if abs(zmax - cavity_length) > tolerance:
        logging.error(f"Mismatch between the length of the field map {zmax = }"
                      + f" and {cavity_length = }.")
        return False

    if abs(norm - 1.) > tolerance:
        logging.warning("Field map scaling factor (second line of the file) "
                        " is different from unity. It may enter in conflict "
                        + "with k_e (6th argument of FIELD_MAP in the .dat).")
    return True

def convert_phi_0(phi_rf_abs, abs_to_rel, rf_field_dict):
    """
    Calculate the missing phi_0 (relative or absolute).

    By default, TW uses relative phases. In other words, it considers that
    particles always enter in the cavity at phi = 0 rad, and phi_0 is
    defined accordingly. This routine recalculates phi_0 so that
    modulo(phi_abs + phi_0_abs, 2pi) = phi_rel + phi_0_rel = phi_0_rel

    All phases in this routine are defined by:
        phi = omega_rf * t

    Parameters
    ----------
    phi_rf_abs : float
        Absolute phase of the particle at the entrance of the cavity.
    abs_to_rel : bool
        True if you want to convert absolute into relative,
        False if you want to convert relative into absolute.
    """
    try:
        phi_0_abs = rf_field_dict['phi_0_abs']
    except KeyError:
        print(rf_field_dict)
    phi_0_rel = rf_field_dict['phi_0_rel']
    if abs_to_rel:
        assert phi_0_abs is not None
        phi_0_rel = np.mod(phi_0_abs + phi_rf_abs, 2. * np.pi)
    else:
        assert phi_0_rel is not None
        phi_0_abs = np.mod(phi_0_rel - phi_rf_abs, 2. * np.pi)
    return phi_0_rel, phi_0_abs


FIELD_MAP_LOADERS = {
    ".edz": _load_electric_field_1d
}
