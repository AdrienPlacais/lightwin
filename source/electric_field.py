#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:43:39 2021.

@author: placais

TODO : remove RfField.norm that should not be used anymore
TODO : phi_s_rad_objective should not be used too
"""
import cmath
import numpy as np


def compute_param_cav(integrated_field, status):
    """Compute synchronous phase and accelerating field."""
    if status == 'failed':
        polar_itg = np.array([np.NaN, np.NaN])
    else:
        polar_itg = cmath.polar(integrated_field)
    cav_params = {'v_cav_mv': polar_itg[0],
                  'phi_s_deg': np.rad2deg(polar_itg[1]),
                  'phi_s_rad': polar_itg[1]}
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

    def __init__(self, norm=np.NaN, absolute_phase_flag=0, phi_0=None):
        # By default, electric field spatial function is null.
        self.e_spat = lambda x: 0.

        self.norm = norm
        self.k_e = norm

        self.phi_0 = {'rel': None,
                      'abs': None,
                      'nominal_rel': None,
                      'abs_phase_flag': bool(absolute_phase_flag)}
        if self.phi_0['abs_phase_flag']:
            self.phi_0['abs'] = phi_0
        else:
            self.phi_0['rel'] = phi_0
            self.phi_0['nominal_rel'] = phi_0

        self.cav_params = {'v_cav_mv': np.NaN,
                           'phi_s_deg': np.NaN,
                           'phi_s_rad': np.NaN}
        self.phi_s_rad_objective = None

        # Initialized later as it depends on the Section the cavity is in
        self.omega0_rf, self.n_cell, self.n_z = None, None, None

    def init_freq_ncell(self, f_mhz, n_cell):
        """Initialize the frequency and the number of cells."""
        self.omega0_rf = 2e6 * np.pi * f_mhz
        self.n_cell = n_cell

    def transfer_data(self, **kwargs):
        """Assign the norm, phi_0 after a fit."""
        self.k_e = kwargs['k_e']
        self.phi_0['rel'] = kwargs['phi_0_rel']
        self.phi_0['abs'] = kwargs['phi_0_abs']

    def rephase_cavity(self, phi_rf_abs):
        """
        Rephase the cavity.

        In other words, we want the particle to enter with the same relative
        phase as in the nominal linac.
        """
        assert self.phi_0['nominal_rel'] is not None
        self.phi_0['rel'] = self.phi_0['nominal_rel']
        self.phi_0['abs'] = np.mod(self.phi_0['nominal_rel'] - phi_rf_abs,
                                   2. * np.pi)


# =============================================================================
# Helper functions dedicated to electric fields
# =============================================================================
def load_field_map_file(elt):
    """
    Select the field map file and call the proper loading function.

    Warning, filename is directly extracted from the .dat file used by
    TraceWin. Thus, the relative filepath may be misunderstood by this
    script.
    Also check that the extension of the file is .edz, or manually change
    this function.
    Finally, only 1D electric field map are implemented.
    """
    # Check nature and geometry of the field map, and select proper file
    # extension and import function
    extension, import_function = _check_geom(elt)
    elt.field_map_file_name = elt.field_map_file_name + extension

    # Load the field map
    n_z, zmax, norm, f_z = import_function(elt.field_map_file_name)
    assert abs(zmax - elt.length_m) < 1e-6
    assert abs(norm - 1.) < 1e-6, 'Warning, imported electric field ' \
        + 'different from 1. Conflict with electric_field_factor?'

    # Interpolation
    z_cavity_array = np.linspace(0., zmax, n_z + 1)

    def e_spat(pos):
        return np.interp(x=pos, xp=z_cavity_array, fp=f_z, left=0., right=0.)
    return e_spat, n_z


def _check_geom(elt):
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
    assert elt.geometry >= 0, \
        "Second order off-axis development not implemented."

    field_nature = int(np.log10(elt.geometry))
    field_geometry = int(str(elt.geometry)[0])

    assert field_nature == 2, "Only RF electric fields implemented."
    assert field_geometry == 1, "Only 1D field implemented."
    assert elt.aperture_flag <= 0, \
        "Warning! Space charge compensation maps not implemented."

    extension = ".edz"
    import_function = _load_electric_field_1d
    return extension, import_function


def _load_electric_field_1d(path):
    """
    Load a 1D electric field (.edz extension).

    Parameters
    ----------
    path: string
        The path to the .edz file to load.

    Returns
    -------
    f_z: np.array
        Array of electric field in MV/m.
    zmax: float
        z position of the filemap end.
    norm: float
        norm of the electric field.

    Currently not returned
    ----------------------
    n_z: int
        Number of points in the array minus one.
    """
    i = 0
    k = 0

    with open(path) as file:
        for line in file:
            if i == 0:
                line_splitted = line.split(' ')

                # Sometimes the separator is a tab and not a space:
                if len(line_splitted) < 2:
                    line_splitted = line.split('\t')

                n_z = int(line_splitted[0])
                # Sometimes there are several spaces or tabs between numbers
                zmax = float(line_splitted[-1])
                f_z = np.full((n_z + 1), np.NaN)

            elif i == 1:
                norm = float(line)

            else:
                f_z[k] = float(line)
                k += 1

            i += 1

    return n_z, zmax, norm, f_z


def convert_phi_0(phi_rf_abs, abs_to_rel, phi_0_rel=None, phi_0_abs=None):
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
    if abs_to_rel:
        assert phi_0_abs is not None
        phi_0_rel = np.mod(phi_0_abs + phi_rf_abs, 2. * np.pi)
    else:
        assert phi_0_rel is not None
        phi_0_abs = np.mod(phi_0_rel - phi_rf_abs, 2. * np.pi)
    return phi_0_rel, phi_0_abs


def convert_phi_02(phi_rf_abs, abs_to_rel, rf_field_dict):
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
    phi_0_abs = rf_field_dict['phi_0_abs']
    phi_0_rel = rf_field_dict['phi_0_rel']
    if abs_to_rel:
        assert phi_0_abs is not None
        phi_0_rel = np.mod(phi_0_abs + phi_rf_abs, 2. * np.pi)
    else:
        assert phi_0_rel is not None
        phi_0_abs = np.mod(phi_0_rel - phi_rf_abs, 2. * np.pi)
    return phi_0_rel, phi_0_abs
