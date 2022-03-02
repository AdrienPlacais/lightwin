#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:43:39 2021

@author: placais
"""
import numpy as np
from scipy.interpolate import interp1d
from constants import c

phi0 = {
    True: lambda field: field.phi_0['abs'],
    False: lambda field: field.phi_0['rel'],
    }


class RfField():
    """
    Cos-like RF field.

    Warning, all phases are defined as:
        phi = omega_0_rf * t
    While in the rest of the code and in particular in Particle, it is defined
    as:
        phi = omega_0_bunch * t
    """

    def __init__(self, frequency_mhz, norm=np.NaN, phi_0=None):
        self.f_mhz_rf = frequency_mhz
        self.omega0_rf = 2e6 * np.pi * frequency_mhz
        self.omega_0 = self.omega0_rf  # FIXME
        try:
            self.lambda_rf = 1e-6 * c / frequency_mhz
        except ZeroDivisionError:
            self.lambda_rf = None

        # By default, electric field spatial function is null.
        self.e_spat = lambda x: 0.

        self.norm = norm
        self.phi_0 = {'rel': phi_0, 'abs': None}
        self.n_cell = 2
        self.f_e = 0.
        self.phi_s_deg = np.NaN
        self.v_cav_mv = np.NaN

    def e_func_norm(self, norm, phi_0, x, phi_rf):
        """
        Template of the cos-like rf field (normed).

        Parameters
        ----------
        norm : float
            Norm of the electric field.
        phi_0 : float
            Initial phase of the field in rad.
        x : float
            Position of the particle in m.
        phi_rf : float
            Phase of the field. phi_rf = omega_RF * t, while in most of the
            code it is written as phi = omega_bunch * t.

        Return
        ------
        e : float
            Electric field at position x and time phi.
        """
        return norm * self.e_spat(x) * np.cos(phi_rf + phi_0)

    def e_func(self, x, phi_rf, flag_phi_abs=False):
        """
        Compute the rf electric field.

        We use the norm and initial phase defined as attribute.

        Parameters
        ----------
        x : float
            Position of the particle in m.
        phi_rf : float
            Phase of the field. phi_rf = omega_RF * t, while in most of the
            code it is written as phi = omega_bunch * t.

        Return
        ------
        e : float
            Electric field defined by self at position x and time phi.
        """
        return self.e_func_norm(self.norm, phi0[flag_phi_abs](self), x, phi_rf)

    def de_dt_func_norm(self, norm, phi_0, x, phi_rf, beta):
        """
        Template of time derivative of the cos-like rf field (normed).

        Parameters
        ----------
        norm : float
            Norm of the electric field.
        phi_0 : float
            Initial phase of the field in rad.
        x : float
            Position of the particle in m.
        phi_rf : float
            Phase of the field. phi_rf = omega_RF * t, while in most of the
            code it is written as phi = omega_bunch * t.
        beta : float
            Lorentz speed factor of the particle.

        Return
        ------
        de/dt : float
            Time-derivative of the electric field at position x and time phi.

        """
        factor = norm * self.omega0_rf / (beta * c)
        return factor * self.e_spat(x) * np.sin(phi_rf + phi_0)

    def de_dt_func(self, x, phi_rf, beta, flag_phi_abs=False):
        """
        Compute the time derivative of the rf field.

        We use the norm and initial phase defined as attribute.

        Parameters
        ----------
        x : float
            Position of the particle in m.
        phi_rf : float
            Phase of the field. phi = omega_RF * t, while in most of the
            code it is written as phi = omega_bunch * t.
        beta : float
            Lorentz speed factor of the particle.

        Return
        ------
        de/dt : float
            Time-derivative of the electric field at position x and time phi.
        """
        return self.de_dt_func_norm(self.norm, phi0[flag_phi_abs](self),
                                    x, phi_rf, beta)

    def phi_0_rel_to_abs(self, phi_rf_abs):
        """
        Convert relative phi_0 to absolute.

        By default, TW uses relative phases. In other words, it considers that
        particles always enter in the cavity at phi = 0 rad, and phi_0 is
        defined accordingly. This function recalculates phi_0 so that
        modulo(phi_abs + phi_0_abs, 2pi) = phi_rel + phi_0_rel = phi_0_rel

        All phases in this routine are defined by:
            phi = omega_rf * t

        Parameters
        ----------
        phi_rf_abs : float
            Absolute phase of the particle at the entrance of the cavity.
        """
        # assert self.phi_0['abs'] is None, 'Absolute phi_0 has been ' + \
            # 'calculated already.'
        if self.phi_0['abs'] is None:
            phi_0_abs = self.phi_0['rel'] - phi_rf_abs
            phi_0_abs = np.mod(phi_0_abs, 2. * np.pi)
            self.phi_0['abs'] = phi_0_abs


# =============================================================================
# Helper functions dedicated to electric fields
# =============================================================================
def load_field_map_file(element, rf_field):
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
    extension, import_function = check_geom(element)
    element.field_map_file_name = element.field_map_file_name + extension

    # Load the field map
    n_z, zmax, norm, f_z = import_function(element.field_map_file_name)
    assert abs(zmax - element.length_m) < 1e-6
    assert abs(norm - 1.) < 1e-6, 'Warning, imported electric field ' \
        + 'different from 1. Conflict with electric_field_factor?'

    # Interpolation
    z_cavity_array = np.linspace(0., zmax, n_z + 1)
    rf_field.e_spat = interp1d(z_cavity_array, f_z, bounds_error=False,
                               kind='linear', fill_value=0.,
                               assume_sorted=True)


def check_geom(element):
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
    assert element.geometry >= 0, \
        "Second order off-axis development not implemented."

    field_nature = int(np.log10(element.geometry))
    field_geometry = int(str(element.geometry)[0])

    assert field_nature == 2, "Only RF electric fields implemented."
    assert field_geometry == 1, "Only 1D field implemented."
    assert element.aperture_flag <= 0, \
        "Warning! Space charge compensation maps not implemented."

    extension = ".edz"
    import_function = load_electric_field_1d
    return extension, import_function


def load_electric_field_1d(path):
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
