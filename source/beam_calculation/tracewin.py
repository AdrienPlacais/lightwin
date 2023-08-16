#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:13:16 2023.

@author: placais

This module holds TraceWin, that inherits from BeamCalculator base class. It
solves the motion of the particles in Envelope or Multipart, in 3D. In contrary
to Envelope1D solver, it is not a real solver but an interface with TraceWin
which must be installed on your machine.

Inherited
---------
    out_folder
    __post_init__()
    _generate_element_to_index_func()

Abstract methods
----------------
    run()
    run_with_this()
    init_solver_parameters()
    _generate_simulation_output()

"""
from dataclasses import dataclass
from typing import Callable
import os
import logging
import subprocess
from functools import partial

import numpy as np

from constants import c
import util.converters as convert

from beam_calculation.output import SimulationOutput
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.single_element_tracewin_parameters import (
    SingleElementTraceWinParameters)

from tracewin_utils.interface import beam_calculator_to_command

from failures.set_of_cavity_settings import SetOfCavitySettings

from core.list_of_elements import ListOfElements
from core.accelerator import Accelerator
from core.particle import ParticleFullTrajectory
from core.beam_parameters import BeamParameters


@dataclass
class TraceWin(BeamCalculator):
    """
    A class to hold a TW simulation and its results.

    The simulation is not necessarily runned.

    Attributes
    ----------
    executable : str
        Path to the TraceWin executable.
    ini_path : str
        Path to the `.ini` TraceWin file.
    base_kwargs : dict[str, str | bool | int | None | float]
        TraceWin optional arguments. Override what is defined in .ini, but
        overriden by arguments from `ListOfElements` and `SimulationOutput`.
    _tracewin_command : list[str] | None, optional
        Attribute to hold the value of the base command to call TraceWin.
    id : str
        Complete name of the solver.
    out_folder : str
        Name of the results folder (not a complete path, just a folder name).
    get_results : Callable
        Function to call to get the output results.
    path_cal : str
        Name of the results folder. Updated at every call of the
        `init_solver_parameters` method, using `Accelerator.accelerator_path`
        and `self.out_folder` attributes.
    dat_file :
        Base name for the `.dat` file. ??

    """

    executable: str
    ini_path: str
    base_kwargs: dict[str, str | int | float | bool | None]
    _tracewin_command: list[str] | None = None

    def __post_init__(self) -> None:
        """Define some other useful methods, init variables."""
        self.ini_path = os.path.abspath(self.ini_path)
        self.id = self.__repr__()
        self.out_folder += "_TraceWin"

        filename = 'tracewin.out'
        if self._is_a_multiparticle_simulation(self.base_kwargs):
            filename = 'partran1.out'
        self.get_results = partial(self._load_results, filename=filename)

        self.path_cal: str
        self.dat_file: str

    def tracewin_command(self, base_path_cal: str, **kwargs
                         ) -> tuple[list[str], str]:
        """
        Define the 'base' command for TraceWin.

        This part of the command is the same for every `ListOfElements` and
        every `Fault`. It sets the TraceWin executable, the `.ini` file.
        It also defines `base_kwargs`, which should be the same for every
        calculation.
        Finally, it sets `path_cal`. But this path is more `ListOfElements`
        dependent...
        `Accelerator.accelerator_path` + `out_folder`
            (+ `fault_optimisation_tmp_folder`)

        """
        kwargs = kwargs.copy()
        for key, val in self.base_kwargs.items():
            if key not in kwargs:
                kwargs[key] = val

        path_cal = os.path.join(base_path_cal, self.out_folder)
        if not os.path.exists(path_cal):
            os.makedirs(path_cal)

        _tracewin_command = beam_calculator_to_command(
            self.executable,
            self.ini_path,
            path_cal,
            **kwargs,
        )
        return _tracewin_command, path_cal

    # TODO what is specific_kwargs for? I should just have a function
    # set_of_cavity_settings_to_kwargs
    def run(self, elts: ListOfElements, **specific_kwargs) -> None:
        """
        Run TraceWin.

        Parameters
        ----------
        elts : ListOfElements
            List of _Elements in which you want the beam propagated.
        **specific_kwargs : dict
            TraceWin optional arguments. Overrides what is defined in
            base_kwargs and .ini.

        """
        return self.run_with_this(set_of_cavity_settings=None, elts=elts,
                                  **specific_kwargs)

    def run_with_this(self, set_of_cavity_settings: SetOfCavitySettings | None,
                      elts: ListOfElements, **specific_kwargs
                      ) -> SimulationOutput:
        """
        Perform a simulation with new cavity settings.

        Calling it with set_of_cavity_settings = None should be the same as
        calling the plain `run` method.

        Parameters
        ----------
        set_of_cavity_settings : SetOfCavitySettings | None
            Holds the norms and phases of the compensating cavities.
        elts: ListOfElements
            List of elements in which the beam should be propagated.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds energy, phase, transfer matrices (among others) packed into a
            single object.

        """
        if set_of_cavity_settings is not None:
            raise NotImplementedError

        kwargs = specific_kwargs.copy()
        for key, val in self.base_kwargs.items():
            if key not in kwargs:
                kwargs[key] = val

        if specific_kwargs is None:
            specific_kwargs = {}

        command, path_cal = self.tracewin_command(elts.get('out_path',
                                                           to_numpy=False),
                                                  **specific_kwargs)
        command.extend(elts.tracewin_command)
        logging.info(f"Running TW with command {command}...")

        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        process.wait()
        for line in process.stdout:
            print(line)

        simulation_output = self._generate_simulation_output(elts, path_cal)
        return simulation_output

    def init_solver_parameters(self, accelerator: Accelerator) -> None:
        """
        Set the `path_cal` variable.

        We also set the `_tracewin_command` attribute to None, as it must be
        updated when `path_cal` changes.

        """
        self.path_cal = os.path.join(accelerator.get('accelerator_path'),
                                     self.out_folder)
        assert os.path.exists(self.path_cal)

        self._tracewin_command = None

    def _generate_simulation_output(self, elts: ListOfElements, path_cal: str,
                                    ) -> SimulationOutput:
        """Create an object holding all relatable simulation results."""
        results = self.get_results(path_cal=path_cal, post_treat=True)

        results = self._shift_if_not_at_linac_beginning(results, elts)

        self._save_tracewin_meshing_in_elements(elts, results['##'],
                                                results['z(m)'])

        synch_trajectory = ParticleFullTrajectory(
            w_kin=results['w_kin'],
            phi_abs=np.deg2rad(results['phi_abs']),
            synchronous=True)

        # WARNING, different meshing for these files
        elt_number, pos, tm_cumul = self._load_transfer_matrices(path_cal)
        logging.warning("Manually extracting only the z transf mat.")
        tm_cumul = tm_cumul[:, 4:, 4:]

        cavity_parameters = self._create_cavity_parameters(path_cal,
                                                           len(elts))

        element_to_index = self._generate_element_to_index_func(elts)
        beam_parameters = self._create_beam_parameters(element_to_index,
                                                       results)

        simulation_output = SimulationOutput(
            out_folder=self.out_folder,
            z_abs=results['z(m)'],
            synch_trajectory=synch_trajectory,
            cav_params=cavity_parameters,
            r_zz_elt=[],
            rf_fields=[],
            beam_parameters=beam_parameters,
            element_to_index=element_to_index
        )
        simulation_output.z_abs = results['z(m)']

        # FIXME attribute was not declared
        simulation_output.pow_lost = results['Powlost']
        return simulation_output

    def _shift_if_not_at_linac_beginning(self,
                                         results: dict[str, np.ndarray],
                                         elts: ListOfElements
                                         ) -> dict[str, np.ndarray]:
        """
        Shift position and phase if `elts` does not start at linac start.

        TraceWin always starts with `z=0` and `phi_abs=0`, even when we are not
        at the beginning of the linac (sub `.dat`).

        """
        results['z(m)'] += elts.input_particle.z_in
        results['phi_abs'] += elts.input_particle.phi_abs
        return results

    def _save_tracewin_meshing_in_elements(self, elts: ListOfElements,
                                           elt_numbers: np.ndarray,
                                           z_abs: np.ndarray) -> None:
        """Take output files to determine where are evaluated w_kin, etc."""
        elt_numbers = elt_numbers.astype(int)

        for elt_number, elt in enumerate(elts, start=1):
            elt_mesh_indexes = np.where(elt_numbers == elt_number)[0]
            s_in = elt_mesh_indexes[0] - 1
            s_out = elt_mesh_indexes[-1]
            z_element = z_abs[s_in:s_out + 1]

            elt.beam_calc_param[self.id] = SingleElementTraceWinParameters(
                elt.length_m, z_element, s_in, s_out)

    def _is_a_multiparticle_simulation(self, kwargs) -> bool:
        """Tells if you should buy Bitcoins now or wait a few months."""
        if 'partran' in kwargs:
            return kwargs['partran'] == 1
        return os.path.isfile(os.path.join(self.path_cal, 'partran1.out'))

    def _load_results(self, filename: str, path_cal: str,
                      post_treat: bool = True
                      ) -> dict[str, np.ndarray]:
        """
        Get the TraceWin results.

        Parameters
        ----------
        filename : str
            Results file produced by TraceWin.
        post_treat : bool, optional
            To compute complementary data. The default is True.

        Returns
        -------
        results : dict[str, np.ndarray]
            Dictionary containing the raw outputs from TraceWin.
        """
        f_p = os.path.join(path_cal, filename)
        n_lines_header = 9
        results = {}

        with open(f_p, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == 1:
                    __mc2, freq, __z, __i, __npart = line.strip().split()
                if i == n_lines_header:
                    headers = line.strip().split()
                    break
        results['freq'] = float(freq)

        out = np.loadtxt(f_p, skiprows=n_lines_header)
        for i, key in enumerate(headers):
            results[key] = out[:, i]
            logging.debug(f"successfully loaded {f_p}")

        if post_treat:
            return _post_treat(results)
        return results

    def _load_transfer_matrices(self, path_cal: str,
                                filename: str = 'Transfer_matrix1.dat',
                                high_def: bool = False
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the full transfer matrices calculated by TraceWin.

        Parameters
        ----------
        filename : str, optional
            The name of the transfer matrix file produced by TraceWin. The
            default is 'Transfer_matrix1.dat'.
        high_def : bool, optional
            To get the transfer matrices at all the solver step, instead at the
            elements exit only. The default is False. Currently not
            implemented.

        Returns
        -------
        element_number : np.ndarray
            Number of the elements.
        position_in_m : np.ndarray
            Position of the elements.
        transfer_matrix : np.ndarray
            Cumulated transfer matrices of the elements.

        """
        if high_def:
            logging.error("High definition not implemented. Can only import"
                          + "transfer matrices @ element positions.")
            high_def = False

        f_p = os.path.join(path_cal, filename)
        data = None
        element_number, position_in_m, transfer_matrix = [], [], []

        with open(f_p, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i % 7 == 0:
                    # Get element # and position
                    data = line.split()
                    element_number.append(int(data[1]))
                    position_in_m.append(float(data[3]))

                    # Re-initialize data
                    data = []
                    continue

                data.append([float(dat) for dat in line.split()])

                # Save transfer matrix
                if (i + 1) % 7 == 0:
                    transfer_matrix.append(data)
        logging.debug(f"successfully loaded {f_p}")

        element_number = np.array(element_number)
        position_in_m = np.array(position_in_m)
        transfer_matrix = np.array(transfer_matrix)
        return element_number, position_in_m, transfer_matrix

    def _create_cavity_parameters(self,
                                  path_cal: str,
                                  n_elts: int,
                                  filename: str = 'Cav_set_point_res.dat',
                                  ) -> list[dict[str, float] | None]:
        """
        Load and format a dict containing v_cav and phi_s.

        It has the same format as `Envelope1D` solver format.

        Parameters
        ----------
        path_cal : str
            Path to the folder where the cavity parameters file is stored.
        n_elts : int
            Number of elements under study.
        filename : str, optional
            The name of the cavity parameters file produced by TraceWin. The
            default is 'Cav_set_point_res.dat'.

        Returns
        -------
        cavity_param : list[dict[str, float] | None]
            Contains the cavity parameters.

        """
        cavity_parameters = _load_cavity_parameters(path_cal, filename)
        cavity_parameters = _cavity_parameters_uniform_with_envelope1d(
            cavity_parameters,
            n_elts
        )
        return cavity_parameters

    def _create_beam_parameters(self,
                                element_to_index: Callable,
                                results: dict[str, np.ndarray]
                                ) -> BeamParameters:
        """Create the `BeamParameters` object, holding eps, Twiss, etc."""
        multipart = self._is_a_multiparticle_simulation(self.base_kwargs)
        beam_parameters = BeamParameters(gamma_kin=results['gamma'],
                                         element_to_index=element_to_index)
        beam_parameters = _beam_param_uniform_with_envelope1d(beam_parameters,
                                                              results)
        beam_parameters = _add_beam_param_not_supported_by_envelope1d(
            beam_parameters,
            results,
            multipart)
        return beam_parameters


# =============================================================================
# Cavity parameters
# =============================================================================
def _load_cavity_parameters(path_cal: str,
                            filename: str) -> dict[[str], np.ndarray]:
    """
    Get the cavity parameters calculated by TraceWin.

    Parameters
    ----------
    path_cal : str
        Path to the folder where the cavity parameters file is stored.
    filename : str
        The name of the cavity parameters file produced by TraceWin.

    Returns
    -------
    cavity_param : dict[float, np.ndarray]
        Contains the cavity parameters.

    """
    f_p = os.path.join(path_cal, filename)
    n_lines_header = 1

    with open(f_p, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == n_lines_header - 1:
                headers = line.strip().split()
                break

    out = np.loadtxt(f_p, skiprows=n_lines_header)
    cavity_parameters = {key: out[:, i] for i, key in enumerate(headers)}
    logging.debug(f"successfully loaded {f_p}")
    return cavity_parameters


def _cavity_parameters_uniform_with_envelope1d(
    cavity_parameters: dict[str, np.ndarray],
    n_elts: int
) -> list[None | dict[str, float]]:
    """Transform the dict so we have the same format as Envelope1D."""
    cavity_numbers = cavity_parameters['Cav#'].astype(int)
    v_cav, phi_s = [], []
    cavity_idx = 0
    for elt_idx in range(1, n_elts + 1):
        if elt_idx not in cavity_numbers:
            v_cav.append(None), phi_s.append(None)
            continue

        v_cav.append(cavity_parameters['Voltage[MV]'][cavity_idx])
        phi_s.append(np.deg2rad(cavity_parameters['SyncPhase[°]'][cavity_idx]))
        cavity_idx += 1

    compliant_cavity_parameters = {'v_cav_mv': v_cav, 'phi_s': phi_s}
    return compliant_cavity_parameters


# =============================================================================
# BeamParameters
# =============================================================================
def _beam_param_uniform_with_envelope1d(
        beam_parameters: BeamParameters, results: dict[str, np.ndarray],
        multiparticle: bool = False) -> BeamParameters:
    """Manually set longitudinal phase-spaces in BeamParameters object."""
    gamma_kin, beta_kin = beam_parameters.gamma_kin, beam_parameters.beta_kin
    beam_parameters.create_phase_spaces('zdelta', 'z', 'phiw')

    sigma_00, sigma_01 = results['SizeZ']**2, results['szdp']
    eps_normalized = results['ezdp']
    beam_parameters.zdelta.reconstruct_full_sigma_matrix(
        sigma_00,
        sigma_01,
        eps_normalized,
        eps_is_normalized=True,
        gamma_kin=gamma_kin,
        beta_kin=beta_kin
    )
    beam_parameters.zdelta.init_from_sigma(gamma_kin, beta_kin)

    beam_parameters.init_other_phase_spaces_from_zdelta(*('phiw', 'z'))
    return beam_parameters


def _add_beam_param_not_supported_by_envelope1d(
        beam_parameters: BeamParameters, results: dict[str, np.ndarray],
        multiparticle: bool = False) -> BeamParameters:
    """Manually set transverse and 99% phase-spaces."""
    gamma_kin, beta_kin = beam_parameters.gamma_kin, beam_parameters.beta_kin
    beam_parameters.create_phase_spaces('x', 'y', 't')

    sigma_x_00, sigma_x_01 = results['SizeX']**2, results["sxx'"]
    eps_x_normalized = results['ex']
    beam_parameters.x.reconstruct_full_sigma_matrix(sigma_x_00,
                                                    sigma_x_01,
                                                    eps_x_normalized,
                                                    eps_is_normalized=True,
                                                    gamma_kin=gamma_kin,
                                                    beta_kin=beta_kin)
    beam_parameters.x.init_from_sigma(gamma_kin, beta_kin)

    sigma_y_00, sigma_y_01 = results['SizeY']**2, results["syy'"]
    eps_y_normalized = results['ey']
    beam_parameters.y.reconstruct_full_sigma_matrix(sigma_y_00,
                                                    sigma_y_01,
                                                    eps_y_normalized,
                                                    eps_is_normalized=True,
                                                    gamma_kin=gamma_kin,
                                                    beta_kin=beta_kin)
    beam_parameters.y.init_from_sigma(gamma_kin, beta_kin)

    beam_parameters.t.init_from_averaging_x_and_y(
        beam_parameters.x,
        beam_parameters.y
    )

    if not multiparticle:
        return beam_parameters

    eps_phiw99 = results['ep99']
    eps_x99, eps_y99 = results['ex99'], results['ey99']
    beam_parameters.create_phase_spaces('phiw99', 'x99', 'y99')
    beam_parameters.init_99percent_phase_spaces(eps_phiw99, eps_x99, eps_y99)
    return beam_parameters


# Not implemented
def elts_to_dat(elts: ListOfElements) -> str:
    """Create a .dat file from elts."""
    return str(elts)


# warning!! As for now, should be called every time...
def _post_treat(results: dict) -> dict:
    """Compute and store the missing quantities (envelope or multipart)."""
    results['gamma'] = 1. + results['gama-1']
    results['w_kin'] = convert.energy(results['gamma'], "gamma to kin")
    results['beta'] = convert.energy(results['w_kin'], "kin to beta")
    results['lambda'] = c / results['freq'] * 1e-6

    omega = 2. * np. pi * results['freq'] * 1e6
    delta_z = np.diff(results['z(m)'])
    beta = .5 * (results['beta'][1:] + results['beta'][:-1])
    delta_phi = omega * delta_z / (beta * c)

    num = results['beta'].shape[0]
    phi_abs = np.full((num), 0.)
    for i in range(num - 1):
        phi_abs[i + 1] = phi_abs[i] + delta_phi[i]
    results['phi_abs'] = np.rad2deg(phi_abs)

    # Transverse emittance, used in evaluate
    results['et'] = 0.5 * (results['ex'] + results['ey'])

    # Twiss parameters, used in evaluate
    for _eps, size, disp, twi in zip(
            ['ex', 'ey', 'ezdp'],
            ['SizeX', 'SizeY', 'SizeZ'],
            ["sxx'", "syy'", 'szdp'],
            ['twiss_x', 'twiss_y', 'twiss_zdp']):
        eps = results[_eps] / (results['gamma'] * results['beta'])
        alpha = -results[disp] / eps
        beta = results[size]**2 / eps
        if _eps == 'ezdp':
            beta /= 10.
        gamma = (1. + alpha**2) / beta
        results[twi] = np.column_stack((alpha, beta, gamma))

    return results
