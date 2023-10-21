#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we define :class:`TraceWin`, to call it from command line.

It inherits from :class:`.BeamCalculator` base class.  It solves the motion of
the particles in envelope or multipart, in 3D. In contrary to
:class:`.Envelope1D` solver, it is not a real solver but an interface with
``TraceWin`` which must be installed on your machine.

.. warning::
    For now, :class:`TraceWin` behavior with relative phases is undetermined.
    You should ensure that you are working with *absolute* phases, i.e. that
    last argument of ``FIELD_MAP`` commands is ``1``.
    You can run a simulation with :class:`.Envelope1D` solver and
    ``flag_phi_abs= True``. The ``.dat`` file created in the ``000001_ref``
    folder should be the original ``.dat`` but converted to absolute phases.

.. todo::
    Already written elsewhere in the code, but a script to convert ``.dat``
    between the different phases would be good.

.. todo::
    Allow TW to work with relative phases. Will have to handle ``rf_fields``
    too.

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
from beam_calculation.tracewin.single_element_tracewin_parameters import (
    SingleElementTraceWinParameters)

from tracewin_utils.interface import beam_calculator_to_command
from tracewin_utils import load

from failures.set_of_cavity_settings import SetOfCavitySettings

from core.elements.field_map import FieldMap
from core.list_of_elements.list_of_elements import ListOfElements
from core.accelerator import Accelerator
from core.particle import ParticleFullTrajectory, ParticleInitialState
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
        Path to the ``.ini`` TraceWin file.
    base_kwargs : dict[str, str | bool | int | None | float]
        TraceWin optional arguments. Override what is defined in ``.ini``, but
        overriden by arguments from :class:`ListOfElements` and
        :class:`SimulationOutput`.
    _tracewin_command : list[str] | None, optional
        Attribute to hold the value of the base command to call TraceWin.
    id : str
        Complete name of the solver.
    out_folder : str
        Name of the results folder (not a complete path, just a folder name).
    load_results : Callable
        Function to call to get the output results.
    path_cal : str
        Name of the results folder. Updated at every call of the
        :func:`init_solver_parameters` method, using
        ``Accelerator.accelerator_path`` and ``self.out_folder`` attributes.
    dat_file :
        Base name for the ``.dat`` file. ??

    """

    executable: str
    ini_path: str
    base_kwargs: dict[str, str | int | float | bool | None]

    def __post_init__(self) -> None:
        """Define some other useful methods, init variables."""
        logging.warning("TraceWin solver currently cannot work with relative "
                        "phases (last arg of FIELD_MAP should be 1). You "
                        "should check this, because I will not.")
        self.ini_path = os.path.abspath(self.ini_path)
        self.id = self.__repr__()
        self.out_folder += "_TraceWin"

        filename = 'tracewin.out'
        if self.is_a_multiparticle_simulation:
            filename = 'partran1.out'
        self.load_results = partial(_load_results_generic, filename=filename)
        self._filename = filename

        self.path_cal: str
        self.dat_file: str
        self._tracewin_command: list[str] | None = None

    def _tracewin_base_command(self, base_path_cal: str, **kwargs
                               ) -> tuple[list[str], str]:
        """
        Define the 'base' command for TraceWin.

        This part of the command is the same for every :class:`ListOfElements`
        and every :class:`Fault`. It sets the TraceWin executable, the ``.ini``
        file.  It also defines ``base_kwargs``, which should be the same for
        every calculation.
        Finally, it sets ``path_cal``.
        But this path is more :class:`ListOfElements`
        dependent...
        ``Accelerator.accelerator_path`` + ``out_folder``
            (+ ``fault_optimisation_tmp_folder``)

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

    def _tracewin_full_command(
        self,
        elts: ListOfElements,
        set_of_cavity_settings: SetOfCavitySettings | None,
        **kwargs,
    ) -> tuple[list[str], str]:
        """
        Set the full TraceWin command.

        It contains the 'base' command, which includes every argument that is
        common to every calculation with this :class:`BeamCalculator`: path to
        ``.ini`` file, to executable...

        It contains the :class:`ListOfElements` command: path to the ``.dat``
        file, initial energy and beam properties.

        It can contain some :class:`SetOfCavitySettings` commands: ``ele``
        arguments to modify some cavities tuning.

        """
        out_path = elts.get('out_path', to_numpy=False)
        command, path_cal = self._tracewin_base_command(out_path, **kwargs)
        command.extend(elts.tracewin_command)
        if set_of_cavity_settings is not None:
            command.extend(set_of_cavity_settings.tracewin_command(
                delta_phi_bunch=elts.input_particle.phi_abs
            ))
        return command, path_cal

    # TODO what is specific_kwargs for? I should just have a function
    # set_of_cavity_settings_to_kwargs
    def run(self, elts: ListOfElements, **specific_kwargs) -> None:
        """
        Run TraceWin.

        Parameters
        ----------
        elts : ListOfElements
        List of :class:`Element` s in which you want the beam propagated.
        **specific_kwargs : dict
            ``TraceWin`` optional arguments. Overrides what is defined in
            ``base_kwargs`` and ``.ini``.

        """
        return self.run_with_this(set_of_cavity_settings=None, elts=elts,
                                  **specific_kwargs)

    def run_with_this(self, set_of_cavity_settings: SetOfCavitySettings | None,
                      elts: ListOfElements,
                      **specific_kwargs
                      ) -> SimulationOutput:
        """
        Perform a simulation with new cavity settings.

        Calling it with ``set_of_cavity_settings = None`` is the same as
        calling the plain :func:`run` method.

        Parameters
        ----------
        set_of_cavity_settings : SetOfCavitySettings | None
            Holds the norms and phases of the compensating cavities.
        elts : ListOfElements
            List of elements in which the beam should be propagated.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds energy, phase, transfer matrices (among others) packed into a
            single object.

        """
        if specific_kwargs not in (None, {}):
            logging.critical(f"{specific_kwargs = }: deprecated.")

        if specific_kwargs is None:
            specific_kwargs = {}

        rf_fields = []
        for elt in elts:
            if isinstance(set_of_cavity_settings, SetOfCavitySettings):
                cavity_settings = set_of_cavity_settings.get(elt)
                if cavity_settings is not None:
                    rf_fields.append({'k_e': cavity_settings.k_e,
                                      'phi_0_abs': cavity_settings.phi_0_abs,
                                      'phi_0_rel': cavity_settings.phi_0_rel})
                    continue

            if isinstance(elt, FieldMap):
                rf_fields.append(
                    {'k_e': elt.acc_field.k_e,
                     'phi_0_abs': elt.acc_field.phi_0['phi_0_abs'],
                     'phi_0_rel': elt.acc_field.phi_0['phi_0_rel']})
                continue
            rf_fields.append({})

        command, path_cal = self._tracewin_full_command(
            elts,
            set_of_cavity_settings,
            **specific_kwargs)
        is_a_fit = set_of_cavity_settings is not None
        exception = _run_in_bash(command, output_command=not is_a_fit)

        simulation_output = self._generate_simulation_output(elts, path_cal,
                                                             rf_fields,
                                                             exception)
        return simulation_output

    def post_optimisation_run_with_this(
        self,
        optimized_cavity_settings: SetOfCavitySettings,
        full_elts: ListOfElements,
        **specific_kwargs
    ) -> SimulationOutput:
        """
        Run TraceWin with optimized cavity settings.

        After the optimisation, we want to re-run TraceWin with the new
        settings. However, we need to tell it that the linac is bigger than
        during the optimisation. Concretely, it means:
            * rephasing the cavities in the compensation zone
            * updating the ``index`` ``n`` of the cavities in the ``ele[n][v]``
              command.

        Note that at this point, the ``.dat`` has not been updated yet.

        Parameters
        ----------
        optimized_cavity_settings : SetOfCavitySettings
            Optimized parameters.
        full_elts : ListOfElements
            Contains the full linac.

        Returns
        -------
        simulation_output : SimulationOutput
            Necessary information on the run.

        """
        optimized_cavity_settings.re_set_elements_index_to_absolute_value()
        full_elts.store_settings_in_dat(full_elts.files['dat_filepath'])

        simulation_output = self.run_with_this(optimized_cavity_settings,
                                               full_elts,
                                               **specific_kwargs)
        return simulation_output

    def init_solver_parameters(self, accelerator: Accelerator) -> None:
        """
        Set the ``path_cal`` variable.

        We also set the ``_tracewin_command`` attribute to None, as it must be
        updated when ``path_cal`` changes.

        """
        self.path_cal = os.path.join(accelerator.get('accelerator_path'),
                                     self.out_folder)
        assert os.path.exists(self.path_cal)

        self._tracewin_command = None

    def _generate_simulation_output(
        self,
        elts: ListOfElements,
        path_cal: str,
        rf_fields: list[dict[str, float | None]],
        exception: bool
    ) -> SimulationOutput:
        """
        Create an object holding all relatable simulation results.

        Parameters
        ----------
        elts : ListOfElements
            Contains all elements or only a fraction or all the elements.
        path_cal : str
            Path to results folder.
        rf_fields : list[dict[str, float | None]]
            List of dicts which are empty if corresponding element has no rf
            field, and has keys ``'k_e'``, ``'phi_0_abs'`` and ``'phi_0_rel'``
            otherwise.
        exception : bool
            Indicates if the run was unsuccessful or not.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds all relatable data in a consistent way between the different
            :class:`BeamCalculator` objects.

        """
        if exception:
            filepath = os.path.join(path_cal, self._filename)
            _remove_incomplete_line(filepath)
            _add_dummy_data(filepath, elts)

        results = self._create_main_results_dictionary(path_cal,
                                                       elts.input_particle)

        if exception:
            results = _remove_invalid_values(results)

        self._save_tracewin_meshing_in_elements(elts,
                                                results['##'],
                                                results['z(m)'])

        synch_trajectory = ParticleFullTrajectory(w_kin=results['w_kin'],
                                                  phi_abs=results['phi_abs'],
                                                  synchronous=True)

        cavity_parameters = self._create_cavity_parameters(path_cal,
                                                           len(elts))
        rf_fields = self._complete_list_of_rf_fields(rf_fields,
                                                     cavity_parameters)

        element_to_index = self._generate_element_to_index_func(elts)
        beam_parameters = self._create_beam_parameters(element_to_index,
                                                       results)

        simulation_output = SimulationOutput(
            out_folder=self.out_folder,
            is_multiparticle=self.is_a_multiparticle_simulation,
            is_3d=self.is_a_3d_simulation,
            z_abs=results['z(m)'],
            synch_trajectory=synch_trajectory,
            cav_params=cavity_parameters,
            r_zz_elt=[],
            rf_fields=rf_fields,
            beam_parameters=beam_parameters,
            element_to_index=element_to_index
        )
        simulation_output.z_abs = results['z(m)']

        # FIXME attribute was not declared
        simulation_output.pow_lost = results['Powlost']

        # FIXME another one
        _, _, simulation_output.transfer_matrix = _load_transfer_matrices(
            path_cal)

        return simulation_output

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

    @property
    def is_a_multiparticle_simulation(self) -> bool:
        """Tell if you should buy Bitcoins now or wait a few months."""
        if 'partran' in self.base_kwargs:
            return self.base_kwargs['partran'] == 1
        return os.path.isfile(os.path.join(self.path_cal, 'partran1.out'))

    @property
    def is_a_3d_simulation(self) -> bool:
        """Tell if the simulation is in 3D."""
        return True

    def _create_main_results_dictionary(self,
                                        path_cal: str,
                                        input_particle: ParticleInitialState
                                        ) -> dict[str, np.ndarray]:
        """Load the TraceWin results, compute common interest quantities."""
        results = self.load_results(path_cal=path_cal)
        results = _set_energy_related_results(results)
        results = _set_phase_related_results(results,
                                             z_in=input_particle.z_in,
                                             phi_in=input_particle.phi_abs)
        return results

    def _create_cavity_parameters(self,
                                  path_cal: str,
                                  n_elts: int,
                                  filename: str = 'Cav_set_point_res.dat',
                                  ) -> dict[str, list[float | None]]:
        """
        Load and format a dict containing v_cav and phi_s.

        It has the same format as :class:`Envelope1D` solver format.

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
        cavity_param : dict[str, list[float | None]]
            Contains the cavity parameters. Keys are ``'v_cav_mv'`` and
            ``'phi_s'``.

        """
        cavity_parameters = _load_cavity_parameters(path_cal, filename)
        cavity_parameters = _cavity_parameters_uniform_with_envelope1d(
            cavity_parameters,
            n_elts
        )
        return cavity_parameters

    def _complete_list_of_rf_fields(
        self,
        rf_fields: list[dict[str, float | None]],
        cavity_parameters: dict[str, list[float | None]]
    ) -> list[dict[float | None]]:
        """Create a list with rf field properties, as :class:`Envelope1D`."""
        for i, (v_cav_mv, phi_s, phi_0) in enumerate(
                zip(cavity_parameters['v_cav_mv'],
                    cavity_parameters['phi_s'],
                    cavity_parameters['phi_0'])):
            if v_cav_mv is None:
                continue

            # patch for superpose_map
            if 'k_e' not in rf_fields[i]:
                cavity_parameters['v_cav_mv'][i] = None
                cavity_parameters['phi_s'][i] = None
                cavity_parameters['phi_0'][i] = None
                continue

            if rf_fields[i]['k_e'] < 1e-10:
                continue
            rf_fields[i]['v_cav_mv'] = v_cav_mv
            rf_fields[i]['phi_s'] = phi_s
            rf_fields[i]['phi_0_abs'] = phi_0

        return rf_fields

    def _create_beam_parameters(self,
                                element_to_index: Callable,
                                results: dict[str, np.ndarray]
                                ) -> BeamParameters:
        """Create the :class:`BeamParameters` object, holding eps, Twiss..."""
        multipart = self.is_a_multiparticle_simulation
        beam_parameters = BeamParameters(z_abs=results['z(m)'],
                                         gamma_kin=results['gamma'],
                                         element_to_index=element_to_index)
        beam_parameters = _beam_param_uniform_with_envelope1d(beam_parameters,
                                                              results)
        beam_parameters = _add_beam_param_not_supported_by_envelope1d(
            beam_parameters,
            results,
            multipart)
        return beam_parameters


# =============================================================================
# Main `results` dictionary
# =============================================================================
def _load_results_generic(filename: str,
                          path_cal: str) -> dict[str, np.ndarray]:
    """
    Load the TraceWin results.

    This function is not called directly. Instead, every instance of
    :class:`TraceWin` object has a :func:`load_results` method which calls this
    function with a default ``filename`` argument.
    The value of ``filename`` depends on the TraceWin simulation that was run:
    multiparticle or envelope.

    Parameters
    ----------
    filename : str
        Results file produced by TraceWin.
    path_cal : str
        Folder where the results file is located.

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
    return results


def _remove_invalid_values(results: dict[str, np.ndarray]
                           ) -> dict[str, np.ndarray]:
    """Remove invalid values that appear when ``exception`` is True."""
    results['SizeX'] = _0_to_NaN(results['SizeX'])
    results['SizeY'] = _0_to_NaN(results['SizeY'])
    results['SizeZ'] = _0_to_NaN(results['SizeZ'])
    return results


def _0_to_NaN(data: np.ndarray) -> np.ndarray:
    """Replace 0 by np.NaN in given array."""
    data[np.where(data == 0.)] = np.NaN
    return data


def _set_energy_related_results(results: dict[str, np.ndarray]
                                ) -> dict[str, np.ndarray]:
    """
    Compute the energy from ``gama-1`` column.

    Parameters
    ----------
    results : dict[str, np.ndarray]
        Dictionary holding the TraceWin results.

    Returns
    -------
    results : dict[str, np.ndarray]
        Same as input, but with ``gamma``, ``w_kin``, ``beta`` keys defined.

    """
    results['gamma'] = 1. + results['gama-1']
    results['w_kin'] = convert.energy(results['gamma'], "gamma to kin")
    results['beta'] = convert.energy(results['w_kin'], "kin to beta")
    return results


def _set_phase_related_results(results: dict[str, np.ndarray],
                               z_in: float,
                               phi_in: float,
                               ) -> dict[str, np.ndarray]:
    """
    Compute the phases, pos, frequencies.

    Also shift position and phase if :class:`ListOfElements` under study does
    not start at the beginning of the linac.

    TraceWin always starts with ``z=0`` and ``phi_abs=0``, even when we are not
    at the beginning of the linac (sub ``.dat``).

    Parameters
    ----------
    results : dict[str, np.ndarray]
        Dictionary holding the TraceWin results.
    z_in : float
        Absolute position in the linac of the beginning of the linac portion
        under study (can be 0.).
    phi_in : float
        Absolute phase of the synch particle at the beginning of the linac
        portion under study (can be 0.).

    Returns
    -------
    results : dict[str, np.ndarray]
        Same as input, but with ``lambda`` and ``phi_abs`` keys defined.
        ``phi_abs``
        and ``z(m)`` keys are modified in order to be 0. at the beginning of
        the linac, not at the beginning of the :class:`ListOfElements` under
        study.

    """
    results['z(m)'] += z_in
    results['lambda'] = c / results['freq'] * 1e-6

    omega = 2. * np. pi * results['freq'] * 1e6
    delta_z = np.diff(results['z(m)'])
    beta = .5 * (results['beta'][1:] + results['beta'][:-1])
    delta_phi = omega * delta_z / (beta * c)

    num = results['beta'].shape[0]
    phi_abs = np.full((num), phi_in)
    for i in range(num - 1):
        phi_abs[i + 1] = phi_abs[i] + delta_phi[i]
    results['phi_abs'] = phi_abs

    return results


# =============================================================================
# Cavity parameters
# =============================================================================
def _load_cavity_parameters(path_cal: str,
                            filename: str) -> dict[str, np.ndarray]:
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
    v_cav, phi_s, phi_0 = [], [], []
    cavity_idx = 0
    for elt_idx in range(1, n_elts + 1):
        if elt_idx not in cavity_numbers:
            v_cav.append(None), phi_s.append(None), phi_0.append(None)
            continue

        v_cav.append(cavity_parameters['Voltage[MV]'][cavity_idx])
        phi_s.append(np.deg2rad(cavity_parameters['SyncPhase[°]'][cavity_idx]))
        phi_0.append(np.deg2rad(cavity_parameters['RF_phase[°]'][cavity_idx]))

        cavity_idx += 1

    compliant_cavity_parameters = {'v_cav_mv': v_cav, 'phi_s': phi_s,
                                   'phi_0': phi_0}
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


# =============================================================================
# Transfer matrix file
# =============================================================================
def _load_transfer_matrices(path_cal: str,
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
    element_numbers : np.ndarray
        Number of the elements.
    position_in_m : np.ndarray
        Position of the elements.
    transfer_matrices : np.ndarray
        Cumulated transfer matrices of the elements.

    """
    if high_def:
        logging.error("High definition not implemented. Can only import"
                      "transfer matrices @ element positions.")
        high_def = False

    path = os.path.join(path_cal, filename)
    elements_numbers, position_in_m, transfer_matrices = \
        load.transfer_matrices(path)
    logging.debug(f"successfully loaded {path}")
    return elements_numbers, position_in_m, transfer_matrices


# =============================================================================
# Handle errors
# =============================================================================
def _remove_incomplete_line(filepath: str) -> None:
    """Remove incomplete line from ``.out`` file."""
    n_lines_header = 9
    i_last_valid = -1
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if i < n_lines_header:
            continue

        if i == n_lines_header:
            n_columns = len(line.split())

        if len(line.split()) != n_columns:
            i_last_valid = i
            break

    if i_last_valid == -1:
        return
    logging.warning(f"Not enough columns in `.out` after line {i_last_valid}. "
                    "Removing all lines after this one...")
    with open(filepath, 'w', encoding='utf-8') as file:
        for i, line in enumerate(lines):
            if i >= i_last_valid:
                break
            file.write(line)


def _add_dummy_data(filepath: str, elts: ListOfElements) -> None:
    """
    Add dummy data at the end of the ``.out`` to reach end of linac.

    We also round the column 'z', to avoid a too big mismatch between the z
    column and what we should have.

    """
    with open(filepath, 'r+', encoding='utf-8') as file:
        for line in file:
            pass
        last_idx_in_file = int(line.split()[0])
        last_element_in_file = elts[last_idx_in_file - 1]

        if last_element_in_file is not elts[-1]:
            logging.warning("Incomplete `.out` file. Trying to complete with "
                            "dummy data...")
            elts_to_add = elts[last_idx_in_file:]
            last_pos = np.round(float(line.split()[1]), 4)
            for i, elt in enumerate(elts_to_add, start=last_idx_in_file + 1):
                last_pos += elt.get('length_m', to_numpy=False)
                new_line = line.split()
                new_line[0] = str(i)
                new_line[1] = str(last_pos)
                new_line = ' '.join(new_line) + '\n'
                file.write(new_line)


# =============================================================================
# Bash
# =============================================================================
def _run_in_bash(command: list[str],
                 output_command: bool = True,
                 output_error: bool = False) -> bool:
    """Run given command in bash."""
    output = "\n\t".join(command)
    if output_command:
        logging.info(f"Running command:\n\t{output}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    process.wait()

    exception = False
    for line in process.stdout:
        if output_error:
            print(line)
        exception = True

    if exception and output_error:
        logging.warning("A message was returned when executing following "
                        f"command:\n\t{output}")
    return exception
