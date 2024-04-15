#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to easily generate the :class:`.SimulationOutput`."""
import logging
from abc import ABCMeta
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

import util.converters as convert
from beam_calculation.simulation_output.factory import SimulationOutputFactory
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from beam_calculation.tracewin.beam_parameters_factory import (
    BeamParametersFactoryTraceWin,
)
from beam_calculation.tracewin.element_tracewin_parameters_factory import (
    ElementTraceWinParametersFactory,
)
from beam_calculation.tracewin.transfer_matrix_factory import (
    TransferMatrixFactoryTraceWin,
)
from constants import c
from core.list_of_elements.list_of_elements import ListOfElements
from core.particle import ParticleFullTrajectory, ParticleInitialState
from failures.set_of_cavity_settings import SetOfCavitySettings


@dataclass
class SimulationOutputFactoryTraceWin(SimulationOutputFactory):
    """A class for creating simulation outputs for :class:`.TraceWin`."""

    out_folder: Path
    _filename: Path
    beam_calc_parameters_factory: ElementTraceWinParametersFactory

    def __post_init__(self) -> None:
        """Set filepath-related attributes and create factories.

        The created factories are :class:`.TransferMatrixFactory` and
        :class:`.BeamParametersFactory`. The sub-class that is used is declared
        in :meth:`._transfer_matrix_factory_class` and
        :meth:`._beam_parameters_factory_class`.

        """
        self.load_results = partial(
            _load_results_generic, filename=self._filename
        )
        # Factories created in ABC's __post_init__
        return super().__post_init__()

    @property
    def _transfer_matrix_factory_class(self) -> ABCMeta:
        """Give the **class** of the transfer matrix factory."""
        return TransferMatrixFactoryTraceWin

    @property
    def _beam_parameters_factory_class(self) -> ABCMeta:
        """Give the **class** of the beam parameters factory."""
        return BeamParametersFactoryTraceWin

    def run(
        self,
        elts: ListOfElements,
        path_cal: Path,
        rf_fields: list[dict[str, float | None]],
        exception: bool,
        set_of_cavity_settings: SetOfCavitySettings,
    ) -> SimulationOutput:
        """
        Create an object holding all relatable simulation results.

        Parameters
        ----------
        elts : ListOfElements
            Contains all elements or only a fraction or all the elements.
        path_cal : Path
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
            filepath = Path(path_cal, self._filename)
            _remove_incomplete_line(filepath)
            _add_dummy_data(filepath, elts)

        results = self._create_main_results_dictionary(
            path_cal, elts.input_particle
        )

        if exception:
            results = _remove_invalid_values(results)

        self._save_tracewin_meshing_in_elements(
            elts, results["##"], results["z(m)"]
        )

        synch_trajectory = ParticleFullTrajectory(
            w_kin=results["w_kin"],
            phi_abs=results["phi_abs"],
            synchronous=True,
        )

        cavity_parameters = self._create_cavity_parameters(path_cal, len(elts))
        rf_fields = self._complete_list_of_rf_fields(
            rf_fields, cavity_parameters
        )

        element_to_index = self._generate_element_to_index_func(elts)

        transfer_matrix = self.transfer_matrix_factory.run(
            elts.tm_cumul_in, path_cal, element_to_index
        )

        z_abs = results["z(m)"]
        gamma_kin = synch_trajectory.get("gamma")
        beam_parameters = self.beam_parameters_factory.factory_method(
            z_abs, gamma_kin, results, element_to_index
        )

        simulation_output = SimulationOutput(
            out_folder=self.out_folder,
            is_multiparticle=True,  # FIXME
            is_3d=True,
            z_abs=results["z(m)"],
            synch_trajectory=synch_trajectory,
            cav_params=cavity_parameters,
            rf_fields=rf_fields,
            beam_parameters=beam_parameters,
            element_to_index=element_to_index,
            transfer_matrix=transfer_matrix,
            set_of_cavity_settings=set_of_cavity_settings,
        )
        simulation_output.z_abs = results["z(m)"]

        # FIXME attribute was not declared
        simulation_output.pow_lost = results["Powlost"]

        return simulation_output

    def _create_main_results_dictionary(
        self, path_cal: Path, input_particle: ParticleInitialState
    ) -> dict[str, np.ndarray]:
        """Load the TraceWin results, compute common interest quantities."""
        results = self.load_results(path_cal=path_cal)
        results = _set_energy_related_results(results)
        results = _set_phase_related_results(
            results, z_in=input_particle.z_in, phi_in=input_particle.phi_abs
        )
        return results

    # TODO FIXME
    def _save_tracewin_meshing_in_elements(
        self, elts: ListOfElements, elt_numbers: np.ndarray, z_abs: np.ndarray
    ) -> None:
        """Take output files to determine where are evaluated ``w_kin``..."""
        elt_numbers = elt_numbers.astype(int)

        for elt_number, elt in enumerate(elts, start=1):
            elt_mesh_indexes = np.where(elt_numbers == elt_number)[0]
            s_in = elt_mesh_indexes[0] - 1
            s_out = elt_mesh_indexes[-1]
            z_element = z_abs[s_in : s_out + 1]

            elt.beam_calc_param[self._solver_id] = (
                self.beam_calc_parameters_factory.run(
                    elt, z_element, s_in, s_out
                )
            )

    def _create_cavity_parameters(
        self,
        path_cal: Path,
        n_elts: int,
        filename: Path = Path("Cav_set_point_res.dat"),
    ) -> dict[str, list[float | None]]:
        """
        Load and format a dict containing v_cav and phi_s.

        It has the same format as :class:`Envelope1D` solver format.

        Parameters
        ----------
        path_cal : Path
            Path to the folder where the cavity parameters file is stored.
        n_elts : int
            Number of elements under study.
        filename : Path, optional
            The name of the cavity parameters file produced by TraceWin. The
            default is Path('Cav_set_point_res.dat').

        Returns
        -------
        cavity_param : dict[str, list[float | None]]
            Contains the cavity parameters. Keys are ``'v_cav_mv'`` and
            ``'phi_s'``.

        """
        cavity_parameters = _load_cavity_parameters(path_cal, filename)
        cavity_parameters = _cavity_parameters_uniform_with_envelope1d(
            cavity_parameters, n_elts
        )
        return cavity_parameters

    def _complete_list_of_rf_fields(
        self,
        rf_fields: list[dict[str, float | None]],
        cavity_parameters: dict[str, list[float | None]],
    ) -> list[dict[float | None]]:
        """Create a list with rf field properties, as :class:`Envelope1D`."""
        for i, (v_cav_mv, phi_s, phi_0) in enumerate(
            zip(
                cavity_parameters["v_cav_mv"],
                cavity_parameters["phi_s"],
                cavity_parameters["phi_0"],
            )
        ):
            if v_cav_mv is None:
                continue

            # patch for superpose_map
            if "k_e" not in rf_fields[i]:
                cavity_parameters["v_cav_mv"][i] = None
                cavity_parameters["phi_s"][i] = None
                cavity_parameters["phi_0"][i] = None
                continue

            if rf_fields[i]["k_e"] < 1e-10:
                continue
            rf_fields[i]["v_cav_mv"] = v_cav_mv
            rf_fields[i]["phi_s"] = phi_s
            rf_fields[i]["phi_0_abs"] = phi_0

        return rf_fields


# =============================================================================
# Main `results` dictionary
# =============================================================================
def _0_to_NaN(data: np.ndarray) -> np.ndarray:
    """Replace 0 by np.NaN in given array."""
    data[np.where(data == 0.0)] = np.NaN
    return data


def _remove_invalid_values(
    results: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Remove invalid values that appear when ``exception`` is True."""
    results["SizeX"] = _0_to_NaN(results["SizeX"])
    results["SizeY"] = _0_to_NaN(results["SizeY"])
    results["SizeZ"] = _0_to_NaN(results["SizeZ"])
    return results


def _load_results_generic(
    filename: Path, path_cal: Path
) -> dict[str, np.ndarray]:
    """
    Load the TraceWin results.

    This function is not called directly. Instead, every instance of
    :class:`TraceWin` object has a :func:`load_results` method which calls this
    function with a default ``filename`` argument.
    The value of ``filename`` depends on the TraceWin simulation that was run:
    multiparticle or envelope.

    Parameters
    ----------
    filename : Path
        Results file produced by TraceWin.
    path_cal : Path
        Folder where the results file is located.

    Returns
    -------
    results : dict[str, np.ndarray]
        Dictionary containing the raw outputs from TraceWin.

    """
    f_p = Path(path_cal, filename)

    n_lines_header = 9
    results = {}

    with open(f_p, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i == 1:
                __mc2, freq, __z, __i, __npart = line.strip().split()
            if i == n_lines_header:
                headers = line.strip().split()
                break
    results["freq"] = float(freq)

    out = np.loadtxt(f_p, skiprows=n_lines_header)
    for i, key in enumerate(headers):
        results[key] = out[:, i]
        logging.debug(f"successfully loaded {f_p}")
    return results


def _set_energy_related_results(
    results: dict[str, np.ndarray]
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
    results["gamma"] = 1.0 + results["gama-1"]
    results["w_kin"] = convert.energy(results["gamma"], "gamma to kin")
    results["beta"] = convert.energy(results["w_kin"], "kin to beta")
    return results


def _set_phase_related_results(
    results: dict[str, np.ndarray],
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
    results["z(m)"] += z_in
    results["lambda"] = c / results["freq"] * 1e-6

    omega = 2.0 * np.pi * results["freq"] * 1e6
    delta_z = np.diff(results["z(m)"])
    beta = 0.5 * (results["beta"][1:] + results["beta"][:-1])
    delta_phi = omega * delta_z / (beta * c)

    num = results["beta"].shape[0]
    phi_abs = np.full((num), phi_in)
    for i in range(num - 1):
        phi_abs[i + 1] = phi_abs[i] + delta_phi[i]
    results["phi_abs"] = phi_abs

    return results


# =============================================================================
# Handle errors
# =============================================================================
def _remove_incomplete_line(filepath: Path) -> None:
    """
    Remove incomplete line from ``.out`` file.

    .. todo::
        fix possible unbound error for ``n_columns``.

    """
    n_lines_header = 9
    i_last_valid = -1
    with open(filepath, "r", encoding="utf-8") as file:
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
    logging.warning(
        f"Not enough columns in `.out` after line {i_last_valid}. "
        "Removing all lines after this one..."
    )
    with open(filepath, "w", encoding="utf-8") as file:
        for i, line in enumerate(lines):
            if i >= i_last_valid:
                break
            file.write(line)


def _add_dummy_data(filepath: Path, elts: ListOfElements) -> None:
    """
    Add dummy data at the end of the ``.out`` to reach end of linac.

    We also round the column 'z', to avoid a too big mismatch between the z
    column and what we should have.

    .. todo::
        another possibly unbound error to handle

    """
    with open(filepath, "r+", encoding="utf-8") as file:
        for line in file:
            pass
        last_idx_in_file = int(line.split()[0])
        last_element_in_file = elts[last_idx_in_file - 1]

        if last_element_in_file is not elts[-1]:
            logging.warning(
                "Incomplete `.out` file. Trying to complete with "
                "dummy data..."
            )
            elts_to_add = elts[last_idx_in_file:]
            last_pos = np.round(float(line.split()[1]), 4)
            for i, elt in enumerate(elts_to_add, start=last_idx_in_file + 1):
                last_pos += elt.get("length_m", to_numpy=False)
                new_line = line.split()
                new_line[0] = str(i)
                new_line[1] = str(last_pos)
                new_line = " ".join(new_line) + "\n"
                file.write(new_line)


# =============================================================================
# Cavity parameters
# =============================================================================
def _load_cavity_parameters(
    path_cal: Path, filename: Path
) -> dict[str, np.ndarray]:
    """
    Get the cavity parameters calculated by TraceWin.

    Parameters
    ----------
    path_cal : Path
        Path to the folder where the cavity parameters file is stored.
    filename : Path
        The name of the cavity parameters file produced by TraceWin.

    Returns
    -------
    cavity_param : dict[float, np.ndarray]
        Contains the cavity parameters.

    """
    f_p = Path(path_cal, filename)
    n_lines_header = 1

    with open(f_p, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i == n_lines_header - 1:
                headers = line.strip().split()
                break

    out = np.loadtxt(f_p, skiprows=n_lines_header)
    cavity_parameters = {key: out[:, i] for i, key in enumerate(headers)}
    logging.debug(f"successfully loaded {f_p}")
    return cavity_parameters


def _cavity_parameters_uniform_with_envelope1d(
    cavity_parameters: dict[str, np.ndarray], n_elts: int
) -> list[None | dict[str, float]]:
    """Transform the dict so we have the same format as Envelope1D."""
    cavity_numbers = cavity_parameters["Cav#"].astype(int)
    v_cav, phi_s, phi_0 = [], [], []
    cavity_idx = 0
    for elt_idx in range(1, n_elts + 1):
        if elt_idx not in cavity_numbers:
            v_cav.append(None), phi_s.append(None), phi_0.append(None)
            continue

        v_cav.append(cavity_parameters["Voltage[MV]"][cavity_idx])
        phi_s.append(np.deg2rad(cavity_parameters["SyncPhase[°]"][cavity_idx]))
        phi_0.append(np.deg2rad(cavity_parameters["RF_phase[°]"][cavity_idx]))

        cavity_idx += 1

    compliant_cavity_parameters = {
        "v_cav_mv": v_cav,
        "phi_s": phi_s,
        "phi_0": phi_0,
    }
    return compliant_cavity_parameters
