#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:13:16 2023.

@author: placais

Different use cases, from easiest to hardest:
    - run it, get its results
    - no run, but get results anyway
    - use it to fit
"""
from dataclasses import dataclass
import os
import logging
import subprocess
from functools import partial
import time
import datetime

import numpy as np

from constants import c
import util.converters as convert


@dataclass
class TraceWinSimulator:
    """
    A class to hold a TW simulation and its results.

    The simulation is not necessarily runned.

    Parameters
    ----------
    simulation_type : ['X11 full', 'noX11 full', 'noX11 minimal', 'no run']
        What kind of simulation you want.
    ini_path : str
        Path to the .ini TraceWin file.
    path_cal : str
        Path to the output folder, where TW results will be stored. Overrides
        the path_cal defined in .ini.
    dat_file : str
        Path to the TraceWin .dat file, with accelerator structure. Overrides
        the dat_file defined in .ini.
    **base_kwargs : dict
        TraceWin optional arguments. Override what is defined in .ini, but
        overriden by specific_arguments in the self.run method.

    """

    executable: str
    ini_path: str
    path_cal: str
    dat_file: str
    base_kwargs: dict[[str], str]

    def __post_init__(self) -> None:
        """Define some other useful methods, init variables."""
        self.get_results_envelope = partial(self.get_results,
                                            filename='tracewin.out')
        self.get_results_multipart = partial(self.get_results,
                                             filename='partran1.out')
        os.makedirs(self.path_cal)
        self.results_envelope: dict
        self.results_multipart: dict | None
        self.transfer_matrices: np.ndarray
        self.cavity_parameters: dict

    def run(self, store_all_outputs: bool = True, post_treat: bool = True,
            **specific_kwargs) -> None:
        """
        Run TraceWin.

        Parameters
        ----------
        store_all_outputs : bool, optional
            Save all the outputs from TraceWin. The default is True.
        post_treat : bool, optional
            Compute quantities that will be required later in results (both
            multipart and envelope). The default is True.
        **specific_kwargs : dict
            TraceWin optional arguments. Overrides what is defined in
            base_kwargs and .ini.

        """
        if self.executable is None:
            logging.warning("TraceWinSimulation has an invalid TraceWin "
                            + "executable. Skipping simulation...")
            return

        start_time = time.monotonic()

        kwargs = specific_kwargs.copy()
        for key, val in self.base_kwargs.items():
            if key not in kwargs:
                kwargs[key] = val

        command = self._set_command(**kwargs)
        logging.info(f"Running TW with command {command}...")
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        process.wait()
        for line in process.stdout:
            print(line)

        end_time = time.monotonic()
        delta_t = datetime.timedelta(seconds=end_time - start_time)
        logging.info(f"TW finished! It took {delta_t}")

        if store_all_outputs:
            self.results_envelope = \
                self.get_results_envelope(post_treat=post_treat)
            self.results_multipart = \
                self.get_results_multipart(post_treat=post_treat) \
                if self._is_a_multiparticle_simulation(kwargs) else None
            _, _, self.transfer_matrices = self.get_transfer_matrices()
            self.cavity_parameters = self.get_cavity_parameters()

    def _set_command(self, **kwargs) -> str:
        """Create the command line to launch TraceWin."""
        arguments_that_tracewin_will_not_understand = ["executable"]
        command = [self.executable,
                   self.ini_path,
                   f"path_cal={self.path_cal}",
                   f"dat_file={self.dat_file}"]
        for key, value in kwargs.items():
            if key in arguments_that_tracewin_will_not_understand:
                continue
            if value is None:
                command.append(key)
                continue
            command.append(key + "=" + str(value))
        return command

    def _is_a_multiparticle_simulation(self, kwargs) -> bool:
        """Tells if you should buy Bitcoins now or wait a few months."""
        if 'partran' in kwargs:
            return kwargs['partran'] == 1
        return os.path.isfile(os.path.join(self.path_cal, 'partran1.out'))

    def get_results(self, filename: str, post_treat: bool = True
                    ) -> dict[[str], np.ndarray]:
        """
        Get the TraceWin results.

        Parameters
        ----------
        filename : str
            Results file produced by TraceWin.

        Returns
        -------
        results : dict[[str], np.ndarray]
            Dictionary contatining the raw outputs from TraceWin.
        """
        f_p = os.path.join(self.path_cal, filename)
        n_lines_header = 9
        results = {}

        with open(f_p, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == n_lines_header:
                    headers = line.strip().split()
                    break

        out = np.loadtxt(f_p, skiprows=n_lines_header)
        for i, key in enumerate(headers):
            results[key] = out[:, i]
            logging.debug(f"successfully loaded {f_p}")

        if post_treat:
            return _post_treat(results)
        return results

    def get_transfer_matrices(self, filename: str = 'Transfer_matrix1.dat',
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

        f_p = os.path.join(self.path_cal, filename)
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

    def get_cavity_parameters(self, filename: str = 'Cav_set_point_res.dat'
                              ) -> dict[[str], np.ndarray]:
        """
        Get the cavity parameters calculated by TraceWin.

        Parameters
        ----------
        filename : str, optional
            The name of the cavity parameters file produced by TraceWin. The
            default is 'Cav_set_point_res.dat'.

        Returns
        -------
        cavity_param : dict[[float], np.ndarray]
            Contains the cavity parameters.

        """
        f_p = os.path.join(self.path_cal, filename)
        n_lines_header = 1
        cavity_param = {}

        with open(f_p, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == n_lines_header - 1:
                    headers = line.strip().split()
                    break

        out = np.loadtxt(f_p, skiprows=n_lines_header)
        for i, key in enumerate(headers):
            cavity_param[key] = out[:, i]
        logging.debug(f"successfully loaded {f_p}")
        return cavity_param


def _post_treat(results: dict) -> dict:
    """Compute and store the missing quantities (envelope or multipart)."""
    results['gamma'] = 1. + results['gama-1']
    results['w_kin'] = convert.energy(results['gamma'], "gamma to kin")
    results['beta'] = convert.energy(results['w_kin'], "kin to beta")
    results['lambda'] = c / 162e6   # FIXME

    num = results['beta'].shape[0]
    results['phi_abs_array'] = np.full((num), 0.)
    for i in range(1, num):
        delta_z = results['z(m)'][i] - results['z(m)'][i - 1]
        beta_i = results['beta'][i]
        delta_phi = 2. * np.pi * delta_z / (results['lambda'] * beta_i)
        results['phi_abs_array'][i] = results['phi_abs_array'][i - 1] \
            + np.rad2deg(delta_phi)

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
