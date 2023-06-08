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
import os.path
import logging
import subprocess
from functools import partial
import time
import datetime

import numpy as np

@dataclass
class TraceWinSimulation:
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
        """Define some other useful methods."""
        self.get_results_envelope = partial(self.get_results,
                                            filepath='tracewin.out')
        self.get_results_multipart = partial(self.get_results,
                                             filepath='partran1.out')

    def run(self, **specific_kwargs) -> None:
        """
        Run TraceWin.

        Parameters
        ----------
        **specific_kwargs : dict
            TraceWin optional arguments. Overrides what is defined in
            base_kwargs and .ini.

        """
        if self.executable is None:
            logging.warning("TraceWinSimulation has an invalid TraceWin "
                            + "executable. Skipping simulation...")
            return

        start_time = time.monotonic()

        kwargs = specific_kwargs
        for key, val in self.base_kwargs:
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

    def _set_command(self, **kwargs) -> str:
        """Creat the command line to launch TraceWin."""
        command = [self.executable,
                   self.ini_path,
                   f"path_cal={self.path_cal}",
                   f"dat_file={self.dat_file}"]
        for key, value in kwargs.items():
            if value is None:
                command.append(key)
                continue
            command.append(key + "=" + str(value))
        return command

    def get_results(self, filename: str) -> dict[[str], np.ndarray]:
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
