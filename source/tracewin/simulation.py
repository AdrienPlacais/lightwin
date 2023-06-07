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
import time
import datetime

EXECUTABLES = {
    "X11 full": "/usr/local/bin/./TraceWin",
    "noX11 full": "/usr/local/bin/./TraceWin_noX11",
    "noX11 minimal": "/home/placais/TraceWin/exe/./tracelx64",
    "no run": None
}


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
    simulation_type: str
    ini_path: str
    path_cal: str
    dat_file: str
    base_kwargs: dict[[str], str]

    def __post_init__(self) -> None:
        """Set additional parameters."""
        if not _is_valid_executable(self.simulation_type):
            self.simulation_type = "no run"
            logging.warning("No valid TraceWin executable was found. LightWin "
                            + "will try to run anyway and skip any TraceWin "
                            + "simulation.")
        self.executable = EXECUTABLES[self.simulation_type]

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
        command = [self.executable, self.ini_path, f"path_cal={self.path_cal}",
                   f"dat_file={self.dat_file}"]
        for key, value in kwargs.items():
            if value is None:
                command.append(key)
                continue
            command.append(key + "=" + str(value))
            return command


def _is_valid_executable(simulation_type: str) -> bool:
    """Test if the executable exists."""
    if simulation_type == "no run":
        return True

    if simulation_type not in EXECUTABLES:
        logging.error(f"The simulation type {simulation_type} was not "
                      + f"recognized. Authorized values: {EXECUTABLES.keys()}")
        return False

    tw_exe = EXECUTABLES[simulation_type]
    if not os.path.isfile(tw_exe):
        logging.error(f"The TraceWin executable was not found: {tw_exe}. You "
                      + "should update the EXECUTABLES dictionary in "
                      + "source/LightWin/tracewin/simulation.py.")
        return False

    return True

