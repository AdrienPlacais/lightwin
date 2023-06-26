#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:34:51 2023.

@author: placais

Everything related to the beam_calculator (and beam_calculator_post) keys of
the config file.
"""
import logging
import os
import configparser

TRACEWIN_EXECUTABLES = {  # Should match with your installation
    "X11 full": "/usr/local/bin/./TraceWin",
    "noX11 full": "/usr/local/bin/./TraceWin_noX11",
    "noX11 minimal": "/home/placais/TraceWin/exe/./tracelx64",
    "no run": None
}


# =============================================================================
# Front end
# =============================================================================
def test(c_beam_calculator: configparser.SectionProxy) -> None:
    """Test the appropriate beam_calculator (LightWin or TraceWin)."""
    passed = True
    mandatory = ["TOOL"]
    for key in mandatory:
        if key not in c_beam_calculator.keys():
            logging.error(f"Key {key} is mandatory and missing.")
            passed = False

    valid_tools = {'LightWin': _test_beam_calculator_lightwin,
                   'TraceWin': _test_beam_calculator_tracewin}
    my_tool = c_beam_calculator["TOOL"]
    if my_tool not in valid_tools:
        logging.error(f"{my_tool} is an invalid value for TOOL. "
                      + f"Authorized values are: {valid_tools.keys()}.")
        passed = False

    if not passed or not valid_tools[my_tool](c_beam_calculator):
        raise IOError("Error treating the beam_calculator parameters.")
    logging.info(f"beam_calculator parameters {c_beam_calculator.name} tested "
                 + "with success.")


def config_to_dict(c_beam_calculator: configparser.SectionProxy) -> dict:
    """Call the proper _config_to_dict function."""
    config_to_dicts = {'LightWin': _config_to_dict_lightwin,
                       'TraceWin': _config_to_dict_tracewin}
    my_tool = c_beam_calculator["TOOL"]
    return config_to_dicts[my_tool](c_beam_calculator)


def make_global(beam_calculator: dict) -> None:
    """Update the values of some variables so they can be used everywhere."""


# =============================================================================
# Testers
# =============================================================================
def _test_beam_calculator_lightwin(
        c_beam_calculator: configparser.SectionProxy) -> bool:
    """
    Test consistency of the LightWin beam_calculator.

    FLAF_PHI_ABS: to determine if the phases in the cavities are absolute or
    relative.
    If True, cavities keep their absolute phi_0 (!! relative phi_0 may be
    changed though !!).
    If False, cavities keep their relative phi_0; all cavities after the first
    modified cavity change their status to 'rephased'.

    METHOD: method to integrate the motion. leapfrog or RK (RK4)

    N_STEPS_PER_CELL: number of spatial steps per RF cavity cell.

    FLAG_CYTHON: to determine if transfer_matrices_c (Cython) should be use
    instead of _p (pure Python). _c is ~2 to 4 times faster than _p.
    Warning, you may have to relaod the kernel to force iPython to take the
    change in FLAG_CYTHON into account.

    """
    mandatory = ["FLAG_CYTHON", "METHOD", "FLAG_PHI_ABS"]
    for key in mandatory:
        if key not in c_beam_calculator.keys():
            logging.error(f"{key} is mandatory and missing.")
            return False

    if c_beam_calculator["METHOD"] not in ["leapfrog", "RK"]:
        logging.error("Wrong value for METHOD, "
                      + "beam_calculator not implemented.")
        return False

    if "N_STEPS_PER_CELL" not in c_beam_calculator.keys():
        logging.warning("Number of integration steps per cell not precised. "
                        + "Will use default values.")
        default = {'leapfrog': '40', 'RK': '20'}
        c_beam_calculator["N_STEPS_PER_CELL"] = default["METHOD"]

    # TODO remove this
    if c_beam_calculator.getboolean("FLAG_CYTHON"):
        c_beam_calculator["METHOD"] += "_c"
    else:
        c_beam_calculator["METHOD"] += "_p"

    return True


def _test_beam_calculator_tracewin(
        c_beam_calculator: configparser.SectionProxy) -> bool:
    """Specific test for the TraceWin simulations."""
    mandatory = ["SIMULATION TYPE", "ini_path"]
    for key in mandatory:
        if key not in c_beam_calculator.keys():
            logging.error(f"{key} is mandatory and missing.")
            return False

    if not os.path.isfile(c_beam_calculator["ini_path"]):
        logging.error(f"{c_beam_calculator['ini_path']} does not exist.")
        return False

    simulation_type = c_beam_calculator["SIMULATION TYPE"]
    if simulation_type not in TRACEWIN_EXECUTABLES:
        logging.error(f"The simulation type {simulation_type} was not "
                      + "recognized. Authorized values: "
                      + f"{TRACEWIN_EXECUTABLES.keys()}")
        return False

    tw_exe = TRACEWIN_EXECUTABLES[simulation_type]
    if tw_exe is None:
        logging.warning("No TraceWin simulation. May clash with other things "
                        "as the executable is None.")
        return True

    if not os.path.isfile(tw_exe):
        logging.error(f"The TraceWin executable was not found: {tw_exe}. You "
                      + "should update the TRACEWIN_EXECUTABLES dictionary in "
                      + "config_manager.py.")
        return False
    c_beam_calculator["executable"] = tw_exe

    # TODO: implement all TW options
    for key in c_beam_calculator.keys():
        if "Ele" in key:
            logging.error("Are you trying to use the Ele[n][v] key? It is not "
                          + "implemented and may clash with LightWin.")
            return False

        if key == "Synoptic_file":
            logging.error("Not implemented as I am not sure how this should "
                          + "work.")
            return False

        if key in ['partran', 'toutatis']:
            if c_beam_calculator.get(key) not in ['0', '1']:
                logging.error("partran and toutatis keys should be 0 or 1.")
                return False
    return True


# =============================================================================
# Config to dict
# =============================================================================
def _config_to_dict_lightwin(
        c_beam_calculator: configparser.SectionProxy) -> dict:
    """Save beam_calculator info into a dict."""
    beam_calculator = {}
    getter = {
        'FLAG_CYTHON': c_beam_calculator.getboolean,
        'FLAG_PHI_ABS': c_beam_calculator.getboolean,
        'N_STEPS_PER_CELL': c_beam_calculator.getint,
    }
    for key in c_beam_calculator.keys():
        key = key.upper()
        if key in getter:
            beam_calculator[key] = getter[key](key)
            continue
        beam_calculator[key] = c_beam_calculator.get(key.lower())

    return beam_calculator


def _config_to_dict_tracewin(c_tw: configparser.SectionProxy) -> dict:
    """Convert tw configparser into a dict."""
    tracew = {}
    # Getters. If a key is not in this dict, it won't be transferred to TW
    getter = {
        'executable': c_tw.get,
        'hide': c_tw.get,
        'tab_file': c_tw.get,
        'nbr_thread': c_tw.getint,
        'dst_file1': c_tw.get,
        'dst_file2': c_tw.get,
        'current1': c_tw.getfloat,
        'current2': c_tw.getfloat,
        'nbr_part1': c_tw.getint,
        'nbr_part2': c_tw.getint,
        'energy1': c_tw.getfloat,
        'energy2': c_tw.getfloat,
        'etnx1': c_tw.getfloat,
        'etnx2': c_tw.getfloat,
        'etny1': c_tw.getfloat,
        'etny2': c_tw.getfloat,
        'eln1': c_tw.getfloat,
        'eln2': c_tw.getfloat,
        'freq1': c_tw.getfloat,
        'freq2': c_tw.getfloat,
        'duty1': c_tw.getfloat,
        'duty2': c_tw.getfloat,
        'mass1': c_tw.getfloat,
        'mass2': c_tw.getfloat,
        'charge1': c_tw.getfloat,
        'charge2': c_tw.getfloat,
        'alpx1': c_tw.getfloat,
        'alpx2': c_tw.getfloat,
        'alpy1': c_tw.getfloat,
        'alpy2': c_tw.getfloat,
        'alpz1': c_tw.getfloat,
        'alpz2': c_tw.getfloat,
        'betx1': c_tw.getfloat,
        'betx2': c_tw.getfloat,
        'bety1': c_tw.getfloat,
        'bety2': c_tw.getfloat,
        'betz1': c_tw.getfloat,
        'betz2': c_tw.getfloat,
        'x1': c_tw.getfloat,
        'x2': c_tw.getfloat,
        'y1': c_tw.getfloat,
        'y2': c_tw.getfloat,
        'z1': c_tw.getfloat,
        'z2': c_tw.getfloat,
        'xp1': c_tw.getfloat,
        'xp2': c_tw.getfloat,
        'yp1': c_tw.getfloat,
        'yp2': c_tw.getfloat,
        'zp1': c_tw.getfloat,
        'zp2': c_tw.getfloat,
        'dw1': c_tw.getfloat,
        'dw2': c_tw.getfloat,
        'spreadw1': c_tw.getfloat,
        'spreadw2': c_tw.getfloat,
        'part_step': c_tw.getint,
        'vfac': c_tw.getfloat,
        'random_seed': c_tw.getint,
        'partran': c_tw.getint,
        'toutatis': c_tw.getint,
    }
    for key in c_tw.keys():
        if key in getter:
            tracew[key] = getter[key](key)
            continue

    return tracew
