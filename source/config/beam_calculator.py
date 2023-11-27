#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Everything related to the ``beam_calculator`` key.

``beam_calculator_post`` is also handled here.

.. todo::
    Handle ``tracelx``. Remember that the ``tracelx`` executable should be in
    the same folder as your project!

.. todo::
    Would be possible to add more lists of implemented options to the
    documentation.

"""
import logging
from pathlib import Path
import os
import configparser

tools = ('Envelope1D', 'TraceWin', 'Envelope3D')  #:
methods = ('leapfrog', 'RK')  #:

TRACEWIN_EXECUTABLES = {  # Should match with your installation
    "X11 full": Path("/", "usr", "local", "bin", "TraceWin", "./TraceWin"),
    "noX11 full": Path("/", "usr", "local", "bin", "TraceWin",
                       "./TraceWin_noX11"),
    "noX11 minimal": Path("/", "home", "placais", "TraceWin", "exe",
                          "./tracelx64"),
    "no run": None
}
simulation_types = tuple(TRACEWIN_EXECUTABLES.keys())  #:


# =============================================================================
# Front end
# =============================================================================
def test(c_beam_calculator: configparser.SectionProxy) -> None:
    """Test the appropriate beam_calculator (Envelope1D or TraceWin)."""
    passed = True
    mandatory = ["tool"]
    for key in mandatory:
        if key not in c_beam_calculator.keys():
            logging.error(f"Key {key} is mandatory and missing.")
            passed = False

    valid_tools = {'Envelope1D': _test_beam_calculator_envelope1d,
                   'TraceWin': _test_beam_calculator_tracewin,
                   'Envelope3D': _test_beam_calculator_envelope3d,
                   }

    my_tool = c_beam_calculator["tool"]
    if my_tool not in tools:
        logging.error(f"{my_tool} is an invalid value for tool. "
                      f"Authorized values are: {tools}.")
        passed = False

    if not passed or not valid_tools[my_tool](c_beam_calculator):
        raise IOError("Error treating the beam_calculator parameters.")
    logging.info(f"beam_calculator parameters {c_beam_calculator.name} tested "
                 "with success.")


def config_to_dict(c_beam_calculator: configparser.SectionProxy) -> dict:
    """Call the proper _config_to_dict function."""
    config_to_dicts = {'Envelope1D': _config_to_dict_envelope1d,
                       'TraceWin': _config_to_dict_tracewin,
                       'Envelope3D': _config_to_dict_envelope3d}
    my_tool = c_beam_calculator["tool"]
    return config_to_dicts[my_tool](c_beam_calculator)


# =============================================================================
# Testers
# =============================================================================
def _test_beam_calculator_envelope1d(
        c_beam_calculator: configparser.SectionProxy) -> bool:
    """
    Test consistency of the Envelope1D beam_calculator.

    FLAG_PHI_ABS: to determine if the phases in the cavities are absolute or
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
    mandatory = ("flag_cython", "method", "flag_phi_abs")
    for key in mandatory:
        if key not in c_beam_calculator.keys():
            logging.error(f"{key} is mandatory and missing.")
            return False

    if c_beam_calculator["method"] not in methods:
        logging.error("Wrong value for method, beam_calculator not "
                      "implemented.")
        return False

    if "n_steps_per_cell" not in c_beam_calculator.keys():
        logging.warning("Number of integration steps per cell not precised. "
                        "Will use default values.")
        default = {'leapfrog': '40',
                   'RK': '20'}
        c_beam_calculator["n_steps_per_cell"] = default["method"]

    return True


def _test_beam_calculator_tracewin(
        c_beam_calculator: configparser.SectionProxy) -> bool:
    """Specific test for the TraceWin simulations."""
    mandatory = ("simulation type", "ini_path")
    for key in mandatory:
        if key not in c_beam_calculator.keys():
            logging.error(f"{key} is mandatory and missing.")
            return False

    if not c_beam_calculator.getpath("ini_path").is_file():
        logging.error(f"{c_beam_calculator['ini_path']} does not exist.")
        return False

    simulation_type = c_beam_calculator["simulation type"]
    if simulation_type not in TRACEWIN_EXECUTABLES:
        logging.error(f"The simulation type {simulation_type} was not "
                      "recognized. Authorized values: "
                      f"{TRACEWIN_EXECUTABLES.keys()}")
        return False

    tw_exe = TRACEWIN_EXECUTABLES[simulation_type]
    if tw_exe is None:
        logging.warning("No TraceWin simulation. May clash with other things "
                        "as the executable is None.")
        return True

    if not tw_exe.is_file():
        logging.error(f"The TraceWin executable was not found: {tw_exe}. You "
                      "should update the TRACEWIN_EXECUTABLES dictionary in "
                      "config/beam_calculator.py.")
        return False
    c_beam_calculator["executable"] = tw_exe

    for key in c_beam_calculator.keys():
        if "Ele" in key:
            logging.error("Are you trying to use the Ele[n][v] key? Please "
                          "directly modify your `.dat` to avoid clash with "
                          "LightWin.")
            return False

        if key == "Synoptic_file":
            logging.error("Not implemented as I am not sure how this should "
                          "work.")
            return False

        if key in ('partran', 'toutatis'):
            if c_beam_calculator.get(key) not in ('0', '1'):
                logging.error("partran and toutatis keys should be 0 or 1.")
                return False
    return True


def _test_beam_calculator_envelope3d(
        c_beam_calculator: configparser.SectionProxy) -> bool:
    """Test consistency of the Envelope3D beam_calculator."""
    mandatory = ("flag_phi_abs", )
    for key in mandatory:
        if key not in c_beam_calculator.keys():
            logging.error(f"{key} is mandatory and missing.")
            return False

    if "method" in c_beam_calculator.keys():
        method = c_beam_calculator.get('method')
        if method != 'RK':
            logging.warning(f"You asked for {method = } but only 'RK' is "
                            "implemented. This is what I will use.")
        return False

    if "n_steps_per_cell" not in c_beam_calculator.keys():
        logging.warning("Number of integration steps per cell not precised. "
                        "Will use default values.")
        default = {'RK': '20'}
        c_beam_calculator["n_steps_per_cell"] = default["method"]

    return True


# =============================================================================
# Config to dict
# =============================================================================
def _config_to_dict_envelope1d(
        c_beam_calculator: configparser.SectionProxy) -> dict:
    """Save beam_calculator info into a dict."""
    beam_calculator = {}
    getter = {
        'flag_cython': c_beam_calculator.getboolean,
        'flag_phi_abs': c_beam_calculator.getboolean,
        'n_steps_per_cell': c_beam_calculator.getint,
    }
    for key in c_beam_calculator.keys():
        if key in getter:
            beam_calculator[key] = getter[key](key)
            continue
        beam_calculator[key] = c_beam_calculator.get(key)

    return beam_calculator


def _config_to_dict_tracewin(c_tw: configparser.SectionProxy) -> dict:
    """
    Convert tw configparser into a dict.

    The TraceWin mandatory command line arguments, as well as the arguments
    necessary for LightWin to run are stored in `arg_for_lightwin`.

    We separate the optional arguments that will be sent in the TraceWin
    commannd line in a separated dictionary: `arg_for_tracewin`.
    """
    args_for_lightwin = {}
    args_for_tracewin = {}
    getter_arg_for_lightwin = {
        'executable': c_tw.getpath,
    }

    getter_arg_for_tracewin = {
        'hide': c_tw.get,
        'tab_file': c_tw.getpath,
        'nbr_thread': c_tw.getint,
        'dst_file1': c_tw.getpath,
        'dst_file2': c_tw.getpath,
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
        if key in getter_arg_for_tracewin:
            args_for_tracewin[key] = getter_arg_for_tracewin[key](key)
            continue
        if key in getter_arg_for_lightwin:
            args_for_lightwin[key] = getter_arg_for_lightwin[key](key)
            continue
        args_for_lightwin[key] = c_tw.get(key)

    args_for_lightwin['base_kwargs'] = args_for_tracewin

    return args_for_lightwin


def _config_to_dict_envelope3d(
        c_beam_calculator: configparser.SectionProxy) -> dict:
    """Save beam_calculator info into a dict."""
    beam_calculator = {}
    getter = {
        'flag_phi_abs': c_beam_calculator.getboolean,
        'n_steps_per_cell': c_beam_calculator.getint,
    }
    for key in c_beam_calculator.keys():
        if key in getter:
            beam_calculator[key] = getter[key](key)
            continue
        beam_calculator[key] = c_beam_calculator.get(key)

    return beam_calculator
