#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:36:08 2023.

@author: placais

Handle simulation parameters. In particular:
    - what are the initial properties of the beam?
    - which cavities are broken?
    - how should they be fixed?
    - simulation parameters to give to TW for a 'post' simulation?

TODO: maybe make test and config to dict more compact?

TODO strategy:
    global_section
    global_section_downstream
    flag to select priority in k out of n when k odd
TODO position:
    element name
    element number
    end_section
TODO allow for different objectives at different positions.
    quickfix for now: simply set some scales to 0.

TODO variable: maybe add this? Unnecessary at this point
"""
import logging
import os
import configparser
import numpy as np
from constants import c

# Values that will be available everywhere
FLAG_CYTHON, FLAG_PHI_ABS = bool, bool
METHOD = str
N_STEPS_PER_CELL = int()

LINAC = str
E_MEV, E_REST_MEV, INV_E_REST_MEV = float(), float(), float()
GAMMA_INIT = float()
F_BUNCH_MHZ, OMEGA_0_BUNCH, LAMBDA_BUNCH = float(), float(), float()
Q_ADIM, Q_OVER_M, M_OVER_Q = float(), float(), float()
SIGMA_ZDELTA = np.ndarray(shape=(2, 2))

TRACEWIN_EXECUTABLES = {
    "X11 full": "/usr/local/bin/./TraceWin",
    "noX11 full": "/usr/local/bin/./TraceWin_noX11",
    "noX11 minimal": "/home/placais/TraceWin/exe/./tracelx64",
    "no run": None
}


def process_config(config_path: str, project_path: str,
                   key_beam_calculator: str, key_beam: str, key_wtf: str,
                   key_post_tw: str | None = None,
                   ) -> tuple[dict, dict, dict, dict | None]:
    """
    Frontend for config: load .ini, test it, return its content as dicts.

    Parameters
    ----------
    config_path : str
        Path to the .ini file.
    project_path : str
        Path to the project folder, to keep a copy of the used .ini.
    key_beam_calculator : str
        Name of the Section containing beam_calculator in the .ini file.
    key_beam : str
        Name of the Section containing beam parameters in the .ini file.
    key_wtf : str
        Name of the Section containing wtf parameters in the .ini file.
    key_post_tw : str | None, optional
        Name of the Section containing the TraceWin simulation parameters in
        the .ini file. This simulation is performed once the new settings are
        found. The default is None.

    Returns
    -------
    beam_calculator : dict
        Holds the beam_calculator used for simulation.
    beam : dict
        Dictionary holding all beam parameters.
    wtf : dict
        Dictionary holding all wtf parameters.
    post_tw : dict
        Holds the TW arguments. Overrides what is defined in the TW .ini file.
        Path to .ini, .dat and to results folder are defined in
        Accelerator.files dict.

    """
    # Load config
    # the converters key allows to have methods to directly convert the strings
    # in the .ini to the proper type
    config = configparser.ConfigParser(
        converters={
            'liststr': lambda x: [i.strip() for i in x.split(',')],
            'listint': lambda x: [int(i.strip()) for i in x.split(',')],
            'listfloat': lambda x: [float(i.strip()) for i in x.split(',')],
            'faults': lambda x: [[int(i.strip()) for i in y.split(',')]
                                 for y in x.split(',\n')],
            'groupedfaults': lambda x: [[[int(i.strip()) for i in z.split(',')]
                                         for z in y.split('|')]
                                        for y in x.split(',\n')],
            'matrixfloat': lambda x: np.array(
                [[float(i.strip()) for i in y.split(',')]
                 for y in x.split(',\n')]),
        },
        allow_no_value=True,
    )
    # FIXME listlistint and matrixfloat: same kind of input, but very different
    # outputs!!
    config.read(config_path)

    _test_config(config, key_beam_calculator, key_beam, key_wtf, key_post_tw)

    beam_calculator, beam, wtf, post_tw = _config_to_dict(
        config, key_beam_calculator=key_beam_calculator, key_beam=key_beam,
        key_wtf=key_wtf, key_post_tw=key_post_tw)

    # Remove unused Sections, save resulting file
    _ = [config.remove_section(key) for key in list(config.keys())
         if key not in ['DEFAULT', key_beam_calculator, key_beam, key_wtf,
                        key_post_tw]]
    with open(os.path.join(project_path, 'lighwin.ini'),
              'w', encoding='utf-8') as file:
        config.write(file)

    _beam_calculator_make_global(beam_calculator)
    _beam_make_global(beam)
    return beam_calculator, beam, wtf, post_tw


def _test_config(config: configparser.ConfigParser, key_beam_calculator: str,
                 key_beam: str, key_wtf: str, key_post_tw: str | None) -> None:
    """Run all the configuration tests, and save the config if ok."""
    _test_beam_calculator(config[key_beam_calculator])
    _test_beam(config[key_beam])
    _test_wtf(config[key_wtf])
    if key_post_tw is not None:
        _test_post_tw(config[key_post_tw])


def _config_to_dict(config: configparser.ConfigParser,
                    key_beam_calculator: str, key_beam: str, key_wtf: str,
                    key_post_tw: str | None
                    ) -> tuple[dict, dict, dict, dict | None]:
    """To convert the configparser into the formats required by LightWin."""
    beam_calculator = _config_to_dict_beam_calculator(
        config[key_beam_calculator])
    beam = _config_to_dict_beam(config[key_beam])
    wtf = _config_to_dict_wtf(config[key_wtf])
    post_tw = None
    if key_post_tw is not None:
        post_tw = _config_to_dict_tw(config[key_post_tw])
    return beam_calculator, beam, wtf, post_tw


# =============================================================================
# Everything related to beam_calculator (particles motion beam_calculator)
# (TraceWin @ bottom)
# =============================================================================
def _test_beam_calculator(c_beam_calculator: configparser.SectionProxy
                          ) -> None:
    """Test the appropriate beam_calculator (LightWin or TraceWin)."""
    passed = True
    mandatory = ["TOOL"]
    for key in mandatory:
        if key not in c_beam_calculator.keys():
            logging.error(f"Key {key} is mandatory and missing.")
            passed = False

    valid_tools = {'LightWin': _test_beam_calculator_lightwin,
                   'TraceWin': _test_tracewin}
    my_tool = c_beam_calculator["TOOL"]
    if my_tool not in valid_tools:
        logging.error(f"{my_tool} is an invalid value for TOOL. "
                      + f"Authorized values are: {valid_tools.keys()}.")
        passed = False

    if not passed or not valid_tools[my_tool](c_beam_calculator):
        raise IOError("Error treating the beam_calculator parameters.")
    logging.info(f"beam_calculator parameters {c_beam_calculator.name} tested "
                 + "with success.")


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

    if c_beam_calculator.getboolean("FLAG_CYTHON"):
        c_beam_calculator["METHOD"] += "_c"
    else:
        c_beam_calculator["METHOD"] += "_p"

    return True


def _config_to_dict_beam_calculator(
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


def _beam_calculator_make_global(beam_calculator: dict) -> None:
    """Update the values of some variables so they can be used everywhere."""
    global FLAG_CYTHON, FLAG_PHI_ABS, N_STEPS_PER_CELL, METHOD
    FLAG_CYTHON = beam_calculator["FLAG_CYTHON"]
    FLAG_PHI_ABS = beam_calculator["FLAG_PHI_ABS"]
    N_STEPS_PER_CELL = beam_calculator["N_STEPS_PER_CELL"]
    METHOD = beam_calculator["METHOD"]


# =============================================================================
# Everything related to beam parameters
# =============================================================================
def _test_beam(c_beam: configparser.SectionProxy) -> None:
    """Test the the beam parameters."""
    passed = True

    # Test that all mandatory keys are here
    mandatory = ["LINAC", "E_REST_MEV", "Q_ADIM", "E_MEV",
                 "F_BUNCH_MHZ", "I_MILLI_A", "SIGMA_ZDELTA"]
    for key in mandatory:
        if key not in c_beam.keys():
            logging.error(f"{key} is mandatory and missing.")
            passed = False

    # Test the values of the keys in beam
    if np.abs(c_beam.getfloat("I_MILLI_A")) > 1e-10:
        logging.warning("You asked LW a beam current different from "
                        + "0mA. Space-charge, transverse dynamics are "
                        + "not implemented yet, so this parameter "
                        + "will be ignored.")

    if not passed:
        raise IOError("Wrong value in c_beam.")

    logging.info(f"beam parameters {c_beam.name} tested with success.")


def _config_to_dict_beam(c_beam: configparser.SectionProxy) -> dict:
    """Convert beam configparser into a dict."""
    beam = {}
    # Special getters
    getter = {
        'E_REST_MEV': c_beam.getfloat,
        'Q_ADIM': c_beam.getfloat,
        'E_MEV': c_beam.getfloat,
        'F_BUNCH_MHZ': c_beam.getfloat,
        'I_MILLI_A': c_beam.getfloat,
        'SIGMA_ZDELTA': c_beam.getmatrixfloat,
    }

    for key in c_beam.keys():
        key = key.upper()
        if key in getter:
            beam[key] = getter[key](key)
            continue

        beam[key] = c_beam.get(key)

    # Add some useful keys
    beam["INV_E_REST_MEV"] = 1. / beam["E_REST_MEV"]
    beam["GAMMA_INIT"] = 1. + beam["E_MEV"] / beam["E_REST_MEV"]
    beam["OMEGA_0_BUNCH"] = 2e6 * np.pi * beam["F_BUNCH_MHZ"]
    beam["LAMBDA_BUNCH"] = c / beam["F_BUNCH_MHZ"]
    beam["Q_OVER_M"] = beam["Q_ADIM"] * beam["INV_E_REST_MEV"]
    beam["M_OVER_Q"] = 1. / beam["Q_OVER_M"]

    return beam


def _beam_make_global(beam: dict) -> None:
    """Update the values of some variables so they can be used everywhere."""
    global Q_ADIM, E_REST_MEV, INV_E_REST_MEV, OMEGA_0_BUNCH, GAMMA_INIT, \
        LAMBDA_BUNCH, Q_OVER_M, M_OVER_Q, F_BUNCH_MHZ, E_MEV, SIGMA_ZDELTA, \
        LINAC

    Q_ADIM = beam["Q_ADIM"]
    E_REST_MEV = beam["E_REST_MEV"]
    INV_E_REST_MEV = beam["INV_E_REST_MEV"]
    OMEGA_0_BUNCH = beam["OMEGA_0_BUNCH"]
    GAMMA_INIT = beam["GAMMA_INIT"]
    LAMBDA_BUNCH = beam["LAMBDA_BUNCH"]
    Q_OVER_M = beam["Q_OVER_M"]
    M_OVER_Q = beam["M_OVER_Q"]
    F_BUNCH_MHZ = beam["F_BUNCH_MHZ"]
    E_MEV = beam["E_MEV"]
    SIGMA_ZDELTA = beam["SIGMA_ZDELTA"]
    LINAC = beam["LINAC"]


# =============================================================================
# Everything related to wtf (what to fit) dicts
# =============================================================================
def _config_to_dict_wtf(c_wtf: configparser.SectionProxy) -> dict:
    """Convert wtf configparser into a dict."""
    wtf = {}
    # Special getters
    getter = {
        'objective': c_wtf.getliststr,
        'scale objective': c_wtf.getlistfloat,
        'position': c_wtf.getliststr,
        'failed': c_wtf.getfaults,
        'manual list': c_wtf.getgroupedfaults,
        'k': c_wtf.getint,
        'l': c_wtf.getint,
        'phi_s fit': c_wtf.getboolean,
    }
    if c_wtf.get('strategy') == 'manual':
        getter['failed'] = c_wtf.getgroupedfaults

    for key in c_wtf.keys():
        if key in getter:
            wtf[key] = getter[key](key)
            continue

        wtf[key] = c_wtf.get(key)

    return wtf


def _test_wtf(c_wtf: configparser.SectionProxy) -> None:
    """Test the 'what_to_fit' dictionaries."""
    tests = {'failed and idx': _test_failed_and_idx,
             'strategy': _test_strategy,
             'objective': _test_objective,
             'scale objective': _test_scale_objective,
             'opti method': _test_objective,
             'position': _test_position,
             'misc': _test_misc,
             }
    for key, test in tests.items():
        if not test(c_wtf):
            raise IOError(f"What to fit {c_wtf.name}: error in entry {key}.")
    logging.info(f"what to fit {c_wtf.name} tested with success.")


def _test_failed_and_idx(c_wtf: configparser.SectionProxy) -> bool:
    """
    Check that failed cavities are given.

    Required keys are:
        - idx:
            If 'cavity', 'failed' and 'manual list' are cavity numbers.
            If 'element', 'failed' and 'manual list' are element numbers. If
            these indexes correspond to element that are not a cavity, an error
            will be raised at the initialisaton of the Fault object.
        - failed:
            The indexes of the cavities that fail.

    Example
    -------
    (we consider that idx is 'cavity')
        1, 2,
        8,
        1, 2, 8
    In this case, LW will first fix the linac with the 1st and 2nd cavity
    failed. In a second time, LW will fix an error with the 8th failed cavity.
    In the last simulation, it will fix together the 1st, 2nd and 8th cavity.
    From LightWin's point of view: one line = one FaultScenario object.
    Each FaultScenario has a list of Fault objects. This is handled by
    the Faults are sorted by the FaultScenario._sort_faults method.

    Example for manual
    ------------------
    If strategy is manual, you must specify which cavities are fixed
    together by adding pipes.
        1, 2, 3 | 98, 99
    In this example, 1 2 and 3 are fixed together. The beam is then propagated
    up to 98 and 99 (even if it is not perfectly matched), and 98 and 99 are
    then fixed together.

    """
    for key in ['failed', 'idx']:
        if key not in c_wtf.keys():
            logging.error(f"You must provide {key}.")
            return False

    val = c_wtf.get('idx')
    if val not in ['cavity', 'element']:
        logging.error(f"'idx' key is {val}, while it must be 'cavity' or "
                      + "element.")
        return False

    return True


def _test_strategy(c_wtf: configparser.SectionProxy) -> bool:
    """
    Specific test for the key 'strategy' of what_to_fit.

    Three compensation strategies are implemented:
        - k out of n:
            k compensating cavities per fault. You must provide the number of
            compensating cavities per faulty cavity k. Nearby broken cavities
            are automatically gathered and fixed together.
        - manual:
            Manual association of faults and errors. In the .ini, 1st line of
            manual list will compensate 1st line of failed cavities, etc.
            Example (we consider that idx is 'element'):
            failed =
              12, 14, -> two cavities that will be fixed together
              155,    -> a single error, fixed after [12, 14] is dealt with
              12, 14 | 155 -> fix 12 and 14 and then 155 in the same simulation
                              (if beam is not perfectly matched after the first
                              error, the mismatch will propagate up to the next
                              error).

            manual_list =
              8, 10, 16, 18,    -> compensate errors at idx 12 & 14
              153, 157          -> compensate error at 155
              8, 10, 16, 18 | 153, 157 -> use 8 10 16 18 to compensate 12 and
              14. Propagate beam up to next fault, which is 155, and compensate
              if with 153 157.
        - l neighboring lattices:
            Every fault will be compensated by l full lattices, direct
            neighbors of the errors. You must provide l, which must be even.
        - global:
            Use every cavity.
        - global downstream:
            Use every cavity after the fault.
    """
    if 'strategy' not in c_wtf.keys():
        logging.error("You must provide 'strategy' to tell LightWin how "
                      + "compensating cavities are chosen.")
        return False

    tests = {'k out of n': _test_strategy_k_out_of_n,
             'manual': _test_strategy_manual,
             'l neighboring lattices': _test_strategy_l_neighboring_lattices,
             'global': _test_strategy_global,
             'global downstream': _test_strategy_global_downstream,
             }

    key = c_wtf['strategy']
    if key not in tests:
        logging.error("The 'strategy' key did not match any authorized value "
                      + f"({c_wtf['strategy']}).")
        return False

    return tests[key](c_wtf)


def _test_strategy_k_out_of_n(c_wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for k out of n strategy."""
    if 'k' not in c_wtf.keys():
        logging.error("You must provide k, the number of compensating "
                      + "cavities per failed cavity.")
        return False

    try:
        c_wtf.getint('k')
    except ValueError:
        logging.error("k must be an integer.")
        return False

    return True


def _test_strategy_manual(c_wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for manual strategy."""
    if 'manual list' not in c_wtf.keys():
        logging.error("You must provide a list of lists of compensating "
                      + "cavities corresponding to each list of failed "
                      + "cavities.")
        return False

    scenarios = c_wtf.getgroupedfaults('failed')
    groupcomp = c_wtf.getgroupedfaults('manual list')
    if len(scenarios) != len(groupcomp):
        logging.error("Discrepancy between the number of FaultScenarios and "
                      + "the number of corresponding list of compensating "
                      + "cavities. In other words: 'failed' and 'manual list' "
                      + "entries must have the same number of lines.")
        return False

    for scen, comp in zip(scenarios, groupcomp):
        if len(scen) != len(comp):
            logging.error("In a FaultScenario, discrepancy between the number "
                          + "of fault groups and group of compensating "
                          + "cavities. In other words: 'failed' and 'manual "
                          + "list' entries must have the same number of "
                          + "pipes on every line.")
            return False

    return True


def _test_strategy_l_neighboring_lattices(c_wtf: configparser.SectionProxy
                                          ) -> bool:
    """Even more specific test for l neighboring lattices strategy."""
    if 'l' not in c_wtf.keys():
        logging.error("You must provide l, the number of compensating "
                      + "lattices.")
        return False

    try:
        c_wtf.getint('l')
    except ValueError:
        logging.error("l must be an integer.")
        return False

    if c_wtf.getint('l') % 2 != 0:
        logging.error("l must be even.")
        return False

    return True


def _test_strategy_global(c_wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for global strategy."""
    logging.warning("Option still under implementation.")
    logging.warning("As for now, field amplitudes are always modified during "
                    + "the fit. If you want the 'classic' global compensation,"
                    + " you should manually set the bounds for k_e to a very "
                    + "low value in optimisation/variables.py.")

    if 'position' not in c_wtf.keys():
        logging.error("You must provide 'position' to tell LightWin where "
                      + "objectives should be evaluated.")
        return False

    if 'end of linac' not in c_wtf.getliststr('position'):
        logging.warning("With global methods, objectives should be evaluated "
                        + "at the end of the linac. LW will run anyway and "
                        + "'position' key will not be modified.")

    return True


def _test_strategy_global_downstream(c_wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for global downstream strategy."""
    return _test_strategy_global(c_wtf)


def _test_objective(c_wtf: configparser.SectionProxy) -> bool:
    """Specific test for the key 'objective' of what_to_fit."""
    if 'objective' not in c_wtf.keys():
        logging.error("You must provide 'objective' to tell LightWin what it "
                      + "should fit.")
        return False

    objectives = c_wtf.getliststr('objective')
    implemented = [
        'w_kin', 'phi_abs_array', 'mismatch_factor',
        'eps_zdelta', 'beta_zdelta', 'gamma_zdelta', 'alpha_zdelta',
        'M_11', 'M_12', 'M_22', 'M_21']

    if not all(obj in implemented for obj in objectives):
        logging.error("At least one objective was not recognized.")
        logging.info("""To add your own objective, make sure that:
                     1. it can be returned by the Accelerator.get() method;
                     2. it is present in the util.d_output dictionaries;
                     3. it is in the above 'implemented' dict.""")
        return False

    return True


def _test_opti_method(c_wtf: configparser.SectionProxy) -> bool:
    """Test the optimisation method."""
    if 'opti method' not in c_wtf.keys():
        logging.error("You must provide 'opti method' to tell LightWin what "
                      + "optimisation algorithm it should use.")
        return False

    implemented = ['least_squares', 'PSO']
    # TODO: specific testing for each method (look at the kwargs)
    if c_wtf['opti method'] not in implemented:
        logging.error("Algorithm not implemented.")
        return False
    return True


def _test_position(c_wtf: configparser.SectionProxy) -> bool:
    """Test where the objectives are evaluated."""
    if 'position' not in c_wtf.keys():
        logging.error("You must provide 'position' to tell LightWin where "
                      + "objectives should be evaluated.")
        return False

    positions = c_wtf.getliststr('position')
    implemented = [
        # End of last lattice with a compensating or failed cavity
        'end of last altered lattice',
        # One lattice after last lattice with a compensating/failed cavity
        'one lattice after last altered lattice',
        # End of last lattice with a failed cavity
        'end of last failed lattice',
        # End of linac
        'end of linac',
    ]
    if not all(pos in implemented for pos in positions):
        logging.error("At least one position was not recognized. Allowed "
                      + f"values are: {implemented}.")
        return False
    return True


def _test_scale_objective(c_wtf: configparser.SectionProxy) -> bool:
    """Specific test for the key 'scale objective' of what_to_fit."""
    objectives = c_wtf.getliststr('objective')
    positions = c_wtf.getliststr('position')

    if 'scale objective' in c_wtf.keys():
        scales = c_wtf.getlistfloat('scale objective')
        if len(scales) != len(objectives) * len(positions):
            logging.error("If you want to scale the objectives by a factor, "
                          + "you must provide a list of scale factors (one "
                          + "scale factor per objective and per position).")
            return False
    else:
        scales = [1. for x in range(len(objectives) * len(positions))]
        c_wtf['scale objective'] = ', '.join(map(str, scales))
        logging.warning("Scale of objectives not specified. Use default.")

    return True


def _test_misc(c_wtf: configparser.SectionProxy) -> bool:
    """Perform some other tests."""
    if 'phi_s fit' not in c_wtf.keys():
        logging.error("Please explicitely precise if you want to fit synch "
                      + "phases (recommended for least squares, which do not "
                      + "handle constraints) or not (for algorithms that can "
                      + "handle it).")
        return False

    try:
        c_wtf.getboolean("phi_s fit")
    except ValueError:
        logging.error("Not a boolean.")
        return False
    return True


# =============================================================================
# Everything related to TraceWin simulations
# =============================================================================
def _test_tracewin(c_beam_calculator: configparser.SectionProxy) -> bool:
    """Specific test for the TraceWin simulations."""
    mandatory = ["SIMULATION TYPE"]
    for key in mandatory:
        if key not in c_beam_calculator.keys():
            logging.error(f"{key} is mandatory and missing.")
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


def _test_post_tw(c_tw: configparser.SectionProxy) -> None:
    """Test for the TraceWin simulation parameters."""
    if not _test_tracewin(c_tw):
        raise IOError("Error when testing the TW configuration.")


def _config_to_dict_tw(c_tw: configparser.SectionProxy) -> dict:
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


# =============================================================================
# Main func
# =============================================================================
if __name__ == '__main__':
    # Init paths
    CONFIG_PATH = 'jaea.ini'
    PROJECT_PATH = 'bla/'

    # Load config
    wtfs = process_config(
        CONFIG_PATH, PROJECT_PATH,
        key_beam_calculator='beam_calculator.lightwin.envelope_longitudinal',
        key_beam='beam.jaea',
        key_wtf='wtf.k_out_of_n',
        key_post_tw='post_tracewin.quick_debug')
    print(f"{wtfs}")
