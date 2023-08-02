#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:54:54 2023.

@author: placais

All the functions to test the `wtf` (what to fit) key of the config file.

"""
import logging
import configparser


# =============================================================================
# Front end
# =============================================================================
def test(c_wtf: configparser.SectionProxy) -> None:
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


def config_to_dict(c_wtf: configparser.SectionProxy) -> dict:
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


# =============================================================================
# Test
# =============================================================================
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
        'w_kin', 'phi_abs', 'mismatch_factor_zdelta',
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
