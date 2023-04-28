#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:36:08 2023.

@author: placais

Handle simulation parameters. In particular:
    - which cavities are broken?
    - how should they be fixed?

TODO: allow for different wtf for every fault. Maybe use different .ini?
"""
import logging
import os
import configparser


def process_config(config_path: str, project_path: str, key_wtf: str = 'wtf',
                   key_tw: str = 'tracewin',
                   ) -> dict:
    """Frontend for config: load .ini, test it, return its content as dicts."""
    # Load config
    config = configparser.ConfigParser(
        # Allow to use the getlist method
        converters={
            'liststr': lambda x: [i.strip() for i in x.split(',')],
            'listint': lambda x: [int(i.strip()) for i in x.split(',')],
            'listfloat': lambda x: [float(i.strip()) for i in x.split(',')],
            'listlistint': lambda x: [[int(i.strip()) for i in y.split(',')]
                                      for y in x.split(',\n')],
        }
    )
    config.read(config_path)
    _test_config(config, key_wtf, key_tw)

    # Remove unused Sections, save resulting file
    [config.remove_section(key) for key in list(config.keys())
     if key not in ['DEFAULT', key_wtf]]
    with open(os.path.join(project_path, 'lighwin.ini'),
              'w', encoding='utf-8') as file:
        config.write(file)

    wtf = _config_to_dict(config, key_wtf=key_wtf)
    return wtf


def _test_config(config: configparser.ConfigParser, key_wtf: str, key_tw: str,
                 ) -> None:
    """Run all the config dic tests, and save the config if ok."""
    _test_wtf(config[key_wtf])
    _test_tw(config[key_tw])


def generate_list_of_faults():
    """Generate a proper (list of) faults."""
    failed = None
    return failed


def _config_to_dict(config: configparser.ConfigParser, key_wtf) -> dict:
    """To convert the configparser into the formats required by LightWin."""
    wtf = _config_to_dict_wtf(config[key_wtf])
    return wtf


def _config_to_dict_wtf(c_wtf: configparser.SectionProxy) -> dict:
    """Convert wtf configparser into a dict."""
    d_wtf = {}
    # Special getters
    getter = {
        'objective': c_wtf.getliststr,
        'scale objective': c_wtf.getlistfloat,
        'manual list': c_wtf.getlistlistint,
        'k': c_wtf.getint,
        'l': c_wtf.getint,
        'phi_s fit': c_wtf.getboolean,
    }

    for key in c_wtf.keys():
        if key in getter:
            d_wtf[key] = getter[key](key)
            continue

        d_wtf[key] = c_wtf.get(key)

    return d_wtf


# =============================================================================
# Testing of wtf (what to fit)
# =============================================================================
def _test_wtf(wtf: configparser.SectionProxy) -> None:
    """Test the 'what_to_fit' dictionaries."""
    if not _test_strategy(wtf):
        raise IOError("Wrong argument in wtf['strategy'].")

    if not _test_objective(wtf):
        raise IOError("Wrong argument in wtf['objective'].")

    if not _test_opti_method(wtf):
        raise IOError("Wrong argument in wtf['opti method'].")

    if not _test_position(wtf):
        raise IOError("Wrong argument in wtf['position'].")

    if not _test_misc(wtf):
        raise IOError("Check _test_misc.")

    logging.info(f"what to fit {wtf.name} tested with success.")


# =============================================================================
# Testing of wtf strategy
# =============================================================================
def _test_strategy(wtf: configparser.SectionProxy) -> bool:
    """Specific test for the key 'strategy' of what_to_fit."""
    if 'strategy' not in wtf.keys():
        logging.error("You must provide 'strategy' to tell LightWin how "
                      + "compensating cavities are chosen.")
        return False

    # You must provide the number of compensating cavities per faulty
    # cavity. Nearby broken cavities are automatically gathered and fixed
    #  together.
    if wtf['strategy'] == 'k out of n':
        return _test_strategy_k_out_of_n(wtf)

    # You must provide a list of lists of broken cavities, and the
    # corresponding list of lists of compensating cavities. Broken cavities
    # in a sublist are fixed together with the provided sublist of
    # compensating cavities.
    # example:
    # failed =
    #   [12, 14], -> two cavities that will be fixed together
    #   [155]     -> a single error, fixed after [12, 14] is dealt with
    #
    # manual_list =
    #   [8, 10, 16, 18],    -> compensate errors at idx 12 & 14
    #   [153, 157]          -> compensate error at 155
    #
    if wtf['strategy'] == 'manual':
        return _test_strategy_manual(wtf)

    # You must provide the number of compensating lattices per faulty
    # cavity. Close broken cavities are gathered and fixed together.
    if wtf['strategy'] == 'l neighboring lattices':
        return _test_strategy_l_neighboring_lattices(wtf)

    logging.error("The 'strategy' key did not match any authorized value "
                  + f"({wtf['strategy']}).")
    return False


def _test_strategy_k_out_of_n(wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for k out of n strategy."""
    if 'k' not in wtf.keys():
        logging.error("You must provide k, the number of compensating "
                      + "cavities per failed cavity.")
        return False

    try:
        wtf.getint('k')
    except ValueError:
        logging.error("k must be an integer.")
        return False

    return True


def _test_strategy_manual(wtf: configparser.SectionProxy) -> bool:
    """Even more specific test for manual strategy."""
    if 'manual list' not in wtf.keys():
        logging.error("You must provide a list of lists of compensating "
                      + "cavities corresponding to each list of failed "
                      + "cavities.")
        return False

    logging.info("You must ensure that all the elements in manual list are "
                 + "cavities.")
    return True


def _test_strategy_l_neighboring_lattices(wtf: configparser.SectionProxy
                                          ) -> bool:
    """Even more specific test for l neighboring lattices strategy."""
    if 'l' not in wtf.keys():
        logging.error("You must provide l, the number of compensating "
                      + "lattices.")
        return False

    try:
        wtf.getint('l')
    except ValueError:
        logging.error("l must be an integer.")
        return False

    return True


# =============================================================================
# Testing of wtf objective
# =============================================================================
def _test_objective(wtf: configparser.SectionProxy) -> bool:
    """Specific test for the key 'objective' of what_to_fit."""
    if 'objective' not in wtf.keys():
        logging.error("You must provide 'objective' to tell LightWin what it "
                      + "should fit.")
        return False

    l_obj = wtf.getliststr('objective')
    implemented = [
        'w_kin', 'phi_abs_array', 'mismatch factor',
        'eps_zdelta', 'beta_zdelta', 'gamma_zdelta', 'alpha_zdelta',
        'M_11', 'M_12', 'M_22', 'M_21']

    if not all(obj in implemented for obj in l_obj):
        logging.error("At least one objective was not recognized.")
        logging.info("""To add your own objective, make sure that:
                     1. it can be returned by the Accelerator.get() method;
                     2. it is present in the util.d_output dictionaries;
                     3. it is in the above 'implemented' dict.""")
        return False

    if 'scale objective' in wtf.keys():
        l_scales = wtf.getlistfloat('scale objective')
        if len(l_scales) != len(l_obj):
            logging.error("If you want to scale the objectives by a factor, "
                          + "you must provide a list of scale factors (one "
                          + "scale factor per objective.")
            return False
    else:
        l_scales = [1. for obj in l_obj]
        wtf['scale objective'] = ', '.join(map(str, l_scales))
        logging.warning("Scale of objectives not specified. Use default.")

    return True


# =============================================================================
# Testing of wtf optimisation method
# =============================================================================
def _test_opti_method(wtf: configparser.SectionProxy) -> bool:
    """Test the optimisation method."""
    if 'opti method' not in wtf.keys():
        logging.error("You must provide 'opti method' to tell LightWin what "
                      + "optimisation algorithm it should use.")
        return False

    implemented = ['least_squares', 'PSO']
    # TODO: specific testing for each method (look at the kwargs)
    if wtf['opti method'] not in implemented:
        logging.error("Algorithm not implemented.")
        return False
    return True


# =============================================================================
# Testing of wtf position
# =============================================================================
def _test_position(wtf: configparser.SectionProxy) -> bool:
    """Test where the objectives are evaluated."""
    if 'position' not in wtf.keys():
        logging.error("You must provide 'position' to tell LightWin where "
                      + "objectives should be evaluated.")
        return False

    implemented = [
        # End of last lattice with a compensating cavity
        'end_mod',
        # End of one lattice after last lattice with a compensating cavity
        '1_mod_after',
        # Both lattices
        'both']
    if wtf['position'] not in implemented:
        logging.error("Position not implemented.")
        return False
    return True


# =============================================================================
# Misc test
# =============================================================================
def _test_misc(wtf: configparser.SectionProxy) -> bool:
    """Some other tests."""
    if 'phi_s fit' not in wtf.keys():
        logging.error("Please explicitely precise if you want to fit synch "
                      + "phases (recommended for least squares, which do not "
                      + "handle constraints) or not (for algorithms that can "
                      + "handle it).")
        return False

    try:
        wtf.getboolean("phi_s fit")
    except ValueError:
        logging.error("Not a boolean.")
        return False
    return True

# =============================================================================
# Test TW
# =============================================================================
def _test_tw(tw: configparser.SectionProxy) -> bool:
    """Test for the TraceWin simulation parameters."""
    logging.warning("Not implemented.")
    return True

# =============================================================================
# Main func
# =============================================================================
if __name__ == '__main__':
    # Init paths
    CONFIG_PATH = 'jaea_default.ini'
    PROJECT_PATH = 'bla/'

    # Load config
    wtf = process_config(CONFIG_PATH, PROJECT_PATH, key_wtf='wtf.k_out_of_n')
    wtf = process_config(CONFIG_PATH, PROJECT_PATH,
                         key_wtf='wtf.l_neighboring_lattices')
    wtf = process_config(CONFIG_PATH, PROJECT_PATH, key_wtf='wtf.manual')

    # Save a copy
    # save_path = os.path.join(PROJECT_PATH, 'config.ini')
    # with open(save_path, 'w', encoding='utf-8') as configfile:
        # config.write(configfile)
